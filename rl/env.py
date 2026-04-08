"""
rl/env.py
=========
Custom Gymnasium environment for quarterly portfolio rebalancing.

MDP Design
----------
Each **step** represents one quarter-end rebalancing decision.

State  s_t :
    Flat vector assembled by ``StateBuilder`` from:
      - Quantitative features (returns, vol, momentum, factor betas, macro signals)
      - Forecasting features (GARCH vol forecast, HMM regime probs, FF betas)
      - Agent embedding (MarketBrief sentence-transformer vector)
      - Current portfolio weights (so the agent knows what it already holds)

Action a_t :
    Delta weights Δw ∈ ℝ^n, clipped to [−max_delta, +max_delta] per asset.
    The new weights are w_{t+1} = clip(w_t + Δw, w_min, w_max), then
    renormalized to sum to 1.  A no-trade threshold zeros out very small
    changes to simulate minimum trade sizes.

Reward R_t :
    Augmented Differential Sharpe Ratio (see ``rl/reward.py``):
      R_t = DSR(r_t) − λ_tax·tax_cost_t + λ_tlh·tlh_benefit_t
                      − λ_turnover·transaction_cost_t

Episode :
    One episode covers the full training or test date sequence (quarterly
    steps).  Episodes are short — typically 12–40 steps — so PPO's rollout
    buffer accumulates multiple episodes before each update.

Tax Model (Simplified US Federal)
----------------------------------
Long-term rate  : 15 %  (held > 4 quarters)
Short-term rate : 37 %  (held ≤ 4 quarters)
Cost basis      : FIFO lot tracking per asset
Wash-sale rule  : not enforced (documented limitation)
"""

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYMNASIUM = True
except ImportError:
    _HAS_GYMNASIUM = False

    # Minimal stubs so the module can be imported without gymnasium installed.
    # Full functionality requires: pip install gymnasium
    class _FakeGym:
        class Env:
            metadata: dict = {}
            observation_space: object = None
            action_space: object = None

            def reset(self, **kw):
                return np.zeros(1), {}

            def step(self, a):
                return np.zeros(1), 0.0, True, False, {}

            def render(self):
                pass

        class spaces:
            class Box:
                def __init__(self, **kw):
                    pass

                def sample(self):
                    return np.zeros(1)

    gym = _FakeGym()
    spaces = gym.spaces

from rl.config import RLConfig
from rl.reward import DifferentialSharpeState, RewardCalculator
from rl.state import StateBuilder

logger = logging.getLogger(__name__)

# Tax rates (simplified US federal, no state tax)
_LONG_TERM_RATE = 0.15  # held > 4 quarters
_SHORT_TERM_RATE = 0.37  # held ≤ 4 quarters
_LONG_TERM_HOLD = 4  # quarters for long-term treatment


# ---------------------------------------------------------------------------
# Lot tracker
# ---------------------------------------------------------------------------


@dataclass
class Lot:
    """A single tax lot representing one purchase of an asset.

    Parameters
    ----------
    cost_basis : float
        Purchase price per unit (normalized to portfolio fraction).
    quarters_held : int
        Number of rebalancing quarters the lot has been held.
    size : float
        Fractional portfolio weight held in this lot.
    """

    cost_basis: float
    quarters_held: int = 0
    size: float = 0.0

    @property
    def is_long_term(self) -> bool:
        """True if the lot qualifies for the long-term capital gains rate."""
        return self.quarters_held > _LONG_TERM_HOLD


class LotTracker:
    """FIFO lot-level cost basis tracker for all assets.

    Tracks purchase price and holding period for each position.  When assets
    are sold, FIFO matching is applied: the oldest lots are liquidated first.

    Parameters
    ----------
    tickers : list[str]
        Asset universe.
    """

    def __init__(self, tickers: list[str]) -> None:
        self.tickers = tickers
        # deque of Lot objects per ticker, ordered oldest→newest
        self._lots: dict[str, deque[Lot]] = {t: deque() for t in tickers}

    def buy(self, ticker: str, amount: float, price: float) -> None:
        """Record a purchase as a new lot.

        Parameters
        ----------
        ticker : str
        amount : float
            Weight fraction purchased.
        price : float
            Price per unit (use 1.0 for weight-based normalization).
        """
        if amount <= 1e-6:
            return
        self._lots[ticker].append(Lot(cost_basis=price, size=amount))

    def sell_fifo(
        self, ticker: str, amount: float, current_price: float
    ) -> tuple[float, float]:
        """Sell ``amount`` of ``ticker`` using FIFO lot matching.

        Parameters
        ----------
        ticker : str
        amount : float
            Weight fraction to sell.
        current_price : float
            Current price per unit used to compute gain/loss.

        Returns
        -------
        tax_cost : float
            Realized capital gains tax (short-term or long-term rate applied
            per lot based on holding period).
        tlh_benefit : float
            Realized losses (available to offset other gains).
        """
        lots = self._lots[ticker]
        remaining = amount
        tax_cost = 0.0
        tlh_benefit = 0.0

        while remaining > 1e-7 and lots:
            lot = lots[0]
            sell = min(lot.size, remaining)
            gain = (current_price - lot.cost_basis) * sell

            if gain > 0:
                rate = _LONG_TERM_RATE if lot.is_long_term else _SHORT_TERM_RATE
                tax_cost += gain * rate
            else:
                tlh_benefit += abs(gain)

            lot.size -= sell
            remaining -= sell
            if lot.size < 1e-7:
                lots.popleft()

        return float(tax_cost), float(tlh_benefit)

    def age_one_quarter(self) -> None:
        """Increment quarters_held for every open lot by one."""
        for lots in self._lots.values():
            for lot in lots:
                lot.quarters_held += 1

    def reset(self) -> None:
        """Clear all lots (call at the start of each episode)."""
        for ticker in self.tickers:
            self._lots[ticker] = deque()


# ---------------------------------------------------------------------------
# PortfolioEnv
# ---------------------------------------------------------------------------


class PortfolioEnv(gym.Env):
    """Quarterly portfolio rebalancing environment.

    Inherits from ``gymnasium.Env`` and is compatible with Stable-Baselines3.

    Parameters
    ----------
    tickers : list[str]
        Asset universe (ordered consistently with all input DataFrames).
    rebalance_dates : pd.DatetimeIndex
        Quarter-end dates defining the MDP steps for this episode.
    prices : pd.DataFrame
        Adjusted close prices.  Index = DatetimeIndex, columns = tickers.
    quant_df : pd.DataFrame
        Quantitative feature matrix (output of StateBuilder-normalized data).
    forecast_df : pd.DataFrame
        Forecasting feature matrix (from ForecastBundle.rl_state_extension).
    embed_df : pd.DataFrame
        Agent embedding matrix (from AgentBundle.embeddings).
    state_builder : StateBuilder
        Pre-fitted state assembler (fitted on training data only).
    config : RLConfig
    initial_weights : np.ndarray | None
        Starting portfolio weights.  Defaults to equal weight.
    sector_map : dict[str, list[str]] | None
        Maps sector names to lists of tickers for sector-weight constraints.

    Notes
    -----
    The observation space dimension is computed dynamically from the input
    DataFrames on construction.  Downstream code (e.g. SB3's policy network)
    reads ``env.observation_space.shape[0]`` to determine network input size.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        tickers: list[str],
        rebalance_dates: pd.DatetimeIndex,
        prices: pd.DataFrame,
        quant_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        embed_df: pd.DataFrame,
        state_builder: StateBuilder,
        config: RLConfig | None = None,
        initial_weights: np.ndarray | None = None,
        sector_map: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()

        self.tickers = tickers
        self.n_assets = len(tickers)
        self.rebalance_dates = list(rebalance_dates)
        self.prices = prices
        self.quant_df = quant_df
        self.forecast_df = forecast_df
        self.embed_df = embed_df
        self.state_builder = state_builder
        self.cfg = config or RLConfig()
        self.sector_map = sector_map or {}

        # Initial weights
        self._initial_weights = (
            initial_weights.copy()
            if initial_weights is not None
            else np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float32)
        )

        # Reward calculator
        self._reward_calc = RewardCalculator(self.cfg.reward)

        # Lot tracker for tax computation
        self._lot_tracker = LotTracker(tickers)

        # ── Spaces ────────────────────────────────────────────────────
        obs_dim = state_builder.obs_dim(
            quant_df, forecast_df, embed_df, include_weights=True
        )
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32
        )
        # Delta-weight actions: one continuous value per asset in [−0.5, 0.5]
        max_delta = self.cfg.constraints.max_turnover
        self.action_space = spaces.Box(
            low=-max_delta, high=max_delta, shape=(self.n_assets,), dtype=np.float32
        )

        # Episode state (reset on each episode)
        self._step_idx: int = 0
        self._weights: np.ndarray = self._initial_weights.copy()
        self._portfolio_val: float = 1.0
        self._ema_state: DifferentialSharpeState | None = None
        self._episode_log: list[dict] = []

        logger.info(
            f"[PortfolioEnv] n_assets={self.n_assets}, "
            f"obs_dim={obs_dim}, "
            f"n_steps={len(self.rebalance_dates)}"
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to the beginning of the episode."""
        super().reset(seed=seed)

        self._step_idx = 0
        self._weights = self._initial_weights.copy()
        self._portfolio_val = 1.0
        self._ema_state = self._reward_calc.new_episode_state()
        self._lot_tracker.reset()
        self._episode_log = []

        # Initialize lot tracker with equal-weight starting position
        for i, ticker in enumerate(self.tickers):
            self._lot_tracker.buy(ticker, self._weights[i], price=1.0)

        obs = self._get_obs()
        info = {"step": 0, "n_steps": len(self.rebalance_dates)}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one rebalancing decision and advance the environment.

        Parameters
        ----------
        action : np.ndarray, shape (n_assets,)
            Proposed delta-weight vector from the policy.

        Returns
        -------
        obs : np.ndarray
            Next observation vector.
        reward : float
        terminated : bool
            True when the final step of the episode is reached.
        truncated : bool
            Always False (no time limit beyond the fixed episode length).
        info : dict
        """
        date = self.rebalance_dates[self._step_idx]

        # 1. Compute the portfolio return that occurred over this quarter
        portfolio_return = self._compute_portfolio_return(date)

        # 2. Compute new weights from the action
        old_weights = self._weights.copy()
        new_weights = self._apply_action(action)

        # 3. Compute weight deltas (for turnover and transaction cost)
        delta = new_weights - old_weights

        # 4. Compute tax cost and TLH benefit from sells
        tax_cost, tlh_benefit = self._execute_trades(old_weights, new_weights, date)

        # 5. Update portfolio state
        self._weights = new_weights
        self._portfolio_val *= 1.0 + portfolio_return
        self._lot_tracker.age_one_quarter()

        # 6. Compute reward
        reward, reward_info = self._reward_calc.step(
            ema_state=self._ema_state,
            portfolio_return=portfolio_return,
            tax_cost=tax_cost,
            tlh_benefit=tlh_benefit,
            weight_delta=np.abs(delta),
        )

        self._step_idx += 1
        terminated = self._step_idx >= len(self.rebalance_dates)

        info = {
            "date": str(date.date()),
            "step": self._step_idx,
            "portfolio_return": portfolio_return,
            "portfolio_value": self._portfolio_val,
            "weights": new_weights.copy(),
            "turnover": float(np.abs(delta).sum()),
            **reward_info,
        }
        self._episode_log.append(info)

        obs = self._get_obs() if not terminated else self._get_obs()
        return obs, reward, terminated, False, info

    def render(self, mode: str = "human") -> None:
        """Print a one-line summary of the current step."""
        if self._step_idx > 0:
            log = self._episode_log[-1]
            print(
                f"Step {log['step']:3d} | {log['date']} | "
                f"ret={log['portfolio_return']:+.4f} | "
                f"val={log['portfolio_value']:.4f} | "
                f"DSR={log['dsr']:+.4f} | "
                f"R={log['reward_scaled']:+.4f}"
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector for the current step."""
        idx = min(self._step_idx, len(self.rebalance_dates) - 1)
        date = self.rebalance_dates[idx]
        return self.state_builder.build(
            date=date,
            quant_df=self.quant_df,
            forecast_df=self.forecast_df,
            embed_df=self.embed_df,
            portfolio_weights=self._weights,
        )

    def _compute_portfolio_return(self, date: pd.Timestamp) -> float:
        """Compute the equal-weight portfolio return for the quarter ending at date."""
        try:
            prev_idx = self._step_idx - 1
            if prev_idx < 0:
                return 0.0
            prev_date = self.rebalance_dates[prev_idx]

            # Slice prices between previous and current quarter-end
            mask = (self.prices.index > prev_date) & (self.prices.index <= date)
            p_slice = self.prices[mask]
            if p_slice.empty or len(p_slice) < 2:
                return 0.0

            # Quarter return for each asset
            asset_returns = (
                (p_slice.iloc[-1] / p_slice.iloc[0] - 1.0)
                .reindex(self.tickers)
                .fillna(0.0)
                .values.astype(np.float32)
            )

            # Weight-average portfolio return
            return float(np.dot(self._weights, asset_returns))
        except Exception as exc:
            logger.warning(f"[PortfolioEnv] Portfolio return computation failed: {exc}")
            return 0.0

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        """Map raw action to valid portfolio weights.

        Steps:
          1. Apply no-trade threshold (zero out tiny changes).
          2. Add delta to current weights.
          3. Clip to [min_position, max_position].
          4. Apply sector-weight constraints.
          5. Renormalize to sum to 1.
        """
        con = self.cfg.constraints
        delta = action.copy().astype(np.float32)

        # No-trade threshold
        delta[np.abs(delta) < con.no_trade_threshold] = 0.0

        # Clip total turnover
        total_turnover = np.abs(delta).sum()
        if total_turnover > con.max_turnover:
            delta = delta * (con.max_turnover / total_turnover)

        # New raw weights
        w = self._weights + delta
        w = np.clip(w, con.min_position, con.max_position)

        # Sector constraints
        w = self._apply_sector_constraints(w)

        # Renormalize
        total = w.sum()
        if total < 1e-6:
            w = self._initial_weights.copy()
        else:
            w = w / total

        return w.astype(np.float32)

    def _apply_sector_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Clip sector-level concentration and rescale within each sector."""
        if not self.sector_map:
            return weights

        ticker_idx = {t: i for i, t in enumerate(self.tickers)}
        max_sw = self.cfg.constraints.max_sector_weight
        w = weights.copy()

        for sector, members in self.sector_map.items():
            idxs = [ticker_idx[t] for t in members if t in ticker_idx]
            if not idxs:
                continue
            sector_total = w[idxs].sum()
            if sector_total > max_sw:
                scale = max_sw / sector_total
                w[idxs] *= scale

        return w

    def _execute_trades(
        self,
        old_w: np.ndarray,
        new_w: np.ndarray,
        date: pd.Timestamp,
    ) -> tuple[float, float]:
        """Compute tax cost and TLH benefit from weight changes.

        For each asset, sells are processed through the lot tracker (FIFO),
        and buys create new lots at the current price.

        Parameters
        ----------
        old_w, new_w : np.ndarray
            Portfolio weights before and after rebalancing.
        date : pd.Timestamp
            Rebalancing date (used to look up current prices).

        Returns
        -------
        tax_cost : float
        tlh_benefit : float
        """
        total_tax = 0.0
        total_tlh = 0.0

        for i, ticker in enumerate(self.tickers):
            delta = new_w[i] - old_w[i]
            if abs(delta) < 1e-7:
                continue

            # Current price (normalized: use 1.0 as default)
            try:
                price_series = self.prices[ticker]
                candidates = price_series[price_series.index <= date]
                current_price = (
                    float(candidates.iloc[-1]) if not candidates.empty else 1.0
                )
            except (KeyError, IndexError):
                current_price = 1.0

            if delta < 0:
                # Selling — triggers potential gain/loss
                tax, tlh = self._lot_tracker.sell_fifo(ticker, -delta, current_price)
                total_tax += tax
                total_tlh += tlh
            else:
                # Buying — creates a new lot
                self._lot_tracker.buy(ticker, delta, current_price)

        return total_tax, total_tlh

    # ------------------------------------------------------------------
    # Episode summary
    # ------------------------------------------------------------------

    def episode_summary(self) -> dict:
        """Return performance metrics for the completed episode.

        Useful for logging and evaluation.  Should be called after
        ``terminated=True`` has been returned by ``step()``.

        Returns
        -------
        dict with keys: total_return, annualized_return, sharpe, sortino,
        max_drawdown, total_turnover, total_tax_cost, n_steps.
        """
        if not self._episode_log:
            return {}

        returns = np.array([s["portfolio_return"] for s in self._episode_log])
        values = np.array([s["portfolio_value"] for s in self._episode_log])
        turnovers = np.array([s["turnover"] for s in self._episode_log])
        tax_costs = np.array([s["tax_cost"] for s in self._episode_log])

        total_return = float(values[-1] / values[0] - 1.0) if len(values) > 0 else 0.0
        n_years = len(returns) / 4.0  # quarterly steps
        ann_return = float((1.0 + total_return) ** (1.0 / max(n_years, 0.25)) - 1.0)
        sharpe = RewardCalculator.annualized_sharpe(returns)
        sortino = RewardCalculator.annualized_sortino(returns)

        # Max drawdown
        cum = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / (peak + 1e-9)
        max_dd = float(dd.min())

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "total_turnover": float(turnovers.sum()),
            "total_tax_cost": float(tax_costs.sum()),
            "n_steps": len(self._episode_log),
        }
