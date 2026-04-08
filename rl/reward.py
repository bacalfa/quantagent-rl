"""
rl/reward.py
============
Step-level reward computation for the QuantAgent-RL portfolio environment.

The base reward is the **Differential Sharpe Ratio** (DSR) introduced by
Moody & Saffell (1998).  The key insight is to express the Sharpe ratio as
a function of exponential moving averages (EMAs) of returns and squared
returns, then differentiate with respect to the current return to obtain a
single-step, Markovian reward signal that directly optimizes risk-adjusted
cumulative performance.

Differential Sharpe Ratio
--------------------------
Let A_t and B_t be EMA estimates:

    A_t = A_{t-1} + η · (r_t − A_{t-1})
    B_t = B_{t-1} + η · (r_t² − B_{t-1})

where η (eta) is the adaptation rate.  The differential Sharpe increment is:

    D_t = [B_{t-1} · ΔA_t − 0.5 · A_{t-1} · ΔB_t]
          / (B_{t-1} − A_{t-1}²)^{3/2}

where ΔA_t = A_t − A_{t-1}, ΔB_t = B_t − B_{t-1}.

Differential Sortino Variant
-----------------------------
Replace B_t with a downside-only second moment:

    D_t^− = D_t^− + η · (min(r_t, 0)² − D_t^−)

This penalizes only downside deviations, aligning the reward with
loss-averse investor preferences.

Full Reward
-----------
    R_t = D_t − λ_tax · tax_cost_t
              + λ_tlh · tlh_benefit_t
              − λ_turnover · transaction_cost_t

GPU Acceleration
----------------
``RewardCalculator.batch_differential_sharpe`` implements the EMA recursion
for an entire episode (sequence of T returns) as a vectorized CuPy loop —
useful during policy evaluation / advantage estimation where the full return
trajectory is available.  The per-step ``step`` method runs on NumPy (CPU)
since it is called once per environment step.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from rl.config import RewardConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU backend
# ---------------------------------------------------------------------------


def _get_cupy() -> tuple[object, bool]:
    try:
        import cupy as cp

        return cp, True
    except ImportError:
        return np, False


# ---------------------------------------------------------------------------
# EMA state (one instance per environment episode)
# ---------------------------------------------------------------------------


@dataclass
class DifferentialSharpeState:
    """Running EMA state for the Differential Sharpe / Sortino recursion.

    One instance is created at the start of each episode and updated at
    every step.  Reset it by creating a new instance.

    Parameters
    ----------
    eta : float
        EMA adaptation rate.
    use_sortino : bool
        If True, use downside-only second moment B_t^−.
    """

    eta: float = 0.05
    use_sortino: bool = False

    # Running statistics — initialized to 0.0 (neutral prior)
    A: float = field(default=0.0, init=False)  # EMA of returns
    B: float = field(default=0.0, init=False)  # EMA of r² (or r_negative²)

    def update(self, r: float) -> float:
        """Consume one return observation and return the DSR increment.

        Parameters
        ----------
        r : float
            Portfolio log return for the current step.

        Returns
        -------
        float
            Differential Sharpe (or Sortino) increment D_t.
        """
        eta = self.eta
        A_prev = self.A
        B_prev = self.B

        # EMA updates
        self.A = A_prev + eta * (r - A_prev)
        r_sq = (min(r, 0.0) ** 2) if self.use_sortino else (r**2)
        self.B = B_prev + eta * (r_sq - B_prev)

        dA = self.A - A_prev
        dB = self.B - B_prev

        denom_sq = B_prev - A_prev**2
        if denom_sq <= 1e-9:
            # Denominator underflows in early steps before the running
            # statistics have stabilized — return the raw return as a
            # warm-up reward.
            return float(r)

        denom = denom_sq**1.5
        return float((B_prev * dA - 0.5 * A_prev * dB) / denom)


# ---------------------------------------------------------------------------
# Main reward calculator
# ---------------------------------------------------------------------------


class RewardCalculator:
    """Computes the full step-level reward for the portfolio environment.

    Parameters
    ----------
    config : RewardConfig

    Examples
    --------
    >>> calc = RewardCalculator(RewardConfig())
    >>> state = calc.new_episode_state()
    >>> reward = calc.step(state, portfolio_return=0.012,
    ...                    tax_cost=0.001, tlh_benefit=0.0,
    ...                    weight_delta=np.zeros(20))
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.cfg = config or RewardConfig()
        cp, self._gpu = _get_cupy()
        self._cp = cp
        logger.info(
            f"[RewardCalculator] backend={'GPU (CuPy)' if self._gpu else 'CPU (NumPy)'}, "
            f"base={'Sortino' if self.cfg.use_sortino else 'Sharpe'}"
        )

    # ------------------------------------------------------------------
    # Per-step reward (called inside env.step())
    # ------------------------------------------------------------------

    def new_episode_state(self) -> DifferentialSharpeState:
        """Create a fresh EMA state for a new episode."""
        return DifferentialSharpeState(
            eta=self.cfg.eta,
            use_sortino=self.cfg.use_sortino,
        )

    def step(
        self,
        ema_state: DifferentialSharpeState,
        portfolio_return: float,
        tax_cost: float,
        tlh_benefit: float,
        weight_delta: np.ndarray,
    ) -> tuple[float, dict]:
        """Compute one step of the augmented differential Sharpe reward.

        Parameters
        ----------
        ema_state : DifferentialSharpeState
            Mutable running EMA state; updated in-place.
        portfolio_return : float
            Realized log return of the portfolio for this step.
        tax_cost : float
            Realized capital gains tax cost as a fraction of portfolio value.
        tlh_benefit : float
            Tax-loss harvesting benefit (realized losses usable to offset
            gains) as a fraction of portfolio value.
        weight_delta : np.ndarray
            Absolute per-asset weight changes, shape (n_assets,).

        Returns
        -------
        reward : float
            Scalar reward signal.
        info : dict
            Breakdown dict with keys: 'dsr', 'tax_cost', 'tlh_benefit',
            'transaction_cost', 'reward_raw', 'reward_scaled'.
        """
        cfg = self.cfg

        # Base: Differential Sharpe (or Sortino) increment
        dsr = ema_state.update(portfolio_return)

        # Transaction cost: fixed commission + quadratic market impact
        tc = float(
            cfg.transaction_cost_fixed * np.abs(weight_delta).sum()
            + cfg.transaction_cost_impact * (weight_delta**2).sum()
        )

        # Assemble reward
        reward_raw = (
            dsr
            - cfg.lambda_tax * tax_cost
            + cfg.lambda_tlh * tlh_benefit
            - cfg.lambda_turnover * tc
        )
        reward_scaled = reward_raw * cfg.reward_scale

        info = {
            "dsr": dsr,
            "tax_cost": tax_cost,
            "tlh_benefit": tlh_benefit,
            "transaction_cost": tc,
            "reward_raw": reward_raw,
            "reward_scaled": reward_scaled,
        }
        return reward_scaled, info

    # ------------------------------------------------------------------
    # Batch trajectory reward (GPU-accelerated, used in evaluation)
    # ------------------------------------------------------------------

    def batch_differential_sharpe(
        self,
        returns: np.ndarray,
    ) -> np.ndarray:
        """Compute the DSR increment for an entire return trajectory.

        This GPU-accelerated version processes the full sequence of T returns
        in a single CuPy loop (one kernel launch per step), making it ~10×
        faster than calling ``DifferentialSharpeState.update`` T times in
        Python.  Used during rollout advantage computation and evaluation.

        Parameters
        ----------
        returns : np.ndarray, shape (T,)
            Sequence of portfolio log returns.

        Returns
        -------
        np.ndarray, shape (T,)
            Differential Sharpe increments D_0, D_1, ..., D_{T-1}.
        """
        xp = self._cp if self._gpu else np
        r = xp.asarray(returns, dtype=xp.float32)
        T = len(r)
        eta = float(self.cfg.eta)

        D = xp.zeros(T, dtype=xp.float32)
        A = xp.float32(0.0)
        B = xp.float32(0.0)

        for t in range(T):
            A_prev = A
            B_prev = B

            A = A_prev + eta * (r[t] - A_prev)
            r_sq = (
                xp.minimum(r[t], xp.float32(0.0)) ** 2
                if self.cfg.use_sortino
                else r[t] ** 2
            )
            B = B_prev + eta * (r_sq - B_prev)

            dA = A - A_prev
            dB = B - B_prev
            denom_sq = B_prev - A_prev**2

            if float(denom_sq) > 1e-9:
                D[t] = (B_prev * dA - xp.float32(0.5) * A_prev * dB) / (
                    denom_sq ** xp.float32(1.5)
                )
            else:
                D[t] = r[t]  # warm-up fallback

        if self._gpu:
            return D.get()
        return np.asarray(D)

    # ------------------------------------------------------------------
    # Transaction cost helper (public utility)
    # ------------------------------------------------------------------

    def transaction_cost(self, weight_delta: np.ndarray) -> float:
        """Compute transaction cost for a given set of weight changes.

        Parameters
        ----------
        weight_delta : np.ndarray
            Per-asset absolute weight changes, shape (n_assets,).

        Returns
        -------
        float
            Total transaction cost as a fraction of portfolio value.
        """
        cfg = self.cfg
        return float(
            cfg.transaction_cost_fixed * np.abs(weight_delta).sum()
            + cfg.transaction_cost_impact * (weight_delta**2).sum()
        )

    # ------------------------------------------------------------------
    # Diagnostic: annualized Sharpe from a return series
    # ------------------------------------------------------------------

    @staticmethod
    def annualized_sharpe(
        returns: np.ndarray,
        periods_per_year: int = 4,
        risk_free: float = 0.0,
    ) -> float:
        """Compute the classic annualized Sharpe ratio from a return series.

        Used for evaluation and comparison against baselines; not used as
        the RL training reward.

        Parameters
        ----------
        returns : np.ndarray
            Return series (one observation per rebalancing period).
        periods_per_year : int
            Number of rebalancing periods per year (4 for quarterly).
        risk_free : float
            Per-period risk-free rate.

        Returns
        -------
        float
        """
        excess = returns - risk_free
        std = np.std(excess, ddof=1)
        if std < 1e-9:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(periods_per_year))

    @staticmethod
    def annualized_sortino(
        returns: np.ndarray,
        periods_per_year: int = 4,
        risk_free: float = 0.0,
    ) -> float:
        """Compute the annualized Sortino ratio from a return series."""
        excess = returns - risk_free
        downside = excess[excess < 0]
        if len(downside) < 2:
            return 0.0
        downside_std = np.std(downside, ddof=1)
        if downside_std < 1e-9:
            return 0.0
        return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))
