"""
rl/pipeline.py
==============
RLPipeline — walk-forward training and evaluation orchestrator for QuantAgent-RL.

Ties together:
  StateBuilder  → builds the RL observation vector
  PortfolioEnv  → Gymnasium environment
  PPOAgent      → Stable-Baselines3 PPO with GPU policy
  RewardCalculator → differential Sharpe reward
  RecencyWeightedSampler → recency-biased episode selection

Walk-Forward Protocol
---------------------
1. For each fold produced by the DataPipeline (expanding training window):
   a. Construct a StateBuilder and fit scalers on training data only.
   b. Build a PortfolioEnv over the training rebalance dates.
   c. Train the PPO agent (fresh on fold 0; warm-start on subsequent folds).
   d. Evaluate the trained agent on the test rebalance dates.
   e. Store the fold result in an RLFoldResult dataclass.
2. Aggregate fold results into a full backtest summary.

Baselines
---------
Each fold evaluation computes the PPO agent's performance against two
baselines on the same test dates:
  - Equal-weight (1/N) portfolio, quarterly rebalanced.
  - Buy-and-hold initial weights (no rebalancing).

Both baselines are evaluated in the same PortfolioEnv (same reward function)
so comparisons are fair.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from agents import AgentBundle
from data import DataPipeline, WalkForwardFold
from forecasting import ForecastBundle
from rl.agent import PPOAgent
from rl.config import RLConfig
from rl.env import PortfolioEnv
from rl.reward import RewardCalculator
from rl.state import StateBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class FoldMetrics:
    """Performance metrics for one agent or baseline over one fold.

    Attributes
    ----------
    total_return : float
    annualized_return : float
    sharpe : float
    sortino : float
    max_drawdown : float
    total_turnover : float
    total_tax_cost : float
    n_steps : int
    quarterly_returns : list[float]
    quarterly_weights : list[np.ndarray]
    """

    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    total_turnover: float = 0.0
    total_tax_cost: float = 0.0
    n_steps: int = 0
    quarterly_returns: list[float] = field(default_factory=list)
    quarterly_weights: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def from_episode_log(cls, log: list[dict]) -> "FoldMetrics":
        """Construct metrics from the environment's episode log."""
        if not log:
            return cls()
        returns = [s["portfolio_return"] for s in log]
        values = [s["portfolio_value"] for s in log]
        turnovers = [s["turnover"] for s in log]
        tax_costs = [s["tax_cost"] for s in log]
        weights = [s["weights"] for s in log]

        r_arr = np.array(returns, dtype=np.float32)
        v_arr = np.array(values, dtype=np.float32)
        n = len(r_arr)

        total_ret = float(v_arr[-1] / v_arr[0] - 1.0) if n > 0 else 0.0
        n_years = n / 4.0
        ann_ret = float((1.0 + total_ret) ** (1.0 / max(n_years, 0.25)) - 1.0)
        sharpe = RewardCalculator.annualized_sharpe(r_arr)
        sortino = RewardCalculator.annualized_sortino(r_arr)

        cum = np.cumprod(1.0 + r_arr)
        peak = np.maximum.accumulate(cum)
        max_dd = float(((cum - peak) / (peak + 1e-9)).min())

        return cls(
            total_return=total_ret,
            annualized_return=ann_ret,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_dd,
            total_turnover=float(sum(turnovers)),
            total_tax_cost=float(sum(tax_costs)),
            n_steps=n,
            quarterly_returns=returns,
            quarterly_weights=weights,
        )


@dataclass
class RLFoldResult:
    """All results for one walk-forward fold.

    Attributes
    ----------
    fold_idx : int
    train_dates, test_dates : pd.DatetimeIndex
    ppo_metrics : FoldMetrics
        Out-of-sample performance of the trained PPO agent.
    equal_weight_metrics : FoldMetrics
        Baseline: equal-weight (1/N) portfolio.
    hold_metrics : FoldMetrics
        Baseline: buy-and-hold from training-end weights.
    train_metrics : FoldMetrics
        In-sample training performance (last training episode).
    agent_path : str
        Path to the saved PPO model checkpoint.
    """

    fold_idx: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    ppo_metrics: FoldMetrics = field(default_factory=FoldMetrics)
    equal_weight_metrics: FoldMetrics = field(default_factory=FoldMetrics)
    hold_metrics: FoldMetrics = field(default_factory=FoldMetrics)
    train_metrics: FoldMetrics = field(default_factory=FoldMetrics)
    agent_path: str = ""

    def __repr__(self) -> str:
        return (
            f"RLFoldResult(fold={self.fold_idx}, "
            f"ppo_sharpe={self.ppo_metrics.sharpe:.3f}, "
            f"ew_sharpe={self.equal_weight_metrics.sharpe:.3f}, "
            f"ppo_ret={self.ppo_metrics.total_return:.3f})"
        )


# ---------------------------------------------------------------------------
# RLPipeline
# ---------------------------------------------------------------------------


class RLPipeline:
    """Walk-forward RL training and evaluation orchestrator.

    Parameters
    ----------
    config : RLConfig
    checkpoint_dir : str
        Directory for saving trained PPO model checkpoints.

    Examples
    --------
    >>> pipeline = RLPipeline(RLConfig())
    >>> results = pipeline.run(data_fold, forecast_bundle, agent_bundle)
    """

    def __init__(
        self,
        config: RLConfig | None = None,
        checkpoint_dir: str = "checkpoints/rl",
    ) -> None:
        self.cfg = config or RLConfig()
        self.checkpoint_dir = Path(checkpoint_dir)
        self._agent: PPOAgent | None = None  # warm-started across folds

    # ------------------------------------------------------------------
    # Single fold
    # ------------------------------------------------------------------

    def run_fold(
        self,
        fold: WalkForwardFold,
        forecast_bundle: ForecastBundle | None = None,
        agent_bundle: AgentBundle | None = None,
        sector_map: dict[str, list[str]] | None = None,
        warm_start: bool = False,
    ) -> RLFoldResult:
        """Train and evaluate the PPO agent on one walk-forward fold.

        Parameters
        ----------
        fold : WalkForwardFold
            Output of DataPipeline.get_fold(i).
        forecast_bundle : ForecastBundle | None
            Output of ForecastingPipeline.run_fold(fold).
        agent_bundle : AgentBundle | None
            Output of AgentPipeline.run_fold(fold).
        sector_map : dict[str, list[str]] | None
            Sector → ticker mapping for sector-weight constraints.
        warm_start : bool
            If True and a previous agent exists, fine-tune rather than
            retrain from scratch.

        Returns
        -------
        RLFoldResult
        """
        fold_idx = fold.fold_idx
        logger.info(
            f"[RLPipeline] Fold {fold_idx}: "
            f"train {fold.train_start.date()} → {fold.train_end.date()}, "
            f"test {fold.test_start.date()} → {fold.test_end.date()}"
        )

        # ── Assemble DataFrames ──────────────────────────────────────
        quant_train, quant_test = self._split_quant(fold)
        fc_train, fc_test = self._split_forecast(fold, forecast_bundle)
        emb_train, emb_test = self._split_embeddings(fold, agent_bundle)

        # ── Fit state builder (training data only) ───────────────────
        sb = StateBuilder(tickers=fold.tickers)
        sb.fit(quant_train, fc_train, emb_train)

        # ── Build training environment ────────────────────────────────
        train_env = PortfolioEnv(
            tickers=fold.tickers,
            rebalance_dates=fold.train_dates,
            prices=fold.train_prices,
            quant_df=pd.concat([quant_train, quant_test]),
            forecast_df=pd.concat([fc_train, fc_test])
            if not fc_test.empty
            else fc_train,
            embed_df=pd.concat([emb_train, emb_test])
            if not emb_test.empty
            else emb_train,
            state_builder=sb,
            config=self.cfg,
            sector_map=sector_map,
        )

        # ── Train PPO ────────────────────────────────────────────────
        obs_dim = train_env.observation_space.shape[0]
        if self._agent is None or not warm_start:
            self._agent = PPOAgent(
                obs_dim=obs_dim,
                n_assets=len(fold.tickers),
                config=self.cfg,
                fold_idx=fold_idx,
            )
        else:
            self._agent.fold_idx = fold_idx

        self._agent.train(
            env=train_env,
            warm_start=warm_start and self._agent._is_fitted,
        )

        # ── Collect training metrics (last episode) ───────────────────
        train_obs, _ = train_env.reset()
        done = False
        while not done:
            action = self._agent.predict(train_obs)
            train_obs, _, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
        train_metrics = FoldMetrics.from_episode_log(train_env._episode_log)

        # ── Save checkpoint ──────────────────────────────────────────
        ckpt_path = self.checkpoint_dir / f"fold_{fold_idx}" / "ppo_agent"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self._agent.save(ckpt_path)

        # ── Build test environment ────────────────────────────────────
        test_prices = pd.concat([fold.train_prices, fold.test_prices])
        test_env = PortfolioEnv(
            tickers=fold.tickers,
            rebalance_dates=fold.test_dates,
            prices=test_prices,
            quant_df=pd.concat([quant_train, quant_test]),
            forecast_df=pd.concat([fc_train, fc_test])
            if not fc_test.empty
            else fc_train,
            embed_df=pd.concat([emb_train, emb_test])
            if not emb_test.empty
            else emb_train,
            state_builder=sb,
            config=self.cfg,
            sector_map=sector_map,
        )

        # ── Evaluate PPO ──────────────────────────────────────────────
        ppo_metrics = self._evaluate_agent(test_env, self._agent)

        # ── Evaluate baselines ────────────────────────────────────────
        ew_metrics = self._evaluate_equal_weight(test_env)
        hold_metrics = self._evaluate_hold(test_env, fold.tickers)

        result = RLFoldResult(
            fold_idx=fold_idx,
            train_dates=fold.train_dates,
            test_dates=fold.test_dates,
            ppo_metrics=ppo_metrics,
            equal_weight_metrics=ew_metrics,
            hold_metrics=hold_metrics,
            train_metrics=train_metrics,
            agent_path=str(ckpt_path) + ".zip",
        )
        logger.info(f"[RLPipeline] Fold {fold_idx}: {result}")
        return result

    # ------------------------------------------------------------------
    # All folds
    # ------------------------------------------------------------------

    def run_all_folds(
        self,
        data_pipeline: DataPipeline,
        forecast_bundles: list[ForecastBundle] | None = None,
        agent_bundles: list[AgentBundle] | None = None,
        sector_map: dict[str, list[str]] | None = None,
    ) -> list[RLFoldResult]:
        """Run the full walk-forward training and evaluation loop.

        Parameters
        ----------
        data_pipeline : DataPipeline
        forecast_bundles : list[ForecastBundle] | None
        agent_bundles : list[AgentBundle] | None
        sector_map : dict[str, list[str]] | None

        Returns
        -------
        list[RLFoldResult]
        """
        results = []
        for i in range(data_pipeline.n_folds):
            fold = data_pipeline.get_fold(i)
            fb = forecast_bundles[i] if forecast_bundles else None
            ab = agent_bundles[i] if agent_bundles else None
            warm = i > 0  # warm-start after first fold

            result = self.run_fold(
                fold=fold,
                forecast_bundle=fb,
                agent_bundle=ab,
                sector_map=sector_map,
                warm_start=warm,
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # DataFrame summary
    # ------------------------------------------------------------------

    @staticmethod
    def summary_dataframe(results: list[RLFoldResult]) -> pd.DataFrame:
        """Assemble a multi-strategy comparison DataFrame from fold results.

        Returns
        -------
        pd.DataFrame
            Index = fold index.  Columns = metrics × strategy
            (PPO, equal_weight, hold).
        """
        rows = []
        for r in results:
            for label, m in [
                ("ppo", r.ppo_metrics),
                ("equal_weight", r.equal_weight_metrics),
                ("hold", r.hold_metrics),
            ]:
                rows.append(
                    {
                        "fold": r.fold_idx,
                        "strategy": label,
                        "total_return": m.total_return,
                        "annualized_return": m.annualized_return,
                        "sharpe": m.sharpe,
                        "sortino": m.sortino,
                        "max_drawdown": m.max_drawdown,
                        "total_turnover": m.total_turnover,
                        "total_tax_cost": m.total_tax_cost,
                        "n_steps": m.n_steps,
                    }
                )
        return pd.DataFrame(rows).set_index("fold")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _split_quant(fold: WalkForwardFold) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extract quantitative feature DataFrames from a fold."""
        train = fold.train_state_matrix
        test = fold.test_state_matrix
        # Flatten MultiIndex columns if present
        if isinstance(train.columns, pd.MultiIndex):
            train = train.copy()
            test = test.copy()
            train.columns = [
                "_".join(str(c) for c in col).strip("_") for col in train.columns
            ]
            test.columns = [
                "_".join(str(c) for c in col).strip("_") for col in test.columns
            ]
        return train, test

    @staticmethod
    def _split_forecast(
        fold: WalkForwardFold,
        bundle: ForecastBundle | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extract forecasting feature DataFrames from a ForecastBundle."""
        if bundle is None:
            empty = pd.DataFrame()
            return empty, empty
        ext = bundle.rl_state_extension
        train = ext[ext.index <= fold.train_end]
        test = ext[(ext.index > fold.train_end) & (ext.index <= fold.test_end)]
        return train, test

    @staticmethod
    def _split_embeddings(
        fold: WalkForwardFold,
        bundle: AgentBundle | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extract embedding DataFrames from an AgentBundle."""
        if bundle is None:
            empty = pd.DataFrame()
            return empty, empty
        emb = bundle.embeddings
        train = emb[emb.index <= fold.train_end]
        test = emb[(emb.index > fold.train_end) & (emb.index <= fold.test_end)]
        return train, test

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_agent(env: PortfolioEnv, agent: PPOAgent) -> FoldMetrics:
        """Run the trained agent deterministically on the test environment."""
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        return FoldMetrics.from_episode_log(env._episode_log)

    @staticmethod
    def _evaluate_equal_weight(env: PortfolioEnv) -> FoldMetrics:
        """Run a simple equal-weight (1/N) rebalancing baseline."""
        n = env.n_assets
        ew = np.full(n, 1.0 / n, dtype=np.float32)
        obs, _ = env.reset()
        done = False
        while not done:
            # Compute delta from current weights to equal weight
            delta = ew - env._weights
            obs, _, terminated, truncated, _ = env.step(delta)
            done = terminated or truncated
        return FoldMetrics.from_episode_log(env._episode_log)

    @staticmethod
    def _evaluate_hold(env: PortfolioEnv, tickers: list[str]) -> FoldMetrics:
        """Run a buy-and-hold baseline (zero delta actions throughout)."""
        obs, _ = env.reset()
        done = False
        no_action = np.zeros(env.n_assets, dtype=np.float32)
        while not done:
            obs, _, terminated, truncated, _ = env.step(no_action)
            done = terminated or truncated
        return FoldMetrics.from_episode_log(env._episode_log)
