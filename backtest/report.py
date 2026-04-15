"""
backtest/report.py
==================
Result containers for the QuantAgent-RL backtest module.

``BacktestReport`` is the primary output of ``BacktestEngine.run()``.
It holds per-fold and aggregate metrics for all strategies, rolling metric
time-series for visualization, and convenience methods for exporting
to DataFrames and CSV files.

``TearsheetData`` is a thin container holding the data needed to render
a strategy tear sheet (cumulative returns, drawdown path, rolling Sharpe,
rolling alpha) so the demo notebook and any future dashboard can consume
it without re-running the engine.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# StrategyReturns — minimal input contract
# ---------------------------------------------------------------------------


@dataclass
class StrategyReturns:
    """Per-period return series and associated metadata for one strategy.

    Parameters
    ----------
    name : str
        Human-readable strategy label (e.g. 'PPO', 'equal_weight').
    returns : np.ndarray, shape (T,)
        Per-period portfolio returns.
    dates : pd.DatetimeIndex, shape (T,)
        Date corresponding to each return observation.
    weights : list[np.ndarray] | None
        Sequence of per-period portfolio weight vectors (one per period).
    tax_costs : np.ndarray | None, shape (T,)
        Per-period realized capital gains tax costs.
    turnovers : np.ndarray | None, shape (T,)
        Per-period one-way turnover (sum of |Δw|).
    fold_idx : int | None
        Walk-forward fold the data came from.
    """

    name: str
    returns: np.ndarray
    dates: pd.DatetimeIndex
    weights: list[np.ndarray] | None = None
    tax_costs: np.ndarray | None = None
    turnovers: np.ndarray | None = None
    fold_idx: int | None = None

    def __len__(self) -> int:
        return len(self.returns)

    def __repr__(self) -> str:
        return (
            f"StrategyReturns(name='{self.name}', "
            f"T={len(self.returns)}, "
            f"fold={self.fold_idx})"
        )


# ---------------------------------------------------------------------------
# TearsheetData
# ---------------------------------------------------------------------------


@dataclass
class TearsheetData:
    """Time-series data required to render a single-strategy tear sheet.

    All arrays have the same length T (number of rebalancing periods).

    Parameters
    ----------
    strategy_name : str
    dates : pd.DatetimeIndex
    cumulative_returns : np.ndarray
        Cumulative return index (1.0 = starting value).
    drawdown : np.ndarray
        Drawdown path (values ≤ 0).
    rolling_sharpe : np.ndarray
        Rolling Sharpe ratio (NaN for early periods).
    rolling_alpha : np.ndarray | None
        Rolling annualized Jensen's alpha vs benchmark.
    rolling_volatility : np.ndarray
        Rolling annualized volatility.
    benchmark_cumulative : np.ndarray | None
        Benchmark cumulative return index for comparison.
    scalar_metrics : dict[str, float]
        Full-period scalar metrics dictionary.
    """

    strategy_name: str
    dates: pd.DatetimeIndex
    cumulative_returns: np.ndarray
    drawdown: np.ndarray
    rolling_sharpe: np.ndarray
    rolling_alpha: np.ndarray | None = None
    rolling_volatility: np.ndarray | None = None
    benchmark_cumulative: np.ndarray | None = None
    scalar_metrics: dict[str, float] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all time-series columns as a tidy DataFrame."""
        cols: dict[str, np.ndarray] = {
            "cumulative_return": self.cumulative_returns,
            "drawdown": self.drawdown,
            "rolling_sharpe": self.rolling_sharpe,
        }
        if self.rolling_alpha is not None:
            cols["rolling_alpha"] = self.rolling_alpha
        if self.rolling_volatility is not None:
            cols["rolling_volatility"] = self.rolling_volatility
        if self.benchmark_cumulative is not None:
            cols["benchmark_cumulative"] = self.benchmark_cumulative

        df = pd.DataFrame(cols, index=self.dates)
        df.index.name = "date"
        return df


# ---------------------------------------------------------------------------
# BacktestReport
# ---------------------------------------------------------------------------


@dataclass
class BacktestReport:
    """Complete walk-forward backtest results.

    Produced by ``BacktestEngine.run()``.  Contains both fold-level and
    aggregate (all-folds-concatenated) metrics for every strategy, plus
    per-strategy tear sheet data for visualization.

    Parameters
    ----------
    strategies : list[str]
        Ordered list of strategy names included in the report.
    fold_metrics : dict[str, list[dict[str, float]]]
        Outer key = strategy name.  Inner list has one metrics dict per fold.
    aggregate_metrics : dict[str, dict[str, float]]
        Outer key = strategy name.  Inner dict = full-series scalar metrics
        computed on the concatenated out-of-sample returns across all folds.
    tearsheets : dict[str, TearsheetData]
        Outer key = strategy name.
    benchmark_name : str
        Ticker or label used as the benchmark.
    config_summary : dict
        Snapshot of the BacktestConfig used to produce this report.
    n_folds : int
    """

    strategies: list[str]
    fold_metrics: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    aggregate_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    tearsheets: dict[str, TearsheetData] = field(default_factory=dict)
    benchmark_name: str = "SPY"
    config_summary: dict = field(default_factory=dict)
    n_folds: int = 0

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Return a compact strategy-comparison DataFrame.

        Index = strategy name.
        Columns = key performance metrics (annualized return, Sharpe,
        Sortino, Calmar, max drawdown, information ratio, tax drag,
        avg turnover).

        Returns
        -------
        pd.DataFrame
        """
        key_metrics = [
            "annualized_return",
            "volatility",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "max_dd_duration",
            "information_ratio",
            "alpha",
            "beta",
            "tracking_error",
            "effective_tax_drag",
            "avg_turnover",
            "hit_rate",
        ]
        rows: dict[str, dict] = {}
        for strat in self.strategies:
            agg = self.aggregate_metrics.get(strat, {})
            rows[strat] = {k: agg.get(k, np.nan) for k in key_metrics}

        df = pd.DataFrame(rows).T
        df.index.name = "strategy"
        return df

    def fold_summary(self, metric: str = "sharpe") -> pd.DataFrame:
        """Return a per-fold comparison table for one metric.

        Parameters
        ----------
        metric : str
            Metric name (must exist in fold_metrics).

        Returns
        -------
        pd.DataFrame
            Index = fold index, columns = strategy names.
        """
        rows = []
        if not self.fold_metrics:
            return pd.DataFrame(columns=self.strategies)

        non_empty = [v for v in self.fold_metrics.values() if v]
        if not non_empty:
            return pd.DataFrame(columns=self.strategies)
        max_folds = max(len(v) for v in non_empty)

        for fold_i in range(max_folds):
            row: dict[str, float] = {"fold": fold_i}
            for strat in self.strategies:
                folds = self.fold_metrics.get(strat, [])
                if fold_i < len(folds):
                    row[strat] = folds[fold_i].get(metric, np.nan)
                else:
                    row[strat] = np.nan
            rows.append(row)

        return pd.DataFrame(rows).set_index("fold")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return all aggregate metrics as a wide DataFrame.

        Equivalent to ``summary()`` but includes all computed metrics,
        not just the key subset.

        Returns
        -------
        pd.DataFrame
            Index = strategy names, columns = all metric names.
        """
        rows: dict[str, dict] = {}
        for strat in self.strategies:
            rows[strat] = dict(self.aggregate_metrics.get(strat, {}))

        return pd.DataFrame(rows).T

    def to_csv(self, path: str | Path) -> None:
        """Save the aggregate metrics summary to a CSV file.

        Parameters
        ----------
        path : str or Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path)

    def to_json(self, path: str | Path) -> None:
        """Serialize the report (without array data) to a JSON file.

        Parameters
        ----------
        path : str or Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "strategies": self.strategies,
            "benchmark_name": self.benchmark_name,
            "n_folds": self.n_folds,
            "config_summary": self.config_summary,
            "aggregate_metrics": {
                strat: {k: (float(v) if np.isfinite(v) else None) for k, v in m.items()}
                for strat, m in self.aggregate_metrics.items()
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def best_strategy(self, metric: str = "sharpe") -> str:
        """Return the strategy with the highest value for ``metric``.

        Parameters
        ----------
        metric : str

        Returns
        -------
        str
        """
        best_strat = None
        best_val = -np.inf
        for strat, m in self.aggregate_metrics.items():
            val = m.get(metric, -np.inf)
            if val > best_val:
                best_val = val
                best_strat = strat
        return best_strat or (self.strategies[0] if self.strategies else "")

    def ppo_vs_benchmark_summary(self) -> str:
        """Return a human-readable PPO-vs-benchmark one-liner."""
        ppo = self.aggregate_metrics.get("ppo", self.aggregate_metrics.get("PPO", {}))
        bm = self.aggregate_metrics.get("equal_weight", {})
        if not ppo or not bm:
            return "Insufficient data for comparison."

        ppo_sh = ppo.get("sharpe", 0.0)
        bm_sh = bm.get("sharpe", 0.0)
        ppo_r = ppo.get("annualized_return", 0.0) * 100
        bm_r = bm.get("annualized_return", 0.0) * 100
        ir = ppo.get("information_ratio", 0.0)

        direction = "outperforms" if ppo_sh > bm_sh else "underperforms"
        return (
            f"PPO {direction} equal-weight: "
            f"Sharpe {ppo_sh:.3f} vs {bm_sh:.3f}, "
            f"Ann. Return {ppo_r:.1f}% vs {bm_r:.1f}%, "
            f"IR = {ir:.3f}."
        )

    def __repr__(self) -> str:
        return (
            f"BacktestReport("
            f"strategies={self.strategies}, "
            f"n_folds={self.n_folds}, "
            f"benchmark='{self.benchmark_name}')"
        )
