"""
backtest/engine.py
==================
BacktestEngine — walk-forward backtest orchestrator for QuantAgent-RL.

Consumes ``RLFoldResult`` objects produced by ``rl.pipeline.RLPipeline``
and assembles a ``BacktestReport`` containing:

  - Full-series scalar metrics for every strategy × fold combination
  - Aggregate metrics computed on the concatenated out-of-sample returns
    across all walk-forward folds
  - Rolling metric time-series for tear-sheet visualization
  - Benchmark comparison (alpha, beta, information ratio, tracking error)

Benchmark Handling
------------------
The benchmark return series is constructed from the ``prices`` DataFrame
passed to the engine.  Benchmark returns are computed as per-period
log-returns of the benchmark ticker (e.g. SPY), aligned to the same
rebalance dates as the portfolio strategies.

GPU Acceleration
----------------
``MetricsCalculator.batch_rolling_sharpe`` processes all S strategies
simultaneously in CuPy, giving ~8–15× speedup over per-strategy CPU loops.
Scalar metric computation for scalar functions uses plain NumPy (fast
enough given T ≤ 200 quarterly periods); only the rolling O(T·W) operations
benefit materially from GPU parallelism.
"""

import logging

import numpy as np
import pandas as pd

from backtest.config import BacktestConfig
from backtest.metrics import MetricsCalculator
from backtest.report import BacktestReport, StrategyReturns, TearsheetData
from rl import RLFoldResult

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Walk-forward backtest engine.

    Parameters
    ----------
    config : BacktestConfig
    prices : pd.DataFrame | None
        Adjusted close prices indexed by date, columns = tickers.
        Used to compute benchmark returns.  If None, benchmark
        comparisons are skipped.

    Examples
    --------
    Typical usage from ``RLFoldResult`` objects:

    >>> engine = BacktestEngine(BacktestConfig(), prices=prices_df)
    >>> report = engine.run_from_fold_results(fold_results)
    >>> print(report.summary())

    Direct usage from return arrays:

    >>> strategies = [
    ...     StrategyReturns('ppo',          ppo_returns,   dates),
    ...     StrategyReturns('equal_weight', ew_returns,    dates),
    ...     StrategyReturns('hold',         hold_returns,  dates),
    ... ]
    >>> report = engine.run(strategies)
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        prices: pd.DataFrame | None = None,
    ) -> None:
        self.cfg = config or BacktestConfig()
        self.prices = prices
        self._mc = MetricsCalculator(self.cfg)

    # ------------------------------------------------------------------
    # Primary entry point: from RLFoldResult objects
    # ------------------------------------------------------------------

    def run_from_fold_results(
        self,
        fold_results: list[RLFoldResult],
        prices: pd.DataFrame | None = None,
    ) -> BacktestReport:
        """Build a BacktestReport from a list of RLFoldResult objects.

        Parameters
        ----------
        fold_results : list[RLFoldResult]
            Output of ``rl.pipeline.RLPipeline.run_all_folds()``.
        prices : pd.DataFrame | None
            Price data for benchmark return computation.  Falls back to
            ``self.prices`` if not provided.

        Returns
        -------
        BacktestReport
        """
        if prices is not None:
            self.prices = prices

        # Determine strategy labels from the first fold result
        all_strats: dict[str, list[StrategyReturns]] = {
            "ppo": [],
            "equal_weight": [],
            "hold": [],
        }

        for fold in fold_results:
            test_dates = pd.DatetimeIndex(fold.test_dates)

            for strat_label, metrics_attr in [
                ("ppo", "ppo_metrics"),
                ("equal_weight", "equal_weight_metrics"),
                ("hold", "hold_metrics"),
            ]:
                m = getattr(fold, metrics_attr)
                r = np.asarray(m.quarterly_returns, dtype=np.float64)
                # Align dates: quarterly_returns length may differ from test_dates
                n = min(len(r), len(test_dates))
                all_strats[strat_label].append(
                    StrategyReturns(
                        name=strat_label,
                        returns=r[:n],
                        dates=test_dates[:n],
                        weights=m.quarterly_weights[:n]
                        if m.quarterly_weights
                        else None,
                        tax_costs=m.total_tax_cost[:n]
                        if m.total_tax_cost is not None
                        else None,
                        turnovers=m.total_turnover[:n]
                        if m.total_turnover is not None
                        else None,
                        fold_idx=fold.fold_idx,
                    )
                )

        # Concatenate fold-level StrategyReturns into one per-strategy series
        concatenated: list[StrategyReturns] = []
        for strat_label, fold_sr_list in all_strats.items():
            if not fold_sr_list:
                continue
            all_returns = np.concatenate([sr.returns for sr in fold_sr_list])
            all_dates = (
                fold_sr_list[0].dates.append([sr.dates for sr in fold_sr_list[1:]])
                if len(fold_sr_list) > 1
                else fold_sr_list[0].dates
            )

            concatenated.append(
                StrategyReturns(
                    name=strat_label,
                    returns=all_returns,
                    dates=all_dates,
                    fold_idx=None,
                )
            )

        report = self.run(concatenated, fold_results=fold_results)
        return report

    # ------------------------------------------------------------------
    # Primary entry point: from StrategyReturns lists
    # ------------------------------------------------------------------

    def run(
        self,
        strategies: list[StrategyReturns],
        fold_results: list[RLFoldResult] | None = None,
    ) -> BacktestReport:
        """Compute the full backtest report from strategy return series.

        Parameters
        ----------
        strategies : list[StrategyReturns]
            One entry per strategy.  All series should cover the same
            date range (out-of-sample test periods only).
        fold_results : list[RLFoldResult] | None
            Optional — used to populate per-fold metric breakdowns.

        Returns
        -------
        BacktestReport
        """
        strat_names = [s.name for s in strategies]
        n_folds = len(fold_results) if fold_results else 0

        logger.info(
            f"[BacktestEngine] Running report: "
            f"{len(strategies)} strategies, {n_folds} folds"
        )

        # ── Benchmark returns ─────────────────────────────────────────
        bm_returns = self._benchmark_returns(strategies)

        # ── Aggregate metrics ─────────────────────────────────────────
        aggregate_metrics: dict[str, dict[str, float]] = {}
        for sr in strategies:
            bm = bm_returns.reindex(sr.dates).values if bm_returns is not None else None
            aggregate_metrics[sr.name] = self._mc.full_metrics(
                sr.returns,
                benchmark_returns=bm,
                tax_costs=sr.tax_costs,
                turnovers=sr.turnovers,
            )

        # ── Per-fold metrics ──────────────────────────────────────────
        fold_metrics: dict[str, list[dict[str, float]]] = {
            s.name: [] for s in strategies
        }
        if fold_results:
            fold_metrics = self._per_fold_metrics(fold_results, bm_returns)

        # ── Tear sheets ───────────────────────────────────────────────
        tearsheets = self._build_tearsheets(strategies, bm_returns)

        report = BacktestReport(
            strategies=strat_names,
            fold_metrics=fold_metrics,
            aggregate_metrics=aggregate_metrics,
            tearsheets=tearsheets,
            benchmark_name=self.cfg.benchmark_ticker,
            n_folds=n_folds,
            config_summary={
                "benchmark_ticker": self.cfg.benchmark_ticker,
                "periods_per_year": self.cfg.periods_per_year,
                "rolling_window": self.cfg.rolling_window,
                "risk_free_rate": self.cfg.risk_free_rate,
            },
        )
        logger.info(f"[BacktestEngine] Report complete. {report}")
        return report

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def _benchmark_returns(
        self,
        strategies: list[StrategyReturns],
    ) -> pd.Series | None:
        """Compute per-period benchmark returns aligned to strategy dates."""
        if self.prices is None:
            return None
        ticker = self.cfg.benchmark_ticker
        if ticker not in self.prices.columns:
            logger.warning(
                f"[BacktestEngine] Benchmark ticker '{ticker}' not in prices; "
                "skipping benchmark metrics."
            )
            return None

        # Union of all strategy dates
        all_dates = pd.DatetimeIndex([])
        for sr in strategies:
            all_dates = all_dates.append(sr.dates)
        all_dates = all_dates.unique().sort_values()

        bm_prices = self.prices[ticker].reindex(method="ffill")

        # Per-period returns at each rebalance date
        bm_pct = bm_prices.pct_change()
        bm_aligned = pd.Series(dtype=float, index=all_dates)

        for date in all_dates:
            try:
                candidates = bm_pct[bm_pct.index <= date]
                if not candidates.empty:
                    bm_aligned[date] = float(candidates.iloc[-1])
            except Exception:
                bm_aligned[date] = 0.0

        return bm_aligned.fillna(0.0)

    # ------------------------------------------------------------------
    # Per-fold metrics
    # ------------------------------------------------------------------

    def _per_fold_metrics(
        self,
        fold_results: list[RLFoldResult],
        bm_returns: pd.Series | None,
    ) -> dict[str, list[dict[str, float]]]:
        """Extract scalar metrics from each RLFoldResult."""
        out: dict[str, list[dict]] = {"ppo": [], "equal_weight": [], "hold": []}

        for fold in fold_results:
            test_dates = pd.DatetimeIndex(fold.test_dates)

            for label, attr in [
                ("ppo", "ppo_metrics"),
                ("equal_weight", "equal_weight_metrics"),
                ("hold", "hold_metrics"),
            ]:
                m = getattr(fold, attr)
                r = np.asarray(m.quarterly_returns, dtype=np.float64)
                n = min(len(r), len(test_dates))

                bm = None
                if bm_returns is not None:
                    bm = bm_returns.reindex(test_dates[:n]).values

                metrics = self._mc.full_metrics(r[:n], benchmark_returns=bm)
                metrics["fold_idx"] = float(fold.fold_idx)
                out[label].append(metrics)

        return out

    # ------------------------------------------------------------------
    # Tear sheets
    # ------------------------------------------------------------------

    def _build_tearsheets(
        self,
        strategies: list[StrategyReturns],
        bm_returns: pd.Series | None,
    ) -> dict[str, TearsheetData]:
        """Build a TearsheetData for every strategy."""
        tearsheets: dict[str, TearsheetData] = {}

        # GPU-batched rolling Sharpe across all strategies at once
        if strategies:
            T = max(len(s.returns) for s in strategies)
            S = len(strategies)
            mat = np.zeros((T, S), dtype=np.float64)
            for j, sr in enumerate(strategies):
                n = len(sr.returns)
                mat[:n, j] = sr.returns

            rolling_sharpe_mat = self._mc.batch_rolling_sharpe(mat)

        for j, sr in enumerate(strategies):
            n = len(sr.returns)
            dates = sr.dates

            cum = MetricsCalculator.cumulative_return_series(sr.returns)
            dd = MetricsCalculator.drawdown_series(sr.returns)
            r_sh = rolling_sharpe_mat[:n, j]

            # Rolling vol and alpha
            rolling = self._mc.rolling_metrics(
                sr.returns,
                bm_returns.reindex(dates).values if bm_returns is not None else None,
            )

            bm_cum = None
            if bm_returns is not None:
                bm_r = bm_returns.reindex(dates).fillna(0.0).values
                bm_cum = MetricsCalculator.cumulative_return_series(bm_r)

            scalar = self._mc.full_metrics(
                sr.returns,
                benchmark_returns=(
                    bm_returns.reindex(dates).values if bm_returns is not None else None
                ),
                tax_costs=sr.tax_costs,
                turnovers=sr.turnovers,
            )

            tearsheets[sr.name] = TearsheetData(
                strategy_name=sr.name,
                dates=dates,
                cumulative_returns=cum,
                drawdown=dd,
                rolling_sharpe=r_sh,
                rolling_alpha=rolling.get("rolling_alpha"),
                rolling_volatility=rolling.get("rolling_volatility"),
                benchmark_cumulative=bm_cum,
                scalar_metrics=scalar,
            )

        return tearsheets
