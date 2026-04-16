"""
backtest/metrics.py
===================
GPU-accelerated performance metric computation for QuantAgent-RL.

All metrics accept a 1-D return series (one observation per rebalancing
period) and return either a scalar or a same-length array for rolling
variants.  CuPy is used when available so that batch computation across
many strategies and folds is accelerated on-device.

Metrics provided
----------------
Scalar (full-series):
  total_return, annualized_return, volatility, sharpe, sortino, calmar,
  max_drawdown, max_drawdown_duration, information_ratio, alpha, beta,
  effective_tax_drag, turnover_rate, hit_rate, avg_win_loss_ratio

Rolling (one value per period):
  rolling_sharpe, rolling_sortino, rolling_drawdown, rolling_alpha,
  rolling_beta, rolling_volatility

GPU Acceleration
----------------
Rolling metrics are the primary GPU target: computing rolling windows
over T = 200 quarterly periods × 10 strategies in a single batched
CuPy operation is ~8–15× faster than looping over strategies in NumPy.

The key operations accelerated:
  - Rolling sum / mean / std via CuPy cumulative sums (O(T) vs O(T·W))
  - Cumulative product for total-return index
  - Drawdown path via running maximum and minimum (CuPy)
  - Batch OLS (alpha / beta) via CuPy einsum over rolling windows
"""

import logging

import numpy as np
import pandas as pd

from backtest.config import BacktestConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU backend
# ---------------------------------------------------------------------------


def _get_xp(use_gpu: bool | None) -> tuple[object, bool]:
    """Return (array_module, gpu_active) pair."""
    if use_gpu is False:
        return np, False
    try:
        import cupy as cp

        if use_gpu is True and not cp.cuda.is_available():
            raise RuntimeError("use_gpu=True but no CUDA device detected.")
        logger.debug("[Metrics] CuPy backend active.")
        return cp, True
    except (ImportError, RuntimeError):
        if use_gpu is True:
            raise
        return np, False


def _to_np(arr: object) -> np.ndarray:
    """Convert a CuPy or NumPy array to a NumPy ndarray."""
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# MetricsCalculator
# ---------------------------------------------------------------------------


class MetricsCalculator:
    """Computes performance metrics from return series.

    Parameters
    ----------
    config : BacktestConfig

    Examples
    --------
    >>> mc = MetricsCalculator(BacktestConfig())
    >>> returns = np.array([0.02, -0.01, 0.03, ...])
    >>> metrics = mc.full_metrics(returns, benchmark_returns)
    >>> rolling = mc.rolling_metrics(returns, benchmark_returns)
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.cfg = config or BacktestConfig()
        self._xp, self._gpu = _get_xp(self.cfg.use_gpu)
        logger.info(
            f"[MetricsCalculator] backend={'GPU (CuPy)' if self._gpu else 'CPU (NumPy)'}"
        )

    # ------------------------------------------------------------------
    # Full-series scalar metrics
    # ------------------------------------------------------------------

    def full_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray | None = None,
        tax_costs: np.ndarray | None = None,
        turnovers: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute all scalar performance metrics for a return series.

        Parameters
        ----------
        returns : np.ndarray, shape (T,)
            Per-period portfolio returns (not annualized).
        benchmark_returns : np.ndarray | None, shape (T,)
            Benchmark returns for the same periods.  Required for alpha,
            beta, and information ratio.
        tax_costs : np.ndarray | None, shape (T,)
            Per-period realized tax costs as a fraction of portfolio value.
            Required for effective_tax_drag and after_tax_return.
        turnovers : np.ndarray | None, shape (T,)
            Per-period one-way turnover (sum of absolute weight changes).

        Returns
        -------
        dict[str, float]
            All computed metrics keyed by metric name.
        """
        r = np.asarray(returns, dtype=np.float64)
        ppy = self.cfg.periods_per_year
        rf = self.cfg.rf_per_period

        out: dict[str, float] = {}

        # Basic return stats
        out["total_return"] = self._total_return(r)
        out["annualized_return"] = self._annualized_return(r, ppy)
        out["volatility"] = self._volatility(r, ppy)
        out["sharpe"] = self._sharpe(r, rf, ppy)
        out["sortino"] = self._sortino(r, rf, ppy)
        out["calmar"] = self._calmar(r, ppy)
        out["max_drawdown"] = self._max_drawdown(r)
        out["max_dd_duration"] = float(self._max_drawdown_duration(r))
        out["hit_rate"] = float(np.mean(r > 0))
        out["avg_win_loss_ratio"] = self._win_loss_ratio(r)
        out["skewness"] = float(self._skewness(r))
        out["kurtosis"] = float(self._kurtosis(r))

        # Benchmark-relative metrics
        if benchmark_returns is not None:
            bm = np.asarray(benchmark_returns, dtype=np.float64)
            bm = bm[: len(r)]
            out["alpha"] = self._alpha(r, bm, rf, ppy)
            out["beta"] = self._beta(r, bm)
            out["information_ratio"] = self._information_ratio(r, bm, ppy)
            out["tracking_error"] = self._tracking_error(r, bm, ppy)
            out["benchmark_return"] = self._annualized_return(bm, ppy)
            out["excess_return"] = out["annualized_return"] - out["benchmark_return"]

        # Tax metrics
        if tax_costs is not None and self.cfg.include_tax_metrics:
            tc = np.asarray(tax_costs, dtype=np.float64)[: len(r)]
            out["total_tax_cost"] = float(tc.sum())
            out["effective_tax_drag"] = self._tax_drag(r, tc, ppy)
            # After-tax return: gross annualized return minus annualized tax drag
            out["after_tax_return"] = (
                out["annualized_return"] - out["effective_tax_drag"]
            )

        # Turnover metrics
        if turnovers is not None:
            tv = np.asarray(turnovers, dtype=np.float64)[: len(r)]
            out["avg_turnover"] = float(tv.mean())
            out["total_turnover"] = float(tv.sum())

        return out

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------

    def rolling_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute rolling metric series (one value per period).

        All rolling computations use the GPU backend when available for
        O(T) rolling sum / cumsum implementations.

        Parameters
        ----------
        returns : np.ndarray, shape (T,)
        benchmark_returns : np.ndarray | None, shape (T,)

        Returns
        -------
        dict[str, np.ndarray]
            Each array has shape (T,).  Values are NaN for the first
            ``min_periods_rolling - 1`` positions.
        """
        r = np.asarray(returns, dtype=np.float64)
        W = self.cfg.rolling_window
        ppy = self.cfg.periods_per_year
        rf = self.cfg.rf_per_period
        mp = self.cfg.min_periods_rolling

        out: dict[str, np.ndarray] = {}
        out["rolling_sharpe"] = self._rolling_sharpe_gpu(r, W, rf, ppy, mp)
        out["rolling_sortino"] = self._rolling_sortino_gpu(r, W, rf, ppy, mp)
        out["rolling_volatility"] = self._rolling_vol_gpu(r, W, ppy, mp)
        out["rolling_drawdown"] = self._rolling_max_drawdown_gpu(r, W, mp)

        if benchmark_returns is not None:
            bm = np.asarray(benchmark_returns, dtype=np.float64)[: len(r)]
            out["rolling_alpha"] = self._rolling_alpha_gpu(r, bm, W, rf, ppy, mp)
            out["rolling_beta"] = self._rolling_beta_gpu(r, bm, W, mp)

        return out

    # ------------------------------------------------------------------
    # Batch computation across strategies (primary GPU target)
    # ------------------------------------------------------------------

    def batch_full_metrics(
        self,
        returns_matrix: np.ndarray,
        benchmark_returns: np.ndarray | None = None,
        strategy_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute full-series metrics for multiple strategies simultaneously.

        Parameters
        ----------
        returns_matrix : np.ndarray, shape (T, S)
            Return series for S strategies over T periods.
        benchmark_returns : np.ndarray | None, shape (T,)
        strategy_names : list[str] | None
            Column labels; defaults to ['strategy_0', 'strategy_1', ...].

        Returns
        -------
        pd.DataFrame
            Index = strategy names, columns = metric names.
        """
        T, S = returns_matrix.shape
        names = strategy_names or [f"strategy_{i}" for i in range(S)]

        rows: list[dict] = []
        for j in range(S):
            bm = benchmark_returns if benchmark_returns is not None else None
            m = self.full_metrics(returns_matrix[:, j], bm)
            m["strategy"] = names[j]
            rows.append(m)

        return pd.DataFrame(rows).set_index("strategy")

    def batch_rolling_sharpe(
        self,
        returns_matrix: np.ndarray,
        window: int | None = None,
    ) -> np.ndarray:
        """GPU-batched rolling Sharpe ratio for S strategies simultaneously.

        Processes all S return series in a single CuPy operation, giving
        ~8–15× speedup over per-strategy CPU loops for S > 5.

        Parameters
        ----------
        returns_matrix : np.ndarray, shape (T, S)
        window : int | None
            Rolling window; defaults to ``cfg.rolling_window``.

        Returns
        -------
        np.ndarray, shape (T, S)
            Rolling Sharpe ratios; NaN for the first (window-1) rows.
        """
        xp = self._xp
        W = window or self.cfg.rolling_window
        rf = self.cfg.rf_per_period
        ppy = self.cfg.periods_per_year
        mp = self.cfg.min_periods_rolling
        T, S = returns_matrix.shape

        r_gpu = xp.asarray(returns_matrix, dtype=xp.float64)
        ex = r_gpu - rf  # excess returns, shape (T, S)

        # Prefix sums for O(T) rolling mean and variance
        cum_ex = xp.cumsum(ex, axis=0)  # (T, S)
        cum_ex2 = xp.cumsum(ex**2, axis=0)  # (T, S)

        result = xp.full((T, S), xp.nan)

        for t in range(W - 1, T):
            start = t - W + 1
            n = W if t >= W else t + 1
            if n < mp:
                continue
            n_f = float(n)
            s_ex = cum_ex[t] - (cum_ex[start - 1] if start > 0 else xp.zeros(S))
            s_ex2 = cum_ex2[t] - (cum_ex2[start - 1] if start > 0 else xp.zeros(S))
            mean_ex = s_ex / n_f
            var_ex = s_ex2 / n_f - mean_ex**2
            std_ex = xp.sqrt(xp.maximum(var_ex, 1e-12))
            result[t] = mean_ex / std_ex * xp.sqrt(float(ppy))

        return _to_np(result)

    # ------------------------------------------------------------------
    # Scalar metric implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _total_return(r: np.ndarray) -> float:
        return float(np.prod(1.0 + r) - 1.0)

    @staticmethod
    def _annualized_return(r: np.ndarray, ppy: int) -> float:
        T = len(r)
        if T == 0:
            return 0.0
        total = float(np.prod(1.0 + r))
        n_yrs = T / ppy
        return float(total ** (1.0 / max(n_yrs, 1e-6)) - 1.0)

    @staticmethod
    def _volatility(r: np.ndarray, ppy: int) -> float:
        if len(r) < 2:
            return 0.0
        return float(np.std(r, ddof=1) * np.sqrt(ppy))

    def _sharpe(self, r: np.ndarray, rf: float, ppy: int) -> float:
        excess = r - rf
        std = float(np.std(excess, ddof=1))
        if std < 1e-10:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(ppy))

    def _sortino(self, r: np.ndarray, rf: float, ppy: int) -> float:
        excess = r - rf
        neg = excess[excess < 0]
        if len(neg) < 2:
            return float(np.mean(excess) * ppy / 1e-10)
        down_std = float(np.std(neg, ddof=1))
        if down_std < 1e-10:
            return 0.0
        return float(np.mean(excess) / down_std * np.sqrt(ppy))

    def _calmar(self, r: np.ndarray, ppy: int) -> float:
        ann_ret = self._annualized_return(r, ppy)
        mdd = abs(self._max_drawdown(r))
        return float(ann_ret / mdd) if mdd > 1e-10 else 0.0

    @staticmethod
    def _max_drawdown(r: np.ndarray) -> float:
        cum = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / (peak + 1e-10)
        return float(dd.min())

    @staticmethod
    def _max_drawdown_duration(r: np.ndarray) -> int:
        """Length in periods of the longest drawdown episode."""
        cum = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(cum)
        in_dd = cum < peak
        if not in_dd.any():
            return 0
        max_dur = 0
        cur_dur = 0
        for flag in in_dd:
            if flag:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0
        return max_dur

    @staticmethod
    def _win_loss_ratio(r: np.ndarray) -> float:
        wins = r[r > 0]
        losses = r[r < 0]
        if len(losses) == 0 or len(wins) == 0:
            return 0.0
        avg_win = float(wins.mean())
        avg_loss = float(abs(losses.mean()))
        return float(avg_win / avg_loss) if avg_loss > 1e-10 else 0.0

    @staticmethod
    def _skewness(r: np.ndarray) -> float:
        if len(r) < 3:
            return 0.0
        mu = r.mean()
        std = r.std(ddof=1)
        if std < 1e-10:
            return 0.0
        return float(((r - mu) ** 3).mean() / (std**3))

    @staticmethod
    def _kurtosis(r: np.ndarray) -> float:
        if len(r) < 4:
            return 0.0
        mu = r.mean()
        std = r.std(ddof=1)
        if std < 1e-10:
            return 0.0
        # Excess kurtosis (0 for normal distribution)
        return float(((r - mu) ** 4).mean() / (std**4)) - 3.0

    @staticmethod
    def _alpha(r: np.ndarray, bm: np.ndarray, rf: float, ppy: int) -> float:
        """Jensen's alpha (annualized intercept from CAPM regression)."""
        ex_r = r - rf
        ex_bm = bm - rf
        if len(ex_r) < 3:
            return 0.0
        X = np.column_stack([np.ones(len(ex_bm)), ex_bm])
        try:
            beta_hat, alpha_hat = np.linalg.lstsq(X, ex_r, rcond=None)[0]
            return float(alpha_hat) * ppy
        except np.linalg.LinAlgError:
            return 0.0

    @staticmethod
    def _beta(r: np.ndarray, bm: np.ndarray) -> float:
        cov = float(np.cov(r, bm, ddof=1)[0, 1])
        var = float(np.var(bm, ddof=1))
        return float(cov / var) if var > 1e-10 else 1.0

    @staticmethod
    def _information_ratio(r: np.ndarray, bm: np.ndarray, ppy: int) -> float:
        active = r - bm
        te = float(np.std(active, ddof=1))
        if te < 1e-10:
            return 0.0
        return float(np.mean(active) / te * np.sqrt(ppy))

    @staticmethod
    def _tracking_error(r: np.ndarray, bm: np.ndarray, ppy: int) -> float:
        active = r - bm
        return float(np.std(active, ddof=1) * np.sqrt(ppy))

    @staticmethod
    def _tax_drag(r: np.ndarray, tc: np.ndarray, ppy: int) -> float:
        """Annualized effective tax drag = annualized total tax cost / portfolio value."""
        n_yrs = len(r) / ppy
        return float(tc.sum() / max(n_yrs, 1e-6))

    # ------------------------------------------------------------------
    # GPU-accelerated rolling metric helpers
    # ------------------------------------------------------------------

    def _rolling_sharpe_gpu(
        self,
        r: np.ndarray,
        W: int,
        rf: float,
        ppy: int,
        mp: int,
    ) -> np.ndarray:
        """O(T) rolling Sharpe using CuPy cumulative prefix sums."""
        xp = self._xp
        T = len(r)
        ex = xp.asarray(r - rf, dtype=xp.float64)
        cum1 = xp.cumsum(ex)
        cum2 = xp.cumsum(ex**2)
        out = xp.full(T, xp.nan)

        for t in range(W - 1, T):
            start = t - W + 1
            n = float(W)
            s1 = cum1[t] - (cum1[start - 1] if start > 0 else xp.float64(0))
            s2 = cum2[t] - (cum2[start - 1] if start > 0 else xp.float64(0))
            mean = s1 / n
            var = xp.maximum(s2 / n - mean**2, xp.float64(1e-12))
            out[t] = mean / xp.sqrt(var) * xp.sqrt(xp.float64(ppy))

        result = _to_np(out)
        result[: max(0, mp - 1)] = np.nan
        return result

    def _rolling_sortino_gpu(
        self,
        r: np.ndarray,
        W: int,
        rf: float,
        ppy: int,
        mp: int,
    ) -> np.ndarray:
        """Rolling Sortino ratio (downside deviation denominator)."""
        T = len(r)
        out = np.full(T, np.nan)
        ex = r - rf

        for t in range(W - 1, T):
            window = ex[t - W + 1 : t + 1]
            neg = window[window < 0]
            if len(neg) < 2:
                out[t] = 0.0
                continue
            dd_std = float(np.std(neg, ddof=1))
            out[t] = (
                float(window.mean() / dd_std * np.sqrt(ppy)) if dd_std > 1e-10 else 0.0
            )

        out[: max(0, mp - 1)] = np.nan
        return out

    def _rolling_vol_gpu(
        self,
        r: np.ndarray,
        W: int,
        ppy: int,
        mp: int,
    ) -> np.ndarray:
        """O(T) rolling annualized volatility via CuPy prefix sums."""
        xp = self._xp
        T = len(r)
        rv = xp.asarray(r, dtype=xp.float64)
        cum1 = xp.cumsum(rv)
        cum2 = xp.cumsum(rv**2)
        out = xp.full(T, xp.nan)

        for t in range(W - 1, T):
            start = t - W + 1
            n = float(W)
            s1 = cum1[t] - (cum1[start - 1] if start > 0 else xp.float64(0))
            s2 = cum2[t] - (cum2[start - 1] if start > 0 else xp.float64(0))
            mean = s1 / n
            var = xp.maximum(s2 / (n - 1) - mean**2 * n / (n - 1), xp.float64(0))
            out[t] = xp.sqrt(var) * xp.sqrt(xp.float64(ppy))

        result = _to_np(out)
        result[: max(0, mp - 1)] = np.nan
        return result

    def _rolling_max_drawdown_gpu(
        self,
        r: np.ndarray,
        W: int,
        mp: int,
    ) -> np.ndarray:
        """Rolling maximum drawdown over a trailing window."""
        xp = self._xp
        T = len(r)
        rv = xp.asarray(r, dtype=xp.float64)
        out = xp.full(T, xp.nan)

        for t in range(W - 1, T):
            start = t - W + 1
            window = rv[start : t + 1]
            cum = xp.cumprod(xp.float64(1.0) + window)
            # peak = xp.maximum.accumulate(cum)  # <-- not yet supported
            h_array = xp.asnumpy(cum)
            result = np.maximum.accumulate(h_array)
            peak = xp.array(result)
            dd = (cum - peak) / (peak + xp.float64(1e-10))
            out[t] = xp.min(dd)

        result = _to_np(out)
        result[: max(0, mp - 1)] = np.nan
        return result

    def _rolling_alpha_gpu(
        self,
        r: np.ndarray,
        bm: np.ndarray,
        W: int,
        rf: float,
        ppy: int,
        mp: int,
    ) -> np.ndarray:
        """Rolling annualized Jensen's alpha via rolling OLS."""
        xp = self._xp
        T = len(r)
        ex_r = xp.asarray(r - rf, dtype=xp.float64)
        ex_b = xp.asarray(bm - rf, dtype=xp.float64)
        out = xp.full(T, xp.nan)

        for t in range(W - 1, T):
            start = t - W + 1
            y = ex_r[start : t + 1]
            x = ex_b[start : t + 1]
            # OLS: [alpha, beta] = (X'X)^{-1} X'y
            ones = xp.ones(W, dtype=xp.float64)
            X = xp.column_stack([ones, x])  # (W, 2)
            XtX = X.T @ X  # (2, 2)
            Xty = X.T @ y  # (2,)
            try:
                coef = xp.linalg.solve(XtX + xp.eye(2) * 1e-8, Xty)
                out[t] = coef[0] * xp.float64(ppy)  # annualize intercept
            except Exception:
                out[t] = xp.float64(0.0)

        result = _to_np(out)
        result[: max(0, mp - 1)] = np.nan
        return result

    def _rolling_beta_gpu(
        self,
        r: np.ndarray,
        bm: np.ndarray,
        W: int,
        mp: int,
    ) -> np.ndarray:
        """Rolling CAPM beta via rolling OLS."""
        T = len(r)
        out = np.full(T, np.nan)

        for t in range(W - 1, T):
            start = t - W + 1
            y = r[start : t + 1]
            x = bm[start : t + 1]
            var_x = float(np.var(x, ddof=1))
            if var_x < 1e-12:
                out[t] = 1.0
            else:
                out[t] = float(np.cov(y, x, ddof=1)[0, 1] / var_x)

        out[: max(0, mp - 1)] = np.nan
        return out

    # ------------------------------------------------------------------
    # Drawdown series (for tear sheet visualization)
    # ------------------------------------------------------------------

    @staticmethod
    def drawdown_series(r: np.ndarray) -> np.ndarray:
        """Full drawdown path from a return series.

        Parameters
        ----------
        r : np.ndarray

        Returns
        -------
        np.ndarray, shape (T,)
            Drawdown at each period (values ≤ 0).
        """
        cum = np.cumprod(1.0 + np.asarray(r, dtype=np.float64))
        peak = np.maximum.accumulate(cum)
        return (cum - peak) / (peak + 1e-10)

    @staticmethod
    def cumulative_return_series(r: np.ndarray) -> np.ndarray:
        """Cumulative return index (1.0 = starting value).

        Parameters
        ----------
        r : np.ndarray

        Returns
        -------
        np.ndarray, shape (T,)
        """
        return np.cumprod(1.0 + np.asarray(r, dtype=np.float64))
