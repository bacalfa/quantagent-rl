"""
forecasting/garch.py
====================
GARCH(1,1) volatility forecaster for QuantAgent-RL.

Design
------
The ``arch`` library (Kevin Sheppard) handles parameter estimation via
maximum likelihood. GPU acceleration via CuPy is applied to the two
computationally intensive steps that benefit most from parallelism:

  1. **Batch variance recursion** — computing the conditional variance
     path h_t = ω + α·ε²_{t-1} + β·h_{t-1} for every asset simultaneously
     after parameters have been estimated.
  2. **Multi-step forecast aggregation** — summing h_{t+1}, ..., h_{t+H}
     across the forecast horizon for all assets in parallel.

Parameter estimation itself stays on CPU via ``arch`` because the optimizer
(scipy/L-BFGS-B) is inherently sequential per asset. The GPU wins on the
recursion and aggregation steps, which dominate at large universe sizes.

Walk-Forward Safety
-------------------
``GARCHForecaster.fit()`` must be called only on in-sample data. The
``forecast()`` method takes a new return series and propagates the
fitted parameters forward — it never re-estimates on out-of-sample data.
The ``refit_every_n_quarters`` config option controls how often parameters
are re-estimated across folds.

Outputs
-------
Per-asset, per-quarter:
  - ``vol_forecast``   : annualized 1-quarter-ahead conditional volatility
  - ``long_run_vol``   : unconditional (long-run) volatility implied by params
  - ``persistence``    : α + β (how quickly shocks decay; close to 1 = persistent)
  - ``half_life``      : implied half-life of a volatility shock in trading days
"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting.config import GARCHConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU backend
# ---------------------------------------------------------------------------


def _get_cupy() -> tuple[object, bool]:
    """Return (cupy_module, gpu_available) without raising on import failure."""
    try:
        import cupy as cp

        return cp, True
    except ImportError:
        return np, False


# ---------------------------------------------------------------------------
# Per-asset GARCH parameter container
# ---------------------------------------------------------------------------


@dataclass
class GARCHParams:
    """Fitted GARCH(1,1) parameters for a single asset.

    Attributes
    ----------
    omega, alpha, beta : float
        GARCH(1,1) parameters: ω, α (ARCH), β (GARCH).
    scale : float
        Return scaling factor used during fitting (100 if rescale=True, else 1).
    last_resid : float
        Last in-sample standardized residual ε_{T} (needed to seed the
        variance recursion at inference time).
    last_var : float
        Last in-sample conditional variance h_{T} (in squared scaled units).
    converged : bool
        Whether the optimizer converged.
    log_likelihood : float
        In-sample log-likelihood value.
    """

    omega: float = 0.0
    alpha: float = 0.1
    beta: float = 0.85
    scale: float = 1.0
    last_resid: float = 0.0
    last_var: float = 1.0
    converged: bool = False
    log_likelihood: float = float("-inf")

    @property
    def persistence(self) -> float:
        """α + β: fraction of a variance shock that persists to the next period."""
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        """Unconditional variance: ω / (1 - α - β)."""
        denom = 1.0 - self.persistence
        if denom <= 0:
            return float("inf")
        return self.omega / denom

    @property
    def half_life(self) -> float:
        """Number of periods for a variance shock to decay by half."""
        p = self.persistence
        if p <= 0 or p >= 1:
            return float("inf")
        return np.log(0.5) / np.log(p)


# ---------------------------------------------------------------------------
# Main forecaster
# ---------------------------------------------------------------------------


class GARCHForecaster:
    """Fits and produces GARCH(1,1) volatility forecasts for a universe of assets.

    Parameters
    ----------
    config : GARCHConfig
        Forecasting hyperparameters.

    Examples
    --------
    >>> gf = GARCHForecaster(GARCHConfig(dist='normal'))
    >>> gf.fit(train_log_returns)           # pd.DataFrame, daily, in-sample
    >>> forecasts = gf.forecast(horizon=63) # pd.Series indexed by ticker
    >>> gf.forecast_quarterly(all_log_returns, rebalance_dates)
    """

    def __init__(self, config: GARCHConfig | None = None) -> None:
        self.cfg = config or GARCHConfig()
        self._params: dict[str, GARCHParams] = {}
        self._fit_date: pd.Timestamp | None = None

        cp, self._gpu = _get_cupy()
        self._cp = cp
        if self.cfg.use_gpu is True and not self._gpu:
            raise RuntimeError(
                "use_gpu=True but CuPy is not installed. "
                "Install via: pip install cupy-cuda12x"
            )
        if self.cfg.use_gpu is False:
            self._gpu = False

        backend = "GPU (CuPy)" if self._gpu else "CPU (NumPy)"
        logger.info(f"[GARCH] Backend: {backend}")

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, log_returns: pd.DataFrame) -> "GARCHForecaster":
        """Estimate GARCH(1,1) parameters for every asset in the universe.

        Parameters
        ----------
        log_returns : pd.DataFrame
            Daily log returns. Index = DatetimeIndex, columns = tickers.
            Must contain only in-sample data (walk-forward safe).

        Returns
        -------
        self
            Enables method chaining: ``gf.fit(returns).forecast()``.
        """
        logger.info(
            f"[GARCH] Fitting {log_returns.shape[1]} assets on "
            f"{len(log_returns)} observations "
            f"({log_returns.index[0].date()} → {log_returns.index[-1].date()})"
        )
        self._fit_date = log_returns.index[-1]
        self._params = {}

        n_converged = 0
        for ticker in log_returns.columns:
            series = log_returns[ticker].dropna()
            if len(series) < self.cfg.min_obs:
                logger.warning(
                    f"[GARCH] {ticker}: only {len(series)} obs "
                    f"(min {self.cfg.min_obs}) — using default params."
                )
                self._params[ticker] = GARCHParams()
                continue

            params = self._fit_single(ticker, series)
            self._params[ticker] = params
            if params.converged:
                n_converged += 1

        logger.info(
            f"[GARCH] Fit complete: {n_converged}/{len(log_returns.columns)} converged."
        )
        return self

    def _fit_single(self, ticker: str, series: pd.Series) -> GARCHParams:
        """Estimate GARCH(1,1) for one asset.

        Tries the ``arch`` library first (preferred — fast, numerically robust
        MLE with scipy/L-BFGS-B). Falls back to a pure-SciPy MLE
        implementation so the module works without ``arch`` installed.
        """
        scale = 100.0 if self.cfg.rescale else 1.0
        y = series * scale

        # --- Primary path: arch library ---
        try:
            from arch import arch_model

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am = arch_model(
                    y,
                    vol="GARCH",
                    p=self.cfg.p,
                    q=self.cfg.q,
                    dist=self.cfg.dist,
                    rescale=False,
                )
                result = am.fit(disp="off", show_warning=False)

                params = result.params
                omega = float(params.get("omega", params.iloc[1]))
                alpha = float(params.get("alpha[1]", params.iloc[2]))
                beta = float(params.get("beta[1]", params.iloc[3]))

                alpha = max(alpha, 1e-6)
                beta = max(beta, 1e-6)
                if alpha + beta >= 1.0:
                    beta -= (alpha + beta) - 0.999

                cond_vol = result.conditional_volatility
                last_var = float(cond_vol.iloc[-1] ** 2)
                last_resid = float(
                    (y.iloc[-1] - float(params.iloc[0])) / (cond_vol.iloc[-1] + 1e-9)
                )
                return GARCHParams(
                    omega=omega,
                    alpha=alpha,
                    beta=beta,
                    scale=scale,
                    last_resid=last_resid,
                    last_var=last_var,
                    converged=result.convergence_flag == 0,
                    log_likelihood=float(result.loglikelihood),
                )

        except ImportError:
            logger.debug(
                "[GARCH] arch library not installed — using SciPy fallback. "
                "Install for better numerical stability: pip install arch"
            )
        except Exception as exc:
            logger.warning(
                f"[GARCH] arch fit failed for {ticker}: {exc} — trying SciPy fallback."
            )

        # --- Fallback path: pure SciPy MLE ---
        return self._fit_single_scipy(ticker, y, scale)

    def _fit_single_scipy(self, ticker: str, y: pd.Series, scale: float) -> GARCHParams:
        """GARCH(1,1) MLE via scipy.optimize.minimize (fallback).

        Minimizes the negative Gaussian log-likelihood:
            L = -0.5 * Σ_t [log(h_t) + ε²_t / h_t]
        where h_t = ω + α·ε²_{t-1} + β·h_{t-1}.
        """
        try:
            from scipy.optimize import minimize

            r = y.values.astype(np.float64)
            T = len(r)
            var_proxy = float(np.var(r))

            def neg_log_likelihood(params: np.ndarray) -> float:
                omega, alpha, beta = params
                if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 1.0:
                    return 1e10
                h = np.empty(T)
                h[0] = var_proxy
                for t in range(1, T):
                    h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
                    if h[t] <= 0:
                        return 1e10
                ll = -0.5 * np.sum(np.log(h) + r**2 / h)
                return -ll

            x0 = np.array([var_proxy * 0.05, 0.08, 0.88])
            bounds = [(1e-8, None), (1e-6, 0.5), (1e-6, 0.9999)]
            result = minimize(
                neg_log_likelihood,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-9},
            )

            omega, alpha, beta = result.x
            # Compute final variance path for seed values
            h = np.empty(T)
            h[0] = var_proxy
            for t in range(1, T):
                h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]

            return GARCHParams(
                omega=float(omega),
                alpha=float(alpha),
                beta=float(beta),
                scale=scale,
                last_resid=float(r[-1] / (np.sqrt(h[-1]) + 1e-9)),
                last_var=float(h[-1]),
                converged=result.success,
                log_likelihood=float(-result.fun),
            )

        except Exception as exc:
            logger.warning(
                f"[GARCH] SciPy fallback failed for {ticker}: {exc} — using defaults."
            )
            var_proxy = float(np.var(y.values))
            return GARCHParams(
                omega=var_proxy * 0.05,
                alpha=0.08,
                beta=0.88,
                scale=scale,
                last_var=var_proxy,
                converged=False,
            )

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    def forecast(self, horizon: int | None = None) -> pd.Series:
        """Produce 1-quarter-ahead annualized volatility forecasts.

        Uses fitted parameters and seeds the variance recursion forward
        ``horizon`` steps using the GPU-accelerated batch recursion.

        Parameters
        ----------
        horizon : int | None
            Override the configured horizon. Defaults to cfg.horizon.

        Returns
        -------
        pd.Series
            Annualized volatility forecasts indexed by ticker.
        """
        if not self._params:
            raise RuntimeError("Call fit() before forecast().")

        h = horizon or self.cfg.horizon
        tickers = list(self._params.keys())

        # Stack parameters into arrays for batch computation
        omega = np.array([self._params[t].omega for t in tickers])
        alpha = np.array([self._params[t].alpha for t in tickers])
        beta = np.array([self._params[t].beta for t in tickers])
        h0 = np.array([self._params[t].last_var for t in tickers])
        e0 = np.array([self._params[t].last_resid for t in tickers])

        vol_forecasts = self._batch_forecast(omega, alpha, beta, h0, e0, h)

        # Convert from scaled daily variance to annualized daily vol
        scales = np.array([self._params[t].scale for t in tickers])
        # var in scaled units → unscale → annualize
        vol_forecasts = np.sqrt(vol_forecasts) / scales
        if self.cfg.annualize:
            vol_forecasts *= np.sqrt(252)

        return pd.Series(vol_forecasts, index=tickers, name="vol_forecast")

    def _batch_forecast(
        self,
        omega: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        h0: np.ndarray,
        e0: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """GPU-accelerated batch GARCH variance recursion and aggregation.

        For each asset i and step s in [1, horizon], the GARCH(1,1) recursion is:

            h_{i,s} = ω_i + α_i · E[ε²_{i,s-1}] + β_i · h_{i,s-1}

        For s > 1, E[ε²_{i,s-1}] = h_{i,s-1} (since ε_t is zero-mean with
        unit conditional variance). For s = 1, ε²_{i,0} = e0² is the last
        observed squared standardized residual.

        The H-step-ahead variance forecast is the average of h_{i,1}, ..., h_{i,H},
        which represents the expected average conditional variance over the
        forecast horizon — appropriate for quarterly return risk estimation.

        Parameters
        ----------
        omega, alpha, beta : np.ndarray, shape (n_assets,)
        h0 : np.ndarray
            Last in-sample conditional variance, shape (n_assets,).
        e0 : np.ndarray
            Last in-sample squared standardized residual, shape (n_assets,).
        horizon : int

        Returns
        -------
        np.ndarray, shape (n_assets,)
            Mean conditional variance over the forecast horizon (scaled units²).
        """
        xp = self._cp if self._gpu else np

        omega = xp.asarray(omega)
        alpha = xp.asarray(alpha)
        beta = xp.asarray(beta)
        h = xp.asarray(h0)
        e_sq = xp.asarray(e0**2)

        cumulative = xp.zeros_like(h)

        for s in range(horizon):
            if s == 0:
                h = omega + alpha * e_sq + beta * h
            else:
                # For s > 1: E[ε²] = h (conditional on information at t)
                h = omega + alpha * h + beta * h
            cumulative += h

        mean_var = cumulative / horizon

        # Return as numpy array regardless of backend
        if self._gpu:
            return mean_var.get()
        return np.asarray(mean_var)

    # ------------------------------------------------------------------
    # Walk-forward quarterly forecasts
    # ------------------------------------------------------------------

    def forecast_quarterly(
        self,
        log_returns: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
        fit_end_dates: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """Produce GARCH volatility forecasts at every rebalancing date.

        This is the primary method called by the ForecastingPipeline.
        Parameters are re-estimated at intervals controlled by
        ``cfg.refit_every_n_quarters``.

        Parameters
        ----------
        log_returns : pd.DataFrame
            Full daily log return history (train + test, all tickers).
        rebalance_dates : pd.DatetimeIndex
            Quarter-end dates at which forecasts are required.
        fit_end_dates : pd.DatetimeIndex | None
            Dates on which to re-estimate parameters (walk-forward safe).
            If None, refits every ``cfg.refit_every_n_quarters`` quarters,
            anchored from the first rebalance date.

        Returns
        -------
        pd.DataFrame
            Columns = tickers, index = rebalance_dates.
            Values = annualized 1-quarter-ahead conditional volatility.
        """
        results: dict[pd.Timestamp, pd.Series] = {}
        last_fit_idx = -1

        for i, date in enumerate(rebalance_dates):
            should_refit = (
                last_fit_idx == -1
                or (i - last_fit_idx) >= self.cfg.refit_every_n_quarters
            )

            in_sample = log_returns[log_returns.index <= date]
            if in_sample.empty:
                continue

            enough_obs = in_sample.dropna().shape[0] >= self.cfg.min_obs

            if should_refit and enough_obs:
                logger.debug(f"[GARCH] Refitting at {date.date()}")
                self.fit(in_sample)
                last_fit_idx = i
            elif last_fit_idx == -1:
                # Insufficient obs and no prior fit — seed with simple variance
                self._params = {
                    t: GARCHParams(
                        omega=max(float(in_sample[t].dropna().var()), 1e-6) * 0.05
                        if t in in_sample.columns
                        else 1e-6,
                        alpha=0.08,
                        beta=0.88,
                        scale=100.0 if self.cfg.rescale else 1.0,
                        last_var=max(float(in_sample[t].dropna().var()), 1e-6)
                        if t in in_sample.columns
                        else 1e-4,
                        converged=False,
                    )
                    for t in log_returns.columns
                }

            if self._params:
                results[date] = self.forecast()

        df = pd.DataFrame(results).T
        df.index.name = "date"
        return df

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def parameter_summary(self) -> pd.DataFrame:
        """Return a DataFrame of fitted parameters for all assets.

        Returns
        -------
        pd.DataFrame
            Columns: omega, alpha, beta, persistence, long_run_vol,
            half_life, converged, log_likelihood.
            Index = ticker.
        """
        rows = {}
        for ticker, p in self._params.items():
            lr_var = p.long_run_variance
            # Convert long-run variance from scaled² to annualized vol
            lr_vol = np.sqrt(lr_var) / p.scale * np.sqrt(252)
            rows[ticker] = {
                "omega": p.omega,
                "alpha": p.alpha,
                "beta": p.beta,
                "persistence": p.persistence,
                "long_run_vol": lr_vol,
                "half_life_days": p.half_life,
                "converged": p.converged,
                "log_likelihood": p.log_likelihood,
            }
        return pd.DataFrame(rows).T
