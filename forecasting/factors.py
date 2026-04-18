"""
forecasting/factors.py
======================
Fama-French factor model for QuantAgent-RL.

Computes time-varying factor exposures (betas) and rolling alpha for each
asset in the universe using the Fama-French 3-factor or 5-factor model.
Factor data is sourced from Ken French's Data Library — completely free,
no registration required.

Factor Models
-------------
**3-Factor (FF3)**:
    r_i - rf = α_i + β_mkt · (r_mkt - rf) + β_smb · SMB + β_hml · HML + ε

**5-Factor (FF5)** (extends FF3):
    r_i - rf = α_i + β_mkt · MKT + β_smb · SMB + β_hml · HML
                    + β_rmw · RMW + β_cma · CMA + ε

Where:
    MKT = Mkt-RF  : Market excess return
    SMB           : Small-minus-Big (size premium)
    HML           : High-minus-Low (value premium)
    RMW           : Robust-minus-Weak (profitability premium)   [FF5 only]
    CMA           : Conservative-minus-Aggressive (investment premium) [FF5 only]

Data Source
-----------
Ken French's Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/
  - 3-factor daily: F-F_Research_Data_Factors_daily_CSV.zip
  - 5-factor daily: F-F_Research_Data_5_Factors_2x3_daily_CSV.zip

Both files are downloaded, cached locally, and parsed automatically.

GPU Acceleration
----------------
Rolling OLS is re-implemented in CuPy to process all assets in parallel.
For a universe of N assets over T observations with a rolling window W:
  - Naive approach: N × (T - W) separate OLS solves on CPU
  - GPU approach: Batch all assets and all windows into a single 3D tensor
    operation using CuPy's matmul broadcasting

The GPU advantage scales with N (universe size) and T (history length).
For N=30 assets and T=3750 days (15 years), expect ~8–15× speedup.

Outputs (per asset, per quarter-end date)
-----------------------------------------
  - alpha_ann       : Annualized rolling intercept
  - beta_mkt        : Market beta
  - beta_smb        : Size beta
  - beta_hml        : Value beta
  - beta_rmw        : Profitability beta (FF5 only)
  - beta_cma        : Investment beta (FF5 only)
  - r_squared       : In-window R²
"""

import io
import logging
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.config import FamaFrenchConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fama-French data source URLs
# ---------------------------------------------------------------------------

_FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
_FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

_FF3_COLS = ["mkt_rf", "smb", "hml", "rf"]
_FF5_COLS = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]

_FF3_FACTORS = ["mkt_rf", "smb", "hml"]
_FF5_FACTORS = ["mkt_rf", "smb", "hml", "rmw", "cma"]


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
# Factor data downloader
# ---------------------------------------------------------------------------


class FamaFrenchDataLoader:
    """Downloads and caches daily Fama-French factor returns.

    Parameters
    ----------
    n_factors : int
        3 for the 3-factor model, 5 for the 5-factor model.
    cache_dir : str
        Directory for caching the downloaded CSV.

    Examples
    --------
    >>> loader = FamaFrenchDataLoader(n_factors=3)
    >>> ff_df = loader.load()   # pd.DataFrame with factor returns
    """

    def __init__(
        self, n_factors: int = 3, cache_dir: str = "../data/cache/ff_factors"
    ) -> None:
        if n_factors not in (3, 5):
            raise ValueError("n_factors must be 3 or 5.")
        self.n_factors = n_factors
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _cache_path(self) -> Path:
        return self.cache_dir / f"ff{self.n_factors}_daily.parquet"

    @property
    def _url(self) -> str:
        return _FF3_URL if self.n_factors == 3 else _FF5_URL

    @property
    def _col_names(self) -> list[str]:
        return _FF3_COLS if self.n_factors == 3 else _FF5_COLS

    @property
    def _factor_names(self) -> list[str]:
        return _FF3_FACTORS if self.n_factors == 3 else _FF5_FACTORS

    def load(self, use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
        """Load factor data, downloading from Ken French's site if needed.

        Returns
        -------
        pd.DataFrame
            Daily factor returns in decimal form (not percent).
            Index = DatetimeIndex, columns = factor names + 'rf'.
        """
        if use_cache and not force_refresh and self._cache_path.exists():
            logger.info(f"[FF] Loading FF{self.n_factors} from cache.")
            return pd.read_parquet(self._cache_path)

        logger.info(f"[FF] Downloading FF{self.n_factors} from Ken French's library...")
        df = self._download()
        df.to_parquet(self._cache_path)
        logger.info(f"[FF] Saved to {self._cache_path} ({len(df)} rows).")
        return df

    def _download(self) -> pd.DataFrame:
        try:
            import requests
        except ImportError as exc:
            raise ImportError("requests is required: pip install requests") from exc

        proxies = {
            "http": os.environ.get("HTTP_PROXY", ""),
            "https": os.environ.get("HTTPS_PROXY", ""),
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        verify = os.environ.get("REQUESTS_CA_BUNDLE", False)

        resp = requests.get(
            self._url, proxies=proxies, headers=headers, verify=verify, timeout=60
        )
        resp.raise_for_status()

        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = [n for n in zf.namelist() if n.upper().endswith(".CSV")][0]
        raw = zf.read(csv_name).decode("utf-8", errors="replace")

        return self._parse_csv(raw)

    def _parse_csv(self, raw: str) -> pd.DataFrame:
        """Parse the Ken French CSV format (skips header and footer text)."""
        lines = raw.splitlines()

        # Find the first data line: 8-digit date YYYYMMDD followed by numbers
        start_row = 0
        for i, line in enumerate(lines):
            parts = line.split(",")
            if len(parts) >= 4 and len(parts[0].strip()) == 8:
                try:
                    int(parts[0].strip())
                    start_row = i
                    break
                except ValueError:
                    continue

        # Find the last data line before the annual summary section
        end_row = len(lines)
        for i in range(start_row, len(lines)):
            parts = lines[i].split(",")
            if parts and len(parts[0].strip()) == 4:  # 4-digit year = annual section
                end_row = i
                break

        # Keep only lines where the first field is an 8-digit date integer
        # (filters out section headers like "  Annual" that Ken French includes
        # between the daily and annual sections of the CSV)
        data_lines = [
            l
            for l in lines[start_row:end_row]
            if l.strip()
            and len(l.split(",")[0].strip()) == 8
            and l.split(",")[0].strip().isdigit()
        ]
        data_str = "\n".join(data_lines)
        df = pd.read_csv(
            io.StringIO(data_str),
            header=None,
            names=["date"] + self._col_names,
        )

        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        df = df.set_index("date").sort_index()

        # Convert from percent to decimal
        for col in self._col_names:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

        df = df.dropna(how="all")
        return df


# ---------------------------------------------------------------------------
# Factor model estimator
# ---------------------------------------------------------------------------


class FamaFrenchFactors:
    """Computes rolling Fama-French factor exposures for a universe of assets.

    Parameters
    ----------
    config : FamaFrenchConfig

    Examples
    --------
    >>> ff = FamaFrenchFactors(FamaFrenchConfig(n_factors=3))
    >>> ff.load_factors()
    >>> betas = ff.rolling_betas(log_returns)
    >>> quarterly = ff.forecast_quarterly(log_returns, rebalance_dates)
    """

    def __init__(self, config: FamaFrenchConfig | None = None) -> None:
        self.cfg = config or FamaFrenchConfig()
        self._ff_data: pd.DataFrame | None = None
        self._factor_names: list[str] = (
            _FF3_FACTORS if self.cfg.n_factors == 3 else _FF5_FACTORS
        )

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
        logger.info(f"[FF{self.cfg.n_factors}] Backend: {backend}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_factors(
        self, use_cache: bool = True, force_refresh: bool = False
    ) -> "FamaFrenchFactors":
        """Download and cache Fama-French factor data.

        Returns
        -------
        self
        """
        loader = FamaFrenchDataLoader(
            n_factors=self.cfg.n_factors,
            cache_dir=self.cfg.cache_dir,
        )
        self._ff_data = loader.load(use_cache=use_cache, force_refresh=force_refresh)
        logger.info(
            f"[FF{self.cfg.n_factors}] Factors loaded: "
            f"{self._ff_data.index[0].date()} → {self._ff_data.index[-1].date()}"
        )
        return self

    def _align_factors(
        self,
        log_returns: pd.DataFrame,
        end_date: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Align factor returns with asset excess returns.

        Parameters
        ----------
        log_returns : pd.DataFrame
            Daily log returns (columns = tickers).
        end_date : pd.Timestamp | None
            If set, restrict data to this date (walk-forward safe).

        Returns
        -------
        excess_returns : pd.DataFrame
            Asset log returns minus the risk-free rate.
        factors : pd.DataFrame
            Factor returns aligned to same dates.
        """
        if self._ff_data is None:
            raise RuntimeError("Call load_factors() first.")

        ff = self._ff_data
        if end_date is not None:
            ff = ff[ff.index <= end_date]

        # Align on common trading dates
        common_idx = log_returns.index.intersection(ff.index)
        if len(common_idx) == 0:
            raise ValueError(
                "No overlapping dates between log_returns and FF factor data. "
                "Check that start_date is after 1926-07-01 (earliest FF data)."
            )

        rf = ff.loc[common_idx, "rf"]
        factors = ff.loc[common_idx, self._factor_names]
        asset_returns = log_returns.loc[common_idx]

        # Excess returns: subtract daily risk-free rate from each asset
        excess = asset_returns.subtract(rf, axis=0)

        return excess, factors

    # ------------------------------------------------------------------
    # Rolling OLS
    # ------------------------------------------------------------------

    def rolling_betas(
        self,
        log_returns: pd.DataFrame,
        end_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Compute rolling OLS factor exposures for all assets.

        Uses CuPy-accelerated batched matrix operations on GPU, falling
        back to a vectorized NumPy implementation on CPU.

        Parameters
        ----------
        log_returns : pd.DataFrame
            Daily log returns.
        end_date : pd.Timestamp | None
            Restrict computation to dates on or before this value.

        Returns
        -------
        pd.DataFrame
            MultiIndex columns: (output_name, ticker).
            Output names: 'alpha_ann', 'beta_mkt', 'beta_smb', 'beta_hml',
            ['beta_rmw', 'beta_cma' for FF5], 'r_squared'.
            Index = DatetimeIndex (same as log_returns, NaN for first W-1 rows).
        """
        excess, factors = self._align_factors(log_returns, end_date=end_date)

        W = self.cfg.rolling_window
        T = len(excess)
        N = len(excess.columns)
        K = len(self._factor_names)
        min_obs = int(W * self.cfg.min_obs_fraction)

        tickers = list(excess.columns)
        factor_cols = (
            ["alpha_ann"]
            + [f"beta_{f.replace('_rf', '')}" for f in self._factor_names]
            + ["r_squared"]
        )

        # Arrays: (T, N) for excess returns, (T, K) for factors
        Y_all = excess.values  # (T, N)
        X_raw = factors.values  # (T, K)

        if self._gpu:
            betas_array, rsq_array = self._rolling_ols_gpu(Y_all, X_raw, W, min_obs)
        else:
            betas_array, rsq_array = self._rolling_ols_cpu(Y_all, X_raw, W, min_obs)

        # betas_array shape: (T, N, K+1) where last dim = [intercept, f1, f2, ...]
        # rsq_array shape:   (T, N)
        result = {}

        alpha_daily = betas_array[:, :, 0]  # (T, N)
        alpha_ann = alpha_daily * 252 if self.cfg.annualize_alpha else alpha_daily
        result["alpha_ann"] = pd.DataFrame(
            alpha_ann, index=excess.index, columns=tickers
        )

        for k, fname in enumerate(self._factor_names):
            col_name = f"beta_{fname.replace('_rf', '')}"
            result[col_name] = pd.DataFrame(
                betas_array[:, :, k + 1], index=excess.index, columns=tickers
            )

        result["r_squared"] = pd.DataFrame(
            rsq_array, index=excess.index, columns=tickers
        )

        return pd.concat(result, axis=1)

    def _rolling_ols_cpu(
        self,
        Y: np.ndarray,
        X_raw: np.ndarray,
        W: int,
        min_obs: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized rolling OLS on CPU using NumPy.

        Parameters
        ----------
        Y : np.ndarray, shape (T, N) — excess asset returns
        X_raw : np.ndarray, shape (T, K) — factor returns
        W : int — rolling window
        min_obs : int — minimum valid observations per window

        Returns
        -------
        betas : np.ndarray, shape (T, N, K+1) — [intercept, factor betas]
        r_squared : np.ndarray, shape (T, N)
        """
        T, N = Y.shape
        K = X_raw.shape[1]
        betas = np.full((T, N, K + 1), np.nan)
        rsq = np.full((T, N), np.nan)

        for t in range(W - 1, T):
            y_win = Y[t - W + 1 : t + 1]  # (W, N)
            x_win = X_raw[t - W + 1 : t + 1]  # (W, K)

            valid_mask = ~np.isnan(y_win).any(axis=1) & ~np.isnan(x_win).any(axis=1)
            if valid_mask.sum() < min_obs:
                continue

            y_w = y_win[valid_mask]  # (V, N)
            x_w = x_win[valid_mask]  # (V, K)
            ones = np.ones((len(y_w), 1))
            X = np.hstack([ones, x_w])  # (V, K+1) with intercept

            try:
                # OLS: β = (X'X)^{-1} X'Y — solved per batch via lstsq
                XtX = X.T @ X  # (K+1, K+1)
                XtY = X.T @ y_w  # (K+1, N)
                beta_t = np.linalg.lstsq(XtX, XtY, rcond=None)[0]  # (K+1, N)
                betas[t] = beta_t.T  # (N, K+1)

                # R² per asset
                y_hat = X @ beta_t  # (V, N)
                ss_res = ((y_w - y_hat) ** 2).sum(axis=0)
                ss_tot = ((y_w - y_w.mean(axis=0)) ** 2).sum(axis=0)
                rsq[t] = 1.0 - ss_res / (ss_tot + 1e-9)

            except np.linalg.LinAlgError:
                continue

        return betas, rsq

    def _rolling_ols_gpu(
        self,
        Y: np.ndarray,
        X_raw: np.ndarray,
        W: int,
        min_obs: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """CuPy-accelerated batched rolling OLS.

        Transfers the full T×N and T×K arrays to the GPU once, then
        executes all rolling windows via vectorized slice + batched matmul.
        This avoids Python-level loops over time steps and is significantly
        faster for large T.

        The key insight: for each window ending at time t, the OLS normal
        equations are (X_t'X_t)^{-1} X_t'Y_t. We stack these into a 3D
        tensor [window, asset, feature] and use cp.linalg.solve in batch.
        Windows with insufficient valid data are handled via masking.

        Parameters
        ----------
        Y : np.ndarray, shape (T, N)
        X_raw : np.ndarray, shape (T, K)
        W : int
        min_obs : int

        Returns
        -------
        betas : np.ndarray, shape (T, N, K+1)
        r_squared : np.ndarray, shape (T, N)
        """
        cp = self._cp
        try:
            T, N = Y.shape
            K = X_raw.shape[1]
            Kp1 = K + 1  # intercept + factors

            Y_gpu = cp.asarray(Y, dtype=cp.float32)
            X_gpu = cp.asarray(X_raw, dtype=cp.float32)
            ones = cp.ones((T, 1), dtype=cp.float32)
            X_aug = cp.hstack([ones, X_gpu])  # (T, K+1) with intercept

            betas_out = np.full((T, N, Kp1), np.nan, dtype=np.float32)
            rsq_out = np.full((T, N), np.nan, dtype=np.float32)

            # Batch all valid windows
            window_ends = range(W - 1, T)

            # Compute cumulative XtX and XtY using prefix sums for efficiency
            # XtX[t] = X[t-W+1:t+1].T @ X[t-W+1:t+1]  → (K+1, K+1)
            # XtY[t] = X[t-W+1:t+1].T @ Y[t-W+1:t+1]  → (K+1, N)
            #
            # Use sliding window via cumulative sum trick:
            # S[t] = Σ_{i=0}^{t} x_i x_i.T
            # XtX for window [t-W+1, t] = S[t] - S[t-W]

            # Outer products: (T, K+1, K+1)
            xx = cp.einsum("ti,tj->tij", X_aug, X_aug)  # (T, Kp1, Kp1)
            xy = cp.einsum("ti,tj->tij", X_aug, Y_gpu)  # (T, Kp1, N)

            # Cumulative sums
            cum_xx = cp.cumsum(xx, axis=0)  # (T, Kp1, Kp1)
            cum_xy = cp.cumsum(xy, axis=0)  # (T, Kp1, N)

            for t in window_ends:
                if t >= W:
                    XtX = cum_xx[t] - cum_xx[t - W]
                    XtY = cum_xy[t] - cum_xy[t - W]
                else:
                    XtX = cum_xx[t]
                    XtY = cum_xy[t]

                # Add ridge regularization for numerical stability
                XtX += cp.eye(Kp1, dtype=cp.float32) * 1e-6

                try:
                    beta_t = cp.linalg.solve(XtX, XtY)  # (Kp1, N)
                    betas_out[t] = beta_t.T.get()  # (N, Kp1)

                    # R² computation
                    y_win = Y_gpu[t - W + 1 : t + 1]  # (W, N)
                    x_win = X_aug[t - W + 1 : t + 1]  # (W, Kp1)
                    y_hat = x_win @ beta_t  # (W, N)
                    ss_res = cp.sum((y_win - y_hat) ** 2, axis=0)
                    ss_tot = cp.sum((y_win - y_win.mean(axis=0)) ** 2, axis=0)
                    rsq_t = 1.0 - ss_res / (ss_tot + 1e-9)
                    rsq_out[t] = rsq_t.get()

                except cp.linalg.LinAlgError:
                    continue

            return betas_out.astype(np.float64), rsq_out.astype(np.float64)

        except Exception as exc:
            logger.warning(
                f"[FF] GPU rolling OLS failed ({exc}) — falling back to CPU."
            )
            return self._rolling_ols_cpu(Y, X_raw, W, min_obs)

    # ------------------------------------------------------------------
    # Walk-forward quarterly output
    # ------------------------------------------------------------------

    def forecast_quarterly(
        self,
        log_returns: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Return quarter-end snapshots of rolling factor exposures.

        Parameters
        ----------
        log_returns : pd.DataFrame
            Full daily log return history (all assets).
        rebalance_dates : pd.DatetimeIndex
            Quarter-end dates at which to sample the rolling betas.

        Returns
        -------
        pd.DataFrame
            MultiIndex columns (output_name, ticker).
            Index = rebalance_dates (inner-joined with available data).
        """
        rolling = self.rolling_betas(log_returns)

        # Sample rolling betas at quarter-end dates (or nearest prior date)
        # The .asof() method handles cases where a rebalance date is not a
        # trading day by returning the last available observation before it.
        result_frames = {}
        for col in rolling.columns.get_level_values(0).unique():
            df_col = rolling[col]  # (T, N)
            sampled_rows = {}
            for date in rebalance_dates:
                try:
                    sampled_rows[date] = df_col.asof(date)
                except KeyError:
                    sampled_rows[date] = pd.Series(np.nan, index=df_col.columns)
            result_frames[col] = pd.DataFrame(sampled_rows).T

        return pd.concat(result_frames, axis=1)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def factor_summary(self, log_returns: pd.DataFrame) -> pd.DataFrame:
        """Compute mean in-sample factor exposures (full history, single window).

        Useful for a quick sanity check on factor loadings.

        Parameters
        ----------
        log_returns : pd.DataFrame
            Daily log returns.

        Returns
        -------
        pd.DataFrame
            Index = tickers. Columns = alpha_ann, beta_mkt, beta_smb,
            beta_hml, [beta_rmw, beta_cma], r_squared.
        """
        excess, factors = self._align_factors(log_returns)
        Y = excess.values
        X_raw = factors.values
        ones = np.ones((len(Y), 1))
        X = np.hstack([ones, X_raw])

        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # (K+1, N)
        y_hat = X @ beta
        ss_res = ((Y - y_hat) ** 2).sum(axis=0)
        ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
        rsq = 1.0 - ss_res / (ss_tot + 1e-9)

        tickers = list(excess.columns)
        rows: dict[str, dict] = {t: {} for t in tickers}

        alpha_ann = beta[0] * 252 if self.cfg.annualize_alpha else beta[0]
        for j, ticker in enumerate(tickers):
            rows[ticker]["alpha_ann"] = alpha_ann[j]
            for k, fname in enumerate(self._factor_names):
                col = f"beta_{fname.replace('_rf', '')}"
                rows[ticker][col] = beta[k + 1, j]
            rows[ticker]["r_squared"] = rsq[j]

        return pd.DataFrame(rows).T
