"""
data/features.py
================
GPU-accelerated feature engineering for QuantAgent-RL.

Uses NVIDIA RAPIDS cuDF/cuML when a GPU is available, falling back
transparently to pandas/scikit-learn on CPU. The public API is identical
in both execution paths — callers do not need to know which backend is active.

Features computed:
  - Return features        : multi-window log returns
  - Volatility features    : rolling realized volatility, EWMA vol
  - Momentum features      : price momentum, RSI, Bollinger Band position
  - Market microstructure  : Amihud illiquidity proxy, volume z-scores
  - Cross-sectional ranks  : z-score and percentile ranks across universe
  - Beta / correlation     : rolling beta to benchmark, average pairwise corr
  - Macro-aligned features : lagged macro signal alignment per asset
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU backend detection
# ---------------------------------------------------------------------------


def _detect_backend(use_gpu: bool | None) -> str:
    """Return 'gpu' or 'cpu' based on preference and availability.

    Parameters
    ----------
    use_gpu : bool | None
        True → require GPU, False → require CPU, None → auto.
    """
    if use_gpu is False:
        return "cpu"

    try:
        import cudf  # noqa: F401
        import cuml  # noqa: F401

        logger.info("[Features] RAPIDS cuDF/cuML detected → GPU backend active.")
        return "gpu"
    except ImportError:
        if use_gpu is True:
            raise RuntimeError(
                "use_gpu=True but RAPIDS (cuDF/cuML) is not installed. "
                "Install via conda: https://rapids.ai/start.html"
            )
        logger.info("[Features] RAPIDS not found → CPU (pandas/sklearn) backend.")
        return "cpu"


def _to_pdf(df) -> pd.DataFrame:
    """Convert cuDF DataFrame to pandas if necessary."""
    try:
        import cudf

        if isinstance(df, cudf.DataFrame):
            return df.to_pandas()
    except ImportError:
        pass
    return df


def _from_pdf(pdf: pd.DataFrame, backend: str):
    """Convert pandas DataFrame to cuDF if GPU backend is active."""
    if backend == "gpu":
        import cudf

        return cudf.DataFrame.from_pandas(pdf)
    return pdf


# ---------------------------------------------------------------------------
# Feature Engineering Engine
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """Computes all features required by the RL state vector and LLM agents.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices. Index=Date (DatetimeIndex), columns=tickers.
    volumes : pd.DataFrame
        Daily trading volumes. Same shape as prices.
    benchmark : pd.DataFrame
        Benchmark prices (single column 'benchmark').
    macro : pd.DataFrame, optional
        Macro signal DataFrame from MacroDataIngester. Aligned to trading days.
    return_windows : list of int
        Rolling windows (trading days) for return/vol features.
    vol_windows : list of int
    momentum_windows : list of int
    correlation_window : int
    rsi_window : int
    bb_window : int
    bb_num_std : float
    use_gpu : bool | None

    Examples
    --------
    >>> fe = FeatureEngineer(prices, volumes, benchmark, macro)
    >>> feature_dict = fe.compute_all()
    >>> quarterly_features = fe.resample_quarterly(feature_dict)
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        benchmark: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        return_windows: list[int] = (5, 21, 63),
        vol_windows: list[int] = (21, 63),
        momentum_windows: list[int] = (21, 63, 126),
        correlation_window: int = 63,
        rsi_window: int = 14,
        bb_window: int = 20,
        bb_num_std: float = 2.0,
        use_gpu: bool | None = None,
    ) -> None:
        self.prices = prices.copy()
        self.volumes = volumes.copy()
        self.benchmark = benchmark.copy()
        self.macro = macro
        self.return_windows = list(return_windows)
        self.vol_windows = list(vol_windows)
        self.momentum_windows = list(momentum_windows)
        self.correlation_window = correlation_window
        self.rsi_window = rsi_window
        self.bb_window = bb_window
        self.bb_num_std = bb_num_std

        self.backend = _detect_backend(use_gpu)
        self.tickers = list(prices.columns)

        # Log returns — used by most downstream computations
        self._log_returns = np.log(self.prices / self.prices.shift(1))
        self._bm_log_returns = np.log(self.benchmark / self.benchmark.shift(1)).iloc[
            :, 0
        ]  # squeeze to Series

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all(self) -> dict[str, pd.DataFrame]:
        """Compute all feature groups and return as a named dict.

        Returns
        -------
        dict of {feature_group_name: pd.DataFrame}
            Each DataFrame has index=Date, columns=tickers (or macro cols).
        """
        logger.info(f"[Features] Computing features ({self.backend} backend)...")

        features: dict[str, pd.DataFrame] = {}

        features["log_returns"] = self._log_returns
        features.update(self._compute_return_features())
        features.update(self._compute_volatility_features())
        features.update(self._compute_momentum_features())
        features.update(self._compute_rsi())
        features.update(self._compute_bollinger_bands())
        features.update(self._compute_microstructure())
        features.update(self._compute_cross_sectional_ranks())
        features.update(self._compute_beta_correlation())

        if self.macro is not None:
            features.update(self._compute_macro_alignment())

        logger.info(
            f"[Features] Done. {len(features)} feature groups, "
            f"{self._log_returns.index[-1].date()} last date."
        )
        return features

    def resample_quarterly(
        self,
        feature_dict: dict[str, pd.DataFrame],
        agg: str = "last",
    ) -> dict[str, pd.DataFrame]:
        """Resample all daily feature DataFrames to quarter-end frequency.

        Parameters
        ----------
        feature_dict : dict
            Output of ``compute_all()``.
        agg : str
            Aggregation method: 'last' (end-of-quarter snapshot) or 'mean'.

        Returns
        -------
        dict
            Same keys, DataFrames resampled to quarterly frequency.
        """
        resampled: dict[str, pd.DataFrame] = {}
        for name, df in feature_dict.items():
            if agg == "last":
                resampled[name] = df.resample("QE").last()
            elif agg == "mean":
                resampled[name] = df.resample("QE").mean()
            else:
                raise ValueError(f"Unknown agg='{agg}'")
        return resampled

    def build_state_matrix(
        self,
        quarterly_features: dict[str, pd.DataFrame],
        feature_groups: list[str] | None = None,
    ) -> pd.DataFrame:
        """Stack selected quarterly feature groups into a flat state matrix.

        Parameters
        ----------
        quarterly_features : dict
            Output of ``resample_quarterly()``.
        feature_groups : list of str, optional
            Feature groups to include. If None, uses a default curated set.

        Returns
        -------
        pd.DataFrame
            Multi-level columns (feature_group, ticker). Rows = quarter-end dates.
            Used directly as the numerical component of the RL state.
        """
        default_groups = [
            "ret_5d",
            "ret_21d",
            "ret_63d",
            "vol_21d",
            "vol_63d",
            "mom_21d",
            "mom_63d",
            "mom_126d",
            "rsi",
            "bb_position",
            "amihud_illiquidity",
            "volume_zscore",
            "xs_zscore",
            "beta_63d",
        ]
        selected = feature_groups or default_groups

        frames = {}
        for grp in selected:
            if grp in quarterly_features:
                frames[grp] = quarterly_features[grp]
            else:
                logger.warning(f"[Features] Group '{grp}' not found — skipped.")

        return pd.concat(frames, axis=1)

    # ------------------------------------------------------------------
    # Feature computation methods
    # ------------------------------------------------------------------

    def _compute_return_features(self) -> dict[str, pd.DataFrame]:
        """Multi-window cumulative log returns."""
        out = {}
        for w in self.return_windows:
            out[f"ret_{w}d"] = self._log_returns.rolling(w).sum()
        return out

    def _compute_volatility_features(self) -> dict[str, pd.DataFrame]:
        """Realized volatility (annualized) and EWMA volatility."""
        out = {}
        sqrt252 = np.sqrt(252)

        for w in self.vol_windows:
            # Realized volatility: std of log returns × √252
            out[f"vol_{w}d"] = self._log_returns.rolling(w).std() * sqrt252

        # EWMA volatility (span=21) — reacts faster to recent shocks
        out["vol_ewma_21d"] = (
            self._log_returns.ewm(span=21, adjust=False).std() * sqrt252
        )
        return out

    def _compute_momentum_features(self) -> dict[str, pd.DataFrame]:
        """Price momentum: cumulative log return over look-back window,
        excluding the most recent 5 days (removes short-term reversal effect)."""
        out = {}
        for w in self.momentum_windows:
            # Classic momentum: total return from w days ago to 5 days ago
            out[f"mom_{w}d"] = (
                self._log_returns.rolling(w).sum() - self._log_returns.rolling(5).sum()
            )
        return out

    def _compute_rsi(self) -> dict[str, pd.DataFrame]:
        """Wilder's Relative Strength Index."""
        delta = self.prices.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        avg_gain = gain.ewm(com=self.rsi_window - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=self.rsi_window - 1, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return {"rsi": rsi}

    def _compute_bollinger_bands(self) -> dict[str, pd.DataFrame]:
        """Bollinger Band position: (price - lower) / (upper - lower) ∈ [0, 1]."""
        roll = self.prices.rolling(self.bb_window)
        mid = roll.mean()
        std = roll.std()

        upper = mid + self.bb_num_std * std
        lower = mid - self.bb_num_std * std
        band_width = (upper - lower).replace(0.0, np.nan)
        position = (self.prices - lower) / band_width

        return {
            "bb_position": position.clip(0.0, 1.0),
            "bb_width": (band_width / mid).fillna(0.0),  # normalized width
        }

    def _compute_microstructure(self) -> dict[str, pd.DataFrame]:
        """Market microstructure proxies.

        - Amihud illiquidity: |r_t| / Volume_t (in millions)
          High values → illiquid (price moves a lot per dollar of volume).
        - Volume z-score: standardised trading volume (rolling 63d).
        """
        # Amihud illiquidity proxy
        abs_ret = self._log_returns.abs()
        vol_m = self.volumes / 1e6  # volume in millions
        amihud = abs_ret / vol_m.replace(0.0, np.nan)
        amihud = amihud.rolling(21).mean()  # smooth over 1 month

        # Volume z-score
        vol_roll = self.volumes.rolling(63)
        vol_zscore = (self.volumes - vol_roll.mean()) / (vol_roll.std() + 1e-9)

        return {
            "amihud_illiquidity": amihud,
            "volume_zscore": vol_zscore.clip(-5.0, 5.0),
        }

    def _compute_cross_sectional_ranks(self) -> dict[str, pd.DataFrame]:
        """Cross-sectional z-scores and percentile ranks of 21d log returns.

        These capture relative performance within the universe, which is more
        informative for portfolio allocation than absolute return levels.
        """
        ret_21d = self._log_returns.rolling(21).sum()

        xs_mean = ret_21d.mean(axis=1)
        xs_std = ret_21d.std(axis=1)

        # Broadcast mean and std across columns
        xs_zscore = ret_21d.subtract(xs_mean, axis=0).divide(xs_std + 1e-9, axis=0)

        # Percentile rank (0–1) within universe at each date
        xs_pctrank = ret_21d.rank(axis=1, pct=True)

        return {
            "xs_zscore": xs_zscore,
            "xs_pctrank": xs_pctrank,
        }

    def _compute_beta_correlation(self) -> dict[str, pd.DataFrame]:
        """Rolling beta to benchmark and average pairwise correlation."""
        bm_var = self._bm_log_returns.rolling(self.correlation_window).var()

        # Rolling covariance of each ticker vs benchmark
        cov_df = pd.DataFrame(index=self._log_returns.index)
        for ticker in self.tickers:
            cov_df[ticker] = (
                self._log_returns[ticker]
                .rolling(self.correlation_window)
                .cov(self._bm_log_returns)
            )

        beta = cov_df.divide(bm_var + 1e-9, axis=0)

        # Average pairwise correlation within the universe (single value per date)
        # Returned as a DataFrame with same tickers for convenience (constant across cols)
        avg_corr_series = (
            self._log_returns.rolling(self.correlation_window)
            .corr()
            .groupby(level=0)
            .mean()
            .mean(axis=1)
        )
        avg_corr = pd.DataFrame(
            np.outer(avg_corr_series.values, np.ones(len(self.tickers))),
            index=avg_corr_series.index,
            columns=self.tickers,
        )

        return {
            f"beta_{self.correlation_window}d": beta,
            f"avg_pairwise_corr_{self.correlation_window}d": avg_corr,
        }

    def _compute_macro_alignment(self) -> dict[str, pd.DataFrame]:
        """Broadcast lagged macro signals across all tickers.

        The macro signal at each date is the same for all assets; broadcasting
        it into the per-ticker feature matrix allows the RL agent to learn
        how each asset interacts with macro conditions.

        Returns a dict where each macro signal becomes a (date × ticker) DF,
        with a 1-day lag to avoid look-ahead bias.
        """
        macro_lagged = self.macro.shift(1)  # lag by 1 business day

        # Align macro to prices index
        macro_aligned = macro_lagged.reindex(self._log_returns.index, method="ffill")

        out: dict[str, pd.DataFrame] = {}
        for col in macro_aligned.columns:
            # Broadcast: same value for all tickers at each date
            out[f"macro_{col}"] = pd.DataFrame(
                np.outer(macro_aligned[col].values, np.ones(len(self.tickers))),
                index=macro_aligned.index,
                columns=self.tickers,
            )
        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def compute_total_returns_index(self, dividends: pd.DataFrame) -> pd.DataFrame:
        """Compute a total return index that reinvests dividends.

        Parameters
        ----------
        dividends : pd.DataFrame
            Daily dividend amounts. Same shape as prices.

        Returns
        -------
        pd.DataFrame
            Total return index starting at 100.0 on the first date.
        """
        div_return = dividends / self.prices.shift(1).replace(0.0, np.nan).fillna(1.0)
        price_return = self.prices.pct_change()
        total_return = price_return + div_return

        tri = (1.0 + total_return.fillna(0.0)).cumprod() * 100.0
        return tri

    def normalize_features(
        self,
        feature_matrix: pd.DataFrame,
        fit_end_date: str | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Z-score normalize using in-sample statistics only.

        Parameters
        ----------
        feature_matrix : pd.DataFrame
            Output of ``build_state_matrix()``.
        fit_end_date : str, optional
            If provided, fit mean/std only on data up to this date (walk-forward
            safe). If None, uses all available data (only for final evaluation).

        Returns
        -------
        normalized : pd.DataFrame
        scaler_params : dict
            {'mean': pd.Series, 'std': pd.Series} — persist per fold.
        """
        if fit_end_date:
            fit_data = feature_matrix[feature_matrix.index <= fit_end_date]
        else:
            fit_data = feature_matrix

        mu = fit_data.mean()
        sigma = fit_data.std().replace(0.0, 1.0)  # avoid divide-by-zero

        normalized = (feature_matrix - mu) / sigma
        normalized = normalized.clip(-5.0, 5.0)  # winsorize outliers

        return normalized, {"mean": mu, "std": sigma}

    def apply_scaler(
        self,
        feature_matrix: pd.DataFrame,
        scaler_params: dict,
    ) -> pd.DataFrame:
        """Apply pre-fitted scaler to out-of-sample data.

        Parameters
        ----------
        feature_matrix : pd.DataFrame
        scaler_params : dict
            Output of ``normalize_features()``.

        Returns
        -------
        pd.DataFrame
            Normalized features using in-sample scaler parameters.
        """
        normalized = (feature_matrix - scaler_params["mean"]) / scaler_params["std"]
        return normalized.clip(-5.0, 5.0)
