"""
forecasting/config.py
=====================
Configuration dataclasses for the QuantAgent-RL forecasting module.

Three sub-configs are provided — one per forecasting component — and a
master ForecastConfig that groups them. Downstream modules (rl, backtest)
only need to import ForecastConfig.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# GARCH Config
# ---------------------------------------------------------------------------


@dataclass
class GARCHConfig:
    """Hyperparameters for the GARCH(1,1) volatility forecaster.

    Parameters
    ----------
    p : int
        Lag order of the ARCH term (squared residuals). Default 1.
    q : int
        Lag order of the GARCH term (conditional variance). Default 1.
    dist : str
        Innovation distribution passed to the ``arch`` library.
        Options: 'normal', 't' (Student-t), 'skewt' (skewed Student-t).
        'normal' is fastest; 't' is more realistic for equity returns.
    rescale : bool
        Rescale returns by 100 before fitting to improve numerical
        conditioning of the optimizer. The ``arch`` library recommends
        this for daily return series.
    horizon : int
        Number of steps ahead to forecast conditional volatility.
        For quarterly rebalancing with daily steps, set to 63 (≈ 1 quarter).
    annualize : bool
        If True, multiply the daily vol forecast by sqrt(252) before
        returning. Keeps units consistent with other modules.
    min_obs : int
        Minimum number of non-null return observations required before
        fitting a GARCH model for a given asset. Assets below this
        threshold receive the cross-sectional median forecast.
    use_gpu : bool | None
        Controls CuPy acceleration for batch log-likelihood computation.
        None = auto-detect.
    refit_every_n_quarters : int
        Re-estimate GARCH parameters every N walk-forward quarters.
        Between refits, only the variance recursion is updated (faster).
        Set to 1 to refit at every fold boundary.
    """

    p: int = 1
    q: int = 1
    dist: str = "normal"
    rescale: bool = True
    horizon: int = 63
    annualize: bool = True
    min_obs: int = 252
    use_gpu: bool | None = None
    refit_every_n_quarters: int = 4


# ---------------------------------------------------------------------------
# Regime Config
# ---------------------------------------------------------------------------


@dataclass
class RegimeConfig:
    """Hyperparameters for the Hidden Markov Model regime detector.

    Parameters
    ----------
    n_states : int
        Number of hidden regime states. 3 is the standard choice:
        bull, bear, and sideways/transition.
    n_iter : int
        Maximum EM iterations for HMM fitting.
    covariance_type : str
        HMM covariance structure: 'diag' or 'full'.
        'diag' is faster and usually sufficient for 1D return inputs.
    random_state : int
        Seed for reproducibility of HMM initialization.
    lookback_window : int
        Rolling window of trading days used as the HMM input sequence
        at inference time (not at fit time — the full training history
        is used for fitting). Controls how quickly the regime signal
        responds to recent market conditions.
    feature_set : list[str]
        Names of the return/vol columns to feed into the HMM as
        observations. Must match columns produced by FeatureEngineer.
        Default uses log returns and realized volatility.
    state_labels : dict[int, str]
        Human-readable labels assigned after sorting states by mean
        return. Populated automatically after fitting.
    use_gpu : bool | None
        Controls CuPy acceleration for Viterbi decoding.
        None = auto-detect.
    """

    n_states: int = 3
    n_iter: int = 200
    covariance_type: str = "diag"
    random_state: int = 42
    lookback_window: int = 252
    feature_set: list[str] = field(
        default_factory=lambda: ["log_return", "realized_vol_21d"]
    )
    state_labels: dict[int, str] = field(default_factory=dict)
    use_gpu: bool | None = None


# ---------------------------------------------------------------------------
# Fama-French Config
# ---------------------------------------------------------------------------


@dataclass
class FamaFrenchConfig:
    """Hyperparameters for the Fama-French factor model.

    Parameters
    ----------
    n_factors : int
        3 = Market, SMB, HML (Fama-French 3-factor).
        5 = Market, SMB, HML, RMW, CMA (Fama-French 5-factor).
    rolling_window : int
        Rolling OLS window in trading days for time-varying factor
        exposure estimation.
    min_obs_fraction : float
        Minimum fraction of the rolling window that must be non-null
        for an OLS estimate to be produced (avoids fitting on thin data).
    cache_dir : str
        Directory for caching downloaded Fama-French factor CSVs.
    use_gpu : bool | None
        Controls CuPy-accelerated batched rolling OLS.
        None = auto-detect.
    annualize_alpha : bool
        Multiply the daily rolling alpha by 252 before returning.
    """

    n_factors: int = 3
    rolling_window: int = 63
    min_obs_fraction: float = 0.8
    cache_dir: str = "data/cache/ff_factors"
    use_gpu: bool | None = None
    annualize_alpha: bool = True


# ---------------------------------------------------------------------------
# Master Forecast Config
# ---------------------------------------------------------------------------


@dataclass
class ForecastConfig:
    """Master configuration for the QuantAgent-RL forecasting module.

    Bundles GARCH, regime, and Fama-French sub-configs.

    Usage
    -----
    >>> cfg = ForecastConfig()                           # all defaults
    >>> cfg = ForecastConfig(garch=GARCHConfig(dist='t'))
    """

    garch: GARCHConfig = field(default_factory=GARCHConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    factors: FamaFrenchConfig = field(default_factory=FamaFrenchConfig)
