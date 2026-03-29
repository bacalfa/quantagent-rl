"""
forecasting/pipeline.py
=======================
Forecasting pipeline orchestrator for QuantAgent-RL.

Ties together GARCHForecaster, RegimeDetector, and FamaFrenchFactors into a
single ``ForecastingPipeline`` that produces one ``ForecastBundle`` per
walk-forward fold. The bundle contains all quarterly forecasts needed to
construct the RL state vector's quantitative component.

Walk-Forward Contract
---------------------
Each ``ForecastBundle`` is produced from a ``WalkForwardFold`` and contains:

  Training forecasts (in-sample, for RL environment training):
    - GARCH volatility at each training quarter-end
    - HMM regime label and probabilities at each training quarter-end
    - Factor betas at each training quarter-end

  Test forecasts (out-of-sample, for walk-forward evaluation):
    - Same outputs, computed without any look-ahead into the test period

The pipeline enforces this boundary: GARCH parameters and HMM are fitted
exclusively on training data and then applied to the test period.

Usage
-----
>>> from forecasting import ForecastingPipeline, ForecastConfig
>>> from data import DataPipeline, DataConfig
>>>
>>> data_pipeline = DataPipeline(DataConfig()).run()
>>> fcst_pipeline = ForecastingPipeline(ForecastConfig())
>>> fcst_pipeline.load_factors()
>>>
>>> fold_0 = data_pipeline.get_fold(0)
>>> bundle = fcst_pipeline.run_fold(fold_0)
>>>
>>> bundle.train_vol          # quarterly GARCH vol, training period
>>> bundle.train_regime       # quarterly HMM regime, training period
>>> bundle.train_betas        # quarterly FF betas, training period
>>> bundle.rl_state_extension # concatenated quarterly forecast features
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.pipeline import WalkForwardFold
from forecasting.config import ForecastConfig
from forecasting.factors import FamaFrenchFactors
from forecasting.garch import GARCHForecaster
from forecasting.regime import RegimeDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ForecastBundle — container for one fold's forecast outputs
# ---------------------------------------------------------------------------


@dataclass
class ForecastBundle:
    """All forecast outputs for one walk-forward fold.

    Attributes
    ----------
    fold_idx : int
    train_dates, test_dates : pd.DatetimeIndex
        Quarter-end dates for each split.
    train_vol : pd.DataFrame
        GARCH annualized volatility forecasts.
        Index = train_dates, columns = tickers.
    test_vol : pd.DataFrame
        Same as train_vol but for the test period.
    train_regime : pd.DataFrame
        HMM regime outputs at each training quarter-end.
        Columns: regime_label, regime_index, p_bear, p_sideways, p_bull.
    test_regime : pd.DataFrame
    train_betas : pd.DataFrame
        Fama-French factor exposures at each training quarter-end.
        MultiIndex columns: (output_name, ticker).
    test_betas : pd.DataFrame
    garch_params : pd.DataFrame
        Fitted GARCH parameters for all assets (from the last training fit).
    regime_transitions : pd.DataFrame
        HMM transition matrix with state labels.
    """

    fold_idx: int

    train_dates: pd.DatetimeIndex = field(default_factory=pd.DatetimeIndex)
    test_dates: pd.DatetimeIndex = field(default_factory=pd.DatetimeIndex)

    train_vol: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_vol: pd.DataFrame = field(default_factory=pd.DataFrame)

    train_regime: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_regime: pd.DataFrame = field(default_factory=pd.DataFrame)

    train_betas: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_betas: pd.DataFrame = field(default_factory=pd.DataFrame)

    garch_params: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_transitions: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def rl_state_extension(self) -> pd.DataFrame:
        """Concatenate all forecast features into a flat quarterly DataFrame.

        Returns a DataFrame indexed by rebalancing dates (train + test) where
        each row is the complete forecasting-module contribution to the RL
        state vector for that quarter. The RL environment stacks this with
        the ``data`` module's technical feature matrix.

        Columns:
          vol_{ticker}         : GARCH vol forecast
          regime_index         : 0/1/2 (bear/sideways/bull)
          p_bear, p_sideways, p_bull : HMM probabilities
          alpha_ann_{ticker}   : FF rolling alpha
          beta_mkt_{ticker}    : FF market beta
          beta_smb_{ticker}    : FF SMB beta
          beta_hml_{ticker}    : FF HML beta
          [beta_rmw, beta_cma if FF5]
          r_squared_{ticker}   : FF in-window R²

        Returns
        -------
        pd.DataFrame
            Index = all rebalance dates (train + test), sorted.
        """
        frames = []

        # --- GARCH vol ---
        vol = pd.concat([self.train_vol, self.test_vol]).sort_index()
        vol.columns = [f"vol_{t}" for t in vol.columns]
        frames.append(vol)

        # --- Regime ---
        regime = pd.concat([self.train_regime, self.test_regime]).sort_index()
        # Keep numeric columns only for state extension (label is str)
        regime_num = regime.drop(columns=["regime_label"], errors="ignore")
        frames.append(regime_num)

        # --- Factor betas ---
        if not self.train_betas.empty:
            betas = pd.concat([self.train_betas, self.test_betas]).sort_index()
            # Flatten MultiIndex: (alpha_ann, AAPL) → alpha_ann_AAPL
            betas.columns = [f"{grp}_{ticker}" for grp, ticker in betas.columns]
            frames.append(betas)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1).sort_index()
        return combined.ffill(limit=1)

    def __repr__(self) -> str:
        return (
            f"ForecastBundle("
            f"fold={self.fold_idx}, "
            f"train={len(self.train_dates)}Q, "
            f"test={len(self.test_dates)}Q, "
            f"n_assets_vol={self.train_vol.shape[1] if not self.train_vol.empty else 0}"
            f")"
        )


# ---------------------------------------------------------------------------
# ForecastingPipeline
# ---------------------------------------------------------------------------


class ForecastingPipeline:
    """Orchestrates GARCH, HMM, and Fama-French forecasting across folds.

    Parameters
    ----------
    config : ForecastConfig
        Master forecasting configuration.

    Examples
    --------
    >>> pipeline = ForecastingPipeline(ForecastConfig())
    >>> pipeline.load_factors()
    >>> bundle = pipeline.run_fold(data_fold)
    """

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.cfg = config or ForecastConfig()

        self._garch = GARCHForecaster(self.cfg.garch)
        self._regime = RegimeDetector(self.cfg.regime)
        self._factors = FamaFrenchFactors(self.cfg.factors)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_factors(
        self, use_cache: bool = True, force_refresh: bool = False
    ) -> "ForecastingPipeline":
        """Download and cache Fama-French factor data.

        Must be called before ``run_fold()``.

        Returns
        -------
        self
        """
        self._factors.load_factors(use_cache=use_cache, force_refresh=force_refresh)
        return self

    # ------------------------------------------------------------------
    # Single fold
    # ------------------------------------------------------------------

    def run_fold(self, fold: WalkForwardFold) -> ForecastBundle:  # type: ignore[name-defined]
        """Produce all forecast outputs for one walk-forward fold.

        Parameters
        ----------
        fold : WalkForwardFold
            Output of DataPipeline.get_fold(i).

        Returns
        -------
        ForecastBundle
        """
        logger.info(
            f"[ForecastPipeline] Running fold {fold.fold_idx}: "
            f"train {fold.train_start.date()} → {fold.train_end.date()}, "
            f"test {fold.test_start.date()} → {fold.test_end.date()}"
        )

        # Combined returns for full date range (train + test)
        train_returns = (
            pd.concat([fold.train_prices]).pct_change().apply(np.log1p).dropna()
        )
        all_prices = pd.concat([fold.train_prices, fold.test_prices])
        all_returns = all_prices.pct_change().apply(np.log1p).dropna()

        train_returns = np.log(fold.train_prices / fold.train_prices.shift(1)).dropna()
        all_returns = np.log(all_prices / all_prices.shift(1)).dropna()

        bundle = ForecastBundle(
            fold_idx=fold.fold_idx,
            train_dates=fold.train_dates,
            test_dates=fold.test_dates,
        )

        # ------------------------------------------------------------------
        # 1. GARCH volatility
        # ------------------------------------------------------------------
        logger.info(f"[ForecastPipeline] Fold {fold.fold_idx}: GARCH")
        bundle.train_vol, bundle.test_vol, bundle.garch_params = self._run_garch(
            train_returns, all_returns, fold
        )

        # ------------------------------------------------------------------
        # 2. HMM regime detection
        # ------------------------------------------------------------------
        logger.info(f"[ForecastPipeline] Fold {fold.fold_idx}: HMM regime")
        bundle.train_regime, bundle.test_regime, bundle.regime_transitions = (
            self._run_regime(train_returns, all_returns, fold)
        )

        # ------------------------------------------------------------------
        # 3. Fama-French factor exposures
        # ------------------------------------------------------------------
        logger.info(f"[ForecastPipeline] Fold {fold.fold_idx}: Fama-French")
        bundle.train_betas, bundle.test_betas = self._run_factors(
            train_returns, all_returns, fold
        )

        logger.info(f"[ForecastPipeline] Fold {fold.fold_idx} complete: {bundle}")
        return bundle

    # ------------------------------------------------------------------
    # All folds
    # ------------------------------------------------------------------

    def run_all_folds(self, data_pipeline: object) -> list[ForecastBundle]:
        """Produce forecast bundles for every walk-forward fold.

        Parameters
        ----------
        data_pipeline : DataPipeline
            Must have been run (``pipeline.run()`` called).

        Returns
        -------
        list[ForecastBundle]
        """
        bundles = []
        for i in range(data_pipeline.n_folds):
            fold = data_pipeline.get_fold(i)
            bundle = self.run_fold(fold)
            bundles.append(bundle)
        return bundles

    # ------------------------------------------------------------------
    # Component runners
    # ------------------------------------------------------------------

    def _run_garch(
        self,
        train_returns: pd.DataFrame,
        all_returns: pd.DataFrame,
        fold: object,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit GARCH on training data and forecast over train + test quarters."""
        # Fit on training data only
        self._garch.fit(train_returns)
        garch_params = self._garch.parameter_summary()

        # Training vol forecasts
        train_vol = self._garch.forecast_quarterly(
            log_returns=train_returns,
            rebalance_dates=fold.train_dates,
        )

        # Test vol forecasts — parameters fixed from training fit
        # Propagate the variance recursion forward using test-period returns
        # without re-estimating parameters (refit_every_n_quarters keeps
        # the fitting anchored to the training window boundary)
        test_vol = self._garch.forecast_quarterly(
            log_returns=all_returns,
            rebalance_dates=fold.test_dates,
        )

        return train_vol, test_vol, garch_params

    def _run_regime(
        self,
        train_returns: pd.DataFrame,
        all_returns: pd.DataFrame,
        fold: object,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit HMM on training data and decode regimes over train + test quarters."""
        self._regime.fit(train_returns)

        train_regime = self._regime.forecast_quarterly(
            log_returns=train_returns,
            rebalance_dates=fold.train_dates,
        )
        test_regime = self._regime.forecast_quarterly(
            log_returns=all_returns,
            rebalance_dates=fold.test_dates,
        )
        transitions = self._regime.transition_matrix()

        return train_regime, test_regime, transitions

    def _run_factors(
        self,
        train_returns: pd.DataFrame,
        all_returns: pd.DataFrame,
        fold: object,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute rolling FF betas over train + test periods (date-bounded)."""
        # Training betas: rolling OLS bounded to training data
        train_betas = self._factors.forecast_quarterly(
            log_returns=train_returns,
            rebalance_dates=fold.train_dates,
        )

        # Test betas: rolling OLS on full (train + test) returns, then slice
        # This is valid — rolling OLS uses only past data within each window,
        # so no future returns are seen when computing betas at test dates.
        all_betas = self._factors.forecast_quarterly(
            log_returns=all_returns,
            rebalance_dates=fold.test_dates,
        )

        return train_betas, all_betas
