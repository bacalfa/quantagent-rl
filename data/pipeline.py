"""
data/pipeline.py
================
End-to-end data pipeline orchestrator for QuantAgent-RL.

Ties together:
  MarketDataIngester → MacroDataIngester → SECFilingIngester
  → Universe validation → FeatureEngineer
  → Walk-forward split generation

This is the single entry point that downstream modules (forecasting, rl,
agents) use to obtain clean, normalized, walk-forward-safe datasets.

Usage
-----
>>> from data.pipeline import DataPipeline
>>> from data.config import DataConfig
>>>
>>> cfg = DataConfig()
>>> pipeline = DataPipeline(cfg)
>>> pipeline.run()
>>>
>>> # Access a specific walk-forward fold
>>> fold = pipeline.get_fold(fold_idx=0)
>>> fold.train_state_matrix    # quarterly feature matrix (in-sample)
>>> fold.test_state_matrix     # quarterly feature matrix (out-of-sample)
>>> fold.train_prices          # adjusted close prices (in-sample)
>>> fold.sec_metadata          # SEC filing metadata DataFrame
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from data.config import DataConfig
from data.features import FeatureEngineer
from data.ingestion import MacroDataIngester, MarketDataIngester, SECFilingIngester
from data.universe import Universe, get_rebalance_dates, get_walk_forward_splits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fold dataclass — clean container for one walk-forward fold
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardFold:
    """One walk-forward train/test fold produced by DataPipeline.

    Attributes
    ----------
    fold_idx : int
    train_dates : pd.DatetimeIndex
        Quarter-end dates in the training window.
    test_dates : pd.DatetimeIndex
        Quarter-end dates in the test window.
    train_prices : pd.DataFrame
        Adjusted close prices (daily) up to the last training date.
    test_prices : pd.DataFrame
        Adjusted close prices (daily) for the test window.
    train_dividends, test_dividends : pd.DataFrame
    train_volumes, test_volumes : pd.DataFrame
    train_state_matrix : pd.DataFrame
        Quarterly normalized feature matrix for training.
    test_state_matrix : pd.DataFrame
        Quarterly normalized feature matrix for test (using train scaler).
    macro : pd.DataFrame
        Full macro signal DataFrame (training + test — agents use this).
    sec_metadata : pd.DataFrame
        SEC filing metadata (all filings up to last test date).
    scaler_params : dict
        {'mean': pd.Series, 'std': pd.Series} fitted on training data.
    tickers : list of str
        Valid tickers in this fold's universe.
    train_total_returns : pd.DataFrame
        Total return index (dividends reinvested) for training period.
    """

    fold_idx: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex

    train_prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_dividends: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_dividends: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_volumes: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_volumes: pd.DataFrame = field(default_factory=pd.DataFrame)

    train_state_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_state_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)

    macro: pd.DataFrame = field(default_factory=pd.DataFrame)
    sec_metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    scaler_params: dict = field(default_factory=dict)
    tickers: list[str] = field(default_factory=list)
    train_total_returns: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def train_start(self) -> pd.Timestamp:
        return self.train_dates[0]

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train_dates[-1]

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test_dates[0]

    @property
    def test_end(self) -> pd.Timestamp:
        return self.test_dates[-1]

    @property
    def n_train_quarters(self) -> int:
        return len(self.train_dates)

    @property
    def n_test_quarters(self) -> int:
        return len(self.test_dates)

    def __repr__(self) -> str:
        return (
            f"WalkForwardFold("
            f"fold={self.fold_idx}, "
            f"train={self.train_start.date()}→{self.train_end.date()} "
            f"({self.n_train_quarters}Q), "
            f"test={self.test_start.date()}→{self.test_end.date()} "
            f"({self.n_test_quarters}Q), "
            f"n_assets={len(self.tickers)})"
        )


# ---------------------------------------------------------------------------
# DataPipeline
# ---------------------------------------------------------------------------


class DataPipeline:
    """Orchestrates the full data pipeline from raw ingestion to fold generation.

    Parameters
    ----------
    config : DataConfig
        Master configuration object.
    log_level : int
        Python logging level for pipeline messages.
    cache_root_dir : str
        Root directory for all cached data (market, macro, SEC).

    Examples
    --------
    >>> pipeline = DataPipeline(DataConfig())
    >>> pipeline.run()
    >>> fold_0 = pipeline.get_fold(0)
    >>> print(fold_0)
    """

    def __init__(
        self,
        config: DataConfig,
        log_level: int = logging.INFO,
        cache_root_dir: str = "../data/cache",
    ) -> None:
        self.cfg = config
        self.cache_root_dir = cache_root_dir
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._prices: pd.DataFrame | None = None
        self._dividends: pd.DataFrame | None = None
        self._volumes: pd.DataFrame | None = None
        self._macro: pd.DataFrame | None = None
        self._sec_metadata: pd.DataFrame | None = None
        self._universe: Universe | None = None
        self._feature_dict: dict[str, pd.DataFrame] | None = None
        self._quarterly_features: dict[str, pd.DataFrame] | None = None
        self._folds: list[WalkForwardFold] | None = None
        self._rebalance_dates: pd.DatetimeIndex | None = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        skip_sec: bool = False,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> "DataPipeline":
        """Execute all pipeline stages in order.

        Parameters
        ----------
        skip_sec : bool
            Skip SEC EDGAR download (useful during development).
        use_cache : bool
            Load cached parquet files where available.
        force_refresh : bool
            Re-download all data, ignoring cache.

        Returns
        -------
        self
            Enables method chaining: ``pipeline.run().get_fold(0)``
        """
        logger.info("=" * 60)
        logger.info("QuantAgent-RL Data Pipeline")
        logger.info("=" * 60)

        self._ingest_market_data(use_cache, force_refresh)
        self._ingest_macro_data(use_cache, force_refresh)

        if not skip_sec:
            self._ingest_sec_filings()
        else:
            self._sec_metadata = pd.DataFrame()
            logger.info("[Pipeline] SEC ingestion skipped.")

        self._build_universe()
        self._engineer_features()
        self._build_folds()

        logger.info(
            f"[Pipeline] Done. {len(self._folds)} walk-forward folds generated."
        )
        return self

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_fold(self, fold_idx: int) -> WalkForwardFold:
        """Retrieve a specific walk-forward fold by index.

        Parameters
        ----------
        fold_idx : int
            0-indexed fold number.

        Returns
        -------
        WalkForwardFold
        """
        if self._folds is None:
            raise RuntimeError("Call pipeline.run() before get_fold().")
        if fold_idx >= len(self._folds):
            raise IndexError(
                f"fold_idx={fold_idx} out of range (n_folds={len(self._folds)})"
            )
        return self._folds[fold_idx]

    @property
    def n_folds(self) -> int:
        """Number of walk-forward folds."""
        if self._folds is None:
            raise RuntimeError("Call pipeline.run() first.")
        return len(self._folds)

    @property
    def universe(self) -> Universe:
        if self._universe is None:
            raise RuntimeError("Call pipeline.run() first.")
        return self._universe

    @property
    def rebalance_dates(self) -> pd.DatetimeIndex:
        if self._rebalance_dates is None:
            raise RuntimeError("Call pipeline.run() first.")
        return self._rebalance_dates

    @property
    def all_prices(self) -> pd.DataFrame:
        if self._prices is None:
            raise RuntimeError("Call pipeline.run() first.")
        return self._prices

    @property
    def macro(self) -> pd.DataFrame:
        if self._macro is None:
            raise RuntimeError("Call pipeline.run() first.")
        return self._macro

    @property
    def sec_metadata(self) -> pd.DataFrame:
        return self._sec_metadata or pd.DataFrame()

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _ingest_market_data(self, use_cache: bool, force_refresh: bool) -> None:
        logger.info("[Pipeline] Stage 1: Market data ingestion")
        ingester = MarketDataIngester(
            tickers=self.cfg.universe.tickers + [self.cfg.universe.benchmark_ticker],
            start_date=self.cfg.dates.start_date,
            end_date=self.cfg.dates.end_date,
            cache_dir=self.cache_root_dir + "/market",
        )
        self._prices, self._dividends, self._volumes = ingester.fetch(
            use_cache=use_cache, force_refresh=force_refresh
        )
        logger.info(
            f"[Pipeline] Market data: {self._prices.shape[1]} tickers, "
            f"{len(self._prices)} trading days "
            f"({self._prices.index[0].date()} → {self._prices.index[-1].date()})"
        )

    def _ingest_macro_data(self, use_cache: bool, force_refresh: bool) -> None:
        logger.info("[Pipeline] Stage 2: Macro data ingestion")
        ingester = MacroDataIngester(
            series_map=self.cfg.macro.series,
            start_date=self.cfg.dates.start_date,
            end_date=self.cfg.dates.end_date,
            api_key=self.cfg.macro.api_key,
            cache_dir=self.cache_root_dir + "/macro",
            ffill_limit=self.cfg.macro.ffill_limit,
        )
        self._macro = ingester.fetch(use_cache=use_cache, force_refresh=force_refresh)
        logger.info(
            f"[Pipeline] Macro data: {len(self._macro.columns)} signals, "
            f"{len(self._macro)} trading days."
        )

    def _ingest_sec_filings(self) -> None:
        logger.info("[Pipeline] Stage 3: SEC filing ingestion")
        ingester = SECFilingIngester(
            tickers=self.cfg.universe.tickers,
            form_types=self.cfg.sec.form_types,
            cache_dir=self.cache_root_dir + "/sec",
            user_agent=self.cfg.sec.user_agent,
            start_date=self.cfg.dates.start_date,
            end_date=self.cfg.dates.end_date,
            max_filings_per_ticker=self.cfg.sec.max_filings_per_ticker,
        )
        self._sec_metadata = ingester.fetch()
        logger.info(
            f"[Pipeline] SEC filings: {len(self._sec_metadata)} filings indexed."
        )

    def _build_universe(self) -> None:
        logger.info("[Pipeline] Stage 4: Universe validation")
        self._universe = Universe(
            tickers=self.cfg.universe.tickers,
            price_data=self._prices,
            start_date=self.cfg.dates.start_date,
            min_history_years=self.cfg.universe.min_history_years,
        )
        # Subset all data to valid tickers
        valid = self._universe.valid_tickers
        self._prices = self._prices[valid]
        self._dividends = self._dividends[valid]
        self._volumes = self._volumes[valid]

        # Rebalance date schedule
        end = self.cfg.dates.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
        self._rebalance_dates = get_rebalance_dates(
            self.cfg.dates.start_date, end, self.cfg.dates.rebalance_freq
        )
        logger.info(
            f"[Pipeline] Universe: {self._universe.n_assets} assets, "
            f"{len(self._rebalance_dates)} rebalancing dates."
        )

    def _engineer_features(self) -> None:
        logger.info("[Pipeline] Stage 5: Feature engineering")
        benchmark_col = (
            self._universe.benchmark_ticker
            if self.cfg.universe.benchmark_ticker in self._prices.columns
            else None
        )

        # Separate benchmark from asset prices
        if benchmark_col and benchmark_col in self._prices.columns:
            benchmark = self._prices[[benchmark_col]].rename(
                columns={benchmark_col: "benchmark"}
            )
            asset_prices = self._prices.drop(columns=[benchmark_col], errors="ignore")
            asset_dividends = self._dividends.drop(
                columns=[benchmark_col], errors="ignore"
            )
            asset_volumes = self._volumes.drop(columns=[benchmark_col], errors="ignore")
        else:
            # Fallback: use first asset as pseudo-benchmark
            logger.warning("[Pipeline] Benchmark not in universe — using first ticker.")
            benchmark = self._prices.iloc[:, [0]].rename(
                columns={self._prices.columns[0]: "benchmark"}
            )
            asset_prices = self._prices
            asset_dividends = self._dividends
            asset_volumes = self._volumes

        fc = self.cfg.features
        fe = FeatureEngineer(
            prices=asset_prices,
            volumes=asset_volumes,
            benchmark=benchmark,
            macro=self._macro,
            return_windows=fc.return_windows,
            vol_windows=fc.vol_windows,
            momentum_windows=fc.momentum_windows,
            correlation_window=fc.correlation_window,
            rsi_window=fc.rsi_window,
            bb_window=fc.bb_window,
            bb_num_std=fc.bb_num_std,
            use_gpu=fc.use_gpu,
        )

        self._feature_dict = fe.compute_all()
        self._quarterly_features = fe.resample_quarterly(self._feature_dict)
        self._state_matrix_raw = fe.build_state_matrix(self._quarterly_features)
        self._fe = fe  # keep reference for per-fold normalization

        logger.info(
            f"[Pipeline] Feature matrix: {self._state_matrix_raw.shape[0]} quarters × "
            f"{self._state_matrix_raw.shape[1]} features."
        )

    def _build_folds(self) -> None:
        logger.info("[Pipeline] Stage 6: Walk-forward fold generation")
        splits = get_walk_forward_splits(
            rebalance_dates=self._rebalance_dates,
            min_train_periods=12,  # 3 years minimum training
            test_periods=4,  # 1 year test window
        )

        self._folds = []
        for i, (train_dates, test_dates) in enumerate(splits):
            fold = self._build_single_fold(i, train_dates, test_dates)
            self._folds.append(fold)
            logger.info(f"[Pipeline] Fold {i}: {fold}")

    def _build_single_fold(
        self,
        fold_idx: int,
        train_dates: pd.DatetimeIndex,
        test_dates: pd.DatetimeIndex,
    ) -> WalkForwardFold:
        """Build one WalkForwardFold with properly partitioned data."""
        train_end = train_dates[-1]
        test_end = test_dates[-1]

        # Daily price slices
        train_prices = self._prices[self._prices.index <= train_end]
        test_prices = self._prices[
            (self._prices.index > train_end) & (self._prices.index <= test_end)
        ]
        train_divs = self._dividends[self._dividends.index <= train_end]
        test_divs = self._dividends[
            (self._dividends.index > train_end) & (self._dividends.index <= test_end)
        ]
        train_vols = self._volumes[self._volumes.index <= train_end]
        test_vols = self._volumes[
            (self._volumes.index > train_end) & (self._volumes.index <= test_end)
        ]

        # Quarterly feature matrix slices
        state = self._state_matrix_raw
        train_state_raw = state[state.index <= train_end]
        test_state_raw = state[(state.index > train_end) & (state.index <= test_end)]

        # Normalize using only in-sample statistics (walk-forward safe)
        train_state_norm, scaler_params = self._fe.normalize_features(
            train_state_raw, fit_end_date=str(train_end.date())
        )
        test_state_norm = self._fe.apply_scaler(test_state_raw, scaler_params)

        # Total return index for training period
        tickers = [
            t
            for t in self._universe.valid_tickers
            if t != self.cfg.universe.benchmark_ticker
        ]
        train_tri = self._fe.compute_total_returns_index(
            train_divs.reindex(columns=tickers)
        )

        # SEC filings available up to test_end (date-bounded for backtesting)
        if self._sec_metadata is not None and not self._sec_metadata.empty:
            sec_fold = self._sec_metadata[
                self._sec_metadata["filing_date"] <= test_end
            ].copy()
        else:
            sec_fold = pd.DataFrame()

        return WalkForwardFold(
            fold_idx=fold_idx,
            train_dates=train_dates,
            test_dates=test_dates,
            train_prices=train_prices,
            test_prices=test_prices,
            train_dividends=train_divs,
            test_dividends=test_divs,
            train_volumes=train_vols,
            test_volumes=test_vols,
            train_state_matrix=train_state_norm,
            test_state_matrix=test_state_norm,
            macro=self._macro,
            sec_metadata=sec_fold,
            scaler_params=scaler_params,
            tickers=tickers,
            train_total_returns=train_tri,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a human-readable pipeline summary."""
        if self._folds is None:
            print("Pipeline has not been run yet. Call pipeline.run() first.")
            return

        print("=" * 60)
        print("QuantAgent-RL Data Pipeline Summary")
        print("=" * 60)
        print(f"Universe          : {self._universe.n_assets} assets")
        print(
            f"Date range        : {self.cfg.dates.start_date} → {self.cfg.dates.end_date or 'today'}"
        )
        print(f"Rebalance freq    : {self.cfg.dates.rebalance_freq}")
        print(f"Rebalance dates   : {len(self._rebalance_dates)}")
        print(f"Walk-forward folds: {len(self._folds)}")
        print(f"Macro signals     : {len(self._macro.columns)}")
        print(
            f"SEC filings       : {len(self._sec_metadata) if self._sec_metadata is not None else 0}"
        )
        print(f"Feature matrix    : {self._state_matrix_raw.shape}")
        print(f"GPU backend       : {self.cfg.features.use_gpu}")
        print()
        print("Sector distribution:")
        print(self._universe.sector_counts.to_string())
        print()
        print("Walk-forward folds:")
        for fold in self._folds:
            print(f"  {fold}")
