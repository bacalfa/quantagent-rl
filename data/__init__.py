"""
data
====
QuantAgent-RL data module.

Public API
----------
DataPipeline   : end-to-end pipeline from ingestion to walk-forward folds
DataConfig     : master configuration dataclass
WalkForwardFold: dataclass for one train/test split
Universe       : validated, sector-annotated stock universe

Quick start
-----------
>>> from data import DataPipeline, DataConfig
>>> cfg = DataConfig()
>>> pipeline = DataPipeline(cfg).run()
>>> fold = pipeline.get_fold(0)
>>> fold.train_state_matrix   # quarterly RL state features (normalised)
>>> fold.train_prices         # daily adjusted close prices
"""

from data.config import (
    DEFAULT_UNIVERSE,
    FRED_SERIES,
    DataConfig,
    DateRangeConfig,
    FeatureConfig,
    MacroConfig,
    SECConfig,
    UniverseConfig,
)
from data.features import FeatureEngineer
from data.ingestion import MacroDataIngester, MarketDataIngester, SECFilingIngester
from data.pipeline import DataPipeline, WalkForwardFold
from data.universe import Universe, get_rebalance_dates, get_walk_forward_splits

__all__ = [
    # Pipeline
    "DataPipeline",
    "WalkForwardFold",
    # Config
    "DataConfig",
    "DateRangeConfig",
    "FeatureConfig",
    "MacroConfig",
    "SECConfig",
    "UniverseConfig",
    "DEFAULT_UNIVERSE",
    "FRED_SERIES",
    # Core classes
    "FeatureEngineer",
    "Universe",
    "MarketDataIngester",
    "MacroDataIngester",
    "SECFilingIngester",
    # Utilities
    "get_rebalance_dates",
    "get_walk_forward_splits",
]
