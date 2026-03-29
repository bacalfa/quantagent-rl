"""
forecasting
===========
QuantAgent-RL forecasting module.

Public API
----------
ForecastingPipeline : end-to-end orchestrator → ForecastBundle per fold
ForecastBundle      : dataclass holding all quarterly forecast outputs
ForecastConfig      : master configuration dataclass

Component classes
-----------------
GARCHForecaster     : GARCH(1,1) volatility with CuPy batch recursion
RegimeDetector      : Gaussian HMM with GPU Viterbi decoding
FamaFrenchFactors   : rolling OLS factor model with CuPy batched matmul
FamaFrenchDataLoader: free Ken French Data Library downloader / parser

Quick start
-----------
>>> from forecasting import ForecastingPipeline, ForecastConfig
>>> pipeline = ForecastingPipeline(ForecastConfig())
>>> pipeline.load_factors()
>>> bundle = pipeline.run_fold(data_fold)   # WalkForwardFold from data module
>>> bundle.train_vol        # quarterly GARCH vol forecasts
>>> bundle.train_regime     # quarterly HMM regime labels + probabilities
>>> bundle.train_betas      # quarterly Fama-French factor exposures
>>> bundle.rl_state_extension  # flat DataFrame ready to append to RL state
"""

from forecasting.config import (
    FamaFrenchConfig,
    ForecastConfig,
    GARCHConfig,
    RegimeConfig,
)
from forecasting.factors import FamaFrenchDataLoader, FamaFrenchFactors
from forecasting.garch import GARCHForecaster, GARCHParams
from forecasting.pipeline import ForecastBundle, ForecastingPipeline
from forecasting.regime import RegimeDetector

__all__ = [
    # Pipeline
    "ForecastingPipeline",
    "ForecastBundle",
    # Config
    "ForecastConfig",
    "GARCHConfig",
    "RegimeConfig",
    "FamaFrenchConfig",
    # Components
    "GARCHForecaster",
    "GARCHParams",
    "RegimeDetector",
    "FamaFrenchFactors",
    "FamaFrenchDataLoader",
]
