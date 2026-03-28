"""
data/config.py
==============
Configuration dataclasses for the QuantAgent-RL data module.

All runtime parameters — date ranges, API keys, feature-engineering
hyperparameters, and GPU preferences — are centralised here so that
experiments can be reproduced by changing a single config object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


@dataclass
class UniverseConfig:
    """Defines the investable stock universe.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to include (e.g. a curated S&P-100 subset).
    benchmark_ticker : str
        Benchmark symbol used for beta and tracking-error calculations.
    min_history_years : float
        Minimum years of price history required to include a ticker.
    """

    tickers: List[str] = field(default_factory=lambda: DEFAULT_UNIVERSE)
    benchmark_ticker: str = "SPY"
    min_history_years: float = 5.0


# Curated 30-stock universe spanning 10 GICS sectors — enough for meaningful
# portfolio diversification while keeping the RL action space tractable.
DEFAULT_UNIVERSE: List[str] = [
    # Information Technology
    "AAPL",
    "MSFT",
    "NVDA",
    "AVGO",
    # Communication Services
    "GOOGL",
    "META",
    # Consumer Discretionary
    "AMZN",
    "TSLA",
    # Health Care
    "UNH",
    "JNJ",
    "LLY",
    "ABBV",
    # Financials
    "JPM",
    "BAC",
    "GS",
    "BRK-B",
    # Industrials
    "RTX",
    "HON",
    "CAT",
    # Energy
    "XOM",
    "CVX",
    # Consumer Staples
    "PG",
    "KO",
    "WMT",
    # Materials
    "LIN",
    "APD",
    # Real Estate
    "PLD",
    "AMT",
    # Utilities
    "NEE",
    "DUK",
]


# ---------------------------------------------------------------------------
# Date Range
# ---------------------------------------------------------------------------


@dataclass
class DateRangeConfig:
    """Training and evaluation date boundaries.

    Parameters
    ----------
    start_date : str
        Earliest date to fetch data for (YYYY-MM-DD).
    end_date : str
        Latest date to fetch data for. Defaults to today if None.
    rebalance_freq : str
        Pandas offset alias for the rebalancing period.
        'QE' = quarter-end, 'ME' = month-end, 'YE' = year-end.
    """

    start_date: str = "2005-01-01"
    end_date: Optional[str] = None  # None → today
    rebalance_freq: str = "QE"  # quarterly rebalancing


# ---------------------------------------------------------------------------
# FRED Macro Signals
# ---------------------------------------------------------------------------

# FRED series IDs → human-readable names used as DataFrame column names.
FRED_SERIES: dict[str, str] = {
    "FEDFUNDS": "fed_funds_rate",  # Federal Funds Effective Rate
    "CPIAUCSL": "cpi_yoy",  # CPI All Urban Consumers
    "DCOILWTICO": "wti_crude_oil",  # WTI Crude Oil Price
    "T10Y2Y": "yield_curve_10y2y",  # 10Y-2Y Treasury Spread
    "VIXCLS": "vix",  # CBOE Volatility Index
    "UNRATE": "unemployment_rate",  # Unemployment Rate
    "UMCSENT": "consumer_sentiment",  # U of M Consumer Sentiment
    "BAMLH0A0HYM2": "hy_spread",  # High-Yield OAS Spread
}


@dataclass
class MacroConfig:
    """Configuration for FRED macro signal ingestion.

    Parameters
    ----------
    api_key : str
        FRED API key. Reads from FRED_API_KEY environment variable.
    series : dict
        Mapping from FRED series ID to output column name.
    ffill_limit : int
        Maximum number of calendar days to forward-fill missing observations.
    """

    api_key: str = field(default_factory=lambda: os.environ.get("FRED_API_KEY", ""))
    series: dict = field(default_factory=lambda: dict(FRED_SERIES))
    ffill_limit: int = 7


# ---------------------------------------------------------------------------
# SEC / Earnings Config
# ---------------------------------------------------------------------------


@dataclass
class SECConfig:
    """Configuration for SEC EDGAR filing retrieval.

    SEC EDGAR is a free US government database — no API key required.
    A descriptive user_agent string is required by SEC fair-use policy.

    Parameters
    ----------
    form_types : list of str
        SEC form types to retrieve: ["10-Q", "10-K"].
    cache_dir : str
        Local directory for caching filing metadata, XBRL JSON, and MD&A text.
    user_agent : str
        Required by SEC EDGAR fair-use policy.
        Format: "Your Name your_email@domain.com"
        Set via SEC_USER_AGENT environment variable.
    max_filings_per_ticker : int
        Cap on the number of filings retrieved per ticker per form type.
    fetch_xbrl : bool
        Download structured XBRL financial facts (revenue, EPS, debt, etc.).
        Recommended: True. Provides clean quantitative inputs for Company Agent.
    fetch_mda : bool
        Extract MD&A text from filings via edgartools (pip install edgartools).
        Falls back to heuristic HTML extraction if edgartools is unavailable.
    """

    form_types: List[str] = field(default_factory=lambda: ["10-Q", "10-K"])
    cache_dir: str = "data/cache/sec"
    user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "SEC_USER_AGENT", "QuantAgentRL research@example.com"
        )
    )
    max_filings_per_ticker: int = 8
    fetch_xbrl: bool = True
    fetch_mda: bool = True


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------


@dataclass
class FeatureConfig:
    """Hyperparameters for GPU-accelerated feature engineering.

    Parameters
    ----------
    return_windows : list of int
        Rolling windows (in trading days) for return computation.
    vol_windows : list of int
        Rolling windows for realised volatility computation.
    momentum_windows : list of int
        Look-back windows for momentum signal construction.
    correlation_window : int
        Window for rolling pairwise correlation (used for regime features).
    rsi_window : int
        Period for Relative Strength Index.
    bb_window : int
        Period for Bollinger Band computation.
    bb_num_std : float
        Number of standard deviations for Bollinger Bands.
    use_gpu : bool | None
        True  → force GPU (raises if unavailable).
        False → force CPU.
        None  → auto-detect (GPU if available, else CPU).
    """

    return_windows: List[int] = field(default_factory=lambda: [5, 21, 63])
    vol_windows: List[int] = field(default_factory=lambda: [21, 63])
    momentum_windows: List[int] = field(default_factory=lambda: [21, 63, 126])
    correlation_window: int = 63
    rsi_window: int = 14
    bb_window: int = 20
    bb_num_std: float = 2.0
    use_gpu: Optional[bool] = None  # auto-detect


# ---------------------------------------------------------------------------
# Master Data Config
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Top-level configuration object for the entire data module.

    Usage
    -----
    >>> cfg = DataConfig()                          # all defaults
    >>> cfg = DataConfig(universe=UniverseConfig(tickers=["AAPL", "MSFT"]))
    """

    universe: UniverseConfig = field(default_factory=UniverseConfig)
    dates: DateRangeConfig = field(default_factory=DateRangeConfig)
    macro: MacroConfig = field(default_factory=MacroConfig)
    sec: SECConfig = field(default_factory=SECConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    def validate(self) -> None:
        """Raise ValueError for obviously invalid configurations."""
        if self.macro.api_key == "":
            raise ValueError(
                "FRED_API_KEY environment variable is not set. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        if not self.universe.tickers:
            raise ValueError("Universe must contain at least one ticker.")
        if self.dates.start_date >= (self.dates.end_date or "2099-12-31"):
            raise ValueError("start_date must be before end_date.")
