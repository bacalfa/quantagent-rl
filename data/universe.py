"""
data/universe.py
================
Stock universe management for QuantAgent-RL.

Provides helpers to:
- Filter tickers by minimum price-history length
- Map tickers to GICS sectors
- Compute equal-weight and market-cap-weight baselines
- Persist / reload the validated universe
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GICS Sector Mapping (subset covering DEFAULT_UNIVERSE)
# ---------------------------------------------------------------------------

TICKER_SECTOR_MAP: dict[str, str] = {
    # Information Technology
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "NVDA": "Information Technology",
    "AVGO": "Information Technology",
    # Communication Services
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    # Health Care
    "UNH": "Health Care",
    "JNJ": "Health Care",
    "LLY": "Health Care",
    "ABBV": "Health Care",
    # Financials
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "BRK-B": "Financials",
    # Industrials
    "RTX": "Industrials",
    "HON": "Industrials",
    "CAT": "Industrials",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    # Consumer Staples
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "WMT": "Consumer Staples",
    # Materials
    "LIN": "Materials",
    "APD": "Materials",
    # Real Estate
    "PLD": "Real Estate",
    "AMT": "Real Estate",
    # Utilities
    "NEE": "Utilities",
    "DUK": "Utilities",
    # Benchmark
    "SPY": "Benchmark",
}


# ---------------------------------------------------------------------------
# Universe class
# ---------------------------------------------------------------------------


class Universe:
    """Validated, sector-annotated stock universe.

    Parameters
    ----------
    tickers : list of str
        Raw candidate ticker list.
    price_data : pd.DataFrame
        Adjusted close prices indexed by date, columns = tickers.
        Fetched externally by ``data.ingestion.MarketDataIngester``.
    start_date : str
        Required start of available history (YYYY-MM-DD).
    min_history_years : float
        Minimum required years of non-null price history.
    sector_map : dict, optional
        Custom ticker → sector mapping. Defaults to TICKER_SECTOR_MAP.
    """

    def __init__(
        self,
        tickers: list[str],
        price_data: pd.DataFrame,
        start_date: str,
        min_history_years: float = 5.0,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        self._raw_tickers = tickers
        self._price_data = price_data
        self._start_date = pd.Timestamp(start_date)
        self._min_history_years = min_history_years
        self._sector_map = sector_map or TICKER_SECTOR_MAP

        self.valid_tickers: list[str] = []
        self.dropped_tickers: list[str] = []
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Filter tickers that do not meet minimum history requirements."""
        min_days = int(self._min_history_years * 252)

        for ticker in self._raw_tickers:
            if ticker not in self._price_data.columns:
                logger.warning(
                    f"[Universe] {ticker}: not found in price data — dropped."
                )
                self.dropped_tickers.append(ticker)
                continue

            series = self._price_data[ticker].dropna()

            # Must have data starting on or before start_date
            if series.empty or series.index[0] > self._start_date:
                logger.warning(
                    f"[Universe] {ticker}: history starts {series.index[0].date()} "
                    f"(required <= {self._start_date.date()}) — dropped."
                )
                self.dropped_tickers.append(ticker)
                continue

            # Must meet minimum observation count
            obs_from_start = series[series.index >= self._start_date]
            if len(obs_from_start) < min_days:
                logger.warning(
                    f"[Universe] {ticker}: only {len(obs_from_start)} observations "
                    f"(required >= {min_days}) — dropped."
                )
                self.dropped_tickers.append(ticker)
                continue

            self.valid_tickers.append(ticker)

        logger.info(
            f"[Universe] {len(self.valid_tickers)} valid / "
            f"{len(self.dropped_tickers)} dropped out of "
            f"{len(self._raw_tickers)} candidates."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_assets(self) -> int:
        """Number of validated assets."""
        return len(self.valid_tickers)

    @property
    def prices(self) -> pd.DataFrame:
        """Adjusted close prices for valid tickers only."""
        return self._price_data[self.valid_tickers]

    @property
    def sectors(self) -> pd.Series:
        """Series mapping ticker → GICS sector."""
        return pd.Series(
            {t: self._sector_map.get(t, "Unknown") for t in self.valid_tickers},
            name="sector",
        )

    @property
    def sector_counts(self) -> pd.Series:
        """Number of tickers per GICS sector."""
        return self.sectors.value_counts()

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def equal_weights(self) -> pd.Series:
        """1/N equal-weight portfolio across valid tickers."""
        w = 1.0 / self.n_assets
        return pd.Series(w, index=self.valid_tickers, name="equal_weight")

    def market_cap_weights(self, market_caps: pd.Series | None = None) -> pd.Series:
        """Market-cap-weighted portfolio.

        Parameters
        ----------
        market_caps : pd.Series, optional
            Series indexed by ticker. If None, falls back to equal weights
            with a warning.
        """
        if market_caps is None:
            logger.warning(
                "[Universe] market_caps not provided — returning equal weights."
            )
            return self.equal_weights()

        caps = market_caps.reindex(self.valid_tickers).fillna(0.0)
        total = caps.sum()
        if total == 0:
            return self.equal_weights()
        return (caps / total).rename("market_cap_weight")

    def sector_constrained_weights(
        self,
        weights: pd.Series,
        max_sector_weight: float = 0.35,
    ) -> pd.Series:
        """Clip sector-level concentration and renormalise.

        Parameters
        ----------
        weights : pd.Series
            Proposed portfolio weights indexed by ticker.
        max_sector_weight : float
            Maximum allowed weight for any single GICS sector.

        Returns
        -------
        pd.Series
            Renormalised weights satisfying sector constraints.
        """
        w = weights.copy().reindex(self.valid_tickers).fillna(0.0)
        sectors = self.sectors

        for sector in sectors.unique():
            mask = sectors == sector
            sector_total = w[mask].sum()
            if sector_total > max_sector_weight:
                scale = max_sector_weight / sector_total
                w[mask] *= scale

        total = w.sum()
        return (w / total).rename("constrained_weight") if total > 0 else w

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise universe metadata for caching."""
        return {
            "valid_tickers": self.valid_tickers,
            "dropped_tickers": self.dropped_tickers,
            "sectors": self.sectors.to_dict(),
            "n_assets": self.n_assets,
            "start_date": str(self._start_date.date()),
            "min_history_years": self._min_history_years,
        }

    def save(self, path: str | Path) -> None:
        """Save universe metadata as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"[Universe] Saved to {path}")

    @classmethod
    def load_tickers(cls, path: str | Path) -> list[str]:
        """Load valid tickers from a previously saved universe JSON."""
        with open(path) as f:
            data = json.load(f)
        return data["valid_tickers"]

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Universe(n_assets={self.n_assets}, "
            f"start={self._start_date.date()}, "
            f"min_years={self._min_history_years})"
        )


# ---------------------------------------------------------------------------
# Utility: quarterly rebalance dates
# ---------------------------------------------------------------------------


def get_rebalance_dates(
    start_date: str,
    end_date: str,
    freq: str = "QE",
) -> pd.DatetimeIndex:
    """Return a DatetimeIndex of quarter-end (or other period-end) dates.

    Parameters
    ----------
    start_date, end_date : str
        Date range in YYYY-MM-DD format.
    freq : str
        Pandas offset alias. 'QE' = quarter-end, 'ME' = month-end.

    Returns
    -------
    pd.DatetimeIndex
        Dates on which rebalancing is performed.

    Examples
    --------
    >>> dates = get_rebalance_dates("2020-01-01", "2023-12-31")
    >>> print(dates)
    DatetimeIndex(['2020-03-31', '2020-06-30', ...], dtype='datetime64[ns]', freq='QE-DEC')
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def get_walk_forward_splits(
    rebalance_dates: pd.DatetimeIndex,
    min_train_periods: int = 12,
    test_periods: int = 4,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Generate expanding-window walk-forward train/test splits.

    Parameters
    ----------
    rebalance_dates : pd.DatetimeIndex
        Full sequence of rebalancing dates.
    min_train_periods : int
        Minimum number of rebalancing periods in the first training window.
    test_periods : int
        Fixed number of rebalancing periods in each test window.

    Returns
    -------
    list of (train_dates, test_dates) tuples
        Each tuple contains the rebalance dates for that fold's train/test split.

    Notes
    -----
    The training window is *anchored* at the start (expanding), so each
    successive fold adds ``test_periods`` rebalancing dates to training.
    This implements anchored/expanding walk-forward cross-validation.
    """
    splits = []
    total = len(rebalance_dates)

    start_idx = min_train_periods
    while start_idx + test_periods <= total:
        train_dates = rebalance_dates[:start_idx]
        test_dates = rebalance_dates[start_idx : start_idx + test_periods]
        splits.append((train_dates, test_dates))
        start_idx += test_periods  # advance by one test window

    logger.info(
        f"[Universe] Walk-forward splits: {len(splits)} folds "
        f"(min_train={min_train_periods}, test={test_periods} periods each)."
    )
    return splits
