"""
data/ingestion.py
=================
Data ingestion layer for QuantAgent-RL.

Provides three independent ingesters:
  - MarketDataIngester   : OHLCV + dividends via yfinance
  - MacroDataIngester    : FRED macro signals (rates, CPI, VIX, ...)
  - SECFilingIngester    : 10-Q / 10-K filings + structured XBRL financials
                          via the free SEC EDGAR REST API (data.sec.gov)
                          and the `edgartools` library.

SEC EDGAR is a free US government database — no API key or registration
required. A User-Agent header identifying your application is required
by SEC fair-use policy (not a paywall).

EDGAR REST API endpoints used:
  https://data.sec.gov/submissions/{CIK}.json     — filing metadata
  https://data.sec.gov/api/xbrl/companyfacts/{CIK}.json — structured XBRL financials

Each ingester is stateless — it fetches, lightly cleans, and returns a
pandas DataFrame. Caching is handled via parquet/JSON files to avoid
redundant API calls during development.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _today() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def _cache_path(cache_dir: str, name: str) -> Path:
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.parquet"


def _load_cache(path: Path) -> pd.DataFrame | None:
    if path.exists():
        logger.debug(f"[Cache] Loading {path}")
        return pd.read_parquet(path)
    return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path)
    logger.debug(f"[Cache] Saved {path}")


# ---------------------------------------------------------------------------
# 1. Market Data Ingester
# ---------------------------------------------------------------------------


class MarketDataIngester:
    """Fetches adjusted OHLCV data and dividends via yfinance.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to download.
    start_date : str
        Earliest date (YYYY-MM-DD).
    end_date : str, optional
        Latest date (YYYY-MM-DD). Defaults to today.
    cache_dir : str
        Directory for parquet caches.
    batch_size : int
        Number of tickers per yfinance download call (avoids rate-limits).
    retry_delay : float
        Seconds to wait between retries on network errors.

    Examples
    --------
    >>> ingester = MarketDataIngester(tickers=["AAPL", "MSFT"], start_date="2010-01-01")
    >>> prices, dividends, volumes = ingester.fetch()
    """

    def __init__(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str | None = None,
        cache_dir: str = "data/cache/market",
        batch_size: int = 50,
        retry_delay: float = 2.0,
    ) -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date or _today()
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self, use_cache: bool = True, force_refresh: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Download market data and return (prices, dividends, volumes).

        Parameters
        ----------
        use_cache : bool
            Load from parquet cache if available.
        force_refresh : bool
            Re-download even if cache exists.

        Returns
        -------
        prices : pd.DataFrame
            Adjusted close prices. Index=Date, columns=ticker.
        dividends : pd.DataFrame
            Dividend amounts. Same structure as prices.
        volumes : pd.DataFrame
            Daily trading volumes. Same structure as prices.
        """
        prices_path = _cache_path(self.cache_dir, "adj_close")
        divs_path = _cache_path(self.cache_dir, "dividends")
        vols_path = _cache_path(self.cache_dir, "volumes")

        if use_cache and not force_refresh:
            prices = _load_cache(prices_path)
            divs = _load_cache(divs_path)
            vols = _load_cache(vols_path)
            if prices is not None and divs is not None and vols is not None:
                logger.info("[MarketData] Loaded from cache.")
                return prices, divs, vols

        prices, divs, vols = self._download_all()

        _save_cache(prices, prices_path)
        _save_cache(divs, divs_path)
        _save_cache(vols, vols_path)

        return prices, divs, vols

    def fetch_benchmark(self, ticker: str = "SPY") -> pd.DataFrame:
        """Fetch benchmark adjusted close series for beta / tracking error.

        Parameters
        ----------
        ticker : str
            Benchmark ticker symbol.

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame with adjusted close prices.
        """
        tmp = MarketDataIngester(
            tickers=[ticker],
            start_date=self.start_date,
            end_date=self.end_date,
            cache_dir=self.cache_dir,
        )
        prices, _, _ = tmp.fetch()
        return prices[[ticker]].rename(columns={ticker: "benchmark"})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _download_all(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Download in batches and concatenate results."""
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError("yfinance is required: pip install yfinance") from e

        all_prices: list[pd.DataFrame] = []
        all_divs: list[pd.DataFrame] = []
        all_vols: list[pd.DataFrame] = []

        batches = [
            self.tickers[i : i + self.batch_size]
            for i in range(0, len(self.tickers), self.batch_size)
        ]

        for batch_idx, batch in enumerate(batches):
            logger.info(
                f"[MarketData] Downloading batch {batch_idx + 1}/{len(batches)}: "
                f"{batch}"
            )
            for attempt in range(3):
                try:
                    raw = yf.download(
                        tickers=batch,
                        start=self.start_date,
                        end=self.end_date,
                        auto_adjust=True,
                        actions=True,  # includes dividends & splits
                        progress=False,
                        group_by="column",
                    )
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise
                    logger.warning(
                        f"[MarketData] Retry {attempt + 1} after error: {exc}"
                    )
                    time.sleep(self.retry_delay)

            prices_batch, divs_batch, vols_batch = self._parse_raw(raw, batch)
            all_prices.append(prices_batch)
            all_divs.append(divs_batch)
            all_vols.append(vols_batch)

            if batch_idx < len(batches) - 1:
                time.sleep(0.5)  # polite rate-limiting

        prices = pd.concat(all_prices, axis=1)
        divs = pd.concat(all_divs, axis=1)
        vols = pd.concat(all_vols, axis=1)

        prices, divs, vols = self._clean(prices, divs, vols)
        return prices, divs, vols

    @staticmethod
    def _parse_raw(
        raw: pd.DataFrame, tickers: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Extract Close, Dividends, Volume from yfinance multi-level output."""
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].reindex(columns=tickers)
            # yfinance returns 'Dividends' when actions=True
            divs = raw.get(
                "Dividends", pd.DataFrame(index=raw.index, columns=tickers)
            ).reindex(columns=tickers)
            vols = raw["Volume"].reindex(columns=tickers)
        else:
            # Single ticker — yfinance collapses MultiIndex
            ticker = tickers[0]
            close = raw[["Close"]].rename(columns={"Close": ticker})
            divs_series = raw.get("Dividends", pd.Series(0.0, index=raw.index))
            divs = divs_series.rename(ticker).to_frame()
            vols = raw[["Volume"]].rename(columns={"Volume": ticker})

        return (
            close.astype(float),
            divs.fillna(0.0).astype(float),
            vols.astype(float),
        )

    @staticmethod
    def _clean(
        prices: pd.DataFrame,
        divs: pd.DataFrame,
        vols: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Remove weekends/holidays with no data and forward-fill gaps."""
        # Drop rows where ALL prices are NaN (non-trading days)
        prices = prices.dropna(how="all")
        divs = divs.reindex(prices.index).fillna(0.0)
        vols = vols.reindex(prices.index).fillna(0.0)

        # Forward-fill individual ticker gaps (up to 5 trading days)
        prices = prices.ffill(limit=5)

        return prices, divs, vols


# ---------------------------------------------------------------------------
# 2. Macro Data Ingester
# ---------------------------------------------------------------------------


class MacroDataIngester:
    """Downloads macro-economic time series from FRED.

    Parameters
    ----------
    series_map : dict
        Mapping {FRED_series_id: column_name}.
    start_date : str
    end_date : str, optional
    api_key : str
        FRED API key. Falls back to FRED_API_KEY env var.
    cache_dir : str
    ffill_limit : int
        Max calendar days to forward-fill missing FRED observations.

    Examples
    --------
    >>> ingester = MacroDataIngester(series_map={"FEDFUNDS": "fed_funds_rate"}, ...)
    >>> macro_df = ingester.fetch()
    """

    def __init__(
        self,
        series_map: dict[str, str],
        start_date: str,
        end_date: str | None = None,
        api_key: str | None = None,
        cache_dir: str = "data/cache/macro",
        ffill_limit: int = 7,
    ) -> None:
        self.series_map = series_map
        self.start_date = start_date
        self.end_date = end_date or _today()
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self.cache_dir = cache_dir
        self.ffill_limit = ffill_limit

        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable "
                "or pass api_key= explicitly."
            )

    def fetch(
        self, use_cache: bool = True, force_refresh: bool = False
    ) -> pd.DataFrame:
        """Download all configured FRED series and return as a single DataFrame.

        Returns
        -------
        pd.DataFrame
            Daily-frequency macro signals aligned to trading calendar.
            Index = pd.DatetimeIndex, columns = human-readable signal names.
        """
        cache_path = _cache_path(self.cache_dir, "macro_signals")
        if use_cache and not force_refresh and cache_path.exists():
            logger.info("[MacroData] Loaded from cache.")
            return pd.read_parquet(cache_path)

        df = self._download_all()
        _save_cache(df, cache_path)
        return df

    def _download_all(self) -> pd.DataFrame:
        try:
            from fredapi import Fred
        except ImportError as e:
            raise ImportError("fredapi is required: pip install fredapi") from e

        fred = Fred(api_key=self.api_key)
        frames: dict[str, pd.Series] = {}

        for series_id, col_name in self.series_map.items():
            try:
                logger.info(f"[MacroData] Fetching {series_id} → {col_name}")
                s = fred.get_series(
                    series_id,
                    observation_start=self.start_date,
                    observation_end=self.end_date,
                )
                frames[col_name] = s
            except Exception as exc:
                logger.warning(f"[MacroData] Failed to fetch {series_id}: {exc}")

        if not frames:
            raise RuntimeError("[MacroData] No FRED series could be downloaded.")

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Forward-fill to align with daily trading calendar
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="B")
        df = df.reindex(full_idx).ffill(limit=self.ffill_limit)

        # Compute derived features
        df = self._add_derived_features(df)

        logger.info(
            f"[MacroData] Downloaded {len(df.columns)} series, {len(df)} trading days."
        )
        return df

    @staticmethod
    def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add change-based and normalised macro features.

        Derived signals:
        - *_mom1m  : 1-month (21-day) change / momentum
        - *_zscore : rolling 252-day z-score (regime-normalised)
        - cpi_yoy_delta : MoM change in CPI YoY (inflation acceleration)
        """
        derived: dict[str, pd.Series] = {}

        for col in df.columns:
            # 1-month momentum (level change)
            derived[f"{col}_mom1m"] = df[col].diff(21)

            # 252-day rolling z-score
            roll = df[col].rolling(252, min_periods=126)
            derived[f"{col}_zscore"] = (df[col] - roll.mean()) / (roll.std() + 1e-9)

        # Inflation acceleration: delta of YoY CPI
        if "cpi_yoy" in df.columns:
            derived["cpi_acceleration"] = df["cpi_yoy"].diff(21)

        # Yield curve inversion flag
        if "yield_curve_10y2y" in df.columns:
            derived["yield_curve_inverted"] = (df["yield_curve_10y2y"] < 0).astype(
                float
            )

        df = pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)
        return df


# ---------------------------------------------------------------------------
# 3. SEC Filing Ingester  (free — SEC EDGAR REST API + edgartools)
# ---------------------------------------------------------------------------
#
# SEC EDGAR is a completely free US government database. No API key or
# registration is required. The only requirement is a descriptive User-Agent
# header per SEC fair-use policy: https://www.sec.gov/os/accessing-edgar-data
#
# Two complementary data sources are used:
#
# A) EDGAR REST API (data.sec.gov) — filing metadata + XBRL structured
#    financials (revenue, net income, EPS, debt, free cash flow). This gives
#    the Company Agent clean, machine-readable numbers it can reason about
#    without having to extract them from raw HTML.
#
# B) edgartools library — retrieves and parses the Management Discussion &
#    Analysis (MD&A) section from 10-Q / 10-K filings. MD&A contains the
#    qualitative narrative that is most useful for LLM sentiment analysis.
#
# Rate limits: SEC requests ≤ 10 req/sec. We enforce a 0.12s delay per call.
# ---------------------------------------------------------------------------

# XBRL concept names for key financial metrics we want from company facts API
XBRL_CONCEPTS: dict[str, str] = {
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",  # alt
    "NetIncomeLoss": "net_income",
    "EarningsPerShareDiluted": "eps_diluted",
    "OperatingIncomeLoss": "operating_income",
    "ResearchAndDevelopmentExpense": "rd_expense",
    "GrossProfit": "gross_profit",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "LongTermDebt": "long_term_debt",
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "CommonStockSharesOutstanding": "shares_outstanding",
}

_EDGAR_BASE = "https://www.sec.gov"
_EDGAR_BASE_DATA = "https://data.sec.gov"
_EDGAR_HEADERS_TMPL = {"User-Agent": "{user_agent}", "Accept-Encoding": "gzip, deflate"}
_SEC_RATE_LIMIT_DELAY = 0.12  # seconds between EDGAR API calls


class SECFilingIngester:
    """Retrieves SEC filing metadata, structured XBRL financials, and MD&A text.

    Data is sourced exclusively from the free SEC EDGAR REST API and
    the ``edgartools`` library — no subscription or API key required.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to retrieve filings for.
    form_types : list of str
        SEC form types: ["10-Q", "10-K"] (8-K supported but not parsed for MD&A).
    user_agent : str
        Required by SEC EDGAR fair-use policy.
        Format: "Your Name your_email@domain.com"
        Set via SEC_USER_AGENT environment variable or pass directly.
    start_date : str
        Earliest filing date to include (YYYY-MM-DD).
    end_date : str, optional
        Latest filing date. Defaults to today.
    max_filings_per_ticker : int
        Cap on filings retrieved per ticker per form type.
    cache_dir : str
        Directory for caching downloaded JSON and parquet data.
    fetch_xbrl : bool
        If True, download structured XBRL financial facts per company.
        Produces a richer dataset for the Company Agent.
    fetch_mda : bool
        If True, download and extract MD&A text from filings via edgartools.
        Slower but provides the qualitative narrative the LLM agent needs.

    Notes
    -----
    The ``fetch()`` method returns a filing metadata DataFrame. Use
    ``fetch_financials()`` to get the structured XBRL financials table,
    and ``get_mda_text()`` to retrieve MD&A text for a specific filing.

    Examples
    --------
    >>> ingester = SECFilingIngester(tickers=["AAPL", "MSFT"], ...)
    >>> metadata   = ingester.fetch()           # filing index
    >>> financials = ingester.fetch_financials() # structured XBRL numbers
    >>> mda = ingester.get_mda_text("AAPL", "10-Q", "2024-01-01")
    """

    def __init__(
        self,
        tickers: list[str],
        form_types: list[str],
        user_agent: str,
        start_date: str,
        end_date: str | None = None,
        max_filings_per_ticker: int = 8,
        cache_dir: str = "data/cache/sec",
        fetch_xbrl: bool = True,
        fetch_mda: bool = True,
    ) -> None:
        self.tickers = tickers
        self.form_types = form_types
        self.user_agent = user_agent or os.environ.get(
            "SEC_USER_AGENT", "QuantAgentRL research@example.com"
        )
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date or _today())
        self.max_filings_per_ticker = max_filings_per_ticker
        self.cache_dir = Path(cache_dir)
        self.fetch_xbrl = fetch_xbrl
        self.fetch_mda = fetch_mda

        self._headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

        # CIK lookup cache (ticker → 10-digit CIK string)
        self._cik_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, use_cache: bool = True) -> pd.DataFrame:
        """Retrieve filing metadata for all configured tickers and form types.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, cik, form_type, filing_date, accession_number,
            report_date, primary_document_url, mda_cache_path (if fetch_mda).
        """
        cache_path = _cache_path(str(self.cache_dir), "filing_metadata")
        if use_cache and cache_path.exists():
            logger.info("[SEC] Filing metadata loaded from cache.")
            return pd.read_parquet(cache_path)

        # Resolve all CIKs first (single bulk request)
        self._load_cik_map()

        records: list[dict] = []
        for ticker in self.tickers:
            cik = self._cik_map.get(ticker.upper())
            if not cik:
                logger.warning(f"[SEC] CIK not found for {ticker} — skipped.")
                continue
            ticker_records = self._fetch_ticker_metadata(ticker, cik)
            records.extend(ticker_records)

        if not records:
            logger.warning("[SEC] No filing metadata retrieved.")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["filing_date"] = pd.to_datetime(df["filing_date"])
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        df = df.sort_values(["ticker", "form_type", "filing_date"])
        _save_cache(df, cache_path)
        logger.info(
            f"[SEC] {len(df)} filings indexed across {df['ticker'].nunique()} tickers."
        )
        return df

    def fetch_financials(self, use_cache: bool = True) -> pd.DataFrame:
        """Retrieve structured XBRL financial facts for all tickers.

        Calls the EDGAR company facts API to extract key metrics (revenue,
        net income, EPS, free cash flow, debt) as a clean time-series table.
        This structured data feeds directly into the Company Agent as
        quantitative context alongside the qualitative MD&A text.

        Returns
        -------
        pd.DataFrame
            Long-format table: ticker, metric, period_end, value, unit, form.
            Pivot on (ticker, period_end) to get a wide feature matrix.
        """
        if not self.fetch_xbrl:
            logger.info("[SEC] XBRL fetch disabled.")
            return pd.DataFrame()

        cache_path = _cache_path(str(self.cache_dir), "xbrl_financials")
        if use_cache and cache_path.exists():
            logger.info("[SEC] XBRL financials loaded from cache.")
            return pd.read_parquet(cache_path)

        self._load_cik_map()
        all_records: list[dict] = []

        for ticker in self.tickers:
            cik = self._cik_map.get(ticker.upper())
            if not cik:
                continue
            records = self._fetch_xbrl_facts(ticker, cik)
            all_records.extend(records)
            time.sleep(_SEC_RATE_LIMIT_DELAY)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
        df = df.dropna(subset=["period_end", "value"])
        df = df[
            (df["period_end"] >= self.start_date) & (df["period_end"] <= self.end_date)
        ]
        df = df.sort_values(["ticker", "metric", "period_end"])
        _save_cache(df, cache_path)
        logger.info(
            f"[SEC] XBRL financials: {len(df)} records, "
            f"{df['ticker'].nunique()} tickers, {df['metric'].nunique()} metrics."
        )
        return df

    def get_mda_text(
        self,
        ticker: str,
        form_type: str = "10-Q",
        as_of_date: str | None = None,
        max_chars: int = 40_000,
    ) -> str:
        """Retrieve the MD&A section text for the most recent filing.

        Uses ``edgartools`` to locate and parse the MD&A section from the
        filing's primary document. If edgartools is unavailable, falls back
        to the raw filing text from EDGAR.

        Parameters
        ----------
        ticker : str
        form_type : str
        as_of_date : str, optional
            Return the most recent filing on or before this date (YYYY-MM-DD).
            Useful for date-bounded backtesting (no look-ahead bias).
        max_chars : int
            Character limit for LLM context window safety.

        Returns
        -------
        str
            MD&A text, truncated to max_chars. Empty string on failure.
        """
        as_of = pd.Timestamp(as_of_date) if as_of_date else self.end_date

        # Try edgartools first (structured MD&A extraction)
        mda = self._fetch_mda_edgartools(ticker, form_type, as_of, max_chars)
        if mda:
            return mda

        # Fallback: raw filing text from EDGAR REST API
        return self._fetch_raw_filing_text(ticker, form_type, as_of, max_chars)

    def pivot_financials(self, financials_df: pd.DataFrame) -> pd.DataFrame:
        """Pivot the long-format financials table to a wide quarterly matrix.

        Parameters
        ----------
        financials_df : pd.DataFrame
            Output of ``fetch_financials()``.

        Returns
        -------
        pd.DataFrame
            Multi-index columns (ticker, metric). Index = quarter-end dates.
            Values are the most recent reported figure for that period.
        """
        if financials_df.empty:
            return pd.DataFrame()

        # Keep only quarterly (10-Q) and annual (10-K) filings; de-duplicate
        # by keeping the last-reported value per ticker/metric/period
        df = financials_df.sort_values("period_end").drop_duplicates(
            subset=["ticker", "metric", "period_end"], keep="last"
        )

        pivot = df.pivot_table(
            index="period_end",
            columns=["ticker", "metric"],
            values="value",
            aggfunc="last",
        )
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.resample("QE").last().ffill(limit=2)
        return pivot

    # ------------------------------------------------------------------
    # Internal — CIK resolution
    # ------------------------------------------------------------------

    def _load_cik_map(self) -> None:
        """Bulk-load ticker→CIK mapping from EDGAR company_tickers.json.

        This single request resolves all tickers at once, avoiding one
        API call per ticker for CIK lookup.
        """
        if self._cik_map:
            return  # already loaded

        cache_file = self.cache_dir / "cik_map.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        import json

        if cache_file.exists():
            with open(cache_file) as f:
                self._cik_map = json.load(f)
            return

        try:
            import requests

            url = f"{_EDGAR_BASE}/files/company_tickers.json"
            resp = requests.get(url, headers=self._headers, timeout=15)
            resp.raise_for_status()
            raw = resp.json()

            # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "..."}, ...}
            self._cik_map = {
                entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
                for entry in raw.values()
            }
            with open(cache_file, "w") as f:
                json.dump(self._cik_map, f)
            logger.info(f"[SEC] CIK map loaded: {len(self._cik_map)} tickers.")
            time.sleep(_SEC_RATE_LIMIT_DELAY)

        except Exception as exc:
            logger.error(f"[SEC] CIK map fetch failed: {exc}")

    # ------------------------------------------------------------------
    # Internal — filing metadata
    # ------------------------------------------------------------------

    def _fetch_ticker_metadata(self, ticker: str, cik: str) -> list[dict]:
        """Fetch filing metadata for one ticker via EDGAR submissions API."""
        try:
            import requests

            url = f"{_EDGAR_BASE_DATA}/submissions/CIK{cik}.json"
            resp = requests.get(url, headers=self._headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            time.sleep(_SEC_RATE_LIMIT_DELAY)
        except Exception as exc:
            logger.warning(f"[SEC] Submissions fetch failed for {ticker}: {exc}")
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        reports = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])

        records = []
        counts: dict[str, int] = {}

        for form, date_str, acc, report, pdoc in zip(
            forms, dates, accessions, reports, primary_docs
        ):
            if form not in self.form_types:
                continue

            filing_dt = pd.Timestamp(date_str)
            if filing_dt < self.start_date or filing_dt > self.end_date:
                continue

            counts[form] = counts.get(form, 0) + 1
            if counts[form] > self.max_filings_per_ticker:
                continue

            acc_clean = acc.replace("-", "")
            doc_url = (
                (
                    f"https://www.sec.gov/Archives/edgar/full-index/"
                    f"{acc_clean[:10]}/{acc}/{pdoc}"
                )
                if pdoc
                else ""
            )

            records.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "form_type": form,
                    "filing_date": date_str,
                    "report_date": report,
                    "accession_number": acc,
                    "primary_document_url": doc_url,
                }
            )

        logger.info(f"[SEC] {ticker}: {len(records)} filings indexed.")
        return records

    # ------------------------------------------------------------------
    # Internal — XBRL structured financials
    # ------------------------------------------------------------------

    def _fetch_xbrl_facts(self, ticker: str, cik: str) -> list[dict]:
        """Fetch structured financial facts from EDGAR company facts API."""
        try:
            import requests

            url = f"{_EDGAR_BASE_DATA}/api/xbrl/companyfacts/CIK{cik}.json"
            resp = requests.get(url, headers=self._headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(f"[SEC] XBRL facts fetch failed for {ticker}: {exc}")
            return []

        records = []
        us_gaap = data.get("facts", {}).get("us-gaap", {})

        for xbrl_concept, friendly_name in XBRL_CONCEPTS.items():
            concept_data = us_gaap.get(xbrl_concept, {})
            units = concept_data.get("units", {})

            # Financials are in USD; EPS in USD/share; shares in shares
            for unit_type, observations in units.items():
                for obs in observations:
                    form = obs.get("form", "")
                    if form not in ("10-Q", "10-K"):
                        continue
                    # Use only point-in-time (instant) or period end values
                    period_end = obs.get("end") or obs.get("instant")
                    if not period_end:
                        continue
                    records.append(
                        {
                            "ticker": ticker,
                            "metric": friendly_name,
                            "xbrl_concept": xbrl_concept,
                            "period_end": period_end,
                            "value": obs.get("val"),
                            "unit": unit_type,
                            "form": form,
                            "accession": obs.get("accn", ""),
                        }
                    )

        logger.info(f"[SEC] {ticker}: {len(records)} XBRL fact records.")
        return records

    # ------------------------------------------------------------------
    # Internal — MD&A text (edgartools + fallback)
    # ------------------------------------------------------------------

    def _fetch_mda_edgartools(
        self,
        ticker: str,
        form_type: str,
        as_of: pd.Timestamp,
        max_chars: int,
    ) -> str:
        """Extract MD&A section using edgartools (preferred path)."""
        try:
            import edgar

            edgar.set_identity(self.user_agent)

            company = edgar.Company(ticker)
            filings = company.get_filings(form=form_type)
            if filings is None or len(filings) == 0:
                return ""

            # Filter to filings on or before as_of date
            filing_list = [
                f
                for f in filings
                if f.filing_date is not None
                and pd.Timestamp(str(f.filing_date)) <= as_of
            ]
            if not filing_list:
                return ""

            # Most recent eligible filing
            filing = sorted(filing_list, key=lambda f: f.filing_date, reverse=True)[0]
            doc = filing.primary_document()
            if doc is None:
                return ""

            # edgartools exposes an MDA property on 10-Q/10-K objects
            tenq = filing.obj()
            mda_text = getattr(tenq, "mda", None)
            if mda_text:
                return str(mda_text)[:max_chars]

            # Fallback within edgartools: get full document text
            full_text = doc.text() if hasattr(doc, "text") else ""
            return self._extract_mda_heuristic(full_text, max_chars)

        except ImportError:
            logger.debug("[SEC] edgartools not installed — using fallback.")
            return ""
        except Exception as exc:
            logger.warning(
                f"[SEC] edgartools MD&A extraction failed for {ticker}: {exc}"
            )
            return ""

    def _fetch_raw_filing_text(
        self,
        ticker: str,
        form_type: str,
        as_of: pd.Timestamp,
        max_chars: int,
    ) -> str:
        """Fallback: download filing HTML from EDGAR and extract MD&A heuristically."""
        try:
            import requests

            cik = self._cik_map.get(ticker.upper(), "")
            if not cik:
                return ""

            # Get submissions to find the right accession number
            url = f"{_EDGAR_BASE_DATA}/submissions/CIK{cik}.json"
            resp = requests.get(url, headers=self._headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            time.sleep(_SEC_RATE_LIMIT_DELAY)

            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])

            # Find most recent eligible filing
            candidates = [
                (d, a, p)
                for f, d, a, p in zip(forms, dates, accessions, primary_docs)
                if f == form_type and pd.Timestamp(d) <= as_of
            ]
            if not candidates:
                return ""

            candidates.sort(reverse=True)
            _, acc, pdoc = candidates[0]
            acc_nodash = acc.replace("-", "")

            doc_url = (
                f"https://www.sec.gov/Archives/edgar/full-index/"
                f"{acc_nodash[:10]}/{acc}/{pdoc}"
            )
            resp2 = requests.get(doc_url, headers=self._headers, timeout=30)
            resp2.raise_for_status()
            time.sleep(_SEC_RATE_LIMIT_DELAY)

            from bs4 import BeautifulSoup

            soup = BeautifulSoup(resp2.content, "lxml")
            for tag in soup(["script", "style"]):
                tag.decompose()
            full_text = " ".join(soup.get_text(separator=" ").split())

            return self._extract_mda_heuristic(full_text, max_chars)

        except Exception as exc:
            logger.warning(f"[SEC] Raw filing fallback failed for {ticker}: {exc}")
            return ""

    @staticmethod
    def _extract_mda_heuristic(text: str, max_chars: int) -> str:
        """Heuristically locate and extract the MD&A section from raw filing text.

        Looks for the standard SEC section heading and extracts text until the
        next major section. This is a best-effort extraction — edgartools is
        more reliable when available.
        """
        import re

        # Common MD&A section header patterns in SEC filings
        mda_patterns = [
            r"ITEM\s+2[\.\s]+MANAGEMENT.{0,50}DISCUSSION",
            r"Item\s+2[\.\s]+Management.{0,50}Discussion",
        ]
        end_patterns = [
            r"ITEM\s+3[\.\s]+",
            r"Item\s+3[\.\s]+",
            r"QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES",
        ]

        mda_start = -1
        for pat in mda_patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                mda_start = match.start()
                break

        if mda_start == -1:
            # Can't locate MD&A — return first chunk of document
            return text[:max_chars]

        mda_end = len(text)
        for pat in end_patterns:
            match = re.search(pat, text[mda_start + 100 :], re.IGNORECASE)
            if match:
                mda_end = mda_start + 100 + match.start()
                break

        return text[mda_start:mda_end][:max_chars]
