"""
agents/tools.py
===============
Tool definitions for the QuantAgent-RL agents module.

Tools are implemented as plain callables so they work both with LangChain's
tool-calling infrastructure and with direct invocation in tests.

Web Search and Look-Ahead Bias
-------------------------------
During live trading, agents use Anthropic's web-search tool (server-side)
so results are naturally current.

During backtesting, *strict date-bounding cannot be guaranteed* for live
web searches. This is a documented limitation: if you backtest with
``enable_web_search=True``, search results may include news published after
the as_of_date.  The recommended backtesting workflow is:

  1. Run the agent pipeline with real web search for the current period only.
  2. Cache all agent outputs (MarketBrief JSON) to disk.
  3. Replay the cached outputs during historical fold simulations.

The ``DateBoundedQuery`` helper appends date qualifiers to every search
query as a best-effort mitigation, but this is not a hard guarantee.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache

import requests

logger = logging.getLogger(__name__)

_EDGAR_BASE = "https://www.sec.gov"
_EDGAR_BASE_DATA = "https://data.sec.gov"
_EDGAR_HEADERS = {"User-Agent": "QuantAgentRL research@example.com"}
_FRED_BASE = "https://api.stlouisfed.org/fred"


# ---------------------------------------------------------------------------
# Date-bounded query helper
# ---------------------------------------------------------------------------


@dataclass
class DateBoundedQuery:
    """Wraps a search query with date qualifiers for best-effort temporal bounding.

    Parameters
    ----------
    query : str
        Raw search query (e.g., 'Federal Reserve rate decision').
    as_of_date : str
        Date in YYYY-MM-DD format. Used to add year/quarter context.
    """

    query: str
    as_of_date: str

    def build(self) -> str:
        """Return the query with date context appended.

        Example
        -------
        >>> DateBoundedQuery("Fed rate decision", "2022-09-21").build()
        'Fed rate decision Q3 2022'
        """
        try:
            import pandas as pd

            ts = pd.Timestamp(self.as_of_date)
            quarter_str = f"Q{ts.quarter} {ts.year}"
            year_str = str(ts.year)
        except Exception:
            # Fallback: extract year from string
            year_match = re.search(r"\d{4}", self.as_of_date)
            quarter_str = year_match.group() if year_match else ""
            year_str = quarter_str

        # Avoid duplicating the year if it's already in the query
        if year_str in self.query:
            return self.query
        return f"{self.query} {quarter_str}"


# ---------------------------------------------------------------------------
# Web search tool
# ---------------------------------------------------------------------------


def web_search(
    query: str,
    as_of_date: str | None = None,
    max_results: int = 5,
    anthropic_client: object | None = None,
) -> list[dict]:
    """Perform a web search and return structured result snippets.

    When an Anthropic client is provided, uses the server-side
    ``web_search_20250305`` tool (higher quality, no key required beyond
    the Anthropic API key).

    Falls back to a direct DuckDuckGo instant-answer request if no client
    is provided (useful for testing or when search is not critical).

    Parameters
    ----------
    query : str
        Search query.
    as_of_date : str | None
        If set, appends date context to the query.
    max_results : int
        Maximum number of results to return.
    anthropic_client : anthropic.Anthropic | None
        Pre-initialized Anthropic client. If None, uses DuckDuckGo fallback.

    Returns
    -------
    list[dict]
        Each dict has keys: 'title', 'snippet', 'url'.
    """
    if as_of_date:
        query = DateBoundedQuery(query, as_of_date).build()

    if anthropic_client is not None:
        return _web_search_anthropic(query, max_results, anthropic_client)
    return _web_search_ddg(query, max_results)


def _web_search_anthropic(
    query: str,
    max_results: int,
    client: object,
) -> list[dict]:
    """Use Anthropic's server-side web-search tool to retrieve results."""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # fast model for tool calls
            max_tokens=1024,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": f"Search for: {query}"}],
        )
        results = []
        for block in response.content:
            if getattr(block, "type", "") == "tool_result":
                try:
                    data = json.loads(block.content[0].text)
                    for item in data.get("results", [])[:max_results]:
                        results.append(
                            {
                                "title": item.get("title", ""),
                                "snippet": item.get("snippet", ""),
                                "url": item.get("url", ""),
                            }
                        )
                except (json.JSONDecodeError, AttributeError, IndexError):
                    pass
        return results
    except Exception as exc:
        logger.warning(f"[Tools] Anthropic web search failed: {exc}")
        return []


def _web_search_ddg(query: str, max_results: int) -> list[dict]:
    """Lightweight DuckDuckGo instant-answer fallback (no API key needed)."""
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_redirect": "1", "no_html": "1"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        # Abstract answer
        if data.get("AbstractText"):
            results.append(
                {
                    "title": data.get("Heading", query),
                    "snippet": data["AbstractText"],
                    "url": data.get("AbstractURL", ""),
                }
            )
        # Related topics
        for topic in data.get("RelatedTopics", [])[: max_results - len(results)]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(
                    {
                        "title": topic.get("Text", "")[:80],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                    }
                )
        return results[:max_results]
    except Exception as exc:
        logger.warning(f"[Tools] DuckDuckGo fallback failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# FRED fetch tool
# ---------------------------------------------------------------------------


def fetch_fred_series(
    series_id: str,
    api_key: str,
    observation_start: str,
    observation_end: str,
    frequency: str = "q",
) -> list[dict]:
    """Fetch a FRED time series and return recent observations.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g., 'FEDFUNDS').
    api_key : str
        FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html).
    observation_start : str
        Start date YYYY-MM-DD.
    observation_end : str
        End date YYYY-MM-DD.
    frequency : str
        Frequency aggregation ('d' = daily, 'm' = monthly, 'q' = quarterly).

    Returns
    -------
    list[dict]
        Each dict has keys 'date' and 'value'.
    """
    try:
        url = f"{_FRED_BASE}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": observation_start,
            "observation_end": observation_end,
            "frequency": frequency,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        return [
            {"date": o["date"], "value": o["value"]}
            for o in observations
            if o["value"] != "."
        ]
    except Exception as exc:
        logger.warning(f"[Tools] FRED fetch failed ({series_id}): {exc}")
        return []


# ---------------------------------------------------------------------------
# EDGAR XBRL fetch tool
# ---------------------------------------------------------------------------


@lru_cache(maxsize=512)
def _fetch_cik(ticker: str, user_agent: str) -> str | None:
    """Return the 10-digit CIK string for a ticker (cached)."""
    try:
        url = f"{_EDGAR_BASE}/files/company_tickers.json"
        hdrs = {"User-Agent": user_agent}
        resp = requests.get(url, headers=hdrs, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        for entry in raw.values():
            if entry["ticker"].upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception as exc:
        logger.warning(f"[Tools] CIK lookup failed for {ticker}: {exc}")
    return None


def fetch_xbrl_facts(
    ticker: str,
    user_agent: str = "QuantAgentRL research@example.com",
    concepts: list[str] | None = None,
    last_n_periods: int = 8,
) -> dict[str, list[dict]]:
    """Fetch structured XBRL financial facts for a ticker from EDGAR.

    Parameters
    ----------
    ticker : str
    user_agent : str
        Required by SEC EDGAR fair-use policy.
    concepts : list[str] | None
        XBRL concept names to fetch. Defaults to key financial metrics.
    last_n_periods : int
        Number of most recent periods to return per concept.

    Returns
    -------
    dict[str, list[dict]]
        Keys = friendly metric names. Values = list of {period_end, value, form}.
    """
    default_concepts = {
        "Revenues": "revenue",
        "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue_alt",
        "NetIncomeLoss": "net_income",
        "EarningsPerShareDiluted": "eps_diluted",
        "GrossProfit": "gross_profit",
        "OperatingIncomeLoss": "operating_income",
        "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
        "LongTermDebt": "long_term_debt",
    }
    if concepts:
        concept_map = {c: c.lower() for c in concepts}
    else:
        concept_map = default_concepts

    cik = _fetch_cik(ticker, user_agent)
    if not cik:
        return {}

    try:
        url = f"{_EDGAR_BASE_DATA}/api/xbrl/companyfacts/CIK{cik}.json"
        hdrs = {"User-Agent": user_agent}
        resp = requests.get(url, headers=hdrs, timeout=30)
        resp.raise_for_status()
        time.sleep(0.12)  # respect SEC rate limit
        data = resp.json()
    except Exception as exc:
        logger.warning(f"[Tools] EDGAR XBRL fetch failed for {ticker}: {exc}")
        return {}

    us_gaap = data.get("facts", {}).get("us-gaap", {})
    result: dict[str, list[dict]] = {}

    for xbrl_name, friendly_name in concept_map.items():
        concept_data = us_gaap.get(xbrl_name, {})
        observations = []
        for unit_type, obs_list in concept_data.get("units", {}).items():
            for obs in obs_list:
                if obs.get("form") not in ("10-Q", "10-K"):
                    continue
                period_end = obs.get("end") or obs.get("instant")
                if not period_end or obs.get("val") is None:
                    continue
                observations.append(
                    {
                        "period_end": period_end,
                        "value": obs["val"],
                        "unit": unit_type,
                        "form": obs["form"],
                    }
                )
        if observations:
            # Sort by period_end descending, keep last_n_periods
            observations.sort(key=lambda x: x["period_end"], reverse=True)
            result[friendly_name] = observations[:last_n_periods]

    return result


# ---------------------------------------------------------------------------
# Utility: format financials for LLM context
# ---------------------------------------------------------------------------


def format_financials_for_llm(
    xbrl_facts: dict[str, list[dict]],
    mda_text: str,
    ticker: str,
    max_chars: int = 4000,
) -> str:
    """Format XBRL facts and MD&A text into a compact LLM-ready string.

    Parameters
    ----------
    xbrl_facts : dict
        Output of ``fetch_xbrl_facts()``.
    mda_text : str
        MD&A section text from the most recent filing.
    ticker : str
    max_chars : int
        Maximum total characters in the output string.

    Returns
    -------
    str
        Formatted string ready for inclusion in a Claude prompt.
    """
    lines = [f"=== {ticker} Financial Summary ==="]

    # XBRL structured financials
    if xbrl_facts:
        lines.append("\n[Recent Financials — Last 4 Quarters]")
        for metric, obs_list in xbrl_facts.items():
            recent = obs_list[:4]
            vals = [
                f"{o['period_end'][:7]}: {o['value']:,.0f}"
                if isinstance(o["value"], (int, float))
                else f"{o['period_end'][:7]}: {o['value']}"
                for o in recent
            ]
            lines.append(f"  {metric}: {' | '.join(vals)}")
    else:
        lines.append("\n[Financials: Not available from EDGAR]")

    # MD&A text
    if mda_text:
        available = max_chars - sum(len(l) for l in lines) - 100
        truncated = mda_text[: max(0, available)]
        lines.append("\n[MD&A Excerpt]")
        lines.append(truncated)
        if len(mda_text) > available:
            lines.append("... [truncated]")

    return "\n".join(lines)


def format_macro_for_llm(
    macro_data: dict[str, float],
    as_of_date: str,
) -> str:
    """Format FRED macro signals into a compact LLM-ready string.

    Parameters
    ----------
    macro_data : dict[str, float]
        Mapping of signal name → current value (e.g., from FRED).
    as_of_date : str

    Returns
    -------
    str
    """
    lines = [f"=== Macro Indicators as of {as_of_date} ==="]
    labels = {
        "fed_funds_rate": "Fed Funds Rate (%)",
        "cpi_yoy": "CPI YoY (%)",
        "vix": "VIX",
        "yield_curve_10y2y": "10Y-2Y Spread (bps)",
        "hy_spread": "HY Credit Spread (bps)",
        "unemployment_rate": "Unemployment Rate (%)",
        "consumer_sentiment": "UMich Consumer Sentiment",
        "wti_crude_oil": "WTI Crude Oil ($/bbl)",
    }
    for key, label in labels.items():
        val = macro_data.get(key)
        if val is not None:
            lines.append(f"  {label}: {val:.2f}")
    return "\n".join(lines)
