"""
agents/company_agent.py
=======================
CompanyAgent — produces a per-stock fundamental assessment from SEC EDGAR
XBRL financials and the MD&A section of the most recent quarterly filing.

Responsibilities
----------------
- Receives a ticker, its structured XBRL financials (revenue, net income,
  EPS, cash flow, debt), and the MD&A text (from SECFilingIngester).
- Optionally receives the SectorBrief for additional context.
- Returns a ``CompanyBrief`` with revenue/margin trend signals, a
  balance-sheet quality score, and a composite fundamental score in [-1, 1].

No Web Search
-------------
The company agent does not issue web searches. Its entire information set
is the EDGAR data already fetched by the data module. This keeps costs
lower (EDGAR is free) and ensures cleaner date-bounding — the XBRL data
and MD&A are intrinsically bounded to the filing date.
"""

import json
import logging

from agents.base import BaseAgent
from agents.config import COMPANY_SYSTEM_PROMPT, AgentConfig
from agents.schemas import CompanyBrief, SectorBrief
from agents.tools import format_financials_for_llm

logger = logging.getLogger(__name__)


class CompanyAgent(BaseAgent):
    """Per-stock fundamental analyst.

    Parameters
    ----------
    config : AgentConfig

    Examples
    --------
    >>> agent = CompanyAgent(AgentConfig(mock_mode=True))
    >>> brief = agent.run(ticker="AAPL", xbrl_facts={}, mda_text="...",
    ...                   sector_brief=None, as_of_date="2023-09-30")
    """

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        ticker: str,
        xbrl_facts: dict[str, list[dict]],
        mda_text: str,
        sector_brief: SectorBrief | None,
        as_of_date: str,
    ) -> CompanyBrief:
        """Produce a CompanyBrief for a single stock.

        Parameters
        ----------
        ticker : str
        xbrl_facts : dict[str, list[dict]]
            Output of ``agents.tools.fetch_xbrl_facts()``.
        mda_text : str
            MD&A section text from the most recent 10-Q or 10-K.
        sector_brief : SectorBrief | None
            Sector-level context for the company's industry.
        as_of_date : str
            Date boundary (YYYY-MM-DD).

        Returns
        -------
        CompanyBrief
        """
        logger.info(f"[CompanyAgent] Running for {ticker} @ {as_of_date}")

        financials_context = format_financials_for_llm(
            xbrl_facts, mda_text, ticker, max_chars=self.cfg.max_mda_chars
        )
        user_message = self._build_user_message(
            ticker, financials_context, sector_brief, as_of_date
        )

        raw = self.call_llm(
            system_prompt=COMPANY_SYSTEM_PROMPT,
            user_message=user_message,
        )
        return self._parse_brief(raw, ticker, xbrl_facts, as_of_date)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_message(
        ticker: str,
        financials_context: str,
        sector_brief: SectorBrief | None,
        as_of_date: str,
    ) -> str:
        parts = [
            f"ticker: {ticker}",
            f"as_of_date: {as_of_date}",
            "",
            financials_context,
        ]
        if sector_brief:
            parts += [
                "",
                "=== Sector Context ===",
                f"Sector: {sector_brief.sector}",
                f"Sector momentum: {sector_brief.momentum_score:.2f}",
                f"Earnings revision trend: {sector_brief.earnings_revision_trend}",
                f"Sector summary: {sector_brief.analyst_summary}",
            ]
        parts += ["", f"Produce the CompanyBrief JSON for {ticker}."]
        return "\n".join(parts)

    def _parse_brief(
        self, raw: str, ticker: str, xbrl_facts: dict[str, list[dict]], as_of_date: str
    ) -> CompanyBrief:
        """Parse LLM output into a CompanyBrief with heuristic fallback."""
        fallback = self._heuristic_brief(ticker, xbrl_facts, as_of_date)
        parsed = self.parse_json_response(raw, fallback)
        parsed["as_of_date"] = as_of_date
        parsed["ticker"] = ticker

        try:
            return CompanyBrief(**parsed)
        except Exception as exc:
            logger.warning(f"[CompanyAgent:{ticker}] Validation failed: {exc}")
            return CompanyBrief(**fallback)

    @staticmethod
    def _heuristic_brief(
        ticker: str, xbrl_facts: dict[str, list[dict]], as_of_date: str
    ) -> dict:
        """Derive heuristic signals from XBRL data when LLM fails.

        Computes simple YoY growth and margin trend signals from the
        structured financial data without relying on an LLM.
        """
        try:
            rev_obs = xbrl_facts.get("revenue", xbrl_facts.get("revenue_alt", []))
            vals = [
                o["value"] for o in rev_obs[:8] if isinstance(o["value"], (int, float))
            ]
            if len(vals) >= 5:
                recent_growth = (vals[0] - vals[4]) / (abs(vals[4]) + 1e-9)
                if recent_growth > 0.10:
                    rev_trend = "accelerating"
                elif recent_growth > 0.0:
                    rev_trend = "stable"
                elif recent_growth > -0.10:
                    rev_trend = "decelerating"
                else:
                    rev_trend = "negative"
                fund_score = max(-1.0, min(1.0, recent_growth * 3))
            else:
                rev_trend = "stable"
                fund_score = 0.0
        except Exception:
            rev_trend = "stable"
            fund_score = 0.0

        return {
            "as_of_date": as_of_date,
            "ticker": ticker,
            "revenue_growth_trend": rev_trend,
            "margin_trend": "stable",
            "balance_sheet_quality": "adequate",
            "earnings_quality": "medium",
            "fundamental_score": round(fund_score, 3),
            "key_risks": ["Heuristic fallback — LLM unavailable"],
            "key_catalysts": ["Heuristic fallback — LLM unavailable"],
            "analyst_summary": f"{ticker} brief computed via heuristic fallback.",
        }

    # ------------------------------------------------------------------
    # Mock response
    # ------------------------------------------------------------------

    # Realistic mock fundamentals for well-known tickers
    _MOCK_FUNDAMENTALS: dict[str, dict] = {
        "AAPL": {
            "rev": "stable",
            "margin": "stable",
            "bs": "strong",
            "eq": "high",
            "score": 0.6,
        },
        "MSFT": {
            "rev": "accelerating",
            "margin": "expanding",
            "bs": "strong",
            "eq": "high",
            "score": 0.8,
        },
        "NVDA": {
            "rev": "accelerating",
            "margin": "expanding",
            "bs": "strong",
            "eq": "high",
            "score": 0.9,
        },
        "AVGO": {
            "rev": "accelerating",
            "margin": "expanding",
            "bs": "adequate",
            "eq": "high",
            "score": 0.7,
        },
        "GOOGL": {
            "rev": "stable",
            "margin": "stable",
            "bs": "strong",
            "eq": "high",
            "score": 0.5,
        },
        "META": {
            "rev": "accelerating",
            "margin": "expanding",
            "bs": "strong",
            "eq": "high",
            "score": 0.7,
        },
        "AMZN": {
            "rev": "stable",
            "margin": "expanding",
            "bs": "adequate",
            "eq": "medium",
            "score": 0.5,
        },
        "TSLA": {
            "rev": "decelerating",
            "margin": "compressing",
            "bs": "adequate",
            "eq": "medium",
            "score": -0.1,
        },
        "JPM": {
            "rev": "stable",
            "margin": "stable",
            "bs": "strong",
            "eq": "high",
            "score": 0.4,
        },
        "BAC": {
            "rev": "stable",
            "margin": "stable",
            "bs": "adequate",
            "eq": "medium",
            "score": 0.2,
        },
        "GS": {
            "rev": "decelerating",
            "margin": "stable",
            "bs": "adequate",
            "eq": "medium",
            "score": 0.1,
        },
        "XOM": {
            "rev": "stable",
            "margin": "stable",
            "bs": "strong",
            "eq": "high",
            "score": 0.4,
        },
        "CVX": {
            "rev": "stable",
            "margin": "stable",
            "bs": "strong",
            "eq": "high",
            "score": 0.4,
        },
        "JNJ": {
            "rev": "stable",
            "margin": "stable",
            "bs": "strong",
            "eq": "high",
            "score": 0.3,
        },
        "LLY": {
            "rev": "accelerating",
            "margin": "expanding",
            "bs": "strong",
            "eq": "high",
            "score": 0.8,
        },
        "UNH": {
            "rev": "stable",
            "margin": "stable",
            "bs": "strong",
            "eq": "high",
            "score": 0.4,
        },
        "RTX": {
            "rev": "stable",
            "margin": "stable",
            "bs": "adequate",
            "eq": "medium",
            "score": 0.3,
        },
        "CAT": {
            "rev": "stable",
            "margin": "stable",
            "bs": "adequate",
            "eq": "medium",
            "score": 0.3,
        },
        "NEE": {
            "rev": "decelerating",
            "margin": "compressing",
            "bs": "stretched",
            "eq": "medium",
            "score": -0.2,
        },
        "PLD": {
            "rev": "stable",
            "margin": "stable",
            "bs": "stretched",
            "eq": "medium",
            "score": 0.1,
        },
    }

    def _mock_response(self, system_prompt: str, user_message: str) -> str:
        # Extract ticker from user message
        ticker = "UNKNOWN"
        for line in user_message.splitlines():
            if line.strip().startswith("ticker:"):
                ticker = line.split(":", 1)[1].strip()
                break

        d = self._MOCK_FUNDAMENTALS.get(
            ticker,
            {
                "rev": "stable",
                "margin": "stable",
                "bs": "adequate",
                "eq": "medium",
                "score": 0.0,
            },
        )
        return json.dumps(
            {
                "ticker": ticker,
                "revenue_growth_trend": d["rev"],
                "margin_trend": d["margin"],
                "balance_sheet_quality": d["bs"],
                "earnings_quality": d["eq"],
                "fundamental_score": d["score"],
                "key_risks": [
                    f"{ticker} faces macro headwinds from elevated interest rates",
                    "Competitive pressure from new market entrants",
                ],
                "key_catalysts": [
                    f"{ticker} AI-related revenue opportunity",
                    "Cost efficiency program expected to boost margins",
                ],
                "analyst_summary": (
                    f"{ticker} demonstrates {d['rev']} revenue growth with "
                    f"{d['margin']} margins and a {d['bs']} balance sheet. "
                    f"Fundamental score of {d['score']:.1f} reflects "
                    f"{'positive' if d['score'] > 0 else 'neutral to negative'} "
                    f"near-term outlook."
                ),
            }
        )
