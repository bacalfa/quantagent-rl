"""
agents/sector_agent.py
======================
SectorAgent — produces a per-sector outlook brief by synthesizing
recent earnings commentary, analyst coverage, and macro context.

Responsibilities
----------------
- Receives a GICS sector name, its constituent tickers, the MacroBrief
  (for macro context), and the as_of_date.
- Issues date-bounded web searches for sector-specific earnings trends,
  analyst revisions, and thematic tailwinds/headwinds.
- Returns a ``SectorBrief`` with a momentum score, earnings revision
  signal, valuation characterization, and narrative summary.

Parallelism
-----------
SectorAgents for different GICS sectors are run concurrently inside
the LangGraph orchestrator when ``sector_agents_parallel=True``.
Each sector gets an independent SectorAgent instance.
"""

import json
import logging

from agents.base import BaseAgent
from agents.config import SECTOR_SYSTEM_PROMPT, AgentConfig
from agents.schemas import MacroBrief, SectorBrief
from agents.tools import web_search

logger = logging.getLogger(__name__)

# Per-sector query templates (the sector name is interpolated at runtime)
_SECTOR_QUERY_TEMPLATES = [
    "{sector} sector earnings outlook analyst estimates",
    "{sector} sector stocks performance recent news",
    "{sector} sector valuation multiples market",
]


class SectorAgent(BaseAgent):
    """Per-GICS-sector outlook analyst.

    Parameters
    ----------
    config : AgentConfig
    sector : str
        GICS sector name (e.g., 'Information Technology').
    tickers : list[str]
        Tickers belonging to this sector in the current universe.

    Examples
    --------
    >>> agent = SectorAgent(AgentConfig(mock_mode=True), "Financials", ["JPM","BAC"])
    >>> brief = agent.run(macro_brief=macro_brief, as_of_date="2023-09-30")
    """

    def __init__(
        self,
        config: AgentConfig,
        sector: str,
        tickers: list[str],
    ) -> None:
        super().__init__(config)
        self.sector = sector
        self.tickers = tickers

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        macro_brief: MacroBrief | None,
        as_of_date: str,
    ) -> SectorBrief:
        """Produce a SectorBrief.

        Parameters
        ----------
        macro_brief : MacroBrief | None
            Current macro environment context. Provides rate/inflation
            regime signals that the sector analysis should condition on.
        as_of_date : str
            Date boundary for web searches (YYYY-MM-DD).

        Returns
        -------
        SectorBrief
        """
        logger.info(f"[SectorAgent:{self.sector}] Running for {as_of_date}")

        search_context = self._gather_search_context(as_of_date)
        user_message = self._build_user_message(macro_brief, search_context, as_of_date)

        raw = self.call_llm(
            system_prompt=SECTOR_SYSTEM_PROMPT,
            user_message=user_message,
        )
        return self._parse_brief(raw)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gather_search_context(self, as_of_date: str) -> str:
        """Run sector-specific web searches and format results."""
        if not self.cfg.enable_web_search or self.cfg.mock_mode:
            return ""

        snippets: list[str] = []
        for template in _SECTOR_QUERY_TEMPLATES:
            query = template.format(sector=self.sector)
            results = web_search(
                query=query,
                as_of_date=as_of_date,
                max_results=2,
                anthropic_client=self._anthropic_client,
            )
            for r in results:
                if r.get("snippet"):
                    snippets.append(f"[{r['title']}] {r['snippet']}")

        return "\n".join(snippets[:8])

    def _build_user_message(
        self,
        macro_brief: MacroBrief | None,
        search_context: str,
        as_of_date: str,
    ) -> str:
        parts = [
            f"as_of_date: {as_of_date}",
            f"sector: {self.sector}",
            f"constituent_tickers: {', '.join(self.tickers)}",
        ]
        if macro_brief:
            parts += [
                "",
                "=== Macro Context ===",
                f"Rate environment: {macro_brief.rate_environment}",
                f"Inflation regime: {macro_brief.inflation_regime}",
                f"Overall sentiment: {macro_brief.overall_sentiment:.2f}",
                f"Recession risk: {macro_brief.recession_risk:.2f}",
            ]
        if search_context:
            parts += ["", "=== Recent News & Earnings Commentary ===", search_context]
        parts += ["", f"Produce the SectorBrief JSON for the {self.sector} sector."]
        return "\n".join(parts)

    def _parse_brief(self, raw: str) -> SectorBrief:
        """Parse LLM output into a SectorBrief."""
        fallback = {
            "sector": self.sector,
            "momentum_score": 0.0,
            "earnings_revision_trend": "neutral",
            "valuation_signal": "fair",
            "key_themes": ["Data unavailable"],
            "risks": ["Data unavailable"],
            "analyst_summary": f"{self.sector} assessment unavailable.",
        }
        parsed = self.parse_json_response(raw, fallback)
        # Ensure sector field is always correct
        parsed["sector"] = self.sector

        try:
            return SectorBrief(**parsed)
        except Exception as exc:
            logger.warning(f"[SectorAgent:{self.sector}] Validation failed: {exc}")
            return SectorBrief(**fallback)

    # ------------------------------------------------------------------
    # Mock response
    # ------------------------------------------------------------------

    # Sector-specific mock momentum scores (realistic directional signals)
    _MOCK_SCORES: dict[str, dict] = {
        "Information Technology": {
            "momentum_score": 0.6,
            "valuation_signal": "stretched",
            "trend": "upgrades",
        },
        "Communication Services": {
            "momentum_score": 0.3,
            "valuation_signal": "fair",
            "trend": "neutral",
        },
        "Consumer Discretionary": {
            "momentum_score": -0.2,
            "valuation_signal": "fair",
            "trend": "neutral",
        },
        "Health Care": {
            "momentum_score": 0.1,
            "valuation_signal": "fair",
            "trend": "neutral",
        },
        "Financials": {
            "momentum_score": 0.2,
            "valuation_signal": "cheap",
            "trend": "upgrades",
        },
        "Industrials": {
            "momentum_score": 0.3,
            "valuation_signal": "fair",
            "trend": "upgrades",
        },
        "Energy": {
            "momentum_score": 0.4,
            "valuation_signal": "cheap",
            "trend": "upgrades",
        },
        "Consumer Staples": {
            "momentum_score": -0.1,
            "valuation_signal": "stretched",
            "trend": "downgrades",
        },
        "Materials": {
            "momentum_score": 0.0,
            "valuation_signal": "fair",
            "trend": "neutral",
        },
        "Real Estate": {
            "momentum_score": -0.4,
            "valuation_signal": "stretched",
            "trend": "downgrades",
        },
        "Utilities": {
            "momentum_score": -0.3,
            "valuation_signal": "stretched",
            "trend": "downgrades",
        },
    }

    def _mock_response(self, system_prompt: str, user_message: str) -> str:
        defaults = self._MOCK_SCORES.get(
            self.sector,
            {"momentum_score": 0.0, "valuation_signal": "fair", "trend": "neutral"},
        )
        return json.dumps(
            {
                "sector": self.sector,
                "momentum_score": defaults["momentum_score"],
                "earnings_revision_trend": defaults["trend"],
                "valuation_signal": defaults["valuation_signal"],
                "key_themes": [
                    f"{self.sector} benefiting from AI infrastructure spending",
                    "Margin improvement driven by cost discipline",
                    f"Tickers: {', '.join(self.tickers[:3])} showing relative strength",
                ],
                "risks": [
                    "Multiple compression risk if rates stay higher for longer",
                    "Geopolitical supply-chain disruption",
                ],
                "analyst_summary": (
                    f"The {self.sector} sector shows {defaults['trend']} earnings revision "
                    f"dynamics with momentum score {defaults['momentum_score']:.1f}. "
                    f"Valuation is {defaults['valuation_signal']} relative to historical "
                    f"ranges, and the macro backdrop is mixed given elevated rates."
                ),
            }
        )
