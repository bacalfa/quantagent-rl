"""
agents/macro_agent.py
=====================
MacroAgent — analyzes quantitative FRED signals and synthesizes recent
macro news to produce a structured MacroBrief.

Responsibilities
----------------
- Receives the current values of all FRED macro signals (fed funds rate,
  CPI YoY, VIX, yield curve spread, HY spread, unemployment).
- When web search is enabled, retrieves recent news on:
    * Federal Reserve policy and rate expectations
    * Inflation trajectory
    * Recession probability (using market signals + analyst consensus)
- Returns a ``MacroBrief`` with discrete regime labels and numeric risk scores.

Walk-Forward Safety
-------------------
All web searches are date-bounded with ``as_of_date``. The macro signal
values passed in must already be restricted to the as-of date by the caller
(DataPipeline enforces this via the fold boundary).
"""

import json
import logging

from agents.base import BaseAgent
from agents.config import MACRO_SYSTEM_PROMPT, AgentConfig
from agents.schemas import MacroBrief
from agents.tools import format_macro_for_llm, web_search

logger = logging.getLogger(__name__)

# Web searches issued by the macro agent (date-bounded at call time)
_MACRO_SEARCH_QUERIES = [
    "Federal Reserve monetary policy rate decision outlook",
    "US inflation CPI trend consumer prices",
    "US economic recession risk GDP growth",
    "credit market stress high yield spreads default rates",
]


class MacroAgent(BaseAgent):
    """Macro-economic environment analyst.

    Parameters
    ----------
    config : AgentConfig

    Examples
    --------
    >>> agent = MacroAgent(AgentConfig(mock_mode=True))
    >>> brief = agent.run(macro_data={...}, as_of_date="2023-09-30")
    """

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        macro_data: dict[str, float],
        as_of_date: str,
    ) -> MacroBrief:
        """Produce a MacroBrief from FRED signals and optional web search.

        Parameters
        ----------
        macro_data : dict[str, float]
            Current values of FRED macro indicators. Keys match the
            ``FRED_SERIES`` mapping in ``data/config.py``.
        as_of_date : str
            Date the analysis is anchored to (YYYY-MM-DD).
            Used to date-bound web searches.

        Returns
        -------
        MacroBrief
        """
        logger.info(f"[MacroAgent] Running for {as_of_date}")

        # Build the formatted context string
        macro_context = format_macro_for_llm(macro_data, as_of_date)

        # Optionally augment with web-search snippets
        search_context = self._gather_search_context(as_of_date)

        user_message = self._build_user_message(
            macro_context, search_context, as_of_date
        )

        raw = self.call_llm(
            system_prompt=MACRO_SYSTEM_PROMPT,
            user_message=user_message,
        )

        return self._parse_brief(raw, macro_data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gather_search_context(self, as_of_date: str) -> str:
        """Run date-bounded web searches and format results as context."""
        if not self.cfg.enable_web_search or self.cfg.mock_mode:
            return ""

        snippets: list[str] = []
        for query in _MACRO_SEARCH_QUERIES[: self.cfg.max_search_results]:
            results = web_search(
                query=query,
                as_of_date=as_of_date,
                max_results=2,
                anthropic_client=self._anthropic_client,
            )
            for r in results:
                if r.get("snippet"):
                    snippets.append(f"[{r['title']}] {r['snippet']}")

        if not snippets:
            return ""
        return "\n".join(snippets[:10])  # cap total snippets

    @staticmethod
    def _build_user_message(
        macro_context: str,
        search_context: str,
        as_of_date: str,
    ) -> str:
        parts = [
            f"as_of_date: {as_of_date}",
            "",
            macro_context,
        ]
        if search_context:
            parts += ["", "=== Recent News Context ===", search_context]
        parts += [
            "",
            "Based on the above data, produce the MacroBrief JSON.",
        ]
        return "\n".join(parts)

    def _parse_brief(self, raw: str, macro_data: dict) -> MacroBrief:
        """Parse LLM response into a MacroBrief, with sensible fallbacks."""
        fallback_data = self._heuristic_brief(macro_data)
        parsed = self.parse_json_response(raw, fallback_data)

        # Validate and coerce via Pydantic
        try:
            return MacroBrief(**parsed)
        except Exception as exc:
            logger.warning(
                f"[MacroAgent] Pydantic validation failed: {exc}; using heuristic fallback."
            )
            return MacroBrief(**fallback_data)

    @staticmethod
    def _heuristic_brief(macro_data: dict) -> dict:
        """Derive a heuristic MacroBrief from raw signal values.

        Used when the LLM call fails or returns invalid JSON.
        Simple threshold-based logic ensures a reasonable fallback.
        """
        vix = macro_data.get("vix", 20.0)
        ff_rate = macro_data.get("fed_funds_rate", 2.5)
        cpi = macro_data.get("cpi_yoy", 2.5)
        ycurve = macro_data.get("yield_curve_10y2y", 0.5)
        hy_spr = macro_data.get("hy_spread", 400.0)

        rate_env = (
            "tightening" if ff_rate > 3.5 else "easing" if ff_rate < 1.5 else "neutral"
        )
        inf_reg = (
            "high"
            if cpi > 5
            else "elevated"
            if cpi > 3
            else "moderate"
            if cpi > 1.5
            else "low"
        )
        yc_sig = "inverted" if ycurve < 0 else "flat" if ycurve < 0.5 else "normal"
        rec_risk = min(
            1.0, max(0.0, (0.3 if ycurve < 0 else 0.1) + (0.2 if vix > 30 else 0.0))
        )
        cr_stress = min(1.0, max(0.0, (hy_spr - 300) / 700))
        sentiment = max(-1.0, min(1.0, -0.5 * rec_risk + 0.3 * (vix < 20)))

        return {
            "rate_environment": rate_env,
            "inflation_regime": inf_reg,
            "recession_risk": round(rec_risk, 3),
            "yield_curve_signal": yc_sig,
            "credit_stress": round(cr_stress, 3),
            "overall_sentiment": round(float(sentiment), 3),
            "key_risks": ["Heuristic fallback — LLM unavailable"],
            "tailwinds": ["Heuristic fallback — LLM unavailable"],
            "analyst_summary": "Macro brief computed via heuristic fallback.",
        }

    # ------------------------------------------------------------------
    # Mock response
    # ------------------------------------------------------------------

    def _mock_response(self, system_prompt: str, user_message: str) -> str:
        return json.dumps(
            {
                "rate_environment": "tightening",
                "inflation_regime": "elevated",
                "recession_risk": 0.28,
                "yield_curve_signal": "flat",
                "credit_stress": 0.35,
                "overall_sentiment": -0.15,
                "key_risks": [
                    "Persistent core inflation limiting Fed pivot",
                    "Tighter financial conditions weighing on growth",
                    "Commercial real estate credit exposure",
                ],
                "tailwinds": [
                    "Resilient labor market supporting consumption",
                    "AI-driven capex cycle boosting productivity",
                    "Energy sector profitability at multi-year highs",
                ],
                "analyst_summary": (
                    "The macro environment remains cautiously risk-off with the Fed "
                    "holding rates elevated to combat persistent services inflation. "
                    "The yield curve flattening and widening HY spreads signal "
                    "deteriorating credit conditions, though consumer spending remains "
                    "resilient on the back of tight labor markets."
                ),
            }
        )
