"""
agents/orchestrator.py
======================
OrchestratorAgent and LangGraph StateGraph for QuantAgent-RL.

Graph Topology
--------------
The LangGraph ``StateGraph`` wires four agent nodes with macro and sector
running in parallel before the company and orchestrator nodes:

    START ──────────────────────────────────────────────────┐
           │                                                │
    [macro_node]                                  [sector_node_*]
           │                                                │
           └──────────────── company_node ──────────────────┘
                                  │
                           orchestrator_node
                                  │
                                END

Parallel Execution
------------------
LangGraph handles parallelism automatically: when both ``macro_node`` and
``sector_node`` have edges pointing to ``company_node``, LangGraph executes
the predecessors concurrently (using asyncio) and waits for both to complete
before advancing to the company step.

State Schema
------------
The shared graph state is a TypedDict. All fields use Annotated with
``operator.add`` (append semantics) or direct assignment as appropriate.
"""

import json
import logging
import operator

# TypedDict and Annotated are required by LangGraph StateGraph — no builtin equivalents exist.
from typing import Annotated, TypedDict

from agents.base import BaseAgent
from agents.config import ORCHESTRATOR_SYSTEM_PROMPT, AgentConfig
from agents.schemas import (
    CompanyBrief,
    MacroBrief,
    MarketBrief,
    SectorBrief,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph state schema
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    """Shared state flowing through the LangGraph StateGraph.

    All mutable fields use append-semantics (operator.add) so that
    parallel branches can safely write to the same state without clobbering
    each other.
    """

    as_of_date: str
    tickers: list[str]
    sector_map: dict[str, list[str]]  # sector → [tickers]
    macro_data: dict[str, float]  # FRED signal values
    xbrl_data: dict[str, dict]  # ticker → XBRL facts
    mda_data: dict[str, str]  # ticker → MD&A text

    # Outputs written by each node
    macro_brief: MacroBrief | None
    sector_briefs: dict[str, SectorBrief]
    company_briefs: dict[str, CompanyBrief]
    market_brief: MarketBrief | None
    errors: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------


class OrchestratorAgent(BaseAgent):
    """Synthesizes all sub-analyses into a unified MarketBrief.

    The orchestrator receives the MacroBrief, all SectorBriefs, and all
    CompanyBriefs and produces the master MarketBrief that feeds the RL
    state vector.

    Parameters
    ----------
    config : AgentConfig

    Examples
    --------
    >>> orch = OrchestratorAgent(AgentConfig(mock_mode=True))
    >>> brief = orch.run(as_of_date="2023-09-30", tickers=[...],
    ...                  macro_brief=..., sector_briefs={...},
    ...                  company_briefs={...})
    """

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        as_of_date: str,
        tickers: list[str],
        macro_brief: MacroBrief | None,
        sector_briefs: dict[str, SectorBrief],
        company_briefs: dict[str, CompanyBrief],
    ) -> MarketBrief:
        """Produce the unified MarketBrief.

        Parameters
        ----------
        as_of_date : str
        tickers : list[str]
        macro_brief : MacroBrief | None
        sector_briefs : dict[str, SectorBrief]
        company_briefs : dict[str, CompanyBrief]

        Returns
        -------
        MarketBrief
        """
        logger.info(
            f"[OrchestratorAgent] Synthesizing {len(company_briefs)} company briefs @ {as_of_date}"
        )

        user_message = self._build_user_message(
            as_of_date, tickers, macro_brief, sector_briefs, company_briefs
        )
        raw = self.call_llm(
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            user_message=user_message,
        )
        return self._parse_brief(
            raw, as_of_date, tickers, macro_brief, sector_briefs, company_briefs
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_message(
        as_of_date: str,
        tickers: list[str],
        macro_brief: MacroBrief | None,
        sector_briefs: dict[str, SectorBrief],
        company_briefs: dict[str, CompanyBrief],
    ) -> str:
        parts = [
            f"as_of_date: {as_of_date}",
            f"universe_tickers: {', '.join(tickers)}",
            "",
        ]

        if macro_brief:
            parts += [
                "=== MACRO ANALYSIS ===",
                f"Rate environment: {macro_brief.rate_environment}",
                f"Inflation regime: {macro_brief.inflation_regime}",
                f"Recession risk: {macro_brief.recession_risk:.2f}",
                f"Yield curve: {macro_brief.yield_curve_signal}",
                f"Credit stress: {macro_brief.credit_stress:.2f}",
                f"Overall sentiment: {macro_brief.overall_sentiment:.2f}",
                f"Key risks: {'; '.join(macro_brief.key_risks)}",
                f"Tailwinds: {'; '.join(macro_brief.tailwinds)}",
                f"Summary: {macro_brief.analyst_summary}",
                "",
            ]

        if sector_briefs:
            parts.append("=== SECTOR ANALYSES ===")
            for sector, sb in sector_briefs.items():
                parts.append(
                    f"{sector}: momentum={sb.momentum_score:.2f}, "
                    f"revisions={sb.earnings_revision_trend}, "
                    f"valuation={sb.valuation_signal}. "
                    f"{sb.analyst_summary}"
                )
            parts.append("")

        if company_briefs:
            parts.append("=== COMPANY ANALYSES ===")
            for ticker, cb in company_briefs.items():
                parts.append(
                    f"{ticker}: revenue_trend={cb.revenue_growth_trend}, "
                    f"margin={cb.margin_trend}, bs={cb.balance_sheet_quality}, "
                    f"score={cb.fundamental_score:.2f}. "
                    f"{cb.analyst_summary}"
                )
            parts.append("")

        parts.append(
            "Synthesize all analyses above into the MarketBrief JSON. "
            "Resolve any contradictions using sound investment judgment."
        )
        return "\n".join(parts)

    def _parse_brief(
        self,
        raw: str,
        as_of_date: str,
        tickers: list[str],
        macro_brief: MacroBrief | None,
        sector_briefs: dict[str, SectorBrief],
        company_briefs: dict[str, CompanyBrief],
    ) -> MarketBrief:
        """Parse LLM output into a MarketBrief with heuristic fallback."""
        fallback = self._heuristic_brief(
            as_of_date, tickers, macro_brief, sector_briefs, company_briefs
        )
        parsed = self.parse_json_response(raw, fallback)
        parsed["as_of_date"] = as_of_date

        try:
            brief = MarketBrief(**parsed)
        except Exception as exc:
            logger.warning(
                f"[OrchestratorAgent] Validation failed: {exc}; using heuristic."
            )
            brief = MarketBrief(**fallback)

        # Attach source briefs for auditability
        brief.macro_brief = macro_brief
        brief.sector_briefs = sector_briefs
        brief.company_briefs = company_briefs
        return brief

    @staticmethod
    def _heuristic_brief(
        as_of_date: str,
        tickers: list[str],
        macro_brief: MacroBrief | None,
        sector_briefs: dict[str, SectorBrief],
        company_briefs: dict[str, CompanyBrief],
    ) -> dict:
        """Score-weighted heuristic MarketBrief when LLM fails."""
        # Macro regime from macro brief
        if macro_brief:
            sent = macro_brief.overall_sentiment
            macro_regime = (
                "risk_on"
                if sent > 0.2
                else "risk_off"
                if sent < -0.2
                else "transitional"
            )
            portfolio_stance = (
                "aggressive"
                if sent > 0.3
                else "defensive"
                if sent < -0.3
                else "neutral"
            )
            conv = abs(sent)
        else:
            macro_regime = "transitional"
            portfolio_stance = "neutral"
            conv = 0.0

        # Rank companies by fundamental score for overweight/underweight lists
        sorted_cb = sorted(
            company_briefs.items(),
            key=lambda kv: kv[1].fundamental_score,
            reverse=True,
        )
        top_over = [t for t, _ in sorted_cb[:5]]
        top_under = [
            t for t, _ in sorted_cb[-5:] if sorted_cb[-1][1].fundamental_score < -0.1
        ]

        sector_tilts = {
            s: round(sb.momentum_score, 2) for s, sb in sector_briefs.items()
        }

        return {
            "as_of_date": as_of_date,
            "macro_regime": macro_regime,
            "portfolio_stance": portfolio_stance,
            "conviction_score": round(min(1.0, conv), 3),
            "top_overweights": top_over,
            "top_underweights": top_under,
            "sector_tilts": sector_tilts,
            "key_themes": ["Heuristic fallback — LLM unavailable"],
            "risk_flags": ["LLM call failed — signals based on heuristics"],
            "executive_summary": "MarketBrief produced via heuristic fallback due to LLM failure.",
        }

    # ------------------------------------------------------------------
    # Mock response
    # ------------------------------------------------------------------

    def _mock_response(self, system_prompt: str, user_message: str) -> str:
        return json.dumps(
            {
                "as_of_date": "2023-09-30",
                "macro_regime": "transitional",
                "portfolio_stance": "neutral",
                "conviction_score": 0.55,
                "top_overweights": ["NVDA", "MSFT", "LLY", "CAT", "XOM"],
                "top_underweights": ["NEE", "KO", "PG", "TSLA", "PLD"],
                "sector_tilts": {
                    "Information Technology": 0.5,
                    "Health Care": 0.2,
                    "Industrials": 0.3,
                    "Energy": 0.3,
                    "Real Estate": -0.5,
                    "Utilities": -0.4,
                    "Consumer Staples": -0.2,
                },
                "key_themes": [
                    "AI infrastructure build-out driving tech capex supercycle",
                    "Re-shoring benefiting domestic industrials",
                    "GLP-1 drug revolution reshaping health care landscape",
                    "Higher-for-longer rates pressuring rate-sensitive sectors",
                    "Energy sector free cash flow generation supports dividends",
                    "Consumer discretionary softening on exhausted excess savings",
                ],
                "risk_flags": [
                    "Fed pivot timing uncertainty",
                    "Commercial real estate credit stress",
                    "China growth deceleration",
                    "Geopolitical risk premium elevated",
                ],
                "executive_summary": (
                    "The investment environment is transitional: AI-driven technology "
                    "spending and resilient corporate earnings are offset by "
                    "higher-for-longer interest rates and deteriorating consumer credit. "
                    "The portfolio tilts toward secular growth (tech, health care) and "
                    "value cyclicals (industrials, energy) while underweighting "
                    "rate-sensitive sectors (real estate, utilities, consumer staples). "
                    "Conviction is moderate given macro uncertainty."
                ),
            }
        )


# ---------------------------------------------------------------------------
# LangGraph StateGraph builder
# ---------------------------------------------------------------------------


def build_agent_graph(config: AgentConfig) -> object:
    """Build and compile the LangGraph StateGraph for the agent pipeline.

    When ``langgraph`` is installed, returns a compiled LangGraph graph that
    runs macro and sector agents in parallel before the company and
    orchestrator nodes.

    When ``langgraph`` is not installed, returns a ``_SequentialGraph``
    fallback that runs the same four nodes sequentially with the same
    ``invoke(state)`` interface, so all downstream code is unaffected.

    Parameters
    ----------
    config : AgentConfig

    Returns
    -------
    object
        A compiled LangGraph graph or SequentialGraph, both with
        ``invoke(state: GraphState) -> GraphState`` interface.
    """
    from agents.company_agent import CompanyAgent
    from agents.macro_agent import MacroAgent
    from agents.sector_agent import SectorAgent

    macro_agent = MacroAgent(config)
    orch_agent = OrchestratorAgent(config)

    # ------------------------------------------------------------------
    # Node functions (shared by both LangGraph and fallback paths)
    # ------------------------------------------------------------------

    def macro_node(state: GraphState) -> dict:
        """Run MacroAgent and write MacroBrief to state."""
        try:
            brief = macro_agent.run(
                macro_data=state.get("macro_data", {}),
                as_of_date=state["as_of_date"],
            )
            return {"macro_brief": brief, "errors": []}
        except Exception as exc:
            logger.error(f"[Graph:macro_node] {exc}")
            return {"macro_brief": MacroBrief.neutral(), "errors": [str(exc)]}

    def sector_node(state: GraphState) -> dict:
        """Run SectorAgents for all sectors and write SectorBriefs to state."""
        sector_map = state.get("sector_map", {})
        macro_brief = state.get("macro_brief")
        as_of_date = state["as_of_date"]
        results: dict[str, SectorBrief] = {}
        errors: list[str] = []

        for sector, tickers in sector_map.items():
            agent = SectorAgent(config, sector=sector, tickers=tickers)
            try:
                results[sector] = agent.run(
                    macro_brief=macro_brief, as_of_date=as_of_date
                )
            except Exception as exc:
                logger.warning(f"[Graph:sector_node:{sector}] {exc}")
                results[sector] = SectorBrief.neutral(sector)
                errors.append(str(exc))

        return {"sector_briefs": results, "errors": errors}

    def company_node(state: GraphState) -> dict:
        """Run CompanyAgent for every ticker and write CompanyBriefs to state."""
        tickers = state.get("tickers", [])
        xbrl_data = state.get("xbrl_data", {})
        mda_data = state.get("mda_data", {})
        sector_briefs = state.get("sector_briefs", {})
        as_of_date = state["as_of_date"]
        results: dict[str, CompanyBrief] = {}
        errors: list[str] = []

        ticker_sector: dict[str, str] = {}
        for sector, tkrs in state.get("sector_map", {}).items():
            for t in tkrs:
                ticker_sector[t] = sector

        company_agent = CompanyAgent(config)
        for ticker in tickers:
            sector = ticker_sector.get(ticker, "")
            sb = sector_briefs.get(sector)
            try:
                results[ticker] = company_agent.run(
                    ticker=ticker,
                    xbrl_facts=xbrl_data.get(ticker, {}),
                    mda_text=mda_data.get(ticker, ""),
                    sector_brief=sb,
                    as_of_date=as_of_date,
                )
            except Exception as exc:
                logger.warning(f"[Graph:company_node:{ticker}] {exc}")
                results[ticker] = CompanyBrief.neutral(ticker)
                errors.append(str(exc))

        return {"company_briefs": results, "errors": errors}

    def orchestrator_node(state: GraphState) -> dict:
        """Run OrchestratorAgent and write the final MarketBrief to state."""
        try:
            brief = orch_agent.run(
                as_of_date=state["as_of_date"],
                tickers=state.get("tickers", []),
                macro_brief=state.get("macro_brief"),
                sector_briefs=state.get("sector_briefs", {}),
                company_briefs=state.get("company_briefs", {}),
            )
            return {"market_brief": brief, "errors": []}
        except Exception as exc:
            logger.error(f"[Graph:orchestrator_node] {exc}")
            return {
                "market_brief": MarketBrief.neutral(state["as_of_date"]),
                "errors": [str(exc)],
            }

    nodes = [macro_node, sector_node, company_node, orchestrator_node]

    # ------------------------------------------------------------------
    # LangGraph path (preferred — macro + sector run in parallel)
    # ------------------------------------------------------------------
    try:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(GraphState)
        graph.add_node("macro_node", macro_node)
        graph.add_node("sector_node", sector_node)
        graph.add_node("company_node", company_node)
        graph.add_node("orchestrator_node", orchestrator_node)

        # START fans out to macro and sector simultaneously
        graph.add_edge(START, "macro_node")
        # graph.add_edge(START, "sector_node")
        # # Both must finish before company node starts
        # graph.add_edge("macro_node", "company_node")
        graph.add_edge("macro_node", "sector_node")
        graph.add_edge("sector_node", "company_node")
        graph.add_edge("company_node", "orchestrator_node")
        graph.add_edge("orchestrator_node", END)

        logger.info(
            "[build_agent_graph] Using LangGraph StateGraph (parallel execution)."
        )
        return graph.compile()

    except ImportError:
        logger.info(
            "[build_agent_graph] langgraph not installed — using sequential fallback. "
            "Install for parallel macro+sector execution: pip install langgraph"
        )

    # ------------------------------------------------------------------
    # Sequential fallback (same interface, no LangGraph dependency)
    # ------------------------------------------------------------------
    return _SequentialGraph(
        macro_fn=macro_node,
        sector_fn=sector_node,
        company_fn=company_node,
        orchestrator_fn=orchestrator_node,
    )


class _SequentialGraph:
    """Sequential fallback that mirrors LangGraph's ``invoke`` interface.

    Runs macro → sector → company → orchestrator in order, merging each
    node's output dict back into the state before calling the next node.
    This preserves the exact same data-flow semantics as the LangGraph
    parallel graph, just without concurrent execution.
    """

    def __init__(
        self,
        macro_fn: object,
        sector_fn: object,
        company_fn: object,
        orchestrator_fn: object,
    ) -> None:
        self._nodes = [macro_fn, sector_fn, company_fn, orchestrator_fn]

    def invoke(self, state: GraphState) -> GraphState:
        """Run all nodes sequentially and return the final state."""
        current = dict(state)
        for node_fn in self._nodes:
            updates = node_fn(current)
            for key, val in updates.items():
                if key == "errors" and isinstance(val, list):
                    current["errors"] = list(current.get("errors") or []) + val
                else:
                    current[key] = val
        return current
