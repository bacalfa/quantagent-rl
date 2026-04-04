"""
agents
======
QuantAgent-RL multi-agent LLM module.

Four specialist agents — coordinated by a LangGraph StateGraph — analyze
macro signals, sector dynamics, and company fundamentals, then synthesize
their outputs into a structured ``MarketBrief`` that feeds the RL state vector.

Public API
----------
AgentPipeline       : end-to-end pipeline → AgentBundle per fold
AgentBundle         : dataclass holding all quarterly MarketBriefs + embeddings
AgentConfig         : master configuration dataclass

Schemas
-------
MarketBrief         : unified orchestrator output (RL state contribution)
MacroBrief          : macro-environment assessment
SectorBrief         : per-GICS-sector outlook
CompanyBrief        : per-stock fundamental assessment

Agent Classes
-------------
MacroAgent          : FRED signals + web-search synthesis
SectorAgent         : per-sector earnings + news synthesis
CompanyAgent        : EDGAR XBRL + MD&A analysis
OrchestratorAgent   : LangGraph orchestration + MarketBrief synthesis

Utilities
---------
MarketBriefEmbedder : GPU-accelerated sentence embedding for RL state
build_agent_graph   : compile the LangGraph StateGraph

Quick Start (mock mode — no API key required)
---------------------------------------------
>>> from agents import AgentPipeline, AgentConfig
>>> pipeline = AgentPipeline(AgentConfig(mock_mode=True))
>>> bundle = pipeline.run_fold(data_fold)       # WalkForwardFold from data module
>>> brief = bundle.get_brief("2023-09-30")      # MarketBrief for that quarter
>>> vec   = bundle.get_embedding("2023-09-30")  # np.ndarray (384,) for RL state
"""

from agents.company_agent import CompanyAgent
from agents.config import DEFAULT_MODEL, EMBEDDING_DIM, AgentConfig, HuggingFaceConfig
from agents.embedder import MarketBriefEmbedder
from agents.llm import (
    BaseLLMClient,
    ClaudeClient,
    HuggingFaceLLMClient,
    build_llm_client,
)
from agents.macro_agent import MacroAgent
from agents.orchestrator import GraphState, OrchestratorAgent, build_agent_graph
from agents.pipeline import AgentBundle, AgentPipeline
from agents.schemas import CompanyBrief, MacroBrief, MarketBrief, SectorBrief
from agents.sector_agent import SectorAgent

__all__ = [
    # Pipeline
    "AgentPipeline",
    "AgentBundle",
    # Config
    "AgentConfig",
    "HuggingFaceConfig",
    "DEFAULT_MODEL",
    "EMBEDDING_DIM",
    # LLM clients
    "BaseLLMClient",
    "ClaudeClient",
    "HuggingFaceLLMClient",
    "build_llm_client",
    # Schemas
    "MarketBrief",
    "MacroBrief",
    "SectorBrief",
    "CompanyBrief",
    # Agents
    "MacroAgent",
    "SectorAgent",
    "CompanyAgent",
    "OrchestratorAgent",
    # Graph
    "build_agent_graph",
    "GraphState",
    # Embedding
    "MarketBriefEmbedder",
]
