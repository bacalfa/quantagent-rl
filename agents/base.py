"""
agents/base.py
==============
BaseAgent — shared LLM call logic, JSON parsing, retry, and mock support.

Every specialist agent (Macro, Sector, Company, Orchestrator) inherits from
BaseAgent so that all LLM plumbing lives in one place:

  * Backend selection  — delegates to ``agents.llm.build_llm_client`` which
                         returns either a ``ClaudeClient`` or a
                         ``HuggingFaceLLMClient`` depending on
                         ``config.llm_backend``.
  * Web search client  — exposes ``self._anthropic_client`` (the raw
                         ``anthropic.Anthropic`` instance) when the Claude
                         backend is active so that ``tools.web_search`` can
                         invoke Anthropic's server-side search tool.  When the
                         HuggingFace backend is active this attribute is
                         ``None`` and ``web_search`` falls back to DuckDuckGo
                         automatically — no code change required in callers.
  * JSON parsing       — three-strategy extractor shared by all agents.
  * Mock mode          — ``call_llm`` bypasses all network calls and returns
                         the agent's deterministic stub response.

Separation of concerns
----------------------
``BaseLLMClient.complete(system, user)`` is the only method called by
``call_llm``.  All backend-specific details (Anthropic API kwargs, HF
generation parameters, retry logic, device management) are encapsulated in
``agents/llm.py``.
"""

import json
import logging
import re
from abc import ABC, abstractmethod

from agents.config import AgentConfig

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all QuantAgent-RL specialist agents.

    Parameters
    ----------
    config : AgentConfig
        Shared agent configuration.  Controls backend selection, mock mode,
        web search, temperature, and other runtime settings.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.cfg = config

        # _llm_client: unified interface for all LLM calls (Claude or HF).
        self._llm_client = None
        # _anthropic_client: raw anthropic.Anthropic instance, or None.
        #   Used exclusively by tools.web_search so it can invoke Anthropic's
        #   server-side search tool when on the Claude backend.
        #   When llm_backend='huggingface' this stays None and tools.web_search
        #   falls back to DuckDuckGo automatically.
        self._anthropic_client = None

        if not config.mock_mode:
            self._init_clients()

    # ------------------------------------------------------------------
    # Client initialization
    # ------------------------------------------------------------------

    def _init_clients(self) -> None:
        """Build the LLM client and extract the raw Anthropic client if applicable."""
        from agents.llm import ClaudeClient, build_llm_client

        self._llm_client = build_llm_client(self.cfg)

        # Expose the underlying anthropic.Anthropic object for web search.
        if isinstance(self._llm_client, ClaudeClient):
            self._anthropic_client = self._llm_client.raw_client

    # ------------------------------------------------------------------
    # Core LLM call
    # ------------------------------------------------------------------

    def call_llm(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """Dispatch a completion request to the active LLM backend.

        In mock mode, returns the agent's deterministic stub response
        immediately without touching any network or GPU resource.

        In live mode, delegates to ``self._llm_client.complete()``.  All
        backend-specific retry logic, prompt formatting, and generation
        parameters are handled inside the client implementation.

        Parameters
        ----------
        system_prompt : str
            Role / persona / output-format instructions for the model.
        user_message : str
            The actual task, including any pre-fetched context (FRED signals,
            web-search snippets, financial data).  For the HuggingFace backend
            this message should already contain all search results, since the
            model has no native tool-calling capability.

        Returns
        -------
        str
            Raw model output text.  May be JSON, prose, or a mix depending
            on the agent and how well the model followed the system prompt.
        """
        if self.cfg.mock_mode:
            return self._mock_response(system_prompt, user_message)

        if self._llm_client is None:
            self._init_clients()

        return self._llm_client.complete(system_prompt, user_message)

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    def parse_json_response(self, raw: str, fallback: dict) -> dict:
        """Extract and parse a JSON object from a raw LLM response.

        Tries three strategies in order, then returns ``fallback``:

        1. Direct ``json.loads`` on the full string (ideal — Claude and
           well-tuned HF models with temperature=0 usually produce clean JSON).
        2. Slice from the first ``{`` to the last ``}`` and parse that substring
           (handles leading prose or trailing explanations).
        3. Extract content between ```json ... ``` fences (some HF models wrap
           JSON in markdown fences despite being told not to).

        Parameters
        ----------
        raw : str
            Raw model output.
        fallback : dict
            Returned when all three strategies fail.

        Returns
        -------
        dict
        """
        # Strategy 1 — direct parse
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2 — first { ... last }
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass

        # Strategy 3 — ```json fences
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fence:
            try:
                return json.loads(fence.group(1))
            except json.JSONDecodeError:
                pass

        logger.warning(
            f"[{self.__class__.__name__}] Could not parse JSON from "
            f"response (length={len(raw)}); using fallback."
        )
        return fallback

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _mock_response(self, system_prompt: str, user_message: str) -> str:
        """Return a deterministic stub JSON string for mock mode.

        Must be overridden by each concrete agent subclass to return a
        valid JSON string matching that agent's output schema.
        """

    @abstractmethod
    def run(self, **kwargs) -> object:
        """Execute the agent and return its structured output schema object."""
