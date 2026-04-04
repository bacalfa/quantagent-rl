"""
agents/config.py
================
Configuration and prompt constants for the QuantAgent-RL agents module.

System prompts are stored here rather than inline so they can be versioned,
A/B tested, and swapped without touching agent logic.
"""

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
DEFAULT_MAX_TOKENS = 2048
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # output dimension of above model

DEFAULT_HF_MODEL = os.environ.get("HUGGINGFACE_MODEL", "microsoft/phi-4")


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

MACRO_SYSTEM_PROMPT = """\
You are a senior macro-economic analyst supporting a quantitative portfolio \
manager. Your task is to assess the current macro-economic environment using \
the provided quantitative indicators and recent news.
 
You must respond ONLY with a valid JSON object — no markdown fences, no prose \
before or after. The JSON must match this schema exactly:
 
{
  "rate_environment": "<one of: tightening | neutral | easing>",
  "inflation_regime": "<one of: high | elevated | moderate | low>",
  "recession_risk": <float between 0.0 and 1.0>,
  "yield_curve_signal": "<one of: inverted | flat | normal>",
  "credit_stress": <float between 0.0 and 1.0>,
  "overall_sentiment": <float between -1.0 (bearish) and 1.0 (bullish)>,
  "key_risks": [<up to 5 concise risk strings>],
  "tailwinds": [<up to 5 concise tailwind strings>],
  "analyst_summary": "<two to three sentence narrative summary>"
}
 
Base your assessment strictly on information available as of the as_of_date \
provided. Do not reference events after that date.
"""

SECTOR_SYSTEM_PROMPT = """\
You are a senior equity sector analyst. Your task is to assess the near-term \
outlook for a specific GICS sector based on recent earnings commentary, \
analyst coverage, and macro context.
 
You must respond ONLY with a valid JSON object — no markdown fences, no prose. \
The JSON must match this schema exactly:
 
{
  "sector": "<GICS sector name>",
  "momentum_score": <float between -1.0 (strong headwinds) and 1.0 (strong tailwinds)>,
  "earnings_revision_trend": "<one of: upgrades | neutral | downgrades>",
  "valuation_signal": "<one of: stretched | fair | cheap>",
  "key_themes": [<up to 5 concise theme strings>],
  "risks": [<up to 3 concise risk strings>],
  "analyst_summary": "<two to three sentence narrative summary>"
}
 
Base your assessment strictly on information available as of the as_of_date \
provided. Do not reference events after that date.
"""

COMPANY_SYSTEM_PROMPT = """\
You are a fundamental equity analyst. Your task is to assess a single stock \
using its most recent SEC filing financials (structured XBRL data) and the \
Management Discussion and Analysis (MD&A) section of its most recent quarterly \
or annual report.
 
You must respond ONLY with a valid JSON object — no markdown fences, no prose. \
The JSON must match this schema exactly:
 
{
  "ticker": "<ticker symbol>",
  "revenue_growth_trend": "<one of: accelerating | stable | decelerating | negative>",
  "margin_trend": "<one of: expanding | stable | compressing>",
  "balance_sheet_quality": "<one of: strong | adequate | stretched>",
  "earnings_quality": "<one of: high | medium | low>",
  "fundamental_score": <float between -1.0 (very negative) and 1.0 (very positive)>,
  "key_risks": [<up to 4 concise risk strings>],
  "key_catalysts": [<up to 4 concise catalyst strings>],
  "analyst_summary": "<two to three sentence narrative summary>"
}
 
Base your assessment strictly on the provided data. Do not fabricate financial \
figures. If data is insufficient, note that in the analyst_summary.
"""

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the chief investment strategist for a quantitative equity fund. You \
receive structured analyses from three specialist agents — a macro analyst, \
sector analysts, and company analysts — and must synthesize them into a unified \
market brief that will guide portfolio rebalancing decisions.
 
You must respond ONLY with a valid JSON object — no markdown fences, no prose. \
The JSON must match this schema exactly:
 
{
  "as_of_date": "<YYYY-MM-DD>",
  "macro_regime": "<one of: risk_on | risk_off | transitional>",
  "portfolio_stance": "<one of: aggressive | neutral | defensive>",
  "conviction_score": <float between 0.0 (low conviction) and 1.0 (high conviction)>,
  "top_overweights": [<up to 5 ticker strings with highest positive signals>],
  "top_underweights": [<up to 5 ticker strings with highest negative signals>],
  "sector_tilts": {<sector_name>: <float between -1.0 and 1.0>},
  "key_themes": [<up to 6 concise investment theme strings>],
  "risk_flags": [<up to 4 portfolio-level risk strings>],
  "executive_summary": "<three to five sentence investment narrative>"
}
 
Reconcile any contradictions between the sub-analyses using sound investment \
judgment. Weight macro signals more heavily during high-uncertainty regimes \
(high VIX, inverted yield curve, elevated HY spreads).
"""


# ---------------------------------------------------------------------------
# HuggingFaceConfig
# ---------------------------------------------------------------------------


@dataclass
class HuggingFaceConfig:
    """Configuration for the HuggingFace local LLM backend.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Examples:

        * ``'microsoft/phi-4'``          (14 B, full capability, ~28 GB BF16)
        * ``'microsoft/phi-4-mini'``     (3.8 B, fast, ~8 GB BF16)
        * ``'microsoft/phi-3.5-mini-instruct'`` (3.8 B, fits 8 GB VRAM)
        * ``'mistralai/Mistral-7B-Instruct-v0.3'``
        * ``'meta-llama/Llama-3.1-8B-Instruct'``

    device : str | None
        Override device placement (e.g. ``'cuda:0'``, ``'cpu'``).
        None = ``device_map='auto'`` — model is auto-sharded across all GPUs.
    load_in_4bit : bool
        Enable NF4 quantization via bitsandbytes. Reduces Phi-4 VRAM from
        ~28 GB to ~7 GB. Requires: pip install bitsandbytes.
    load_in_8bit : bool
        Enable 8-bit quantization. Mutually exclusive with load_in_4bit.
    torch_dtype : str
        Model weight precision: ``'bfloat16'`` (recommended for Ampere+),
        ``'float16'`` (Turing / older GPUs), ``'float32'`` (CPU), or
        ``'auto'`` (selected based on GPU capability at load time).
    max_new_tokens : int
        Maximum tokens to generate per agent call.
    trust_remote_code : bool
        Passed to ``from_pretrained``. Required by Phi-4 and some other models.
    cache_dir : str | None
        Local directory for caching model weights. None = HuggingFace default
        (typically ``~/.cache/huggingface/hub``).
    attn_implementation : str | None
        Attention backend. Use ``'flash_attention_2'`` for 2–4× speedup on
        Ampere and later GPUs. Requires: pip install flash-attn.
        None = model default (scaled dot-product attention).
    """

    model_name: str = DEFAULT_HF_MODEL
    device: str | None = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: str = "auto"
    max_new_tokens: int = 2048
    trust_remote_code: bool = True
    cache_dir: str | None = None
    attn_implementation: str | None = None

    def __post_init__(self) -> None:
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are mutually exclusive.")
        valid_dtypes = ("bfloat16", "float16", "float32", "auto")
        if self.torch_dtype not in valid_dtypes:
            raise ValueError(
                f"torch_dtype must be one of {valid_dtypes}, got '{self.torch_dtype}'."
            )


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Master configuration for the QuantAgent-RL agents module.

    Parameters
    ----------
    anthropic_api_key : str
        Anthropic API key. Reads from ANTHROPIC_API_KEY env var if not set.
    model : str
        Claude model identifier to use for all agents.
    max_tokens : int
        Maximum token budget per agent call.
    mock_mode : bool
        If True, all LLM calls are bypassed and deterministic stub responses
        are returned. Useful for testing, CI, and offline demos.
    enable_web_search : bool
        If True, macro and sector agents use the Anthropic web-search tool.
        Disable for pure backtesting runs to avoid look-ahead bias.
    embedding_model : str
        Sentence-transformers model name used for MarketBrief embedding.
    embedding_device : str | None
        Torch device for sentence-transformer inference ('cuda', 'cpu', None).
        None = auto-detect (CUDA if available, else CPU).
    max_mda_chars : int
        Maximum characters of MD&A text passed to the company agent.
    max_search_results : int
        Number of web-search results to include in agent context.
    temperature : float
        LLM sampling temperature. 0.0 for reproducibility.
    sector_agents_parallel : bool
        If True, sector agents for different GICS sectors run concurrently.
    request_timeout : float
        HTTP timeout (seconds) for Anthropic API calls.
    """

    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    mock_mode: bool = False
    llm_backend: str = "claude"
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    enable_web_search: bool = True
    embedding_model: str = EMBEDDING_MODEL
    embedding_device: str | None = None
    max_mda_chars: int = 30_000
    max_search_results: int = 5
    temperature: float = 0.0
    sector_agents_parallel: bool = True
    request_timeout: float = 60.0

    def validate(self) -> None:
        """Raise ValueError for misconfigured settings."""
        valid_backends = ("claude", "huggingface")
        if self.llm_backend not in valid_backends:
            raise ValueError(
                f"llm_backend must be one of {valid_backends}, "
                f"got '{self.llm_backend}'."
            )
        if (
            not self.mock_mode
            and self.llm_backend == "claude"
            and not self.anthropic_api_key
        ):
            raise ValueError(
                "ANTHROPIC_API_KEY is not set and llm_backend='claude'. "
                "Either set the environment variable, switch to "
                "llm_backend='huggingface', or pass mock_mode=True."
            )
        if self.temperature < 0.0 or self.temperature > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0.")
        if not self.mock_mode and self.llm_backend == "huggingface":
            self.huggingface.__post_init__()  # validate HF config
