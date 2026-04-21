"""
agents/llm.py
=============
Unified LLM client abstraction for QuantAgent-RL.

Two backends are supported and share the same ``BaseLLMClient`` interface so
that all agent logic is backend-agnostic:

  ClaudeClient
      Wraps the Anthropic Python SDK.  Supports native tool-calling (including
      Anthropic's server-side web-search tool).  Requires an ``ANTHROPIC_API_KEY``
      environment variable.

  HuggingFaceLLMClient
      Runs any HuggingFace causal-LM locally via ``transformers``.  The default
      model is ``microsoft/phi-4`` (14 B parameters); any instruction-tuned model
      that exposes a ``chat_template`` in its tokenizer will work.  GPU
      acceleration is applied automatically when CUDA is available.  Optional
      4-bit / 8-bit quantization via ``bitsandbytes`` reduces VRAM requirements.

GPU Acceleration
----------------
``HuggingFaceLLMClient`` uses ``device_map="auto"`` so the model is sharded
across all available GPUs automatically.  With ``load_in_4bit=True``, Phi-4
fits comfortably in ~7 GB of VRAM (one RTX 3090 / A6000 or similar).
``torch_dtype="bfloat16"`` is recommended for Ampere and later architectures;
``"float16"`` for Turing (T4, RTX 20-series).

Web Search
----------
``ClaudeClient`` can attach Anthropic's native ``web_search_20250305`` tool to
every API call, giving Claude the ability to issue real-time searches during
generation.

``HuggingFaceLLMClient`` has no native tool-calling capability.  Web search
results are instead retrieved externally (via DuckDuckGo in ``tools.web_search``)
by the caller and injected into the prompt as plain text before the LLM is
called.  The ``complete()`` method itself is search-unaware.

Factory
-------
Use ``build_llm_client(config)`` rather than instantiating clients directly.
It reads ``config.llm_backend`` and returns the appropriate concrete client.

Usage
-----
>>> from agents.llm import build_llm_client
>>> from agents.config import AgentConfig
>>> client = build_llm_client(AgentConfig(llm_backend="huggingface"))
>>> text = client.complete("You are a finance expert.", "What is the yield curve?")
"""

import bitsandbytes as bnb

original_new = bnb.nn.Int8Params.__new__


def patched_new(cls, data=None, requires_grad=False, has_fp16_weights=False, **kwargs):
    kwargs.pop("_is_hf_initialized", None)
    return original_new(cls, data, requires_grad, has_fp16_weights, **kwargs)


bnb.nn.Int8Params.__new__ = patched_new

import logging
import threading
import time
from abc import ABC, abstractmethod

from agents.config import AgentConfig

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds; multiplied by attempt number


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseLLMClient(ABC):
    """Common interface for all LLM backends.

    Subclasses implement ``complete()`` which takes a system prompt and a
    user message and returns the model's plain-text response.

    Parameters
    ----------
    temperature : float
        Sampling temperature.  0.0 = greedy / deterministic.
    max_tokens : int
        Maximum number of tokens to generate.
    """

    def __init__(self, temperature: float = 0.0, max_tokens: int = 2048) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> str:
        """Generate a completion given a system prompt and user message.

        Parameters
        ----------
        system_prompt : str
            Role / persona / output-format instructions for the model.
        user_message : str
            The actual task or question, including any pre-fetched context
            (search snippets, financial data, etc.).

        Returns
        -------
        str
            Raw model output (may be JSON, prose, or a mix).
        """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable identifier for this backend (e.g. ``'claude'``)."""


# ---------------------------------------------------------------------------
# Claude backend
# ---------------------------------------------------------------------------


class ClaudeClient(BaseLLMClient):
    """LLM client that wraps the Anthropic Python SDK.

    Parameters
    ----------
    api_key : str
        Anthropic API key.
    model : str
        Claude model identifier (e.g. ``'claude-sonnet-4-6'``).
    temperature : float
    max_tokens : int
    enable_web_search : bool
        When True, Anthropic's ``web_search_20250305`` tool is attached to
        every API call so Claude can issue searches during generation.
        This is *in addition to* any search results already injected by the
        caller into ``user_message``.
    request_timeout : float
        HTTP timeout (seconds) for each API call.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        enable_web_search: bool = False,
        request_timeout: float = 60.0,
    ) -> None:
        super().__init__(temperature=temperature, max_tokens=max_tokens)
        self.model = model
        self.enable_web_search = enable_web_search
        self.request_timeout = request_timeout
        self._client = self._build_anthropic_client(api_key)

    @staticmethod
    def _build_anthropic_client(api_key: str) -> object:
        try:
            import anthropic

            return anthropic.Anthropic(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "anthropic library is required for the Claude backend: "
                "pip install anthropic"
            ) from exc

    @property
    def backend_name(self) -> str:
        return "claude"

    @property
    def raw_client(self) -> object:
        """Expose the underlying ``anthropic.Anthropic`` instance.

        Needed by ``tools.web_search`` so it can invoke Anthropic's
        server-side search tool on behalf of the caller.
        """
        return self._client

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Call the Claude API with optional native web-search tool.

        Retries up to ``_MAX_RETRIES`` times on transient errors with
        linear back-off.
        """
        tools: list[dict] = []
        if self.enable_web_search:
            tools.append({"type": "web_search_20250305", "name": "web_search"})

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.messages.create(**kwargs)
                return self._extract_text(response)
            except Exception as exc:
                if attempt == _MAX_RETRIES - 1:
                    logger.error(
                        f"[ClaudeClient] Call failed after {_MAX_RETRIES} retries: {exc}"
                    )
                    raise
                logger.warning(f"[ClaudeClient] Retry {attempt + 1} after: {exc}")
                time.sleep(_RETRY_DELAY * (attempt + 1))

        return ""  # unreachable

    @staticmethod
    def _extract_text(response: object) -> str:
        """Collect all text blocks from an Anthropic response object."""
        parts = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", "") == "text":
                parts.append(block.text)
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------


class HuggingFaceLLMClient(BaseLLMClient):
    """LLM client that runs a local model via HuggingFace ``transformers``.

    The model is loaded lazily on the first ``complete()`` call so that
    importing this module does not trigger a GPU memory allocation.

    GPU Acceleration
    ----------------
    ``device_map="auto"`` is passed to ``from_pretrained`` so that the model
    is automatically distributed across all available CUDA devices.
    Single-GPU users get the full model on device 0; multi-GPU users benefit
    from automatic tensor-parallel sharding.

    Memory Reduction
    ----------------
    Passing ``load_in_4bit=True`` enables NF4 quantization via
    ``bitsandbytes``.  For Phi-4 (14 B parameters) this reduces peak VRAM
    from ~28 GB (BF16) to ~7 GB — fitting on a single RTX 3090 or A6000.

    Chat Template
    -------------
    The model's ``tokenizer.apply_chat_template`` is used to format the
    system-prompt + user-message conversation into the correct input string
    for the specific model.  Any instruction-tuned model that ships a
    ``tokenizer_config.json`` with a ``chat_template`` field is supported
    (Phi-4, Phi-3.5, Mistral, LLaMA-3, Qwen-2.5, etc.).

    Web Search
    ----------
    HuggingFace models have no native tool-calling.  Callers are expected to
    retrieve search results externally (e.g. via DuckDuckGo) and inject them
    into ``user_message`` as plain text before calling ``complete()``.  This
    client is completely search-unaware.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Default: ``microsoft/phi-4``.
    device : str | None
        Override device string (e.g. ``'cuda:0'``, ``'cpu'``).
        ``None`` → ``device_map="auto"`` for automatic multi-GPU sharding.
    load_in_4bit : bool
        Enable 4-bit NF4 quantization via ``bitsandbytes``.  Requires
        ``bitsandbytes`` to be installed.
    load_in_8bit : bool
        Enable 8-bit quantization.  Mutually exclusive with ``load_in_4bit``.
    torch_dtype : str
        Weight dtype for full-precision loading.  One of ``"bfloat16"``,
        ``"float16"``, ``"float32"``, or ``"auto"`` (model default).
    temperature : float
        Sampling temperature.  0.0 uses greedy decoding (``do_sample=False``).
    max_new_tokens : int
        Maximum tokens to generate per call.
    trust_remote_code : bool
        Passed to ``from_pretrained``.  Required by some models (e.g. Phi-4).
    cache_dir : str | None
        Local directory for caching downloaded model weights.
    attn_implementation : str | None
        Attention backend.  ``"flash_attention_2"`` gives a 2–4× speedup on
        Ampere (A100, RTX 30/40-series) and later GPUs but requires the
        ``flash-attn`` package.  ``None`` uses the model's default.
    llm_int8_enable_fp32_cpu_offload: bool
        This allows the library to keep offloaded modules in their original
        32-bit (FP32) precision, as the bitsandbytes 8-bit kernels are currently
        optimized for GPU execution only.
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-4",
        device: str | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: str = "auto",
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
        trust_remote_code: bool = True,
        cache_dir: str | None = None,
        attn_implementation: str | None = None,
        llm_int8_enable_fp32_cpu_offload: bool = False,
    ) -> None:
        super().__init__(temperature=temperature, max_tokens=max_new_tokens)
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.torch_dtype_str = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        self.attn_implementation = attn_implementation
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload

        # Populated on first call
        self._model = None
        self._tokenizer = None
        self._device_str: str | None = None

    @property
    def backend_name(self) -> str:
        return "huggingface"

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load model and tokenizer on the first call (lazy initialization).

        Lazy loading avoids a lengthy GPU allocation at import time and
        allows the client to be constructed in the main process and used
        in a subprocess without double-loading the weights.
        """
        if self._model is not None:
            return

        logger.info(
            f"[HuggingFaceLLMClient] Loading '{self.model_name}' … "
            f"(4bit={self.load_in_4bit}, 8bit={self.load_in_8bit}, "
            f"dtype={self.torch_dtype_str})"
        )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for the HuggingFace backend: "
                "pip install transformers torch"
            ) from exc

        # ── Resolve torch dtype ──────────────────────────────────────
        _dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if self.torch_dtype_str in _dtype_map:
            torch_dtype = _dtype_map[self.torch_dtype_str]
        elif self.torch_dtype_str == "auto":
            # Prefer bfloat16 on Ampere+ for better numerical stability;
            # fall back to float16 on older GPUs; CPU always uses float32.
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(0)
                torch_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
            else:
                torch_dtype = torch.float32
        else:
            raise ValueError(
                f"Unknown torch_dtype '{self.torch_dtype_str}'. "
                "Use 'bfloat16', 'float16', 'float32', or 'auto'."
            )

        # ── Quantization config (optional) ───────────────────────────
        quantization_config = None
        if self.load_in_4bit or self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    "bitsandbytes is required for quantized loading: "
                    "pip install bitsandbytes"
                ) from exc
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,  # nested quantization
                    bnb_4bit_quant_type="nf4",  # NormalFloat4
                    llm_int8_enable_fp32_cpu_offload=self.llm_int8_enable_fp32_cpu_offload,
                )
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=self.llm_int8_enable_fp32_cpu_offload,
                )

        # ── Model kwargs ─────────────────────────────────────────────
        model_kwargs: dict = {
            "trust_remote_code": self.trust_remote_code,
            "cache_dir": self.cache_dir,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            # bitsandbytes handles device placement internally
            model_kwargs["device_map"] = "auto"
        elif self.device is not None:
            # Explicit device override — load on that device
            model_kwargs["torch_dtype"] = torch_dtype
            model_kwargs["device_map"] = self.device
        else:
            # Auto-shard across all GPUs (or CPU if none available)
            model_kwargs["torch_dtype"] = torch_dtype
            model_kwargs["device_map"] = "auto"

        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        # ── Load tokenizer ───────────────────────────────────────────
        tokenizer_kwargs: dict = {
            "trust_remote_code": self.trust_remote_code,
            "cache_dir": self.cache_dir,
        }
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, **tokenizer_kwargs
        )
        # Ensure pad token is set (some tokenizers omit it)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # ── Load model ───────────────────────────────────────────────
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )
        self._model.eval()

        # Determine the device string for logging
        if hasattr(self._model, "device"):
            self._device_str = str(self._model.device)
        elif torch.cuda.is_available():
            self._device_str = "cuda"
        else:
            self._device_str = "cpu"

        logger.info(
            f"[HuggingFaceLLMClient] '{self.model_name}' loaded on {self._device_str}."
        )

    # ------------------------------------------------------------------
    # complete()
    # ------------------------------------------------------------------

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Generate a response using the local HuggingFace model.

        The system prompt and user message are formatted using the
        tokenizer's built-in chat template before being tokenized and
        passed to the model.  If the model does not support a system role
        in its template, the system prompt is prepended to the user message
        as plain text.

        Parameters
        ----------
        system_prompt : str
            Role / persona / output-format instructions.
        user_message : str
            The task or question, with any pre-fetched context already
            incorporated (search snippets, financial data, etc.).

        Returns
        -------
        str
            The newly generated tokens only — the input prompt is stripped
            from the output before returning.
        """
        self._ensure_loaded()

        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Apply the model's chat template to produce the input string
        try:
            input_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback for tokenizers that do not support a system role:
            # merge system prompt into the user turn.
            logger.debug(
                "[HuggingFaceLLMClient] Chat template does not support 'system' "
                "role — merging into user message."
            )
            merged = f"{system_prompt}\n\n{user_message}"
            input_text = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": merged}],
                tokenize=False,
                add_generation_prompt=True,
            )

        # Tokenize and move to the model's device
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,  # cap input length to avoid OOM on long prompts
        )
        # Move each tensor to the primary model device
        target_device = next(self._model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Generation kwargs
        gen_kwargs: dict = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if self.temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.9
            gen_kwargs["top_k"] = 50
        else:
            gen_kwargs["do_sample"] = False  # greedy decoding

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # Slice off the input tokens — return only the generated portion
        new_tokens = output_ids[0][input_length:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release GPU memory by deleting the model and tokenizer.

        Call this between walk-forward folds if memory is tight and the
        model needs to be reloaded for the next fold's fresh inference.
        """
        import torch

        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"[HuggingFaceLLMClient] Model '{self.model_name}' unloaded.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LLM client registry — model weights are loaded exactly once per process
# ---------------------------------------------------------------------------

_CLIENT_REGISTRY: dict[tuple, BaseLLMClient] = {}
_REGISTRY_LOCK = threading.Lock()


def _client_cache_key(config: AgentConfig) -> tuple:
    """Return a hashable key that identifies a unique client configuration.

    Two ``AgentConfig`` objects that map to the same key will share a single
    ``BaseLLMClient`` instance, so HuggingFace model weights are loaded only
    once per process regardless of how many agent objects are instantiated.
    """
    backend = getattr(config, "llm_backend", "claude").lower()
    if backend == "claude":
        return (
            "claude",
            getattr(config, "anthropic_api_key", ""),
            getattr(config, "model", ""),
            getattr(config, "temperature", 0.0),
            getattr(config, "max_tokens", 2048),
            getattr(config, "enable_web_search", False),
        )
    if backend == "huggingface":
        hf_cfg = getattr(config, "huggingface", None)
        if hf_cfg is None:
            return ("huggingface",)
        return (
            "huggingface",
            getattr(hf_cfg, "model_name", ""),
            getattr(hf_cfg, "load_in_4bit", False),
            getattr(hf_cfg, "load_in_8bit", False),
            getattr(hf_cfg, "torch_dtype", "auto"),
            getattr(config, "temperature", 0.0),
            getattr(hf_cfg, "max_new_tokens", 2048),
        )
    return (backend,)


def build_llm_client(config: AgentConfig) -> BaseLLMClient:
    """Return a shared ``BaseLLMClient`` for the given configuration.

    The first call for a given configuration creates and caches the client;
    subsequent calls with an equivalent configuration return the cached
    instance.  This ensures that HuggingFace model weights are loaded into
    GPU memory exactly once per process, regardless of how many agent
    objects are created.

    Supported backends (``config.llm_backend``):

    * ``"claude"``       — ``ClaudeClient`` (Anthropic API)
    * ``"huggingface"``  — ``HuggingFaceLLMClient`` (local model)

    Parameters
    ----------
    config : AgentConfig

    Returns
    -------
    BaseLLMClient
    """
    key = _client_cache_key(config)
    with _REGISTRY_LOCK:
        if key not in _CLIENT_REGISTRY:
            _CLIENT_REGISTRY[key] = _create_llm_client(config)
        return _CLIENT_REGISTRY[key]


def _create_llm_client(config: AgentConfig) -> BaseLLMClient:
    """Internal factory — create a brand-new client, bypassing the registry.

    Raises
    ------
    ValueError
        If ``config.llm_backend`` is not a recognized value.
    """
    backend = getattr(config, "llm_backend", "claude").lower()

    if backend == "claude":
        return ClaudeClient(
            api_key=config.anthropic_api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            enable_web_search=config.enable_web_search,
            request_timeout=config.request_timeout,
        )

    if backend == "huggingface":
        hf_cfg = getattr(config, "huggingface", None)
        if hf_cfg is None:
            raise ValueError(
                "AgentConfig.huggingface must be set when llm_backend='huggingface'. "
                "Example: AgentConfig(llm_backend='huggingface', "
                "huggingface=HuggingFaceConfig())"
            )
        return HuggingFaceLLMClient(
            model_name=hf_cfg.model_name,
            device=hf_cfg.device,
            load_in_4bit=hf_cfg.load_in_4bit,
            load_in_8bit=hf_cfg.load_in_8bit,
            torch_dtype=hf_cfg.torch_dtype,
            temperature=config.temperature,
            max_new_tokens=hf_cfg.max_new_tokens,
            trust_remote_code=hf_cfg.trust_remote_code,
            cache_dir=hf_cfg.cache_dir,
            attn_implementation=hf_cfg.attn_implementation,
            llm_int8_enable_fp32_cpu_offload=hf_cfg.llm_int8_enable_fp32_cpu_offload,
        )

    raise ValueError(
        f"Unknown llm_backend '{backend}'. Supported values: 'claude', 'huggingface'."
    )
