# `agents` — QuantAgent-RL Multi-Agent LLM Module

The `agents` module implements a **four-agent LLM pipeline** that reads macro signals, sector trends, and company fundamentals at each quarter-end rebalancing date and synthesizes them into a structured `MarketBrief`. The brief is embedded into a dense numeric vector that becomes part of the RL state — giving the reinforcement-learning agent access to *qualitative* investment context alongside the quantitative features produced by the `data` and `forecasting` modules.

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Quick Start](#2-quick-start)
3. [Architecture](#3-architecture)
4. [Configuration (`config.py`)](#4-configuration-configpy)
5. [LLM Abstraction Layer (`llm.py`)](#5-llm-abstraction-layer-llmpy)
   5.1 [Claude Backend](#51-claude-backend)
   5.2 [HuggingFace Backend](#52-huggingface-backend)
   5.3 [Singleton Client Registry](#53-singleton-client-registry)
6. [Structured Output Schemas (`schemas.py`)](#6-structured-output-schemas-schemaspy)
7. [Specialist Agents](#7-specialist-agents)
   7.1 [MacroAgent](#71-macroagent)
   7.2 [SectorAgent](#72-sectoragent)
   7.3 [CompanyAgent](#73-companyagent)
   7.4 [OrchestratorAgent](#74-orchestratoragent)
8. [LangGraph Orchestration (`orchestrator.py`)](#8-langgraph-orchestration-orchestratorpy)
   8.1 [Graph Topology](#81-graph-topology)
   8.2 [Intra-node Parallel Execution](#82-intra-node-parallel-execution)
   8.3 [No-LangGraph Fallback](#83-no-langgraph-fallback)
9. [MarketBrief Embedding (`embedder.py`)](#9-marketbrief-embedding-embedderpy)
10. [Web Search and Tools (`tools.py`)](#10-web-search-and-tools-toolspy)
11. [Pipeline Orchestration (`pipeline.py`)](#11-pipeline-orchestration-pipelinepy)
12. [Walk-Forward Safety and Backtesting](#12-walk-forward-safety-and-backtesting)
13. [Mock Mode](#13-mock-mode)
14. [Environment Variables](#14-environment-variables)
15. [References](#15-references)

---

## 1. Module Overview

```
agents/
├── __init__.py        Public API — all exports
├── config.py          AgentConfig, HuggingFaceConfig, system prompts
├── schemas.py         MacroBrief, SectorBrief, CompanyBrief, MarketBrief
├── base.py            BaseAgent — shared LLM call, JSON parsing, mock support
├── llm.py             ClaudeClient, HuggingFaceLLMClient, singleton registry
├── macro_agent.py     MacroAgent — FRED signals + web search
├── sector_agent.py    SectorAgent — per-GICS sector earnings + news
├── company_agent.py   CompanyAgent — EDGAR XBRL financials + MD&A
├── orchestrator.py    OrchestratorAgent + LangGraph StateGraph builder
├── embedder.py        MarketBriefEmbedder — sentence-transformer encoding
├── tools.py           web_search, format_macro_for_llm, format_financials_for_llm
└── pipeline.py        AgentPipeline, AgentBundle
```

**Design principles:**

| Principle | Implementation |
|---|---|
| Backend agnosticism | All agents call `BaseLLMClient.complete()` — Claude or HuggingFace is a configuration switch |
| Walk-forward safety | All agent inputs are sliced to `as_of_date`; web searches are date-bounded |
| Graceful degradation | Every schema has a `.neutral()` fallback; the orchestrator has a heuristic fallback; the LangGraph graph has a sequential fallback |
| Reproducibility | `temperature=0.0` by default; mock mode returns deterministic stubs |
| Cost efficiency | Singleton LLM client registry prevents reloading HuggingFace model weights; MarketBrief JSON is cached to disk |

---

## 2. Quick Start

```python
from agents import AgentPipeline, AgentConfig

# Zero cost: mock mode bypasses all API calls
pipeline = AgentPipeline(AgentConfig(mock_mode=True))
bundle = pipeline.run_fold(data_fold)            # WalkForwardFold from data module

# Inspect a specific quarter-end brief
brief = bundle.get_brief("2023-09-30")
print(brief.macro_regime)                        # 'risk_on' | 'risk_off' | 'transitional'
print(brief.portfolio_stance)                    # 'aggressive' | 'neutral' | 'defensive'
print(brief.top_overweights)                     # ['NVDA', 'MSFT', ...]
print(brief.sector_tilts)                        # {'Information Technology': 0.5, ...}

# Get the RL state vector contribution (384-dimensional embedding)
vec = bundle.get_embedding("2023-09-30")         # np.ndarray (384,)

# Live mode: Claude with web search
live_pipeline = AgentPipeline(AgentConfig(
    llm_backend="claude",
    enable_web_search=True,
))
```

---

## 3. Architecture

The module follows a **hierarchical information flow**: macro context is established first, then sector-level analysis is conditioned on it, then company-level analysis is conditioned on both macro and sector context. The orchestrator synthesizes all three into a single `MarketBrief`.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         AgentPipeline.run_fold()                         │
│                                                                          │
│  For each quarter-end date in the fold:                                  │
│                                                                          │
│    Input data (date-bounded)                                             │
│    ├── macro_data   : FRED signal snapshot (dict[str, float])            │
│    ├── xbrl_data    : EDGAR XBRL financials per ticker                   │
│    ├── mda_data     : SEC MD&A text per ticker                           │
│    └── sector_map   : GICS sector → [tickers] mapping                   │
│                │                                                         │
│                ▼                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐   │
│    │              LangGraph StateGraph                               │   │
│    │                                                                 │   │
│    │   [macro_node]                                                   │   │
│    │        │   (macro_brief now available)                          │   │
│    │   [sector_node] (one per GICS sector, parallel within node)     │   │
│    │        │   (sector_briefs now available)                        │   │
│    │   [company_node] (one per ticker, parallel within node)         │   │
│    │                            │                                   │   │
│    │                   [orchestrator_node]                           │   │
│    │                            │                                   │   │
│    └────────────────────────────┼───────────────────────────────────┘   │
│                                 │                                        │
│                                 ▼                                        │
│                           MarketBrief                                    │
│                                 │                                        │
│                                 ▼                                        │
│                      MarketBriefEmbedder                                 │
│                                 │                                        │
│                                 ▼                                        │
│                      np.ndarray (384,)   ──► RL state vector             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Configuration (`config.py`)

All runtime behavior is controlled through two dataclasses.

### `AgentConfig` — master configuration

| Parameter | Default | Description |
|---|---|---|
| `llm_backend` | `"claude"` | `'claude'` = Anthropic API, `'huggingface'` = local model |
| `mock_mode` | `False` | Bypass all LLM/API calls; return deterministic stubs |
| `model` | `"claude-sonnet-4-6"` | Claude model ID (overridden by `ANTHROPIC_MODEL` env var) |
| `max_tokens` | `2048` | Token budget per agent call |
| `temperature` | `0.0` | Sampling temperature; 0 = greedy / deterministic |
| `enable_web_search` | `True` | Augment macro/sector agents with live web search |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Sentence-transformers model for embedding |
| `embedding_device` | `None` | `None` = auto-detect CUDA; `'cpu'` to force CPU |
| `max_mda_chars` | `4000` | Character limit for MD&A text passed to company agent |
| `max_search_results` | `5` | Web-search result snippets per query |
| `sector_agents_parallel` | `True` | Run GICS sector agents concurrently |
| `request_timeout` | `60.0` | Anthropic API HTTP timeout (seconds) |

### `HuggingFaceConfig` — local model configuration

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `"microsoft/phi-4"` | HuggingFace model ID |
| `load_in_4bit` | `False` | 4-bit NF4 quantization via bitsandbytes (~7 GB VRAM for Phi-4) |
| `load_in_8bit` | `False` | 8-bit quantization (mutually exclusive with 4-bit) |
| `torch_dtype` | `"auto"` | `'bfloat16'` (Ampere+), `'float16'` (Turing), `'float32'` (CPU) |
| `device` | `None` | `None` = `device_map='auto'` (auto-shard across all GPUs) |
| `attn_implementation` | `None` | `'flash_attention_2'` for 2–4× speedup on Ampere+ GPUs |
| `trust_remote_code` | `True` | Required by Phi-4 and some other models |
| `max_new_tokens` | `2048` | Generation token limit |

**Recommended HuggingFace models:**

| Model | Parameters | VRAM (BF16) | VRAM (4-bit) | Notes |
|---|---|---|---|---|
| `microsoft/phi-4` | 14 B | ~28 GB | ~7 GB | Best quality |
| `microsoft/phi-4-mini` | 3.8 B | ~8 GB | ~3 GB | Fast, good for development |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7 B | ~14 GB | ~5 GB | Widely supported |
| `meta-llama/Llama-3.1-8B-Instruct` | 8 B | ~16 GB | ~5 GB | Strong instruction following |

---

## 5. LLM Abstraction Layer (`llm.py`)

All agents communicate with LLMs exclusively through the `BaseLLMClient` interface. This makes every agent completely backend-agnostic — swapping from Claude to a local model requires only a configuration change.

```
BaseLLMClient (abstract)
    │
    ├── ClaudeClient          ← Anthropic API, optional native web search
    └── HuggingFaceLLMClient  ← local transformers, GPU auto-shard
```

### 5.1 Claude Backend

`ClaudeClient` wraps the `anthropic` Python SDK with:

- **Retry logic:** up to 3 attempts with linear back-off (2s, 4s, 6s) on transient errors
- **Native web search:** when `enable_web_search=True`, the `web_search_20250305` tool is attached to every API call, allowing Claude to issue real-time searches *during* generation
- **Text extraction:** handles multi-block responses (text + tool-result blocks)

```python
# Under the hood for each agent call:
response = client.messages.create(
    model="claude-sonnet-4-6",
    system=MACRO_SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_message}],
    tools=[{"type": "web_search_20250305", "name": "web_search"}],  # if enabled
    temperature=0.0,
    max_tokens=2048,
)
```

### 5.2 HuggingFace Backend

`HuggingFaceLLMClient` runs any instruction-tuned model locally via `transformers`:

- **Lazy loading:** the model is loaded on the **first `complete()` call**, not at import or `__init__`, avoiding GPU memory allocation until needed
- **Chat template:** `tokenizer.apply_chat_template` formats the system + user message correctly for each model's expected input format
- **Auto-sharding:** `device_map="auto"` distributes model layers across all available GPUs
- **Web search:** no native tool calling — callers inject search results directly into `user_message` as plain text

```
First complete() call:
  AutoTokenizer.from_pretrained(model_name, ...)
  AutoModelForCausalLM.from_pretrained(model_name,
      device_map="auto",                 ← auto multi-GPU shard
      torch_dtype=torch.bfloat16,        ← half-precision
      load_in_4bit=True,                 ← optional NF4 quantization
      attn_implementation="flash_attention_2"  ← optional FlashAttn
  )

Subsequent calls: model already loaded, no re-initialization
```

### 5.3 Singleton Client Registry

A process-wide registry prevents multiple agents from re-loading the same HuggingFace model weights when the pipeline is initialized:

```python
_CLIENT_REGISTRY: dict[tuple, BaseLLMClient] = {}
_REGISTRY_LOCK = threading.Lock()

def build_llm_client(config: AgentConfig) -> BaseLLMClient:
    key = _client_cache_key(config)           # (backend, model, dtype, ...)
    with _REGISTRY_LOCK:
        if key not in _CLIENT_REGISTRY:
            _CLIENT_REGISTRY[key] = _create_llm_client(config)
        return _CLIENT_REGISTRY[key]
```

**Without the registry:** each of the four agents (`MacroAgent`, `SectorAgent`, `CompanyAgent`, `OrchestratorAgent`) would independently call `AutoModelForCausalLM.from_pretrained()`, loading 28 GB of weights four times.

**With the registry:** weights are loaded exactly once per process. All four agents share the same `HuggingFaceLLMClient` instance. Thread safety is enforced by `threading.Lock`.

---

## 6. Structured Output Schemas (`schemas.py`)

Every agent is instructed via its system prompt to respond **exclusively** with a valid JSON object matching a fixed schema. The system prompts include strict instructions like:

> "CRITICAL: You must strictly respond ONLY with a valid JSON object — no markdown fences, no prose before or after, no `<think>` tokens, no preamble, no postscript."

The JSON is parsed into dataclasses with `__post_init__` validation:

### `MacroBrief` — MacroAgent output

| Field | Type | Allowed values |
|---|---|---|
| `rate_environment` | `str` | `'tightening'` \| `'neutral'` \| `'easing'` |
| `inflation_regime` | `str` | `'high'` \| `'elevated'` \| `'moderate'` \| `'low'` |
| `recession_risk` | `float` | $[0, 1]$ |
| `yield_curve_signal` | `str` | `'inverted'` \| `'flat'` \| `'normal'` |
| `credit_stress` | `float` | $[0, 1]$ |
| `overall_sentiment` | `float` | $[-1, 1]$ (−1 = bearish, +1 = bullish) |
| `key_risks` | `list[str]` | Up to 5 items |
| `tailwinds` | `list[str]` | Up to 5 items |
| `analyst_summary` | `str` | 2–3 sentence narrative |

### `SectorBrief` — SectorAgent output

| Field | Type | Allowed values |
|---|---|---|
| `sector` | `str` | GICS sector name |
| `momentum_score` | `float` | $[-1, 1]$ |
| `earnings_revision_trend` | `str` | `'upgrades'` \| `'neutral'` \| `'downgrades'` |
| `valuation_signal` | `str` | `'stretched'` \| `'fair'` \| `'cheap'` |
| `key_themes` | `list[str]` | Up to 5 items |
| `risks` | `list[str]` | Up to 3 items |

### `CompanyBrief` — CompanyAgent output

| Field | Type | Allowed values |
|---|---|---|
| `revenue_growth_trend` | `str` | `'accelerating'` \| `'stable'` \| `'decelerating'` \| `'negative'` |
| `margin_trend` | `str` | `'expanding'` \| `'stable'` \| `'compressing'` |
| `balance_sheet_quality` | `str` | `'strong'` \| `'adequate'` \| `'stretched'` |
| `earnings_quality` | `str` | `'high'` \| `'medium'` \| `'low'` |
| `fundamental_score` | `float` | $[-1, 1]$ |
| `key_risks` / `key_catalysts` | `list[str]` | Up to 4 items each |

### `MarketBrief` — OrchestratorAgent output

| Field | Type | Description |
|---|---|---|
| `macro_regime` | `str` | `'risk_on'` \| `'risk_off'` \| `'transitional'` |
| `portfolio_stance` | `str` | `'aggressive'` \| `'neutral'` \| `'defensive'` |
| `conviction_score` | `float` | $[0, 1]$ — how much the signals agree |
| `top_overweights` | `list[str]` | Up to 5 tickers with strongest positive signal |
| `top_underweights` | `list[str]` | Up to 5 tickers with strongest negative signal |
| `sector_tilts` | `dict[str, float]` | Sector name → $[-1, 1]$ tilt signal |
| `key_themes` | `list[str]` | Up to 6 investment themes |
| `risk_flags` | `list[str]` | Up to 4 portfolio-level risks |
| `executive_summary` | `str` | 3–5 sentence investment narrative |

Every schema provides a `.neutral()` class method that returns a zero-signal placeholder used when the LLM call fails or when data is unavailable.

### JSON parsing — three-strategy extractor

`BaseAgent.parse_json_response()` handles the reality that models occasionally wrap JSON in prose or markdown:

```
Strategy 1: json.loads(raw)
  → works for Claude and well-tuned HF models at temperature=0

Strategy 2: json.loads(raw[raw.index('{') : raw.rindex('}')+1])
  → handles leading prose or trailing explanation text

Strategy 3: regex match between ```json ... ``` fences
  → handles models that wrap JSON in markdown fences despite instructions

Strategy 4: return fallback dict
  → logs a warning; downstream validation constructs a neutral brief
```

---

## 7. Specialist Agents

All four agents inherit from `BaseAgent`, which provides the unified `call_llm()` method, JSON parsing, and mock support.

### 7.1 MacroAgent

**Information sources:**
- FRED macro signal snapshot (fed funds rate, CPI, VIX, yield curve, HY spread, unemployment) formatted as a structured text table
- Optional web search (4 queries: Fed policy, inflation, recession risk, credit stress)

**Web search date-bounding:**
Each query is appended with the quarter label (e.g., `"Federal Reserve monetary policy rate decision outlook Q3 2022"`) to bias results toward the relevant period. This is a best-effort mitigation; see Section 12 for the full discussion.

**Output:** `MacroBrief`

### 7.2 SectorAgent

**Information sources:**
- GICS sector name and constituent tickers
- `MacroBrief` for macro context (rate environment, inflation, overall sentiment)
- Optional web search (3 queries per sector: earnings outlook, recent performance, valuation)

**Intra-node parallelism:** When `sector_agents_parallel=True`, all GICS sector agents in the universe run concurrently *within* `sector_node`, after `macro_node` has already completed (see Section 8.2).

**Output:** `SectorBrief`

### 7.3 CompanyAgent

**Information sources:**
- EDGAR XBRL financials: revenue, net income, EPS (diluted), operating income, R&D expense, gross profit, cash, long-term debt, operating cash flow, shares outstanding
- MD&A text from the most recent 10-Q or 10-K (truncated to `max_mda_chars`)
- `SectorBrief` for sector-level context (optional)

**No web search:** The company agent intentionally uses only EDGAR data. This keeps analysis costs low (EDGAR is free), ensures cleaner date-bounding (filings are intrinsically bounded to their filing date), and avoids injecting potentially noisy web content into a highly structured financial analysis task.

**Output:** `CompanyBrief`

### 7.4 OrchestratorAgent

**Information sources:**
- All outputs from the three specialist agents (`MacroBrief`, all `SectorBrief`s, all `CompanyBrief`s)

**Synthesis approach:** The system prompt instructs the orchestrator to:
> "Reconcile any contradictions between the sub-analyses using sound investment judgment. Weight macro signals more heavily during high-uncertainty regimes (high VIX, inverted yield curve, elevated HY spreads)."

**Heuristic fallback:** If the LLM call fails or returns unparseable output, a deterministic heuristic constructs a `MarketBrief` from the numeric scores in the sub-briefs:
- `macro_regime` is derived from `MacroBrief.overall_sentiment`
- `top_overweights` are ranked by `CompanyBrief.fundamental_score`
- `sector_tilts` are copied directly from `SectorBrief.momentum_score`

**Output:** `MarketBrief`

---

## 8. LangGraph Orchestration (`orchestrator.py`)

### 8.1 Graph Topology

The agent execution order is wired as a **LangGraph `StateGraph`** [1] with four nodes running **sequentially**: macro → sector → company → orchestrator. The sector node must follow the macro node because it receives the `MacroBrief` as input context. Nodes communicate through a shared `GraphState` TypedDict.

```
        START
          │
    [macro_node]
     MacroAgent.run()
      → macro_brief
          │  (macro_brief now in state)
    [sector_node]
     ThreadPoolExecutor
     (all GICS sector agents
      run concurrently within
      this single node)
      → sector_briefs
          │  (sector_briefs now in state)
    [company_node]
     ThreadPoolExecutor
     (all per-ticker company agents
      run concurrently within
      this single node)
      → company_briefs
          │
    [orchestrator_node]
     OrchestratorAgent.run()
      → market_brief
          │
         END
```

### 8.2 Intra-node Parallel Execution

While the four nodes execute sequentially, parallelism occurs *within* two of those nodes using `concurrent.futures.ThreadPoolExecutor`:

**Sector node — sector agents in parallel:**
All GICS sectors in the universe are analyzed concurrently inside `sector_node`. The `MacroBrief` produced by the preceding `macro_node` is already available in state and is passed to each `SectorAgent`. Since each sector has its own `SectorAgent` instance, there are no shared resources to contend over.

**Company node — company agents in parallel:**
All stocks are analyzed concurrently. A single `CompanyAgent` instance is shared (stateless between tickers), so the LLM client is not re-initialized per ticker.

```python
with ThreadPoolExecutor(max_workers=_max_workers(n_tickers)) as pool:
    futures = {pool.submit(company_agent.run, ticker, ...): ticker
               for ticker in tickers}
    for future in as_completed(futures):
        company_briefs[futures[future]] = future.result()
```

**Thread pool sizing:**

| Backend | `max_workers` | Reason |
|---|---|---|
| Claude | `min(N, 8)` | API calls are I/O-bound → many threads beneficial |
| HuggingFace | `1` | GPU inference serializes at hardware level; concurrent `model.generate` calls risk CUDA OOM |
| Mock mode | `min(N, 8)` | Stubs return instantly; parallelism is harmless |

### 8.3 No-LangGraph Fallback

When `langgraph` is not installed, `build_agent_graph()` returns a `_SequentialGraph` object with the identical `invoke(state: GraphState) -> GraphState` interface:

```python
class _SequentialGraph:
    def invoke(self, state: GraphState) -> GraphState:
        for node_fn in [macro_fn, sector_fn, company_fn, orchestrator_fn]:
            updates = node_fn(state)
            state.update(updates)
        return state
```

All downstream code (including `AgentPipeline`) is completely unaffected — `build_agent_graph()` is the only call site, and both implementations expose the same interface.

---

## 9. MarketBrief Embedding (`embedder.py`)

The `MarketBrief` is a structured object containing both numeric scores and natural-language text (themes, risk flags, executive summary). To make its full information content available to the RL agent, it is converted to a dense vector via `MarketBriefEmbedder`.

### Embedding strategy

**Primary path — sentence-transformers [2]:**

```
MarketBrief.to_text()
  → "macro_regime: risk_on. portfolio_stance: aggressive.
     top_overweights: NVDA, MSFT, LLY. sector_tilts: IT=0.5, HC=0.2.
     themes: AI infrastructure build-out, ... risk_flags: ..."
        │
        ▼
SentenceTransformer("all-MiniLM-L6-v2")
        │  [bert-style encoding + mean pooling]
        ▼
np.ndarray (384,)  ← L2-normalized
```

The `all-MiniLM-L6-v2` model [3] produces 384-dimensional embeddings with strong semantic properties: briefs describing similar market environments map to nearby points in embedding space, allowing the RL agent to generalize across similar regimes even when surface-level wording differs.

**Fallback path — TF-IDF bag-of-words [4]:**

When `sentence-transformers` is not installed, a deterministic TF-IDF representation over a fixed 200-term finance-domain vocabulary is used. The vocabulary covers macro regime terms, sector names, fundamental signal words, ticker symbols, and risk/sentiment terms. The output is a 200-dimensional L2-normalized vector.

```
Fixed vocab (200 terms):
  ['tightening', 'easing', 'risk_on', 'risk_off', 'inflation',
   'technology', 'healthcare', 'financials', 'accelerating',
   'expanding', 'overweight', 'underweight', 'aapl', 'nvda', ...]

  term_freq[i] = count(term_i in brief_text) / total_terms
  idf[i]       = log(N_quarters / (1 + df[i]))
  tfidf[i]     = term_freq[i] × idf[i]
  → L2-normalize
```

### GPU acceleration

`sentence-transformers` uses PyTorch internally. The `embedding_device` parameter routes inference to the GPU:

| `embedding_device` | Behavior |
|---|---|
| `None` (default) | Auto-detect: uses `'cuda'` if PyTorch finds a GPU, else `'cpu'` |
| `'cuda'` | Force GPU |
| `'cpu'` | Force CPU |

For batch encoding across all quarters of a fold, GPU speedup is 10–25× vs. CPU for batches of 40+ strings.

---

## 10. Web Search and Tools (`tools.py`)

### `web_search()` — two-path implementation

```
web_search(query, as_of_date, max_results, anthropic_client)
        │
        ├── anthropic_client provided?
        │      YES → _web_search_anthropic()
        │              Uses "web_search_20250305" tool via claude-haiku
        │              Higher quality, model-curated snippets
        │
        └── NO → _web_search_ddg()
                  Direct DDGS (DuckDuckGo) query
                  No API key required; used for testing / offline
```

### `DateBoundedQuery` — date-bounded search queries

Every search query issued by macro and sector agents is wrapped in a `DateBoundedQuery` that appends the quarter label:

```python
DateBoundedQuery("Federal Reserve rate decision", "2022-09-21").build()
# → "Federal Reserve rate decision Q3 2022"
```

This biases search results toward the relevant time period. See Section 12 for the full caveat.

### `format_macro_for_llm()` and `format_financials_for_llm()`

These helpers convert raw data dictionaries into clean, human-readable text tables that are injected into the agent's `user_message`:

```
format_macro_for_llm(macro_data, as_of_date):
  === Macro Indicators as of 2023-09-30 ===
  fed_funds_rate     : 5.33
  cpi_yoy            : 3.7
  vix                : 17.52
  yield_curve_10y2y  : -0.44   ← inverted
  hy_spread          : 4.21
  unemployment_rate  : 3.8
  ...

format_financials_for_llm(xbrl_facts, mda_text, ticker, max_chars):
  === AAPL Financial Data ===
  Revenues: $94.8B (2023-Q3)
  NetIncomeLoss: $19.9B
  EarningsPerShareDiluted: $1.26
  ...
  === MD&A (truncated to 4000 chars) ===
  "During the third quarter of fiscal 2023, we generated total net sales ..."
```

---

## 11. Pipeline Orchestration (`pipeline.py`)

`AgentPipeline` runs the LangGraph graph for every quarter-end date in a `WalkForwardFold` and collects results into an `AgentBundle`.

### `AgentBundle` fields

| Field | Type | Description |
|---|---|---|
| `briefs` | `dict[str, MarketBrief]` | Keys = `'YYYY-MM-DD'` quarter-end dates |
| `embeddings` | `DataFrame[dates × embed_dims]` | Columns `embed_0` … `embed_383` |
| `embedding_dim` | `int` | 384 (sentence-transformers) or 200 (TF-IDF fallback) |
| `errors` | `list[str]` | Non-fatal errors from individual agent calls |

### Disk caching

`MarketBrief` outputs are serialized as JSON and written to disk after each quarter-end date:

```
data/cache/agent_briefs/
└── fold_0/
    ├── 2018-03-31.json
    ├── 2018-06-30.json
    ├── 2018-09-30.json
    └── 2018-12-31.json
```

On the next call to `run_fold()`, cached briefs are loaded without any LLM invocation. Use `force_refresh=True` to re-run all agents.

### `run_fold()` data preparation

For each quarter-end date, `AgentPipeline` prepares date-bounded inputs before invoking the graph:

```python
# Macro snapshot: last non-null row of each FRED signal up to as_of_date
macro_data = fold.macro[fold.macro.index <= as_of_date].iloc[-1].to_dict()

# XBRL facts: filings with filing_date <= as_of_date
xbrl_data = {ticker: fetch_xbrl_facts(..., end_date=as_of_date)
             for ticker in fold.tickers}

# MD&A text: most recent filing before as_of_date
mda_data = {ticker: get_mda_text(ticker, as_of_date=as_of_date)
            for ticker in fold.tickers}
```

---

## 12. Walk-Forward Safety and Backtesting

The agents module provides strong walk-forward guarantees for structured data and a best-effort constraint for web search.

| Data source | Walk-forward guarantee | Mechanism |
|---|---|---|
| FRED macro signals | **Hard** | Sliced to `as_of_date` before formatting |
| EDGAR XBRL / MD&A | **Hard** | Filtered by `filing_date <= as_of_date` |
| Web search | **Soft (best-effort)** | Date-bounded query construction |

**Web search caveat:**
Live web searches during backtesting may surface articles published after the `as_of_date`, introducing look-ahead bias. The recommended workflow for clean historical backtesting is:

```
Recommended backtesting workflow:
  1. Run live agents for the current/recent period with web search enabled.
     → AgentPipeline.run_fold(fold, force_refresh=True)

  2. All MarketBriefs are cached to disk as JSON.

  3. For historical fold simulation, replay from cache:
     → AgentPipeline.run_fold(fold, force_refresh=False)
     → LLM calls are skipped; cached JSON is loaded directly.

  4. For pure offline backtesting with no internet access:
     → Set enable_web_search=False (default).
     → Agents rely solely on FRED signals and EDGAR data.
```

---

## 13. Mock Mode

Setting `mock_mode=True` in `AgentConfig` bypasses all LLM and API calls:

- All `call_llm()` calls return each agent's deterministic `_mock_response()` string
- No `ANTHROPIC_API_KEY` required
- No HuggingFace model weights loaded
- Run time reduces from minutes to seconds

The mock `MarketBrief` is a realistic example reflecting a transitional macro environment with AI-driven technology tailwinds, which is a useful test baseline.

```python
# Development workflow
dev_pipeline = AgentPipeline(AgentConfig(mock_mode=True))
bundle = dev_pipeline.run_fold(data_fold)          # completes in seconds

# When ready for live analysis
live_pipeline = AgentPipeline(AgentConfig(
    mock_mode=False,
    llm_backend="claude",
    enable_web_search=True,
))
```

---

## 14. Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | **Yes** (Claude backend, non-mock) | Anthropic API key |
| `ANTHROPIC_MODEL` | No | Override Claude model ID; defaults to `claude-sonnet-4-6` |
| `HUGGINGFACE_MODEL` | No | Override HuggingFace model; defaults to `microsoft/phi-4` |
| `EMBEDDING_MODEL` | No | Override sentence-transformers model; defaults to `all-MiniLM-L6-v2` |
| `HTTP_PROXY` / `HTTPS_PROXY` | No | Proxy settings for Anthropic API and DuckDuckGo calls |

---

## 15. References

[1] LangChain, Inc. *LangGraph: Build stateful, multi-actor LLM applications*. [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
The `StateGraph` framework used to wire the four-node agent graph, manage shared state, and handle parallel branch execution.

[2] Reimers, N. and Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *Proceedings of EMNLP 2019*, 3982–3992.
Introduces the sentence-transformers framework used by `MarketBriefEmbedder`. The `all-MiniLM-L6-v2` model is a distilled, high-performance variant from the same line of work.

[3] Wang, W. et al. (2020). "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers." *Advances in Neural Information Processing Systems (NeurIPS 2020)*.
Architecture basis for the `all-MiniLM-L6-v2` embedding model used as the default.

[4] Salton, G. and Buckley, C. (1988). "Term-Weighting Approaches in Automatic Text Retrieval." *Information Processing and Management*, 24(5), 513–523.
Foundation for the TF-IDF bag-of-words fallback embedding used when `sentence-transformers` is unavailable.

[5] Yao, S. et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023*.
Establishes the Reason + Act paradigm where LLM agents interleave reasoning steps with external tool calls (web search, data lookup) — the pattern used by `MacroAgent` and `SectorAgent`.

[6] Wei, J. et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS 2022*.
Motivates the structured analyst persona prompting strategy used in all four system prompts, which encourages the model to reason step-by-step before producing a structured output.

[7] Anthropic. *Claude API Documentation*. [https://docs.anthropic.com](https://docs.anthropic.com)
Reference for the `messages.create` API, native web-search tool (`web_search_20250305`), and retry behavior implemented in `ClaudeClient`.

[8] Wolf, T. et al. (2020). "HuggingFace's Transformers: State-of-the-Art Natural Language Processing." *EMNLP 2020 Findings*.
The `transformers` library powering `HuggingFaceLLMClient`, including `AutoModelForCausalLM`, `AutoTokenizer`, and `apply_chat_template`.

[9] Dettmers, T. et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *NeurIPS 2022*.
The quantization technique implemented in `bitsandbytes`, used by `HuggingFaceLLMClient` when `load_in_8bit=True`.

[10] Dettmers, T. et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS 2023*.
Introduces the NF4 4-bit quantization scheme used when `load_in_4bit=True`, reducing Phi-4 VRAM from 28 GB to ~7 GB.

[11] Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*.
The attention kernel enabled by `attn_implementation='flash_attention_2'`, providing a 2–4× speedup for HuggingFace inference on Ampere and later NVIDIA GPUs.

[12] Abdin, M. et al. (2024). "Phi-4 Technical Report." *Microsoft Research*.
The default HuggingFace model (`microsoft/phi-4`): a 14B parameter instruction-tuned model with strong structured-output following and JSON generation capabilities.
