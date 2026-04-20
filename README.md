# QuantAgent-RL

**Tax-Aware Portfolio Rebalancing via Multi-Agent LLM Analysis and Reinforcement Learning**

A production-grade, end-to-end system that combines GPU-accelerated quantitative finance, agentic AI (LangGraph orchestration), and reinforcement learning to produce quarterly portfolio rebalancing decisions that maximize risk-adjusted returns while managing tax exposure.

---

## System Architecture

```
quantagent-rl/
├── data/           # GPU-accelerated ingestion & feature engineering
├── forecasting/    # GARCH volatility, HMM regime detection, factor models
├── agents/         # LangGraph multi-agent LLM system
├── rl/             # Gymnasium environment, PPO agent, differential Sharpe reward
├── backtest/       # Walk-forward engine, performance metrics
└── demo/           # Streamlit dashboard
```

---

## Key Design Decisions

### Reward Function
Uses the **Differential Sharpe Ratio** (Moody & Saffell, 1998) as the base reward, augmented with tax and transaction cost penalties:

```
R_t = DifferentialSharpe_t - λ₁·tax_cost_t - λ₂·transaction_cost_t + λ₃·tlh_benefit_t
```

### Walk-Forward Training
Expanding-window cross-validation with recency-weighted episode sampling and warm-start fine-tuning across folds — implementing continual learning under non-stationary market conditions.

### GPU Acceleration
Every data processing stage uses RAPIDS cuDF/cuML when a GPU is available, with automatic fallback to pandas/scikit-learn on CPU.

---

## Modules

| Module | Description |
|--------|-------------|
| `data` | Market data (yfinance), macro signals (FRED), SEC filings ingestion + GPU feature engineering |
| `agents` | LangGraph multi-agent: Macro, Sector, Company, and Orchestrator agents |
| `forecasting` | GARCH(1,1) volatility, HMM regime detection, Fama-French factor exposures |
| `rl` | Custom Gymnasium environment (including FIFO lot tracker, short/long-term capital gains, tax-loss harvesting calculator),<br> PPO via Stable-Baselines3, differential Sharpe reward |
| `backtest` | Walk-forward engine, Sharpe, Sortino, max drawdown, turnover, tax drag metrics |

---

## Tax Model Assumptions (Simplified US Federal)

- Long-term capital gains rate: **15%** (held > 1 year)
- Short-term capital gains rate: **37%** (held ≤ 1 year)
- Cost basis method: **FIFO**
- No wash-sale rule enforcement (documented limitation — future work)
- No state income tax
- No dividend withholding tax (dividends assumed reinvested at NAV)

---

## GPU Acceleration Stack

| Layer | NVIDIA Technology |
|-------|------------------|
| Feature engineering | cuDF, cuML |
| Volatility forecasting | CuPy-accelerated GARCH batch recursion |
| HMM regime decoding | CuPy Viterbi + log-emission computation |
| Factor model (rolling OLS) | CuPy batched einsum + linalg.solve |
| Reward computation | CuPy differential Sharpe EMA loop |
| RL policy + value networks | PyTorch CUDA via Stable-Baselines3 |
| MarketBrief embedding | PyTorch CUDA via sentence-transformers |
| Backtesting | vectorbt with CUDA support |

---

## Setup

1. Clone the repository

```shell
git clone git@github.com:bacalfa/quantagent-rl.git
cd quantagent-rl
```

2. Create virtual or a conda environment (example using `uv`)

```shell
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies (example using `uv`)

```shell
uv sync
```

4. Create file `.env` in top folder and containing the following text (replace API keys with yours)

```
# Create file called .env and add your actual API keys

# Your Anthropic API key from https://console.anthropic.com/ (optional if using HuggingFace model)
ANTHROPIC_API_KEY=sk-*********************************************************************************************************

# LLM selection
ANTHROPIC_MODEL=claude-sonnet-4-6
HUGGINGFACE_MODEL=mistralai/mistral-7B-instruct-v0.3

# Embedding model selection (sentence-transformers)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# FRED API Key from https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=********************************

# EDGAR User Agent
SEC_USER_AGENT=First Name Last Name username@email.com

# HuggingFace user access token
HF_TOKEN=hf_**********************************
```

---

## Simplified Tax Disclaimer

This project is for educational and research purposes only. The tax model is a significant simplification of real US tax law. Nothing in this project constitutes financial or tax advice.
