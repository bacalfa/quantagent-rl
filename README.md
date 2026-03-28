# QuantAgent-RL

**Tax-Aware Portfolio Rebalancing via Multi-Agent LLM Analysis and Reinforcement Learning**

A production-grade, end-to-end system that combines GPU-accelerated quantitative finance, agentic AI (LangGraph orchestration), and reinforcement learning to produce quarterly portfolio rebalancing decisions that maximize risk-adjusted returns while managing tax exposure.

---

## System Architecture

```
quantagent-rl/
├── data/           # GPU-accelerated ingestion & feature engineering
├── agents/         # (TODO) LangGraph multi-agent LLM system
├── forecasting/    # (TODO) GARCH volatility, HMM regime detection, factor models
├── rl/             # (TODO) Gymnasium environment, PPO agent, differential Sharpe reward
├── tax/            # (TODO) Lot-level cost basis tracker, tax cost calculator
├── backtest/       # (TODO) Walk-forward engine, performance metrics
└── demo/           # (TODO) Streamlit dashboard
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
| `rl` | Custom Gymnasium environment, PPO via Stable-Baselines3, differential Sharpe reward |
| `tax` | FIFO lot tracker, short/long-term capital gains, tax-loss harvesting calculator |
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
| Volatility forecasting | CuPy-accelerated GARCH |
| RL training | PyTorch (CUDA) via Stable-Baselines3 |
| Backtesting | vectorbt with CUDA support |

---

## Setup

```bash
pip install -r requirements.txt
```

Set environment variables:
```bash
export FRED_API_KEY="your_fred_api_key"         # https://fred.stlouisfed.org/docs/api/api_key.html
export ANTHROPIC_API_KEY="your_anthropic_key"   # https://console.anthropic.com
```

---

## Simplified Tax Disclaimer

This project is for educational and research purposes only. The tax model is a significant simplification of real US tax law. Nothing in this project constitutes financial or tax advice.
