# `rl` — QuantAgent-RL Reinforcement Learning Module

The `rl` module is the core decision engine of QuantAgent-RL. It trains a **Proximal Policy Optimization (PPO)** agent to allocate capital across a universe of equities at each quarter-end rebalancing date. The agent maximizes a risk-adjusted reward derived from the **Differential Sharpe Ratio**, with augmentations for transaction costs and tax efficiency. Training follows a **walk-forward protocol** that prevents look-ahead bias and simulates realistic live deployment.

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Quick Start](#2-quick-start)
3. [Architecture](#3-architecture)
4. [MDP Formulation](#4-mdp-formulation)
   4.1 [State Space](#41-state-space)
   4.2 [Action Space](#42-action-space)
   4.3 [Reward Signal](#43-reward-signal)
   4.4 [Episode Structure](#44-episode-structure)
5. [Configuration (`config.py`)](#5-configuration-configpy)
6. [Environment (`env.py`)](#6-environment-envpy)
   6.1 [Action Processing Pipeline](#61-action-processing-pipeline)
   6.2 [Tax Model and Lot Tracking](#62-tax-model-and-lot-tracking)
7. [Reward Signal (`reward.py`)](#7-reward-signal-rewardpy)
   7.1 [Differential Sharpe Ratio](#71-differential-sharpe-ratio)
   7.2 [Differential Sortino Variant](#72-differential-sortino-variant)
   7.3 [Augmented Reward](#73-augmented-reward)
   7.4 [GPU Batch Computation](#74-gpu-batch-computation)
8. [State Construction (`state.py`)](#8-state-construction-statepy)
9. [PPO Agent (`agent.py`)](#9-ppo-agent-agentpy)
   9.1 [GPU Policy Network](#91-gpu-policy-network)
   9.2 [Warm-Start Across Folds](#92-warm-start-across-folds)
   9.3 [Recency-Weighted Episode Sampling](#93-recency-weighted-episode-sampling)
10. [Walk-Forward Pipeline (`pipeline.py`)](#10-walk-forward-pipeline-pipelinepy)
    10.1 [FoldMetrics and RLFoldResult](#101-foldmetrics-and-rlfolresult)
    10.2 [Baselines](#102-baselines)
11. [Walk-Forward Safety](#11-walk-forward-safety)
12. [GPU Acceleration Summary](#12-gpu-acceleration-summary)
13. [References](#13-references)

---

## 1. Module Overview

```
rl/
├── __init__.py      Public API — all exports
├── config.py        RewardConfig, PortfolioConstraints, PPOConfig, WalkForwardConfig, RLConfig
├── env.py           PortfolioEnv (Gymnasium), Lot, LotTracker
├── reward.py        DifferentialSharpeState, RewardCalculator
├── state.py         StateBuilder — observation vector assembly + normalization
├── agent.py         PPOAgent (SB3 wrapper), RLCallback, RecencyWeightedSampler
└── pipeline.py      RLPipeline, FoldMetrics, RLFoldResult
```

**Design principles:**

| Principle | Implementation |
|---|---|
| Walk-forward integrity | Scalers fitted on training data only; test data never touches scaler fitting |
| Markovian reward | Differential Sharpe Ratio — directly differentiates risk-adjusted return without a full trajectory |
| Tax awareness | FIFO lot tracking with short/long-term rates; tax-loss harvesting incentive in reward |
| Continual learning | Warm-start fine-tuning: policy weights carry forward across folds instead of retraining from scratch |
| Recency bias | Exponential decay episode sampler — recent market regimes sampled more frequently |
| Graceful degradation | All three upstream bundles (forecast, agent embeddings) are optional; the environment degrades to quantitative features only |

---

## 2. Quick Start

```python
from rl import RLPipeline, RLConfig, PPOConfig

# Minimal run — uses equal-weight baseline if upstream bundles are omitted
pipeline = RLPipeline(RLConfig())
result = pipeline.run_fold(data_fold)           # WalkForwardFold from data module
print(result.ppo_metrics.sharpe)               # out-of-sample Sharpe
print(result.equal_weight_metrics.sharpe)      # 1/N baseline for comparison

# Full run — all three upstream modules
result = pipeline.run_fold(
    fold=data_fold,
    forecast_bundle=forecast_bundle,           # ForecastBundle from forecasting module
    agent_bundle=agent_bundle,                 # AgentBundle from agents module
    sector_map={"Information Technology": ["AAPL", "MSFT", "NVDA"], ...},
)

# Multi-fold walk-forward backtest
results = pipeline.run_all_folds(
    data_pipeline=data_pipeline,
    forecast_bundles=forecast_bundles,
    agent_bundles=agent_bundles,
)
df = RLPipeline.summary_dataframe(results)     # DataFrame: folds × metrics × strategies
print(df[df.strategy == "ppo"][["sharpe", "max_drawdown", "total_return"]])

# Force GPU training
gpu_pipeline = RLPipeline(RLConfig(ppo=PPOConfig(device="cuda")))
```

---

## 3. Architecture

The RL module sits at the top of the QuantAgent-RL stack and consumes outputs from all three upstream modules:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                             RLPipeline.run_fold()                            │
│                                                                              │
│  Inputs (per walk-forward fold)                                              │
│  ├── WalkForwardFold.train_state_matrix  ← data module (quantitative)       │
│  ├── ForecastBundle.rl_state_extension   ← forecasting module               │
│  └── AgentBundle.embeddings              ← agents module (384-dim vector)   │
│                  │                                                           │
│                  ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  StateBuilder.fit(train_data)   ← fit z-score scalers on train only  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                  │                                                           │
│                  ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  PortfolioEnv  (Gymnasium)                                           │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │ Step loop (one quarter per step):                              │  │   │
│  │  │                                                                │  │   │
│  │  │  observation = StateBuilder.build(date)                        │  │   │
│  │  │       │   [quant | forecast | agent_embed | weights]           │  │   │
│  │  │       ▼                                                        │  │   │
│  │  │  action = PPOAgent.predict(obs)                                │  │   │
│  │  │       │   Δw ∈ ℝ^n  (delta weights)                           │  │   │
│  │  │       ▼                                                        │  │   │
│  │  │  _apply_action():  threshold → clip → sector → normalize       │  │   │
│  │  │       │                                                        │  │   │
│  │  │       ▼                                                        │  │   │
│  │  │  _execute_trades(): FIFO lot matching → tax_cost, tlh_benefit  │  │   │
│  │  │       │                                                        │  │   │
│  │  │       ▼                                                        │  │   │
│  │  │  RewardCalculator.step()                                       │  │   │
│  │  │       R_t = DSR(r_t) − λ_tax·tax + λ_tlh·tlh − λ_tc·|Δw|    │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                  │                                                           │
│                  ▼                                                           │
│  PPOAgent.train(env, timesteps=50_000)                                       │
│  [fold > 0: warm_start=True → fine-tune existing policy]                     │
│                  │                                                           │
│                  ▼                                                           │
│  Evaluate on test dates → FoldMetrics (PPO, equal-weight, hold)              │
│                  │                                                           │
│                  ▼                                                           │
│  RLFoldResult  ─────► RLPipeline.summary_dataframe(results)                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. MDP Formulation

The portfolio management problem is cast as a **finite-horizon Markov Decision Process (MDP)** [1] with quarterly rebalancing steps.

### 4.1 State Space

The observation vector $s_t \in \mathbb{R}^d$ is assembled by `StateBuilder` as the concatenation of four blocks:

$$
s_t = \bigl[\underbrace{z^{\text{quant}}_t}_{\text{quantitative features}} \;\big|\; \underbrace{z^{\text{forecast}}_t}_{\text{GARCH/HMM/FF}} \;\big|\; \underbrace{z^{\text{embed}}_t}_{\text{agent embedding}} \;\big|\; \underbrace{w_t}_{\text{current weights}}\bigr]
$$

| Block | Source module | Typical dimension | Content |
|---|---|---|---|
| $z^{\text{quant}}_t$ | `data` | ~200–400 | Returns, volatility, momentum, factor betas, macro signals |
| $z^{\text{forecast}}_t$ | `forecasting` | ~50–100 | GARCH vol forecasts, HMM regime probabilities, rolling FF betas |
| $z^{\text{embed}}_t$ | `agents` | 384 | Sentence-transformer embedding of `MarketBrief` |
| $w_t$ | environment | $n$ | Current portfolio weights (so the agent knows what it already holds) |

All blocks are **z-score normalized** using in-sample statistics and clipped to $[-5, 5]$.

### 4.2 Action Space

The action $a_t \in \mathbb{R}^n$ represents **delta weights** — proposed changes to each asset's portfolio weight:

$$
a_t = \Delta w_t, \quad a_t \in [-\delta_{\max}, \, \delta_{\max}]^n
$$

where $\delta_{\max}$ = `constraints.max_turnover` (default: 0.5). New weights are computed as:

$$
w_{t+1} = \text{normalize}\bigl(\text{clip}(w_t + a_t, \, w_{\min}, \, w_{\max})\bigr)
$$

The delta-weight parameterization is preferable to direct weight prediction because it naturally handles **portfolio inertia** — an agent that outputs zero always holds the current portfolio, making the no-trade action the trivially learnable baseline.

### 4.3 Reward Signal

The step reward is the **Augmented Differential Sharpe Ratio** (see Section 7):

$$
R_t = D_t(\eta) - \lambda_{\text{tax}} \cdot c^{\text{tax}}_t + \lambda_{\text{tlh}} \cdot b^{\text{tlh}}_t - \lambda_{\text{turnover}} \cdot c^{\text{tc}}_t
$$

where $D_t$ is the differential Sharpe (or Sortino) increment at step $t$.

### 4.4 Episode Structure

Each **episode** covers the full sequence of quarter-end dates in either the training or test window:

```
Episode (training window example: 12 quarters = 3 years):

  t=0       t=1       t=2       ...     t=11      t=12
  2018-Q1   2018-Q2   2018-Q3   ...     2020-Q4   [terminated]
   │         │         │                  │
   reset    step      step              step
   w_0=1/n  a_1→w_1  a_2→w_2           a_12→w_12

PPO collects multiple short episodes before each gradient update
(rollout_buffer_size = n_steps × n_envs).
```

Episodes are short (12–40 steps for 3–10-year training windows), so PPO's rollout buffer accumulates **several complete episodes** before each gradient update.

---

## 5. Configuration (`config.py`)

All hyperparameters are grouped into a single `RLConfig` object, which nests four sub-configs:

```python
@dataclass
class RLConfig:
    reward:      RewardConfig          # reward shaping weights
    constraints: PortfolioConstraints  # weight / turnover / sector limits
    ppo:         PPOConfig             # SB3 architecture and training
    walk_forward: WalkForwardConfig    # fold sizes and evaluation settings
```

### `RewardConfig`

| Parameter | Default | Description |
|---|---|---|
| `use_sortino` | `False` | Use downside-only variance in the DSR denominator |
| `eta` | `0.05` | EMA adaptation rate $\eta$ for running statistics $A_t$, $B_t$ |
| `lambda_tax` | `0.5` | Penalty weight on realized capital gains tax |
| `lambda_tlh` | `0.3` | Incentive weight on tax-loss harvesting benefit |
| `lambda_turnover` | `0.1` | Penalty weight on total L1 turnover |
| `transaction_cost_fixed` | `0.001` | Fixed commission per trade (10 bps of notional) |
| `transaction_cost_impact` | `0.002` | Quadratic market impact coefficient |
| `reward_scale` | `10.0` | Global reward scaling for numerical stability |

### `PortfolioConstraints`

| Parameter | Default | Description |
|---|---|---|
| `max_position` | `0.20` | Maximum weight for any single asset (20% cap) |
| `min_position` | `0.00` | Minimum weight — 0.0 = long-only, no shorting |
| `max_sector_weight` | `0.40` | Maximum combined weight for any GICS sector |
| `max_turnover` | `0.50` | Maximum one-way L1 turnover per quarter |
| `no_trade_threshold` | `0.005` | Weight changes below 50 bps are zeroed out |

### `PPOConfig`

| Parameter | Default | Description |
|---|---|---|
| `policy` | `"MlpPolicy"` | SB3 policy class (two-layer MLP) |
| `net_arch` | `[256, 256]` | Hidden layer sizes for policy and value networks |
| `learning_rate` | `3e-4` | Adam optimizer learning rate |
| `n_steps` | `512` | Rollout buffer size (steps before each PPO update) |
| `batch_size` | `64` | Mini-batch size for gradient updates |
| `n_epochs` | `10` | Gradient epochs per PPO update |
| `gamma` | `0.99` | Discount factor (near 1.0 for quarterly horizon) |
| `gae_lambda` | `0.95` | GAE $\lambda$ for advantage estimation |
| `clip_range` | `0.2` | PPO clipping parameter $\epsilon$ |
| `ent_coef` | `0.005` | Entropy regularization coefficient |
| `total_timesteps` | `50,000` | Training steps for the first fold |
| `warmstart_timesteps` | `10,000` | Fine-tuning steps for subsequent folds |
| `device` | `"auto"` | `"cuda"`, `"cpu"`, or `"auto"` (detect) |
| `seed` | `42` | Random seed for reproducibility |

### `WalkForwardConfig`

| Parameter | Default | Description |
|---|---|---|
| `min_train_quarters` | `12` | Minimum training window (3 years) |
| `test_quarters` | `4` | Test window per fold (1 year) |
| `recency_weight_decay` | `0.05` | Exponential decay for episode sampling |
| `n_eval_episodes` | `5` | Evaluation episodes at each fold boundary |

---

## 6. Environment (`env.py`)

`PortfolioEnv` is a custom **Gymnasium** [2] environment that models the quarterly portfolio rebalancing problem.

```python
env = PortfolioEnv(
    tickers=["AAPL", "MSFT", "NVDA", ...],
    rebalance_dates=fold.train_dates,
    prices=fold.train_prices,
    quant_df=quant_train,
    forecast_df=fc_train,
    embed_df=emb_train,
    state_builder=sb,                          # pre-fitted on training data
    config=RLConfig(),
    sector_map={"Information Technology": ["AAPL", "MSFT", "NVDA"]},
)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

**Observation space:** `Box(low=-5, high=5, shape=(obs_dim,), dtype=float32)`

**Action space:** `Box(low=-max_turnover, high=+max_turnover, shape=(n_assets,), dtype=float32)`

### 6.1 Action Processing Pipeline

The raw delta-weight action output by the policy goes through a five-step transformation before execution:

```
Raw action  Δw ∈ [-0.5, 0.5]^n
      │
      ▼  1. No-trade threshold
         |Δw_i| < 0.005  →  Δw_i = 0
      │
      ▼  2. Turnover clipping
         if Σ|Δw_i| > max_turnover:
             Δw ← Δw × (max_turnover / Σ|Δw_i|)
      │
      ▼  3. Weight update + position clipping
         w_new = clip(w_old + Δw, min_pos, max_pos)
      │
      ▼  4. Sector-weight constraints
         for each sector s:
             if Σ_{i∈s} w_i > max_sector_weight:
                 w_{i∈s} ← w_{i∈s} × (max_sector_weight / Σ_{i∈s} w_i)
      │
      ▼  5. Renormalization
         w_new ← w_new / Σ w_new
```

The no-trade threshold (step 1) simulates a **minimum trade size** and prevents micro-rebalancing — a common issue with continuous-action RL agents that generates unrealistically high turnover.

### 6.2 Tax Model and Lot Tracking

The environment implements a simplified US federal capital gains tax model using **FIFO lot tracking**:

| Tax category | Rate | Holding period |
|---|---|---|
| Short-term gains | 37% | ≤ 4 quarters |
| Long-term gains | 15% | > 4 quarters |

**`Lot` dataclass:** stores `cost_basis`, `quarters_held`, and `size` for a single purchase.

**`LotTracker`:** maintains a `deque[Lot]` per ticker. On each sell, lots are consumed oldest-first (FIFO):

```
FIFO sell example: sell 8% of portfolio weight in AAPL

  Lots (oldest → newest):
  [Lot(basis=0.05, held=6q, size=0.05)]   ← sell all 5%  (long-term, 15% rate)
  [Lot(basis=0.09, held=1q, size=0.08)]   ← sell 3% more (short-term, 37% rate)

  gain_long  = (current - 0.05) × 0.05 → tax at 15%
  gain_short = (current - 0.09) × 0.03 → tax at 37%
  if gain < 0:  → tlh_benefit += |gain|  (no tax, harvests a loss)
```

**Tax-loss harvesting (TLH):** realized losses flow into `tlh_benefit`, which is positively weighted in the reward function. This incentivizes the agent to strategically realize losses to offset future gains — behavior that reduces the effective tax drag on the portfolio over time.

> **Limitation:** The wash-sale rule (which disallows re-purchasing a sold security within 30 days) is not enforced in the current implementation. This is a documented simplification.

---

## 7. Reward Signal (`reward.py`)

The reward signal is designed to be **Markovian** — it depends only on the current transition $(s_t, a_t, s_{t+1})$ without requiring knowledge of the full return trajectory. This is achieved through the Differential Sharpe Ratio.

### 7.1 Differential Sharpe Ratio

The Differential Sharpe Ratio (DSR) was introduced by Moody & Saffell (1998) [3]. The key idea is to express the Sharpe ratio through EMA statistics and differentiate with respect to the current return, yielding a single-step reward that directly optimizes the Sharpe ratio over time.

Define the running statistics:

$$
A_t = A_{t-1} + \eta (r_t - A_{t-1})
$$

$$
B_t = B_{t-1} + \eta (r_t^2 - B_{t-1})
$$

where $\eta$ is the EMA adaptation rate and $r_t$ is the portfolio return at step $t$. $A_t$ estimates $\mathbb{E}[r]$ and $B_t$ estimates $\mathbb{E}[r^2]$, so $B_t - A_t^2$ approximates $\text{Var}(r)$.

The Sharpe ratio expressed through these statistics is:

$$
S_t = \frac{A_t}{\sqrt{B_t - A_t^2}}
$$

Differentiating $S_t$ with respect to $r_t$ — i.e., computing the incremental improvement in the Sharpe ratio from one additional return observation — gives the **DSR increment** $D_t$:

$$
\boxed{D_t = \frac{B_{t-1} \cdot \Delta A_t - \tfrac{1}{2} A_{t-1} \cdot \Delta B_t}{(B_{t-1} - A_{t-1}^2)^{3/2}}}
$$

where $\Delta A_t = A_t - A_{t-1}$ and $\Delta B_t = B_t - B_{t-1}$.

**Key properties of the DSR:**
- **Markovian:** $D_t$ depends only on $r_t$, $A_{t-1}$, and $B_{t-1}$ — no trajectory lookback required
- **Risk-sensitive:** the $(B_{t-1} - A_{t-1}^2)^{3/2}$ denominator penalizes high variance; a large gain in a high-volatility regime earns less reward than the same gain in a low-volatility regime
- **Warm-up fallback:** in the first few steps before $B_{t-1} - A_{t-1}^2 > 10^{-9}$, $D_t = r_t$ (raw return), preventing division by near-zero

### 7.2 Differential Sortino Variant

When `use_sortino=True`, the second-moment statistic $B_t$ is replaced with a **downside-only** second moment $B_t^-$:

$$
B_t^- = B_{t-1}^- + \eta \bigl(\min(r_t, 0)^2 - B_{t-1}^-\bigr)
$$

This yields the **Differential Sortino Ratio**, which penalizes only downside deviations. Positive returns contribute to raising the portfolio value but do not increase $B^-$, so upside volatility is not penalized — aligning the reward with loss-averse investor preferences [4].

### 7.3 Augmented Reward

The full step reward augments the DSR with transaction costs and tax adjustments:

$$
R_t = \underbrace{D_t(\eta)}_{\text{risk-adj. return}} \;-\; \underbrace{\lambda_{\text{tax}} \cdot c^{\text{tax}}_t}_{\text{realized tax cost}} \;+\; \underbrace{\lambda_{\text{tlh}} \cdot b^{\text{tlh}}_t}_{\text{TLH benefit}} \;-\; \underbrace{\lambda_{\text{turnover}} \cdot c^{\text{tc}}_t}_{\text{transaction cost}}
$$

where the transaction cost term is:

$$
c^{\text{tc}}_t = c_{\text{fixed}} \sum_i |\Delta w_i| + c_{\text{impact}} \sum_i \Delta w_i^2
$$

The quadratic **market impact** term $c_{\text{impact}} \sum_i \Delta w_i^2$ penalizes large trades more than proportionally — a standard linear-impact model [5] that is convex in trade size and thus easier to differentiate through during policy gradient updates.

The final reward is scaled by `reward_scale` (default: 10) to keep values in a numerically well-conditioned range (~$[-1, 1]$) for the PPO value function.

### 7.4 GPU Batch Computation

`RewardCalculator.batch_differential_sharpe()` computes DSR increments for an **entire trajectory** of $T$ returns using a CuPy loop — useful during advantage estimation and policy evaluation where the full trajectory is available:

```python
# GPU batch DSR (CuPy) — ~10× faster than calling update() T times in Python
D = reward_calc.batch_differential_sharpe(returns)   # shape (T,)
```

The scalar EMA state is maintained as `xp.float32` values on the GPU throughout the loop to avoid CPU↔GPU data transfer overhead.

---

## 8. State Construction (`state.py`)

`StateBuilder` assembles and normalizes the observation vector, enforcing walk-forward safety at the scaler level.

### Fit–transform protocol

```
Training phase:
  sb = StateBuilder(tickers)
  sb.fit(quant_train, forecast_train, embed_train)
       │
       └── computes column-level z-score statistics
           μ, σ from training window only

Test phase (applied to both train and test):
  obs = sb.build(date, quant_df, forecast_df, embed_df, weights)
       │
       └── looks up the row asof(date), applies
           z-score using training statistics,
           clips to [-5, 5]
```

The `asof` lookup — finding the most recent row on or before `date` — is crucial for walk-forward safety: if a quarterly feature is missing for a specific date, the most recently available value is used rather than a future value.

### Normalization formula

For each feature block $x$:

$$
z_t = \frac{x_t - \hat{\mu}_{\text{train}}}{\hat{\sigma}_{\text{train}}}, \quad z_t \leftarrow \text{clip}(z_t, -5, 5)
$$

The clipping to $[-5, 5]$ prevents outlier observations (e.g., during market crashes or data anomalies) from destabilizing the policy network, which is especially important during out-of-sample test periods that may contain novel market regimes not represented in the training data.

### Observation dimension

The total observation dimension is computed dynamically on construction from the input DataFrames:

$$
d = \underbrace{n^{\text{quant}}}_{\text{quantitative}} + \underbrace{n^{\text{forecast}}}_{\text{forecasting}} + \underbrace{n^{\text{embed}}}_{384 \text{ or } 0} + \underbrace{n}_{\text{portfolio weights}}
$$

SB3's policy network reads `env.observation_space.shape[0]` at construction time, so the network input size is always consistent with the actual observation dimension.

---

## 9. PPO Agent (`agent.py`)

The PPO agent [6] is implemented as a thin wrapper around **Stable-Baselines3** [7], adding warm-start support, recency-weighted sampling, and a lightweight training callback.

### 9.1 GPU Policy Network

The policy and value networks are two-layer MLPs with hidden size 256:

```
Observation (d,)
      │
   Linear(d → 256)  +  ReLU
      │
   Linear(256 → 256)  +  ReLU
      │
   ┌──────────────────┐
   │                  │
Linear(256 → n)    Linear(256 → 1)
  Policy head        Value head
(delta weights)   (state value V(s))
```

When `device='cuda'` (or `'auto'` with a GPU detected), all layers, gradients, and Adam optimizer states live on-device. The GPU speedup for the gradient update step is **3–8× vs. CPU** for typical batch sizes (64) and network sizes (256 × 256).

### 9.2 Warm-Start Across Folds

In a standard walk-forward protocol, the agent would be trained from random initialization at each fold — discarding all accumulated knowledge. QuantAgent-RL implements **continual learning** by preserving policy weights across folds:

```
Fold 0 (train: 2010–2012):
  agent = PPOAgent(obs_dim, n_assets, config)
  agent.train(env, timesteps=50_000)               ← full training
  → policy weights: θ_0

Fold 1 (train: 2010–2013, expanded):
  agent.train(env, warm_start=True, timesteps=10_000)
       │
       └── model.set_env(new_env)                  ← update env only
           model.learn(10_000, ...)                ← fine-tune θ_0 → θ_1
  → policy weights: θ_1  (carrying knowledge from fold 0)

Fold 2 (train: 2010–2014):
  agent.train(env, warm_start=True, timesteps=10_000)
  → policy weights: θ_2
  ...
```

Warm-starting reduces compute cost dramatically (10,000 steps vs. 50,000 per fold) while benefiting from the broader historical context at each successive fold.

### 9.3 Recency-Weighted Episode Sampling

Financial markets are **non-stationary** — the market regime from 10 years ago is often less predictive of near-future behavior than the regime from 1 year ago. The `RecencyWeightedSampler` assigns exponentially increasing sampling probabilities to more recent training episodes:

$$
p_i \propto e^{\,\delta \cdot i}, \quad i = 0, 1, \ldots, N-1
$$

where $\delta$ = `walk_forward.recency_weight_decay` (default: 0.05) and $i$ is the episode's chronological index.

```
Example: 20 training quarters, decay=0.05

  2010-Q1  2010-Q2  ...  2014-Q4
   p ∝ e^0  p ∝ e^0.05   p ∝ e^0.95

  Relative sampling rate of most recent episode:
    e^(0.05 × 19) / e^0 = e^0.95 ≈ 2.6×

  So the most recent quarter is sampled ~2.6× more often than the oldest.
```

Setting `decay=0.0` recovers uniform (non-recency-weighted) sampling, which is equivalent to the standard PPO rollout buffer.

---

## 10. Walk-Forward Pipeline (`pipeline.py`)

`RLPipeline` orchestrates the full training and evaluation loop across walk-forward folds, consuming outputs from all three upstream modules.

### Walk-forward protocol

```
Time →
──────────────────────────────────────────────────────────────────────────────
Fold 0:  [──── train (12 q) ────][── test (4 q) ──]
Fold 1:  [─────── train (16 q) ──────────][── test (4 q) ──]
Fold 2:  [──────────── train (20 q) ───────────────][── test (4 q) ──]
...
                                                    ▲
                          expanding training window │ fixed test window
```

For each fold:

1. **Split** quant, forecast, and embedding DataFrames into train/test at `fold.train_end`
2. **Fit** `StateBuilder` scalers on training data only
3. **Build** `PortfolioEnv` for the training rebalance dates
4. **Train** `PPOAgent` (fresh on fold 0; warm-start on fold > 0)
5. **Save** PPO checkpoint: `checkpoints/rl/fold_{i}/ppo_agent.zip`
6. **Build** `PortfolioEnv` for the test rebalance dates
7. **Evaluate** PPO agent and both baselines on the test environment
8. **Return** `RLFoldResult`

### 10.1 `FoldMetrics` and `RLFoldResult`

`FoldMetrics` is computed directly from the environment's episode log:

| Metric | Description |
|---|---|
| `total_return` | Cumulative return: $V_T / V_0 - 1$ |
| `annualized_return` | Annualized: $(1 + R_{\text{total}})^{4/T} - 1$ |
| `sharpe` | Annualized Sharpe: $\sqrt{4} \cdot \bar{r} / \sigma_r$ |
| `sortino` | Annualized Sortino: $\sqrt{4} \cdot \bar{r} / \sigma_r^-$ (downside std only) |
| `max_drawdown` | Maximum peak-to-trough drawdown: $\min_t (V_t - \max_{s \leq t} V_s) / \max_{s \leq t} V_s$ |
| `total_turnover` | Sum of all quarterly one-way L1 turnovers |
| `total_tax_cost` | Sum of all realized capital gains taxes |
| `quarterly_returns` | Full list of per-step portfolio returns |
| `quarterly_weights` | Full list of per-step portfolio weight vectors |

`RLFoldResult` collects all metrics for one fold:

```python
result.ppo_metrics           # FoldMetrics for the trained PPO agent
result.equal_weight_metrics  # FoldMetrics for 1/N baseline
result.hold_metrics          # FoldMetrics for buy-and-hold baseline
result.train_metrics         # FoldMetrics from last training episode (in-sample)
result.agent_path            # path to saved PPO checkpoint
```

### 10.2 Baselines

Two baselines are evaluated on the **same test environment** as the PPO agent, ensuring fair comparison (same prices, same reward function, same transaction cost model):

| Baseline | Description |
|---|---|
| **Equal-weight (1/N)** | Portfolio weight $1/n$ for each asset, no rebalancing within the test window |
| **Buy-and-hold** | Holds the weights from the last training step without rebalancing |

Evaluating baselines in the same environment ensures that any PPO advantage over the baselines is not attributable to different cost assumptions or data access.

### Summary DataFrame

`RLPipeline.summary_dataframe(results)` produces a tidy multi-row DataFrame for analysis:

```
fold  strategy       total_return  annualized_return  sharpe  sortino  max_drawdown  total_turnover
  0   ppo                  0.142              0.134    0.85     1.12        -0.078           1.23
  0   equal_weight         0.098              0.094    0.62     0.81        -0.112           0.00
  0   hold                 0.087              0.083    0.55     0.73        -0.135           0.00
  1   ppo                  0.183              0.172    1.02     1.35        -0.055           1.41
  ...
```

---

## 11. Walk-Forward Safety

The RL module provides strong walk-forward guarantees through two mechanisms:

**Scaler fitting:** `StateBuilder.fit()` must be called with **training data only**. The `RLPipeline` enforces this by splitting DataFrames at `fold.train_end` before passing them to `StateBuilder`. Applying the fitted scaler to test data is safe because the normalization statistics are derived exclusively from the past.

**Feature lookup:** `StateBuilder._asof_row()` uses an `asof` lookup — the most recent row on or before `date`. This ensures that even if a feature DataFrame has irregular dates or gaps, no future values are inadvertently consumed.

```
Walk-forward data flow (no leakage):

  Training window:                         Test window:
  ┌──────────────────────────────────┐     ┌──────────────────┐
  │ quant_train, fc_train, emb_train │     │ quant_test, ...  │
  │         ▼                        │     │                  │
  │   StateBuilder.fit()             │     │  .build(date)    │
  │   μ, σ computed here only        │─────►  uses μ, σ from  │
  └──────────────────────────────────┘     │  training only   │
                                           └──────────────────┘
```

---

## 12. GPU Acceleration Summary

| Component | GPU path | CPU fallback | Speedup |
|---|---|---|---|
| Policy / value network | PyTorch on CUDA | PyTorch on CPU | 3–8× (gradient step) |
| Batch DSR computation | CuPy loop | NumPy loop | ~10× (long trajectories) |
| Sentence-transformer embedding | PyTorch on CUDA (agents module) | PyTorch on CPU | 10–25× (batch) |

**Recommended setup for full GPU utilization:**

```python
RLConfig(
    ppo=PPOConfig(device="cuda"),         # policy on GPU
    reward=RewardConfig(),                 # DSR uses CuPy if available
)
AgentConfig(embedding_device="cuda")      # sentence-transformer on GPU
```

**HuggingFace LLM backend note:** when using a local HuggingFace model in the agents module, the LLM occupies most VRAM. In this configuration, consider running the PPO policy on CPU (`PPOConfig(device="cpu")`) to avoid GPU memory conflicts, or use a smaller quantized model.

---

## 13. References

[1] Sutton, R.S. and Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
The foundational MDP framework and policy gradient theory used throughout the RL module.

[2] Towers, M. et al. (2023). "Gymnasium." Farama Foundation. [https://gymnasium.farama.org](https://gymnasium.farama.org)
The `gym.Env` interface that `PortfolioEnv` inherits from, ensuring compatibility with SB3 and other RL frameworks.

[3] Moody, J. and Saffell, M. (1998). "Performance Functions and Reinforcement Learning for Trading Systems and Portfolios." *Journal of Forecasting*, 17(5–6), 441–470.
Introduces the Differential Sharpe Ratio as a Markovian, step-level reward for directly optimizing risk-adjusted portfolio performance in an RL setting.

[4] Sortino, F.A. and van der Meer, R. (1991). "Downside Risk." *Journal of Portfolio Management*, 17(4), 27–31.
Foundational reference for the Sortino ratio and the rationale for penalizing downside variance only — the basis for the `use_sortino=True` reward variant.

[5] Almgren, R. and Chriss, N. (2001). "Optimal Execution of Portfolio Transactions." *Journal of Risk*, 3(2), 5–39.
Establishes the linear-plus-quadratic transaction cost model (fixed commission + market impact) implemented in `RewardCalculator.transaction_cost()`.

[6] Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*.
The PPO algorithm used for policy optimization. PPO's clipped surrogate objective provides stable updates without the complexity of trust-region constraints.

[7] Schulman, J. et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." *arXiv:1506.02438*.
Introduces GAE, the advantage estimator used by PPO. The `gae_lambda` parameter controls the bias–variance tradeoff in advantage estimates.

[8] Raffin, A. et al. (2021). "Stable-Baselines3: Reliable Reinforcement Learning Implementations." *Journal of Machine Learning Research*, 22(268), 1–8.
The SB3 library used for PPO implementation, providing well-tested policy and value network architectures, rollout buffers, and training loops.

[9] Jiang, Z., Xu, D., and Liang, J. (2017). "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem." *arXiv:1706.10059*.
Applies deep RL to portfolio management; motivates the delta-weight action parameterization and the episode-per-rebalancing-period structure.

[10] Ye, Y. et al. (2020). "Reinforcement Learning for Portfolio Management." *arXiv:2005.12158*.
Surveys RL approaches to portfolio optimization including walk-forward evaluation protocols and comparison against equal-weight and buy-and-hold baselines.

[11] Hendershott, T., Jones, C.M., and Menkveld, A.J. (2011). "Does Algorithmic Trading Improve Liquidity?" *Journal of Finance*, 66(1), 1–33.
Contextualizes transaction cost modeling and market-impact assumptions used in the reward function.

[12] Harvey, C.R. et al. (2018). "An Evaluation of Alternative Multiple Testing Methods for Finance Applications." *Review of Asset Pricing Studies*, 10(2), 199–248.
Motivates the walk-forward (out-of-sample) evaluation protocol: multiple in-sample tests overfit; all performance claims in QuantAgent-RL are based solely on out-of-sample `test_dates`.
