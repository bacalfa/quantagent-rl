# `forecasting` — QuantAgent-RL Forecasting Module

The `forecasting` module produces the **quantitative context** that the reinforcement-learning agent and LLM agents need to make informed portfolio allocation decisions.
It computes three complementary signals for every walk-forward fold — volatility, market regime, and factor exposures — and packages them into a single `ForecastBundle` that slots directly into the RL state vector.

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Quick Start](#2-quick-start)
3. [Architecture](#3-architecture)
4. [Configuration (`config.py`)](#4-configuration-configpy)
5. [GARCH Volatility Forecasting (`garch.py`)](#5-garch-volatility-forecasting-garchpy)
   5.1 [Model](#51-model)
   5.2 [Estimation](#52-estimation)
   5.3 [GPU-Accelerated Batch Forecasting](#53-gpu-accelerated-batch-forecasting)
   5.4 [Walk-Forward Refitting Schedule](#54-walk-forward-refitting-schedule)
   5.5 [Outputs](#55-outputs)
6. [Market Regime Detection (`regime.py`)](#6-market-regime-detection-regimepy)
   6.1 [Model](#61-model)
   6.2 [Feature Construction](#62-feature-construction)
   6.3 [Fitting](#63-fitting)
   6.4 [Inference — Viterbi and Forward-Backward](#64-inference--viterbi-and-forward-backward)
   6.5 [State Labeling](#65-state-labeling)
   6.6 [GPU Viterbi Decoding](#66-gpu-viterbi-decoding)
   6.7 [Fallback HMM](#67-fallback-hmm)
   6.8 [Outputs](#68-outputs)
7. [Fama-French Factor Model (`factors.py`)](#7-fama-french-factor-model-factorspy)
   7.1 [Model](#71-model)
   7.2 [Factor Data](#72-factor-data)
   7.3 [Rolling OLS](#73-rolling-ols)
   7.4 [GPU-Accelerated Batched OLS](#74-gpu-accelerated-batched-ols)
   7.5 [Outputs](#75-outputs)
8. [Pipeline Orchestration (`pipeline.py`)](#8-pipeline-orchestration-pipelinepy)
9. [Walk-Forward Safety](#9-walk-forward-safety)
10. [GPU Acceleration Summary](#10-gpu-acceleration-summary)
11. [Environment Variables](#11-environment-variables)
12. [References](#12-references)

---

## 1. Module Overview

```
forecasting/
├── __init__.py       Public API — all exports live here
├── config.py         GARCHConfig, RegimeConfig, FamaFrenchConfig, ForecastConfig
├── garch.py          GARCH(1,1) volatility forecaster + GARCHParams
├── regime.py         Gaussian HMM regime detector + fallback NumPy HMM
├── factors.py        Fama-French factor loader + rolling OLS estimator
└── pipeline.py       ForecastingPipeline orchestrator + ForecastBundle
```

**Design principles:**

| Principle | Implementation |
|---|---|
| Walk-forward safety | Every component is **fit on training data only**; inference on test data uses frozen parameters |
| Modularity | Each component (`GARCHForecaster`, `RegimeDetector`, `FamaFrenchFactors`) is fully independent |
| Graceful degradation | All three components have CPU fallbacks that require no GPU or optional dependencies |
| RL integration | `ForecastBundle.rl_state_extension` produces a ready-to-use quarterly DataFrame that appends to the `data` module's technical feature matrix |

---

## 2. Quick Start

```python
from data import DataPipeline, DataConfig
from forecasting import ForecastingPipeline, ForecastConfig

# Build data folds first
data_pipeline = DataPipeline(DataConfig()).run()

# Initialize and load Fama-French data (one-time download, cached)
fcst = ForecastingPipeline(ForecastConfig())
fcst.load_factors()

# Run a single fold
fold_0 = data_pipeline.get_fold(0)
bundle = fcst.run_fold(fold_0)

# Inspect outputs
print(bundle)
# ForecastBundle(fold=0, train=12Q, test=4Q, n_assets_vol=30)

print(bundle.train_vol)            # annualized GARCH volatility (quarters × tickers)
print(bundle.train_regime)         # regime_label, p_bear, p_sideways, p_bull
print(bundle.train_betas)          # MultiIndex: (alpha_ann/beta_mkt/..., ticker)
print(bundle.rl_state_extension)   # flat DataFrame — all forecast features combined

# Run all folds at once
bundles = fcst.run_all_folds(data_pipeline)
```

---

## 3. Architecture

Each `ForecastingPipeline.run_fold()` call produces one `ForecastBundle` by executing three components sequentially on a `WalkForwardFold` from the `data` module.

```
WalkForwardFold (from data module)
       │
       ├── train_prices / train_dates
       └── test_prices  / test_dates
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                   ForecastingPipeline.run_fold()                 │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ GARCHForecaster │  │  RegimeDetector  │  │ FamaFrench     │  │
│  │                 │  │                  │  │ Factors        │  │
│  │ fit(train)      │  │ fit(train)       │  │ rolling_betas  │  │
│  │ forecast(train) │  │ decode(train)    │  │ (train + test) │  │
│  │ forecast(test)* │  │ decode(test)*    │  │                │  │
│  └────────┬────────┘  └────────┬─────────┘  └───────┬────────┘  │
│           │                    │                     │           │
│           ▼                    ▼                     ▼           │
│     train_vol            train_regime           train_betas      │
│     test_vol             test_regime            test_betas       │
│     garch_params         regime_transitions                      │
│                                                                  │
│  * Parameters frozen from training fit; no re-estimation        │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
                     ForecastBundle
                           │
                           └── .rl_state_extension  ──► RL state vector
```

---

## 4. Configuration (`config.py`)

All parameters are organized in four dataclasses composed into a single `ForecastConfig`.

```python
@dataclass
class ForecastConfig:
    garch   : GARCHConfig        # GARCH model hyperparameters
    regime  : RegimeConfig       # HMM hyperparameters
    factors : FamaFrenchConfig   # Fama-French model hyperparameters
```

### GARCH configuration

| Parameter | Default | Description |
|---|---|---|
| `p` | `1` | ARCH lag order |
| `q` | `1` | GARCH lag order |
| `dist` | `"normal"` | Innovation distribution: `'normal'`, `'t'`, `'skewt'` |
| `rescale` | `True` | Multiply returns by 100 before fitting (improves numerical conditioning) |
| `horizon` | `63` | Steps ahead to forecast (~1 quarter of trading days) |
| `annualize` | `True` | Multiply daily vol forecast by $\sqrt{252}$ |
| `min_obs` | `252` | Minimum observations required to fit (assets below this get cross-sectional median) |
| `use_gpu` | `None` | `None` = auto-detect CuPy; `True`/`False` to force |
| `refit_every_n_quarters` | `4` | Re-estimate parameters every N quarters; variance recursion updated every quarter |

### Regime configuration

| Parameter | Default | Description |
|---|---|---|
| `n_states` | `3` | Number of latent regimes (bull / sideways / bear) |
| `n_iter` | `200` | Maximum Baum-Welch EM iterations |
| `covariance_type` | `"diag"` | Emission covariance structure |
| `random_state` | `42` | Seed for reproducible initialization |
| `lookback_window` | `252` | Recent trading days used at inference time |
| `use_gpu` | `None` | Auto-detect CuPy for Viterbi decoding |

### Fama-French configuration

| Parameter | Default | Description |
|---|---|---|
| `n_factors` | `3` | `3` = FF3 (Market, SMB, HML); `5` = FF5 (adds RMW, CMA) |
| `rolling_window` | `63` | Rolling OLS window in trading days (~1 quarter) |
| `min_obs_fraction` | `0.8` | Minimum fraction of window with valid data |
| `cache_dir` | `"../data/cache/ff_factors"` | Local parquet cache directory |
| `annualize_alpha` | `True` | Multiply daily rolling alpha by 252 |
| `use_gpu` | `None` | Auto-detect CuPy for batched OLS |

---

## 5. GARCH Volatility Forecasting (`garch.py`)

### 5.1 Model

The GARCH(1,1) model [1,2] specifies that the conditional variance $h_t$ evolves as:

$$h_t = \omega + \alpha \varepsilon_{t-1}^2 + \beta h_{t-1}$$

where:
- $\varepsilon_t = r_t - \mu$ is the demeaned return
- $\omega > 0$, $\alpha \geq 0$, $\beta \geq 0$ are the model parameters
- $\alpha + \beta < 1$ ensures covariance stationarity

Three key diagnostics are derived from the fitted parameters:

| Diagnostic | Formula | Interpretation |
|---|---|---|
| **Persistence** | $\alpha + \beta$ | Fraction of a volatility shock that persists to the next period. Values near 1 → long-memory vol |
| **Long-run variance** | $\omega / (1 - \alpha - \beta)$ | Unconditional (steady-state) variance the process reverts to |
| **Half-life** | $\ln(0.5) / \ln(\alpha + \beta)$ | Days for a variance shock to decay by 50% |

**Example:** With $\alpha=0.08$, $\beta=0.88$, persistence is 0.96, half-life is ~17 trading days — typical for equity volatility.

### 5.2 Estimation

`GARCHForecaster.fit()` estimates parameters per asset via **maximum likelihood estimation (MLE)** with the Gaussian log-likelihood:

$$\mathcal{L} = -\frac{1}{2} \sum_{t=1}^{T} \left[ \ln h_t + \frac{\varepsilon_t^2}{h_t} \right]$$

**Two-path implementation:**

```
Primary path: arch library (Kevin Sheppard)
  ├── arch_model(returns, vol='GARCH', p=1, q=1, dist=cfg.dist)
  ├── L-BFGS-B optimizer, analytical gradients
  └── Returns GARCHParams with: omega, alpha, beta,
      last_var (h_T), last_resid (ε_T), converged flag

Fallback path: pure SciPy MLE (no arch dependency)
  ├── scipy.optimize.minimize with L-BFGS-B
  ├── Explicit recursion for h_t in the objective
  └── Same GARCHParams output
```

Assets with fewer than `min_obs` observations receive the **cross-sectional median** forecast to avoid degenerate fits on thin data.

### 5.3 GPU-Accelerated Batch Forecasting

Once parameters are estimated (per-asset, on CPU), the **multi-step variance recursion** and **aggregation** are accelerated with CuPy, which is the computationally dominant step for large universes.

For each asset $i$ and forecast step $s \in \{1, \ldots, H\}$:

$$h_{i,s} = \omega_i + \left(\alpha_i + \beta_i\right) h_{i,s-1} \quad \text{for } s > 1$$
$$h_{i,1} = \omega_i + \alpha_i \varepsilon_{i,0}^2 + \beta_i h_{i,T}$$

The **H-step average variance** is used as the quarterly forecast:

$$\hat{\sigma}^2_{i,\text{quarterly}} = \frac{1}{H} \sum_{s=1}^{H} h_{i,s}$$

This is annualized as $\hat{\sigma}_{i} = \sqrt{\hat{\sigma}^2_{i,\text{quarterly}} \cdot 252}$.

All $N$ assets are processed in a single vectorized GPU kernel — no Python loop over assets:

```
omega, alpha, beta, h0, e0  ← stacked numpy arrays, shape (N,)
        │
        ▼  cp.asarray(...)  [host → device, one transfer]
        │
   for s in range(H):        ← H iterations, fully vectorized over N
       h = omega + (alpha + beta) * h_prev  [element-wise, shape (N,)]
       cumulative += h
        │
        ▼  .get()  [device → host, one transfer]
mean_var = cumulative / H    ← annualized volatility per asset
```

### 5.4 Walk-Forward Refitting Schedule

GARCH parameters are **not re-estimated every quarter** — doing so would be both slow and statistically unnecessary, since volatility parameters evolve slowly. The `refit_every_n_quarters` config controls the schedule:

```
Quarter: Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8  Q9  ...
         ▼               ▼               ▼
       [FIT]           [FIT]           [FIT]   ← full MLE (refit_every=4)
         │    │    │    │    │    │    │
       [recursion only — parameters frozen, new h_t values propagated]
```

Between refits, only the **variance path** is updated using new return observations. This delivers nearly identical accuracy at a fraction of the computation cost.

### 5.5 Outputs

`forecast_quarterly()` returns a `DataFrame[rebalance_dates × tickers]` of annualized volatility.

`parameter_summary()` returns a `DataFrame` with one row per ticker containing: `omega`, `alpha`, `beta`, `persistence`, `long_run_vol`, `half_life`, `converged`, `log_likelihood`.

---

## 6. Market Regime Detection (`regime.py`)

### 6.1 Model

A **Gaussian Hidden Markov Model (HMM)** [3] models the market as switching between $K$ latent states (regimes). At each time $t$, the hidden state $z_t \in \{0, 1, \ldots, K-1\}$ evolves according to a transition matrix, and the observable features $\mathbf{x}_t$ are drawn from a Gaussian emission distribution:

$$P(z_t = j \mid z_{t-1} = i) = A_{ij}$$

$$P(\mathbf{x}_t \mid z_t = k) = \mathcal{N}(\mathbf{x}_t;\, \boldsymbol{\mu}_k,\, \boldsymbol{\Sigma}_k)$$

With three states (the default), the model captures the three canonical equity market regimes:

| State (sorted) | Label | Characteristics |
|---|---|---|
| 0 | **bear** | Negative mean return, high cross-sectional volatility |
| 1 | **sideways** | Near-zero mean return, moderate dispersion |
| 2 | **bull** | Positive mean return, low dispersion |

### 6.2 Feature Construction

Rather than feeding individual asset returns into the HMM (which would make the model sensitive to universe composition), the `RegimeDetector` constructs **two market-level features** for each trading day:

$$\text{feature}_1 = \frac{1}{N} \sum_{i=1}^{N} r_{i,t} \quad \text{(equally-weighted return — market direction)}$$

$$\text{feature}_2 = \text{std}_N(r_{i,t}) \quad \text{(cross-sectional dispersion — risk)}$$

These features are **z-score normalized** using training-set statistics before being passed to the HMM, ensuring consistent emission distributions across folds.

### 6.3 Fitting

`RegimeDetector.fit()` runs the **Baum-Welch EM algorithm** [4] on the full training history:

```
E-step: Forward-backward algorithm
  → α_t(i) = P(x_1,...,x_t, z_t=i)    (forward variable)
  → β_t(i) = P(x_{t+1},...,x_T | z_t=i)  (backward variable)
  → γ_t(i) = P(z_t=i | x_1,...,x_T)  (smoothed posterior)
  → ξ_t(i,j) = P(z_t=i, z_{t+1}=j | x_1,...,x_T)

M-step: Re-estimate A, μ_k, Σ_k from sufficient statistics
  → A_ij = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
  → μ_k  = Σ_t γ_t(k) x_t / Σ_t γ_t(k)
  → Σ_k  = Σ_t γ_t(k)(x_t−μ_k)(x_t−μ_k)' / Σ_t γ_t(k)

Repeat until |ΔLL| < 1e-4 or n_iter reached.
```

**Primary path:** `hmmlearn.hmm.GaussianHMM` (Cython-optimized Baum-Welch).
**Fallback path:** `_NumpyGaussianHMM` — a fully self-contained NumPy implementation with k-means initialization (see Section 6.7).

### 6.4 Inference — Viterbi and Forward-Backward

Two inference algorithms are provided, suited for different use cases:

| Algorithm | Method | Output | Use case |
|---|---|---|---|
| **Forward-Backward** [4] | `decode_latest()` | Smoothed posteriors $P(z_t \mid \mathbf{x}_{1:T})$ | Soft regime probabilities for RL state vector |
| **Viterbi** [5] | `decode_sequence()` | Most likely state sequence $z_1^*, \ldots, z_T^*$ | Visualizing historical regime transitions |

`decode_latest()` is the primary method used by the pipeline: it returns the posterior probability vector $[p_{\text{bear}}, p_{\text{sideways}}, p_{\text{bull}}]$ at the most recent quarter-end, allowing the RL agent to act on **regime uncertainty** rather than a hard assignment.

### 6.5 State Labeling

After fitting, HMM states are permuted to a canonical ordering that is consistent across folds and initializations. States are **sorted by their mean market-return emission**:

```
Raw HMM states (arbitrary ordering from EM)
   State 2: μ_return = +0.0003   ──► sorted index 2 = "bull"
   State 0: μ_return = -0.0001   ──► sorted index 1 = "sideways"
   State 1: μ_return = -0.0009   ──► sorted index 0 = "bear"
```

This deterministic relabeling ensures that `p_bull` always refers to the high-return state regardless of which random initialization EM converged from.

### 6.6 GPU Viterbi Decoding

For the Viterbi decoding path, a CuPy implementation accelerates the log-domain trellis forward pass:

$$\delta_t(j) = \max_i \left[ \delta_{t-1}(i) + \log A_{ij} \right] + \log P(\mathbf{x}_t \mid z_t = j)$$

The recurrence is computed on-device; only the final backtracked state sequence is transferred back to the host. Falls back silently to CPU (`hmmlearn.predict`) if any CuPy operation fails.

### 6.7 Fallback HMM

When `hmmlearn` is not installed, `_NumpyGaussianHMM` provides an equivalent implementation:

- **Initialization:** k-means clustering for `means_`, empirical per-cluster variances for `covars_`
- **E-step:** Scaled forward-backward with per-step log-scaling to prevent underflow
- **M-step:** Vectorized $\xi$ computation via `np.einsum`; convergence check on $|\Delta \text{LL}| < 10^{-4}$
- **Inference:** Both Viterbi decoding and smoothed posteriors (`predict_proba`)

For production use, `hmmlearn` is strongly recommended for better numerical stability and faster convergence on long series.

### 6.8 Outputs

`forecast_quarterly()` returns a `DataFrame[rebalance_dates]` with columns:

| Column | Type | Description |
|---|---|---|
| `regime_label` | `str` | `'bull'`, `'bear'`, or `'sideways'` |
| `regime_index` | `int` | Sorted state index (0 = bear, 1 = sideways, 2 = bull) |
| `p_bear` | `float` | Posterior probability of bear state |
| `p_sideways` | `float` | Posterior probability of sideways state |
| `p_bull` | `float` | Posterior probability of bull state |

`transition_matrix()` returns the learned $A$ matrix as a `DataFrame` with state labels as row/column names.

---

## 7. Fama-French Factor Model (`factors.py`)

### 7.1 Model

The Fama-French factor model [6,7] explains the excess return of asset $i$ as a linear combination of common risk factors:

**3-Factor model (FF3) [6]:**

$$r_{i,t} - r_{f,t} = \alpha_i + \beta^{\text{mkt}}_i \cdot (r_{m,t} - r_{f,t}) + \beta^{\text{smb}}_i \cdot \text{SMB}_t + \beta^{\text{hml}}_i \cdot \text{HML}_t + \varepsilon_{i,t}$$

**5-Factor model (FF5) [7]:**

$$r_{i,t} - r_{f,t} = \alpha_i + \beta^{\text{mkt}}_i \cdot \text{MKT}_t + \beta^{\text{smb}}_i \cdot \text{SMB}_t + \beta^{\text{hml}}_i \cdot \text{HML}_t + \beta^{\text{rmw}}_i \cdot \text{RMW}_t + \beta^{\text{cma}}_i \cdot \text{CMA}_t + \varepsilon_{i,t}$$

**Factor definitions:**

| Factor | Full name | Description |
|---|---|---|
| MKT-RF | Market excess return | Broad market premium over the risk-free rate |
| SMB | Small-Minus-Big | Size premium: small-cap returns minus large-cap returns |
| HML | High-Minus-Low | Value premium: high book-to-market minus low book-to-market |
| RMW | Robust-Minus-Weak | Profitability premium: high vs. low operating profitability *(FF5 only)* |
| CMA | Conservative-Minus-Aggressive | Investment premium: low vs. high investment firms *(FF5 only)* |

Betas are estimated with a **rolling OLS window** (default: 63 trading days, ≈ 1 quarter) so that exposures reflect the current factor sensitivities rather than static full-history estimates.

### 7.2 Factor Data

`FamaFrenchDataLoader` downloads daily factor returns **free of charge** from [Ken French's Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/) and caches them locally as parquet files.

```
Ken French Data Library
        │
        ▼  HTTP GET (requests library)
F-F_Research_Data_Factors_daily_CSV.zip  [FF3]
F-F_Research_Data_5_Factors_2x3_daily_CSV.zip  [FF5]
        │
        ▼  _parse_csv()
         ├── Skip header / footer text blocks
         ├── Detect data rows: 8-digit YYYYMMDD date
         ├── Stop at annual-summary section (4-digit year)
         └── Convert percent → decimal (÷ 100)
        │
        ▼
data/cache/ff_factors/ff{3|5}_daily.parquet
```

The cache is read on all subsequent calls, avoiding redundant network traffic. `force_refresh=True` re-downloads unconditionally.

### 7.3 Rolling OLS

For each rolling window ending at time $t$, the OLS system is:

$$\mathbf{Y}_t = \mathbf{X}_t \boldsymbol{\beta} + \boldsymbol{\varepsilon}_t$$

where $\mathbf{X}_t \in \mathbb{R}^{W \times (K+1)}$ includes an intercept column and $\mathbf{Y}_t \in \mathbb{R}^{W \times N}$ contains all $N$ asset excess returns. The solution is:

$$\hat{\boldsymbol{\beta}}_t = (\mathbf{X}_t^\top \mathbf{X}_t)^{-1} \mathbf{X}_t^\top \mathbf{Y}_t$$

solved via `np.linalg.lstsq` (CPU) or `cp.linalg.solve` on the normal equations (GPU). The in-window $R^2$ is computed per asset:

$$R^2_{i,t} = 1 - \frac{\sum_\tau (y_{i,\tau} - \hat{y}_{i,\tau})^2}{\sum_\tau (y_{i,\tau} - \bar{y}_{i,\tau})^2}$$

### 7.4 GPU-Accelerated Batched OLS

The CPU implementation loops over all $T - W + 1$ time steps — expensive for long histories. The GPU implementation eliminates this loop using **cumulative sum prefix arrays**:

```
Step 1: Compute outer products for every time step
  xx[t] = x[t] ⊗ x[t]    shape (T, K+1, K+1)   ← einsum, one pass
  xy[t] = x[t] ⊗ y[t]    shape (T, K+1, N)      ← einsum, one pass

Step 2: Prefix sums
  cum_xx = cumsum(xx, axis=0)   shape (T, K+1, K+1)
  cum_xy = cumsum(xy, axis=0)   shape (T, K+1, N)

Step 3: Each window via subtraction (O(1) per window)
  XtX[t] = cum_xx[t] - cum_xx[t-W]   ← no Python loop needed
  XtY[t] = cum_xy[t] - cum_xy[t-W]

Step 4: Batch solve
  β[t] = solve(XtX[t], XtY[t])   ← cp.linalg.solve on all windows at once
```

For $N=30$ assets, $T=3750$ days (15 years), $K=3$ factors: expected **8–15× speedup** over the CPU loop.

### 7.5 Outputs

`rolling_betas()` returns a `DataFrame` with MultiIndex columns `(output_name, ticker)`:

| Output name | Description |
|---|---|
| `alpha_ann` | Annualized rolling intercept (Jensen's alpha) |
| `beta_mkt` | Rolling market beta |
| `beta_smb` | Rolling size beta |
| `beta_hml` | Rolling value beta |
| `beta_rmw` | Rolling profitability beta *(FF5 only)* |
| `beta_cma` | Rolling investment beta *(FF5 only)* |
| `r_squared` | In-window $R^2$ |

`forecast_quarterly()` samples these outputs at each quarter-end rebalancing date.

---

## 8. Pipeline Orchestration (`pipeline.py`)

`ForecastingPipeline` sequences the three components and packages results into a `ForecastBundle`.

### `ForecastBundle` fields

| Field | Shape | Description |
|---|---|---|
| `train_vol` / `test_vol` | `[dates × tickers]` | Annualized GARCH vol forecasts |
| `train_regime` / `test_regime` | `[dates × 5 columns]` | Regime label, index, and probabilities |
| `train_betas` / `test_betas` | `[dates × (output, ticker)]` | Factor exposures |
| `garch_params` | `[tickers × params]` | Fitted GARCH(1,1) parameters from last training fit |
| `regime_transitions` | `[K × K]` | Learned HMM transition matrix (with state labels) |

### `rl_state_extension` property

The `rl_state_extension` property assembles all forecast outputs into a single flat `DataFrame` indexed by all rebalancing dates (train + test). This is the interface consumed by the RL environment:

```
rl_state_extension columns:
  vol_{ticker}              GARCH annualized vol         (one per ticker)
  regime_index              int 0/1/2                    (market-wide)
  p_bear                    HMM posterior probability    (market-wide)
  p_sideways                HMM posterior probability    (market-wide)
  p_bull                    HMM posterior probability    (market-wide)
  alpha_ann_{ticker}        annualized Jensen's alpha    (one per ticker)
  beta_mkt_{ticker}         market beta                  (one per ticker)
  beta_smb_{ticker}         size beta                    (one per ticker)
  beta_hml_{ticker}         value beta                   (one per ticker)
  [beta_rmw_{ticker}]       profitability beta  (FF5 only)
  [beta_cma_{ticker}]       investment beta     (FF5 only)
  r_squared_{ticker}        factor model R²              (one per ticker)
```

---

## 9. Walk-Forward Safety

The forecasting module enforces strict walk-forward discipline at every stage. The boundary rule is: **no test-period information is ever used when fitting models.**

```
Timeline:
  ───────────────────────────────────────────────────────────────────
  Train                          │ Test
  Q1 ── Q2 ── ... ── Q12        │ Q13 ── Q14 ── Q15 ── Q16
  ───────────────────────────────┼───────────────────────────────────
                                 │
  GARCH.fit(train_returns)       │ GARCH.forecast() ← params frozen
  RegimeDetector.fit(train)      │ decode() ← model frozen
  FF factors: rolling OLS        │ rolling OLS (valid — each window
                                 │ uses only past data within W days)
  ───────────────────────────────────────────────────────────────────
```

**Factor model note:** Rolling OLS for test dates is walk-forward safe because each $W$-day window at test date $t$ only sees returns up to $t$ — no future data enters the $W$-day lookback, even when the computation runs over the combined train+test series.

**Normalization note:** The HMM feature normalizer's mean and standard deviation are fitted on training data only (`fit=True`). At inference time, the same statistics are applied to test-period features (`fit=False`).

---

## 10. GPU Acceleration Summary

All three components transparently switch between CuPy (GPU) and NumPy/SciPy (CPU):

| Component | GPU-accelerated operation | Expected speedup (N=30, T=3750) |
|---|---|---|
| `GARCHForecaster` | Batch variance recursion + aggregation over all assets | 3–8× |
| `RegimeDetector` | Viterbi trellis forward pass + log-emission computation | 2–5× |
| `FamaFrenchFactors` | Batched rolling OLS via cumulative outer products | 8–15× |

**Backend selection logic** (same for all three components):

| `use_gpu` | Behavior |
|---|---|
| `None` (default) | Auto-detect: uses GPU if `import cupy` succeeds, else CPU |
| `True` | Require GPU; raises `RuntimeError` if CuPy not installed |
| `False` | Force CPU regardless of hardware |

CuPy installation: `pip install cupy-cuda12x` (adjust CUDA version as needed).

---

## 11. Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HTTP_PROXY` / `HTTPS_PROXY` | No | Proxy settings forwarded to Ken French Data Library downloads |
| `REQUESTS_CA_BUNDLE` | No | Custom CA bundle path for corporate TLS inspection environments |

No API keys are required. Ken French's Data Library is publicly available without authentication.

---

## 12. References

[1] Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica*, 50(4), 987–1007.
Original introduction of the ARCH model; the foundational paper for the GARCH family.

[2] Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroscedasticity." *Journal of Econometrics*, 31(3), 307–327.
Extends ARCH to GARCH(p,q), introducing the GARCH(1,1) specification used throughout this module.

[3] Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE*, 77(2), 257–286.
Canonical tutorial on HMMs, covering the Baum-Welch EM algorithm and the Viterbi decoding algorithm implemented in `regime.py`.

[4] Baum, L. E., Petrie, T., Soules, G., and Weiss, N. (1970). "A Maximization Technique Occurring in the Statistical Analysis of Probabilistic Functions of Markov Chains." *The Annals of Mathematical Statistics*, 41(1), 164–171.
The original Baum-Welch EM algorithm used by `_NumpyGaussianHMM` and `hmmlearn`.

[5] Viterbi, A. J. (1967). "Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm." *IEEE Transactions on Information Theory*, 13(2), 260–269.
Original Viterbi algorithm; the log-domain version implemented in `_NumpyGaussianHMM.predict()` and `_viterbi_gpu()`.

[6] Fama, E. F. and French, K. R. (1993). "Common Risk Factors in the Returns on Stocks and Bonds." *Journal of Financial Economics*, 33(1), 3–56.
Introduces the 3-factor model (MKT, SMB, HML) used as the default in `FamaFrenchFactors`.

[7] Fama, E. F. and French, K. R. (2015). "A Five-Factor Asset Pricing Model." *Journal of Financial Economics*, 116(1), 1–22.
Extends FF3 with profitability (RMW) and investment (CMA) factors; supported via `FamaFrenchConfig(n_factors=5)`.

[8] Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357–384.
Seminal paper on Markov regime-switching models in economics; motivates the HMM regime detection approach.

[9] Sheppard, K. et al. *arch: Autoregressive Conditional Heteroscedasticity models in Python*. [https://arch.readthedocs.io](https://arch.readthedocs.io)
The primary estimation back-end used by `GARCHForecaster._fit_single()`.

[10] Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825–2830.
`sklearn.cluster.KMeans` is used for initialization of `_NumpyGaussianHMM`.

[11] Serafini, L. et al. *hmmlearn: Hidden Markov Models in Python, with scikit-learn like API*. [https://hmmlearn.readthedocs.io](https://hmmlearn.readthedocs.io)
Primary Baum-Welch implementation used by `RegimeDetector` when available.

[12] NVIDIA / CuPy Development Team. *CuPy: NumPy & SciPy-compatible Array Library for GPU-accelerated Computing*. [https://cupy.dev](https://cupy.dev)
GPU acceleration backend used across all three forecasting components.

[13] French, K. R. *Data Library*. Tuck School of Business, Dartmouth College. [https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
Free daily factor return data downloaded by `FamaFrenchDataLoader`.
