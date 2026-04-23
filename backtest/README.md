# `backtest` — QuantAgent-RL Walk-Forward Backtest Module

The `backtest` module evaluates the out-of-sample performance of the RL portfolio agent and compares it against standard investment baselines. It consumes the `RLFoldResult` objects produced by the `rl` module, aggregates per-period returns across all walk-forward folds, and computes a comprehensive set of scalar and rolling performance metrics — many of which are GPU-accelerated via CuPy for fast batch evaluation across multiple strategies.

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Quick Start](#2-quick-start)
3. [Architecture](#3-architecture)
4. [Configuration (`config.py`)](#4-configuration-configpy)
5. [Performance Metrics (`metrics.py`)](#5-performance-metrics-metricspy)
   5.1 [Scalar Metrics](#51-scalar-metrics)
   5.2 [Rolling Metrics](#52-rolling-metrics)
   5.3 [GPU Acceleration](#53-gpu-acceleration)
6. [Backtest Engine (`engine.py`)](#6-backtest-engine-enginepy)
   6.1 [Entry Points](#61-entry-points)
   6.2 [Benchmark Handling](#62-benchmark-handling)
   6.3 [Tear Sheet Construction](#63-tear-sheet-construction)
7. [Result Containers (`report.py`)](#7-result-containers-reportpy)
   7.1 [StrategyReturns](#71-strategyreturns)
   7.2 [TearsheetData](#72-tearsheetdata)
   7.3 [BacktestReport](#73-bactestreport)
8. [Walk-Forward Aggregation](#8-walk-forward-aggregation)
9. [Metric Reference](#9-metric-reference)
10. [GPU Acceleration Summary](#10-gpu-acceleration-summary)
11. [References](#11-references)

---

## 1. Module Overview

```
backtest/
├── __init__.py      Public API — all exports
├── config.py        BacktestConfig
├── metrics.py       MetricsCalculator — scalar + rolling + GPU batch
├── engine.py        BacktestEngine — orchestrates report construction
└── report.py        BacktestReport, TearsheetData, StrategyReturns
```

**Design principles:**

| Principle | Implementation |
|---|---|
| Out-of-sample integrity | Only test-period returns from `RLFoldResult` are evaluated; training-period returns are excluded from all aggregate metrics |
| Fair comparison | All strategies (PPO, equal-weight, buy-and-hold) are evaluated in the same environment with the same cost model |
| GPU-first rolling metrics | O(T·W) rolling windows are replaced with O(T) prefix-sum algorithms on CuPy for batch strategy evaluation |
| Flexible input | Two entry points: one for `RLFoldResult` objects, one for raw return arrays — the module can be used standalone |
| Rich export | Results export to DataFrame, CSV, and JSON; tear sheet time-series export to per-strategy DataFrames |

---

## 2. Quick Start

```python
from backtest import BacktestEngine, BacktestConfig

# ── From RLFoldResult objects (typical workflow) ──────────────────────────
engine = BacktestEngine(BacktestConfig(), prices=prices_df)
report = engine.run_from_fold_results(rl_fold_results)

# Strategy comparison table
print(report.summary())
#                     annualized_return  sharpe  sortino  max_drawdown  information_ratio
# strategy
# ppo                         0.142      0.89     1.14       -0.076               0.41
# equal_weight                0.097      0.63     0.82       -0.114               0.00
# hold                        0.088      0.55     0.71       -0.134                NaN

# Per-fold Sharpe breakdown
print(report.fold_summary(metric="sharpe"))

# One-liner PPO vs equal-weight summary
print(report.ppo_vs_benchmark_summary())
# "PPO outperforms equal-weight: Sharpe 0.890 vs 0.628, Ann. Return 14.2% vs 9.7%, IR = 0.410."

# Export
report.to_csv("results/backtest_summary.csv")
report.to_json("results/backtest_summary.json")

# ── From raw return arrays (standalone usage) ─────────────────────────────
from backtest import StrategyReturns

strategies = [
    StrategyReturns("ppo",          ppo_returns,   dates),
    StrategyReturns("equal_weight", ew_returns,    dates),
]
report = BacktestEngine(BacktestConfig(), prices=prices_df).run(strategies)

# ── Tear sheet data for visualization ─────────────────────────────────────
ts = report.tearsheets["ppo"]
df = ts.to_dataframe()       # DataFrame: cumulative_return, drawdown,
                              #             rolling_sharpe, rolling_alpha, ...
```

---

## 3. Architecture

The backtest module sits downstream of the `rl` module and is the final evaluation layer in the QuantAgent-RL stack:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          BacktestEngine.run_from_fold_results()          │
│                                                                          │
│  Input: list[RLFoldResult]                                               │
│  ├── fold.ppo_metrics.quarterly_returns        ← PPO agent               │
│  ├── fold.equal_weight_metrics.quarterly_returns ← 1/N baseline         │
│  └── fold.hold_metrics.quarterly_returns       ← buy-and-hold baseline  │
│                  │                                                       │
│                  ▼                                                       │
│   ┌──────────────────────────────────────────┐                          │
│   │  Concatenate across folds                 │                          │
│   │  ppo: [fold_0_returns | fold_1_returns | ...]                        │
│   │  ew:  [fold_0_returns | fold_1_returns | ...]                        │
│   │  hold:[fold_0_returns | fold_1_returns | ...]                        │
│   └──────────────────────────────────────────┘                          │
│                  │                                                       │
│                  ▼                                                       │
│   ┌──────────────────────────────────────────┐                          │
│   │  Benchmark returns                        │                          │
│   │  prices[benchmark_ticker].pct_change()    │                          │
│   │  aligned to rebalance dates               │                          │
│   └──────────────────────────────────────────┘                          │
│                  │                                                       │
│                  ▼                                                       │
│   MetricsCalculator                                                      │
│   ├── full_metrics()       → aggregate scalar metrics per strategy       │
│   ├── rolling_metrics()    → rolling time-series per strategy            │
│   ├── batch_rolling_sharpe() → GPU-batched across all S strategies       │
│   └── _per_fold_metrics()  → scalar metrics per fold per strategy        │
│                  │                                                       │
│                  ▼                                                       │
│   BacktestReport                                                         │
│   ├── aggregate_metrics   dict[strategy → dict[metric → float]]          │
│   ├── fold_metrics        dict[strategy → list[dict per fold]]           │
│   └── tearsheets          dict[strategy → TearsheetData]                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Configuration (`config.py`)

All backtest behavior is controlled through `BacktestConfig`:

| Parameter | Default | Description |
|---|---|---|
| `benchmark_ticker` | `"SPY"` | Ticker for market benchmark (alpha, beta, IR calculations) |
| `periods_per_year` | `4` | Rebalancing frequency for annualization (4 = quarterly) |
| `rolling_window` | `12` | Periods in rolling metric windows (12 = 3-year rolling) |
| `risk_free_rate` | `0.04` | Annualized risk-free rate; divided by `periods_per_year` per step |
| `min_periods_rolling` | `6` | Minimum periods before a rolling metric is emitted (avoids NaN-heavy early windows) |
| `use_gpu` | `None` | `True` = force CuPy, `False` = force NumPy, `None` = auto-detect |
| `strategies` | `[]` | Strategy labels to include; empty = all available |
| `include_tax_metrics` | `True` | Compute effective tax drag and after-tax return |
| `include_attribution` | `False` | Compute rolling sector-tilt return attribution |

The per-period risk-free rate is derived as:

$$
r_f^{\text{period}} = \frac{r_f^{\text{annual}}}{\text{periods\_per\_year}}
$$

All annualization multipliers (Sharpe, volatility) use $\sqrt{\text{periods\_per\_year}}$.

---

## 5. Performance Metrics (`metrics.py`)

`MetricsCalculator` is the computational core of the backtest module. It computes two categories of metrics from a 1-D return series.

### 5.1 Scalar Metrics

`full_metrics(returns, benchmark_returns, tax_costs, turnovers)` returns a flat `dict[str, float]` with all scalar metrics:

**Return metrics:**

| Metric | Formula | Notes |
|---|---|---|
| `total_return` | $\prod_t (1 + r_t) - 1$ | Compound total return |
| `annualized_return` | $(1 + R_{\text{total}})^{P/T} - 1$ | $P$ = periods per year, $T$ = periods |
| `volatility` | $\sigma_r \cdot \sqrt{P}$ | Annualized standard deviation |

**Risk-adjusted metrics:**

| Metric | Formula | Description |
|---|---|---|
| `sharpe` | $\frac{\bar{r} - r_f}{\sigma_{r-r_f}} \cdot \sqrt{P}$ | Annualized Sharpe ratio [1] |
| `sortino` | $\frac{\bar{r} - r_f}{\sigma^-_{r-r_f}} \cdot \sqrt{P}$ | Annualized Sortino ratio [2]; downside std only |
| `calmar` | $\frac{r^{\text{ann}}}{\|\text{MDD}\|}$ | Calmar ratio [3]; annualized return / max drawdown |
| `max_drawdown` | $\min_t \frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s}$ | Worst peak-to-trough loss |
| `max_dd_duration` | length of longest drawdown episode | In rebalancing periods |

**Return distribution:**

| Metric | Formula | Description |
|---|---|---|
| `hit_rate` | $\frac{1}{T} \sum_t \mathbf{1}[r_t > 0]$ | Fraction of positive-return periods |
| `avg_win_loss_ratio` | $\bar{r}^+ / \|\bar{r}^-\|$ | Average win / average loss magnitude |
| `skewness` | $\mathbb{E}[(r - \bar{r})^3] / \sigma^3$ | Positive = right-skewed (desirable) |
| `kurtosis` | $\mathbb{E}[(r - \bar{r})^4] / \sigma^4 - 3$ | Excess kurtosis (0 for normal distribution) |

**Benchmark-relative metrics** (requires `benchmark_returns`):

| Metric | Formula | Description |
|---|---|---|
| `alpha` | $(\hat{\alpha}_{\text{OLS}}) \cdot P$ | Annualized Jensen's alpha from CAPM regression [4] |
| `beta` | $\text{Cov}(r, r^B) / \text{Var}(r^B)$ | CAPM market beta |
| `information_ratio` | $\frac{\bar{r} - \bar{r}^B}{\sigma_{r - r^B}} \cdot \sqrt{P}$ | Active return per unit of active risk [5] |
| `tracking_error` | $\sigma_{r - r^B} \cdot \sqrt{P}$ | Annualized active return volatility |
| `excess_return` | $r^{\text{ann}} - r^{B,\text{ann}}$ | Annualized return minus benchmark |

**Tax metrics** (requires `tax_costs`):

| Metric | Formula | Description |
|---|---|---|
| `effective_tax_drag` | $\sum_t c^{\text{tax}}_t \;/\; (T / P)$ | Annualized realized tax cost |
| `after_tax_return` | $r^{\text{ann}} - \text{tax\_drag}$ | Gross annualized return net of taxes |
| `total_tax_cost` | $\sum_t c^{\text{tax}}_t$ | Total realized capital gains taxes over the period |

**Jensen's alpha — CAPM regression:**

The alpha metric runs an OLS regression of excess portfolio returns on excess benchmark returns:

$$
(r_t - r_f) = \alpha + \beta (r_t^B - r_f) + \varepsilon_t
$$

The intercept $\hat{\alpha}$ (annualized by multiplying by $P$) measures the return attributable to manager skill — i.e., the component of portfolio return unexplained by simple market exposure.

### 5.2 Rolling Metrics

`rolling_metrics(returns, benchmark_returns)` returns one time-series array per metric, enabling visualization of how performance evolves over time:

| Metric | Window | Notes |
|---|---|---|
| `rolling_sharpe` | W periods | Annualized; NaN for first `min_periods_rolling - 1` periods |
| `rolling_sortino` | W periods | Downside deviation denominator |
| `rolling_volatility` | W periods | Annualized standard deviation |
| `rolling_drawdown` | W periods | Maximum drawdown within the trailing window |
| `rolling_alpha` | W periods | Rolling Jensen's alpha (requires benchmark) |
| `rolling_beta` | W periods | Rolling CAPM beta (requires benchmark) |

**Rolling window illustration** (W = 12 quarters, quarterly data):

```
t=0   t=1   t=2   ...  t=11   t=12   t=13   ...
 │     │     │          │      │      │
 └─────────────────────────────┘
         First valid window (W=12)
               rolling_sharpe[11] = first non-NaN value

                     └────────────────────┘
                           Window at t=12

                            └────────────────────┘
                                  Window at t=13
```

### 5.3 GPU Acceleration

The key computational bottleneck in multi-strategy backtesting is the **rolling window** calculation: naively, computing rolling Sharpe over $T$ periods with window $W$ for $S$ strategies requires $O(T \cdot W \cdot S)$ operations.

`MetricsCalculator` replaces this with an **O(T) prefix-sum** algorithm:

For rolling Sharpe, the sum of excess returns $\sum_{i=t-W+1}^{t} e_i$ and sum of squared excess returns $\sum_{i=t-W+1}^{t} e_i^2$ over any window are computed in $O(1)$ per step using cumulative sums:

$$
\text{SumEx}_{[s,t]} = C_t - C_{s-1}, \quad \text{where } C_t = \sum_{i=0}^{t} e_i
$$

This gives:

$$
\bar{e}_{[s,t]} = \frac{\text{SumEx}_{[s,t]}}{W}, \quad
\text{Var}_{[s,t]} = \frac{\sum e_i^2}{W} - \bar{e}^2, \quad
\text{Sharpe}_{[s,t]} = \frac{\bar{e}}{\sqrt{\text{Var}}} \cdot \sqrt{P}
$$

**Batch GPU rolling Sharpe** (`batch_rolling_sharpe`) processes all $S$ strategies simultaneously on the GPU in a single CuPy operation:

```python
# GPU batch path (CuPy)
r_gpu = xp.asarray(returns_matrix, dtype=xp.float64)   # shape (T, S)
cum_ex  = xp.cumsum(r_gpu - rf, axis=0)                # prefix sum of excess returns
cum_ex2 = xp.cumsum((r_gpu - rf)**2, axis=0)           # prefix sum of squared excess

# For each window ending at t:
mean_ex = (cum_ex[t] - cum_ex[start-1]) / W
var_ex  = (cum_ex2[t] - cum_ex2[start-1]) / W - mean_ex**2
sharpe[t] = mean_ex / sqrt(var_ex) * sqrt(ppy)         # computed for all S at once
```

This delivers an **8–15× speedup** over per-strategy CPU loops when evaluating $S \geq 5$ strategies simultaneously.

---

## 6. Backtest Engine (`engine.py`)

`BacktestEngine` orchestrates the full report construction. It combines metric computation, benchmark alignment, per-fold breakdown, and tear sheet assembly into a single `.run()` call.

### 6.1 Entry Points

**Entry point 1: `run_from_fold_results(fold_results)`** — typical workflow

```
list[RLFoldResult]
      │
      ▼
  For each fold:
    extract quarterly_returns for ppo / equal_weight / hold
    construct StrategyReturns(name, returns, dates)

      │
      ▼
  Concatenate fold-level StrategyReturns into full out-of-sample series
  (fold 0 test + fold 1 test + ... = full backtest history)

      │
      ▼
  Delegate to run(strategies, fold_results)
```

**Entry point 2: `run(strategies, fold_results=None)`** — direct / standalone

```
list[StrategyReturns]
      │
      ▼
  1. Compute benchmark returns (aligned to strategy dates)
  2. Compute aggregate scalar metrics per strategy
  3. Compute per-fold scalar metrics (if fold_results provided)
  4. Build tear sheets (cumulative returns, drawdown, rolling metrics)
  5. Return BacktestReport
```

### 6.2 Benchmark Handling

The benchmark return series is computed from the `prices` DataFrame passed to the engine:

```
prices["SPY"]
      │
      ▼  pct_change()
      │
      ▼  align to union of all strategy rebalance dates (asof lookup)
      │
benchmark_returns: pd.Series[date → float]
```

If the benchmark ticker is not in the `prices` DataFrame, benchmark-relative metrics (`alpha`, `beta`, `information_ratio`, `tracking_error`) are omitted from the report and a warning is logged. The engine degrades gracefully — all non-benchmark metrics are still computed.

### 6.3 Tear Sheet Construction

For each strategy, a `TearsheetData` object is assembled:

```
StrategyReturns.returns
      │
      ├── cumulative_return_series()  → cumulative return index (1.0 = start)
      │
      ├── drawdown_series()           → drawdown path (≤ 0 at each period)
      │
      ├── batch_rolling_sharpe()      → GPU-batched across all strategies
      │
      ├── rolling_metrics()           → rolling_alpha, rolling_volatility,
      │                                 rolling_drawdown, rolling_beta
      │
      └── full_metrics()              → all scalar metrics for the panel
```

The `batch_rolling_sharpe` call processes **all strategies simultaneously** before the per-strategy loop, so GPU memory is loaded once rather than once per strategy.

---

## 7. Result Containers (`report.py`)

### 7.1 `StrategyReturns`

Minimal input contract for one strategy:

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Strategy label (e.g. `'ppo'`, `'equal_weight'`) |
| `returns` | `ndarray (T,)` | Per-period portfolio returns |
| `dates` | `DatetimeIndex (T,)` | Quarter-end dates for each return |
| `weights` | `list[ndarray] \| None` | Per-period portfolio weight vectors |
| `tax_costs` | `ndarray (T,) \| None` | Per-period realized tax costs |
| `turnovers` | `ndarray (T,) \| None` | Per-period one-way L1 turnover |
| `fold_idx` | `int \| None` | Walk-forward fold the data came from |

### 7.2 `TearsheetData`

Container for all time-series data needed to render a single-strategy tear sheet. All arrays have the same length $T$.

| Field | Type | Description |
|---|---|---|
| `cumulative_returns` | `ndarray (T,)` | Cumulative return index; starts at 1.0 |
| `drawdown` | `ndarray (T,)` | Drawdown path ($\leq 0$) |
| `rolling_sharpe` | `ndarray (T,)` | Rolling Sharpe; NaN in early periods |
| `rolling_alpha` | `ndarray (T,) \| None` | Rolling Jensen's alpha |
| `rolling_volatility` | `ndarray (T,) \| None` | Rolling annualized volatility |
| `benchmark_cumulative` | `ndarray (T,) \| None` | Benchmark cumulative return for overlay |
| `scalar_metrics` | `dict[str, float]` | Full-period scalar metrics dictionary |

`TearsheetData.to_dataframe()` returns a tidy `pd.DataFrame` indexed by date, suitable for direct plotting:

```python
df = report.tearsheets["ppo"].to_dataframe()
df.plot(subplots=True)
```

### 7.3 `BacktestReport`

The primary output of `BacktestEngine.run()`:

**Data attributes:**

| Attribute | Type | Description |
|---|---|---|
| `aggregate_metrics` | `dict[str, dict[str, float]]` | Full-series metrics per strategy |
| `fold_metrics` | `dict[str, list[dict[str, float]]]` | Per-fold metrics per strategy |
| `tearsheets` | `dict[str, TearsheetData]` | Visualization data per strategy |
| `benchmark_name` | `str` | Benchmark ticker used |
| `n_folds` | `int` | Number of walk-forward folds |
| `config_summary` | `dict` | Snapshot of `BacktestConfig` |

**Key methods:**

| Method | Returns | Description |
|---|---|---|
| `summary()` | `pd.DataFrame` | Compact strategy comparison (key metrics as rows) |
| `fold_summary(metric)` | `pd.DataFrame` | Per-fold values for one metric across strategies |
| `to_dataframe()` | `pd.DataFrame` | All aggregate metrics (wide format) |
| `to_csv(path)` | — | Save aggregate metrics to CSV |
| `to_json(path)` | — | Serialize scalar metrics to JSON (arrays excluded) |
| `best_strategy(metric)` | `str` | Strategy with highest value for `metric` |
| `ppo_vs_benchmark_summary()` | `str` | Human-readable PPO vs equal-weight one-liner |

**Example `summary()` output:**

```
                    annualized_return  sharpe  sortino  calmar  max_drawdown  information_ratio  alpha   beta  effective_tax_drag  avg_turnover
strategy
ppo                         0.142      0.89     1.14    1.87       -0.076              0.41       0.031  0.87        0.012             1.23
equal_weight                0.097      0.63     0.82    0.86       -0.114              0.00        NaN   1.00         NaN             0.00
hold                        0.088      0.55     0.71    0.65       -0.134               NaN        NaN   1.00         NaN             0.00
```

---

## 8. Walk-Forward Aggregation

A key design decision in the backtest module is **how fold-level returns are combined** for aggregate metric computation.

Rather than averaging per-fold metrics (which would treat each fold equally regardless of length), the engine **concatenates** the out-of-sample return series across all folds into a single continuous series before computing aggregate metrics:

```
Walk-forward folds (1-year test windows):

Fold 0:  [2013-Q1, 2013-Q2, 2013-Q3, 2013-Q4]   ← 4 quarterly returns
Fold 1:  [2014-Q1, 2014-Q2, 2014-Q3, 2014-Q4]   ← 4 quarterly returns
Fold 2:  [2015-Q1, 2015-Q2, 2015-Q3, 2015-Q4]   ← 4 quarterly returns
...
Fold N:  [...]

Concatenated out-of-sample series (for aggregate metrics):
  [2013-Q1, 2013-Q2, ..., 2013-Q4, 2014-Q1, ..., 2015-Q4, ..., ...]
   └────── fold 0 ──────┘└────── fold 1 ──────┘└── fold 2 ──┘

Aggregate Sharpe = f( [all concatenated returns] )
```

This approach:
1. **Preserves temporal ordering** — the concatenated series is a real chronological return history
2. **Weights folds by length** — longer folds have more influence on aggregate metrics than shorter ones
3. **Avoids double-counting** — training-period returns are never included (only `fold.test_dates`)

Per-fold metrics are separately available via `fold_metrics` and `fold_summary(metric)` for diagnosing whether performance is consistent across regimes.

**Example `fold_summary("sharpe")` output:**

```
fold  ppo     equal_weight  hold
   0  0.91        0.67      0.58
   1  1.08        0.72      0.63
   2  0.74        0.55      0.49
   3  0.92        0.61      0.55
   4  0.83        0.64      0.51
```

Consistency across folds (low variance in the `ppo` column) is a stronger signal of genuine skill than a single high aggregate Sharpe.

---

## 9. Metric Reference

Complete list of all metrics computed by `full_metrics()`:

| Metric key | Unit | Higher is better? | Benchmark required |
|---|---|---|---|
| `total_return` | fraction | ✓ | |
| `annualized_return` | fraction/year | ✓ | |
| `volatility` | fraction/year | ✗ | |
| `sharpe` | dimensionless | ✓ | |
| `sortino` | dimensionless | ✓ | |
| `calmar` | dimensionless | ✓ | |
| `max_drawdown` | fraction (≤ 0) | ✗ (less negative = better) | |
| `max_dd_duration` | periods | ✗ | |
| `hit_rate` | fraction [0,1] | ✓ | |
| `avg_win_loss_ratio` | dimensionless | ✓ | |
| `skewness` | dimensionless | ✓ (positive preferred) | |
| `kurtosis` | dimensionless | ✗ (lower preferred) | |
| `alpha` | fraction/year | ✓ | ✓ |
| `beta` | dimensionless | — (neutral ~ 1) | ✓ |
| `information_ratio` | dimensionless | ✓ | ✓ |
| `tracking_error` | fraction/year | ✗ | ✓ |
| `benchmark_return` | fraction/year | reference | ✓ |
| `excess_return` | fraction/year | ✓ | ✓ |
| `total_tax_cost` | fraction | ✗ | |
| `effective_tax_drag` | fraction/year | ✗ | |
| `after_tax_return` | fraction/year | ✓ | |
| `avg_turnover` | fraction/period | ✗ | |
| `total_turnover` | fraction | ✗ | |

---

## 10. GPU Acceleration Summary

| Operation | GPU path | CPU fallback | Speedup (S ≥ 5 strategies) |
|---|---|---|---|
| `batch_rolling_sharpe` | CuPy prefix-sum (T, S) | NumPy loop | 8–15× |
| `rolling_sharpe` (single) | CuPy prefix-sum (T,) | NumPy | 2–4× |
| `rolling_volatility` | CuPy prefix-sum (T,) | NumPy | 2–4× |
| `rolling_drawdown` | CuPy cumprod (per window) | NumPy | 2–4× |
| `rolling_alpha` | CuPy rolling OLS (W×2 system) | NumPy | 1–2× |
| Scalar metrics | NumPy (T ≤ 200 is fast enough) | NumPy | — |

**Auto-detection:** setting `use_gpu=None` (default) automatically uses CuPy when available and falls back to NumPy silently. Only `use_gpu=True` raises an error if no GPU is found.

**Memory:** for typical walk-forward backtests ($T \leq 200$ quarterly periods, $S = 3$ strategies), all data fits in a few MB — GPU is not strictly necessary but provides a material speedup when evaluating many strategies or running parameter sweeps.

---

## 11. References

[1] Sharpe, W.F. (1966). "Mutual Fund Performance." *Journal of Business*, 39(1), 119–138.
Introduces the Sharpe ratio as a reward-to-variability measure. The annualized Sharpe computed here follows the standard convention: $(\bar{r}_{\text{excess}} / \sigma_{\text{excess}}) \cdot \sqrt{P}$.

[2] Sortino, F.A. and van der Meer, R. (1991). "Downside Risk." *Journal of Portfolio Management*, 17(4), 27–31.
Motivates using downside deviation (standard deviation of negative excess returns) as the risk denominator, penalizing only harmful volatility.

[3] Young, T.W. (1991). "Calmar Ratio: A Smoother Tool." *Futures Magazine*, October 1991.
Introduces the Calmar ratio (annualized return / maximum drawdown) as a measure of return per unit of worst-case loss.

[4] Jensen, M.C. (1968). "The Performance of Mutual Funds in the Period 1945–1964." *Journal of Finance*, 23(2), 389–416.
Defines Jensen's alpha as the OLS intercept of the CAPM regression, measuring risk-adjusted excess return attributable to manager skill.

[5] Grinold, R.C. and Kahn, R.N. (2000). *Active Portfolio Management* (2nd ed.). McGraw-Hill.
Establishes the information ratio (active return / tracking error) as the canonical measure of active management efficiency. The Fundamental Law of Active Management ($IR \approx IC \cdot \sqrt{BR}$) relates IR to signal quality and breadth.

[6] Lo, A.W. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal*, 58(4), 36–52.
Discusses the statistical properties of the Sharpe ratio estimator under non-normal return distributions (relevant given the skewness and kurtosis metrics computed alongside the Sharpe).

[7] Harvey, C.R. et al. (2018). "An Evaluation of Alternative Multiple Testing Methods for Finance Applications." *Review of Asset Pricing Studies*, 10(2), 199–248.
Motivates the walk-forward out-of-sample evaluation framework: in-sample Sharpe ratios are upwardly biased, making out-of-sample performance on held-out test folds the only credible performance measure.

[8] Bailey, D.H. and Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk*, 15(2), 3–44.
Discusses the interpretation of Sharpe ratio confidence intervals and the minimum track record length needed for statistical significance — relevant context for interpreting fold-level Sharpe variability.

[9] Okkan, U. and Serbe, Z.A. (2012). "Drawdown-Based Measures in Portfolio Management." *International Journal of Economics and Finance*, 4(5).
Reviews max drawdown and drawdown duration as risk measures complementary to variance-based metrics.

[10] NVIDIA / CuPy Developers (2017). *CuPy: NumPy/SciPy-compatible Array Library for GPU-accelerated Computing*. [https://cupy.dev](https://cupy.dev)
The GPU array library used for O(T) prefix-sum rolling metric computation, enabling 8–15× speedup over NumPy for batch multi-strategy evaluation.
