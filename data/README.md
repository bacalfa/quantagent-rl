# `data` — QuantAgent-RL Data Module

The `data` module is the **single source of truth** for all raw and processed data that flows into the forecasting, agent, and reinforcement-learning components of QuantAgent-RL.  
It handles everything from downloading raw prices and macroeconomic signals to producing fully normalized, walk-forward-safe feature matrices ready for model training.

---

## Table of Contents

1. [Module Overview](#1-module-overview)  
2. [Quick Start](#2-quick-start)  
3. [Architecture](#3-architecture)  
4. [Configuration (`config.py`)](#4-configuration-configpy)  
5. [Data Ingestion (`ingestion.py`)](#5-data-ingestion-ingestionpy)  
   5.1 [Market Data (yfinance)](#51-market-data-yfinance)  
   5.2 [Macro Data (FRED)](#52-macro-data-fred)  
   5.3 [SEC Filings (EDGAR)](#53-sec-filings-edgar)  
6. [Universe Validation (`universe.py`)](#6-universe-validation-universepy)  
7. [Feature Engineering (`features.py`)](#7-feature-engineering-featurespy)  
   7.1 [Feature Catalogue](#71-feature-catalogue)  
   7.2 [Walk-Forward-Safe Normalization](#72-walk-forward-safe-normalization)  
   7.3 [GPU Acceleration](#73-gpu-acceleration)  
8. [Walk-Forward Splitting](#8-walk-forward-splitting)  
9. [Pipeline Orchestration (`pipeline.py`)](#9-pipeline-orchestration-pipelinepy)  
10. [Caching Strategy](#10-caching-strategy)  
11. [Environment Variables](#11-environment-variables)  
12. [References](#12-references)  

---

## 1. Module Overview

```
data/
├── __init__.py        Public API — all exports live here
├── config.py          All configuration dataclasses
├── ingestion.py       Three independent data ingesters
├── universe.py        Ticker validation, GICS sectors, weight helpers
├── features.py        GPU-accelerated feature engineering
├── pipeline.py        End-to-end orchestrator + WalkForwardFold
└── cache/
    ├── market/        Parquet: adj_close, dividends, volumes
    ├── macro/         Parquet: macro_signals
    └── sec/           Parquet: filing_metadata, xbrl_financials; JSON: MD&A text
```

**Design principles:**

| Principle | How it is implemented |
|---|---|
| Walk-forward safety | Normalizers are **fitted only on training data** at each fold; test data is transformed with in-sample statistics |
| Reproducibility | All configuration lives in dataclasses; the same `DataConfig` always produces the same splits |
| Separation of concerns | Each ingester is stateless; caching is a distinct concern handled via parquet files |
| Zero look-ahead bias | Macro signals are **lagged by one business day** before being added to the feature matrix |

---

## 2. Quick Start

```python
from data import DataPipeline, DataConfig, UniverseConfig, DateRangeConfig

# Minimal run — 4 tickers, quarterly rebalancing
cfg = DataConfig(
    universe=UniverseConfig(tickers=["AAPL", "MSFT", "RTX", "XOM"]),
    dates=DateRangeConfig(start_date="2015-01-02"),
)

pipeline = DataPipeline(cfg).run(skip_sec=True)  # skip SEC for speed

# Inspect any fold
fold = pipeline.get_fold(0)
print(fold)
# WalkForwardFold(fold=0, train=2015-03-31→2018-03-31 (12Q),
#                 test=2018-06-30→2019-03-31 (4Q), n_assets=4)

# Ready-to-use RL state matrix (quarters × features, z-score normalized)
print(fold.train_state_matrix.shape)
# (12, 56)   ← 14 feature groups × 4 tickers

# Daily adjusted-close prices for the training window
print(fold.train_prices.head())
```

---

## 3. Architecture

The pipeline is a **linear six-stage DAG**. Each stage writes into `DataPipeline`'s internal state so that later stages can consume the results.

```
┌─────────────────────────────────────────────────────────────────────┐
│                       DataPipeline.run()                            │
│                                                                     │
│  Stage 1          Stage 2          Stage 3                          │
│  ────────         ────────         ────────                         │
│  Market           Macro            SEC                              │
│  DataIngester ──► DataIngester ──► FilingIngester                  │
│  (yfinance)       (FRED API)       (EDGAR REST +                    │
│                                    edgartools)                      │
│       │                │                │                           │
│       └────────────────┴────────────────┘                           │
│                        │                                            │
│  Stage 4          Stage 5          Stage 6                          │
│  ────────         ────────         ────────                         │
│  Universe    ──►  Feature     ──►  Walk-forward                     │
│  Validation       Engineering      Fold Generation                  │
│  (ticker          (GPU/CPU)        (expanding window)               │
│   filtering,                                                        │
│   GICS sectors)                                                     │
│                        │                                            │
│                        ▼                                            │
│              list[WalkForwardFold]                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Configuration (`config.py`)

All tunable parameters are grouped into **five nested dataclasses** that are
composed into a single `DataConfig` object.

```python
@dataclass
class DataConfig:
    universe : UniverseConfig   # which tickers, benchmark, min history
    dates    : DateRangeConfig  # start/end date, rebalance frequency
    macro    : MacroConfig      # FRED series IDs, forward-fill limit
    sec      : SECConfig        # form types, MD&A, XBRL, rate limiting
    features : FeatureConfig    # rolling windows, GPU preference
```

### Key configuration parameters

| Config class | Parameter | Default | Description |
|---|---|---|---|
| `UniverseConfig` | `tickers` | 30-stock S&P subset | Investable universe |
| `UniverseConfig` | `benchmark_ticker` | `"SPY"` | Beta / tracking error benchmark |
| `UniverseConfig` | `min_history_years` | `5.0` | Years of price history required |
| `DateRangeConfig` | `start_date` | `"2005-01-01"` | Earliest data to fetch |
| `DateRangeConfig` | `rebalance_freq` | `"QE"` | Pandas offset for rebalance calendar |
| `MacroConfig` | `ffill_limit` | `7` | Max calendar days to forward-fill FRED gaps |
| `SECConfig` | `max_filings_per_ticker` | `8` | Cap on filings retrieved per ticker |
| `FeatureConfig` | `return_windows` | `[5, 21, 63]` | Look-back windows in trading days |
| `FeatureConfig` | `use_gpu` | `None` (auto) | `True`/`False`/`None` |

---

## 5. Data Ingestion (`ingestion.py`)

### 5.1 Market Data (yfinance)

`MarketDataIngester` downloads **adjusted close prices, dividends, and trading volumes** for all configured tickers via [yfinance](https://github.com/ranaroussi/yfinance).

```
tickers × date_range
        │
        ▼
yfinance.download()   ← batched (50 tickers per call, polite 0.5 s delay)
        │
        ▼
┌────────────────────────────────────┐
│ _parse_raw()                       │
│  MultiIndex  →  Close / Dividends  │
│  or single-ticker collapse path    │
└────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────┐
│ _clean()                           │
│  • Drop all-NaN rows (non-trading) │
│  • Forward-fill gaps ≤ 5 days      │
└────────────────────────────────────┘
        │
        ▼
Cache: data/cache/market/{adj_close,dividends,volumes}.parquet
```

**Outputs:**  
- `prices` — `DataFrame[Date × Ticker]` adjusted close prices  
- `dividends` — `DataFrame[Date × Ticker]` per-share dividend amounts  
- `volumes` — `DataFrame[Date × Ticker]` daily trading volume

### 5.2 Macro Data (FRED)

`MacroDataIngester` pulls the nine FRED time series listed below via [fredapi](https://github.com/mortada/fredapi). Observations are published at various native frequencies (daily, weekly, monthly); everything is resampled to the **business-day calendar** with forward-filling capped at `ffill_limit` days.

| FRED Series ID | Column name | Description |
|---|---|---|
| `FEDFUNDS` | `fed_funds_rate` | Federal funds effective rate |
| `CPIAUCSL` | `cpi_yoy` | CPI all-urban consumers YoY |
| `DCOILWTICO` | `wti_crude_oil` | WTI crude oil spot price |
| `T10Y2Y` | `yield_curve_10y2y` | 10-year minus 2-year Treasury spread |
| `VIXCLS` | `vix` | CBOE Volatility Index |
| `UNRATE` | `unemployment_rate` | US unemployment rate |
| `UMCSENT` | `consumer_sentiment` | University of Michigan consumer sentiment |
| `BAMLH0A0HYM2` | `hy_spread` | High-yield corporate OAS spread |

Beyond the raw series, `_add_derived_features()` automatically appends:

| Derived feature | Formula | Intuition |
|---|---|---|
| `{col}_mom1m` | $x_t - x_{t-21}$ | One-month momentum / change |
| `{col}_zscore` | $(x_t - \mu_{252}) / \sigma_{252}$ | Regime-normalized level |
| `cpi_acceleration` | $\Delta_{21}\,\text{cpi\_yoy}$ | Is inflation re-accelerating? |
| `yield_curve_inverted` | $\mathbf{1}[\text{T10Y2Y} < 0]$ | Inversion flag (recession signal) |

### 5.3 SEC Filings (EDGAR)

`SECFilingIngester` retrieves two complementary data types from the **free** [SEC EDGAR](https://www.sec.gov/os/accessing-edgar-data) API — no API key required. The only requirement is a descriptive `User-Agent` header per SEC fair-use policy.

```
Ticker list
    │
    ├─► _load_cik_map()
    │     Bulk CIK resolution: company_tickers.json endpoint
    │
    └─► _fetch_ticker_metadata()
          For each CIK:
          data.sec.gov/submissions/{CIK}.json  →  filing index
                │
                ├─► XBRL financials (if fetch_xbrl=True)
                │     data.sec.gov/api/xbrl/companyfacts/{CIK}.json
                │     Extracts: revenue, net income, EPS, operating income,
                │               R&D expense, gross profit, cash, long-term
                │               debt, operating cash flow, shares outstanding
                │
                └─► MD&A text (if fetch_mda=True)
                      edgartools library → parsed 10-Q / 10-K HTML
                      Fallback: heuristic HTML extraction
```

**Rate limiting:** SEC policy requires ≤ 10 requests/second. The ingester enforces a 0.12-second delay between all EDGAR API calls.

**Outputs:**
- `fetch()` → `DataFrame` with columns `ticker, cik, form_type, filing_date, accession_number, report_date, primary_document_url`
- `fetch_financials()` → long-format XBRL table (ticker, metric, period_end, value)
- `get_mda_text(ticker, form_type, as_of_date)` → `str` MD&A section text

---

## 6. Universe Validation (`universe.py`)

`Universe` validates tickers, assigns GICS sectors, and provides weight-construction helpers.

### Validation algorithm (two-pass)

Validating with a fixed start date would silently drop any recently-listed ticker whose IPO post-dates that start. Instead, the validator **keeps all tickers and adjusts the start date**:

```
Pass 1 — Start-date reconciliation
─────────────────────────────────
For each ticker present in price_data:
  first_date[ticker] = earliest non-NaN price date

effective_start = max(first_date.values())

If effective_start > requested_start:
  ┌─────────────────────────────────────────────────┐
  │  start_date_adjusted = True                     │
  │  All fold boundaries shift to effective_start   │
  │  Streamlit dashboard warns the user             │
  └─────────────────────────────────────────────────┘

Pass 2 — Minimum-history filter
───────────────────────────────
min_days = min_history_years × 252

For each ticker:
  obs = count(price[ticker] where date ≥ effective_start)
  if obs < min_days:
      dropped_tickers.append(ticker)   ← insufficient history
  else:
      valid_tickers.append(ticker)
```

### GICS sector mapping

The built-in `TICKER_SECTOR_MAP` covers all 30 default tickers across 10 GICS sectors. Custom mappings are injected via `Universe(sector_map=...)`.

```
Information Technology   : AAPL, MSFT, NVDA, AVGO
Communication Services   : GOOGL, META
Consumer Discretionary   : AMZN, TSLA
Health Care              : UNH, JNJ, LLY, ABBV
Financials               : JPM, BAC, GS, BRK-B
Industrials              : RTX, HON, CAT
Energy                   : XOM, CVX
Consumer Staples         : PG, KO, WMT
Materials                : LIN, APD
Real Estate              : PLD, AMT
Utilities                : NEE, DUK
```

### Weight helpers

| Method | Description |
|---|---|
| `equal_weights()` | $w_i = 1/N$ — equal weight across all valid tickers |
| `market_cap_weights(market_caps)` | Cap-weighted; falls back to equal weight if caps unavailable |
| `sector_constrained_weights(w, max_sector_weight=0.35)` | Clips and renormalises to prevent sector over-concentration |

---

## 7. Feature Engineering (`features.py`)

`FeatureEngineer` computes all numerical signals needed by the RL agent's state vector and by the LLM agents for quantitative context.

### 7.1 Feature Catalogue

All features are computed daily and then **resampled to quarter-end** snapshots before being assembled into the RL state matrix.

#### Return features

$$r^{(w)}_t = \sum_{\tau=t-w+1}^{t} \ln\frac{P_\tau}{P_{\tau-1}}, \quad w \in \{5, 21, 63\}$$

Cumulative log returns over 1-week, 1-month, and 3-month windows.

#### Volatility features

**Realized volatility** — annualized standard deviation of log returns:

$$\sigma^{(w)}_t = \sqrt{252} \cdot \text{std}\left(\{r_\tau\}_{\tau=t-w+1}^{t}\right), \quad w \in \{21, 63\}$$

**EWMA volatility** — exponentially weighted, reacts faster to recent shocks:

$$\hat{\sigma}^{\text{EWMA}}_t = \sqrt{252} \cdot \text{ewmstd}_{21}(r_t)$$

#### Momentum features

Classic momentum [1] skips the most recent 5 days to strip out short-term reversal:

$$\text{mom}^{(w)}_t = r^{(w)}_t - r^{(5)}_t, \quad w \in \{21, 63, 126\}$$

#### RSI (Relative Strength Index)

Wilder's RSI [2], computed with EWM smoothing:

$$\text{RSI}_t = 100 - \frac{100}{1 + \frac{\bar{G}_t}{\bar{L}_t}}$$

where $\bar{G}_t$ and $\bar{L}_t$ are the EWM averages of daily gains and losses over `rsi_window=14` days.

#### Bollinger Band position [3]

$$\text{BB\_position}_t = \frac{P_t - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t} \in [0, 1]$$

$$\text{Upper}_t = \mu^{(20)}_t + 2\sigma^{(20)}_t, \quad \text{Lower}_t = \mu^{(20)}_t - 2\sigma^{(20)}_t$$

A value near 1 indicates the price is near the upper band (overbought region); near 0 indicates the lower band (oversold).

#### Market microstructure

**Amihud illiquidity** [4] — price impact per unit of dollar volume:

$$\text{ILLIQ}^{(21)}_t = \frac{1}{21}\sum_{\tau=t-20}^{t} \frac{|r_\tau|}{V_\tau / 10^6}$$

High values indicate an illiquid stock where small trades move prices substantially.

**Volume z-score**:

$$z^{\text{vol}}_t = \frac{V_t - \mu^{(63)}_t}{\sigma^{(63)}_t}, \quad \text{clipped to } [-5, 5]$$

#### Cross-sectional features

These capture **relative** performance within the universe, which is more informative for portfolio allocation than absolute levels [5]:

$$\text{xs\_zscore}^i_t = \frac{r^{(21),i}_t - \bar{r}^{(21)}_t}{\text{std}_N\left(r^{(21)}_t\right)}$$

$$\text{xs\_pctrank}^i_t = \text{rank}\left(r^{(21),i}_t\right) / N$$

#### Beta and correlation

**Rolling beta** to the SPY benchmark [6]:

$$\beta^i_t = \frac{\text{Cov}_{63}\left(r^i, r^{\text{SPY}}\right)}{\text{Var}_{63}\left(r^{\text{SPY}}\right)}$$

**Average pairwise correlation** — a regime indicator: high values signal crowded trades and systemic risk [7]:

$$\bar{\rho}_t = \frac{1}{N(N-1)} \sum_{i \neq j} \rho^{(63)}_{ij,t}$$

#### Macro-aligned features

Each FRED macro signal (and all its derived variants) is **lagged by one business day** and broadcast across all assets. This gives the RL agent the ability to learn asset-specific responses to macro regimes without introducing look-ahead bias.

```
Macro signal at date t  →  lagged to t-1  →  broadcasted to all N tickers
```

### 7.2 Walk-Forward-Safe Normalization

The RL state vector requires z-score normalized features. To prevent information leakage from future data into the training set, `normalize_features` fits the mean and standard deviation **exclusively on the training window** of each fold:

```
Fold k:
  training window: [t_0, t_train_end]
  test window:     (t_train_end, t_test_end]

Step 1 — Fit on training data only:
  μ_k = mean(X[t_0 : t_train_end])
  σ_k = std (X[t_0 : t_train_end])

Step 2 — Transform train:
  X_train_norm = clip((X_train - μ_k) / σ_k, −5, 5)

Step 3 — Apply same scaler to test (no re-fitting):
  X_test_norm  = clip((X_test  - μ_k) / σ_k, −5, 5)
```

The clipping to $[-5, 5]$ winsorizes extreme outliers (e.g., COVID crash or meme-stock spikes) that would otherwise dominate gradients in the RL policy network.

### 7.3 GPU Acceleration

`FeatureEngineer` transparently switches between two execution backends:

```
import cudf   # RAPIDS GPU DataFrame
    │
    ▼ success
GPU backend (cuDF/cuML)   ◄── 5–20× faster on large universes
    │
    ▼ ImportError
CPU backend (pandas)      ◄── always available, identical API
```

The `use_gpu` parameter in `FeatureConfig` controls selection:

| `use_gpu` | Behavior |
|---|---|
| `None` (default) | Auto-detect: GPU if RAPIDS installed, else CPU |
| `True` | Require GPU; raise `RuntimeError` if RAPIDS unavailable |
| `False` | Force CPU regardless of hardware |

RAPIDS installation: `conda install -c rapidsai -c conda-forge cudf cuml`.

---

## 8. Walk-Forward Splitting

The core evaluation methodology is **anchored (expanding-window) walk-forward cross-validation** [8], which closely mimics the constraints of live portfolio management: the model is always trained on the full available history and evaluated strictly out-of-sample.

```
Quarterly rebalance dates:  Q1  Q2  Q3  ... Q12 | Q13 Q14 Q15 Q16 | Q17 ...
                            ─────────────────────┼────────────────
Fold 0:  train [Q1–Q12]                          | test [Q13–Q16]
         ────────────────────────────────────────┤
Fold 1:  train [Q1–Q16]                          | test [Q17–Q20]
         ──────────────────────────────────────────────────────────
Fold 2:  train [Q1–Q20]                          | test [Q21–Q24]
         ...
```

Parameters (configured in `_build_folds`):

| Parameter | Default | Description |
|---|---|---|
| `min_train_periods` | 12 | Minimum quarters (3 years) in first training window |
| `test_periods` | 4 | Quarters (1 year) in each test window |

The number of folds generated for a date range $[t_0, t_T]$ with $Q$ total quarters is:

$$N_{\text{folds}} = \left\lfloor \frac{Q - Q_{\text{min\_train}}}{Q_{\text{test}}} \right\rfloor$$

For a 20-year history with quarterly rebalancing (~80 quarters): **17 folds**, each providing one out-of-sample year of RL evaluation.

---

## 9. Pipeline Orchestration (`pipeline.py`)

`DataPipeline` sequences all six stages and packages each fold's data into a clean `WalkForwardFold` dataclass.

### `WalkForwardFold` fields

| Field | Type | Content |
|---|---|---|
| `fold_idx` | `int` | Zero-indexed fold number |
| `train_dates` / `test_dates` | `DatetimeIndex` | Quarter-end rebalance dates |
| `train_prices` / `test_prices` | `DataFrame` | Daily adjusted close prices |
| `train_dividends` / `test_dividends` | `DataFrame` | Daily dividend amounts |
| `train_volumes` / `test_volumes` | `DataFrame` | Daily trading volumes |
| `train_state_matrix` | `DataFrame` | Normalized RL state features (quarters × features) |
| `test_state_matrix` | `DataFrame` | Test features, same scaler as train |
| `macro` | `DataFrame` | Full macro signal history (used by agents) |
| `sec_metadata` | `DataFrame` | Filings up to `test_end` (date-bounded) |
| `scaler_params` | `dict` | `{mean: Series, std: Series}` for audit / replay |
| `tickers` | `list[str]` | Valid non-benchmark tickers in this fold |
| `train_total_returns` | `DataFrame` | Total return index (dividends reinvested) |

### `DataPipeline.run()` usage

```python
pipeline = DataPipeline(cfg).run(
    skip_sec=False,       # True → skip SEC download (faster for development)
    use_cache=True,       # True → load parquet files if they exist
    force_refresh=False,  # True → re-download even if cache is fresh
)
```

Method chaining is supported: `DataPipeline(cfg).run().get_fold(0)`.

---

## 10. Caching Strategy

All three ingesters write parquet files to `data/cache/` on first fetch and read from them on subsequent calls. This eliminates redundant API traffic during iterative development.

```
data/cache/
├── market/
│   ├── adj_close.parquet      Adjusted close prices (Date × Ticker)
│   ├── dividends.parquet
│   └── volumes.parquet
│
├── macro/
│   └── macro_signals.parquet  Daily FRED signals + derived features
│
└── sec/
    ├── filing_metadata.parquet  Filing index (ticker, date, accession number)
    ├── xbrl_financials.parquet  Structured XBRL facts (long format)
    └── mda_{ticker}_{date}.txt  MD&A text cache (one file per filing)
```

Cache invalidation:

```python
# Re-download everything, overwrite cache
pipeline.run(force_refresh=True)

# Skip cache reads (but still write on completion)
pipeline.run(use_cache=False)
```

---

## 11. Environment Variables

| Variable | Required | Description |
|---|---|---|
| `FRED_API_KEY` | **Yes** (unless cache exists) | Free key from [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `SEC_USER_AGENT` | No | Defaults to `"QuantAgentRL research@example.com"`. Override to identify your application per SEC policy. |

FRED API key registration: [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

---

## 12. References

[1] Jegadeesh, N. and Titman, S. (1993). "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." *Journal of Finance*, 48(1), 65–91.  
Classic momentum strategy paper establishing the 12-1 formation period that skips the most recent month.

[2] Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.  
Original introduction of the Relative Strength Index (RSI) with the 14-period EWM smoothing scheme implemented here.

[3] Bollinger, J. (2002). *Bollinger on Bollinger Bands*. McGraw-Hill.  
Defines the (20, 2) Bollinger Band parameters used as defaults in `FeatureConfig`.

[4] Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects." *Journal of Financial Markets*, 5(1), 31–56.  
Introduces the $|r_t| / V_t$ illiquidity ratio used in `_compute_microstructure()`.

[5] Grinold, R. C. and Kahn, R. N. (1999). *Active Portfolio Management: A Quantitative Approach*. McGraw-Hill.  
Establishes the importance of cross-sectional (relative) signals over absolute levels for portfolio allocation; motivates `xs_zscore` and `xs_pctrank`.

[6] Sharpe, W. F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk." *Journal of Finance*, 19(3), 425–442.  
Foundation of the CAPM beta used in `_compute_beta_correlation()`.

[7] Pollet, J. M. and Wilson, M. (2010). "Average Correlation and Stock Market Returns." *Journal of Financial Economics*, 96(3), 364–380.  
Documents the predictive content of average pairwise correlation for market risk; motivates including `avg_pairwise_corr` in the RL state vector.

[8] Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley.  
Comprehensive treatment of walk-forward analysis and the expanding-window methodology implemented in `get_walk_forward_splits()`.

[9] Federal Reserve Bank of St. Louis. *Federal Reserve Economic Data (FRED)*. [https://fred.stlouisfed.org](https://fred.stlouisfed.org)  
Primary source for all macroeconomic time series fetched by `MacroDataIngester`.

[10] U.S. Securities and Exchange Commission. *Electronic Data Gathering, Analysis, and Retrieval (EDGAR)*. [https://www.sec.gov/edgar](https://www.sec.gov/edgar)  
Free public database providing all 10-Q and 10-K filings, XBRL structured financials, and MD&A text retrieved by `SECFilingIngester`.

[11] MSCI / S&P Dow Jones Indices. *Global Industry Classification Standard (GICS)*. [https://www.spglobal.com/spdji/en/landing/investment-themes/gics/](https://www.spglobal.com/spdji/en/landing/investment-themes/gics/)  
Sector taxonomy used in `TICKER_SECTOR_MAP` and `Universe.sector_constrained_weights()`.
