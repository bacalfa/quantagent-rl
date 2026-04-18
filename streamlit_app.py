"""
QuantAgent-RL — Streamlit Dashboard
====================================
Run with:  streamlit run streamlit_app.py

Uses the full analysis pipeline with actual market data (yfinance),
macro data (FRED / cached), and Fama-French factor data.

Optional environment variables:
  FRED_API_KEY       – FRED API key for live macro data (cached parquet used
                       when absent and cache exists).
  ANTHROPIC_API_KEY  – Enables live Claude LLM agent analysis; falls back to
                       mock mode when absent.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Allow imports from the project root regardless of where the app is launched
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.WARNING)  # keep pipeline logs quiet in UI

_PROJECT_ROOT = Path(__file__).parent
_CACHE_ROOT = str(_PROJECT_ROOT / "data" / "cache")

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantAgent-RL",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — Bloomberg-meets-fintech aesthetic
# IBM Plex Mono for data, IBM Plex Sans for UI.  Deep navy / amber / teal.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --navy:    #080d1a;
    --navy-2:  #0d1526;
    --navy-3:  #111d33;
    --navy-4:  #162040;
    --amber:   #f59e0b;
    --amber-d: #b45309;
    --teal:    #06b6d4;
    --teal-d:  #0891b2;
    --green:   #10b981;
    --red:     #ef4444;
    --white:   #f0f4ff;
    --muted:   #6b7fa3;
    --border:  #1e2d4a;
    --mono:    'IBM Plex Mono', monospace;
    --sans:    'IBM Plex Sans', sans-serif;
}

/* ── App shell ─────────────────────────────────────────────── */
.stApp { background: var(--navy); color: var(--white); font-family: var(--sans); }
section[data-testid="stSidebar"] { background: var(--navy-2) !important; border-right: 1px solid var(--border); }
header { background: transparent !important; }

/* ── Typography ─────────────────────────────────────────────── */
h1,h2,h3,h4 { font-family: var(--mono); color: var(--white); letter-spacing: -0.02em; }
h1 { font-size: 1.6rem; font-weight: 600; }
h2 { font-size: 1.1rem; font-weight: 500; color: var(--teal); text-transform: uppercase; letter-spacing: 0.08em; }
p, label, .stMarkdown { font-family: var(--sans); color: var(--white); }
.stMetric label { font-family: var(--mono); font-size: 0.68rem; color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.06em; }
.stMetric .metric-value, [data-testid="stMetricValue"] { font-family: var(--mono) !important; font-size: 1.5rem !important; color: var(--amber) !important; font-weight: 600 !important; }
[data-testid="stMetricDelta"] { font-family: var(--mono) !important; font-size: 0.72rem !important; }

/* ── Cards ──────────────────────────────────────────────────── */
.card {
    background: var(--navy-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.card-header {
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
}
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; }
.metric-pill {
    background: var(--navy-3);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0.3rem 0.7rem;
    font-family: var(--mono);
    font-size: 0.75rem;
}
.metric-pill .label { color: var(--muted); font-size: 0.62rem; display: block; }
.metric-pill .value { color: var(--amber); font-weight: 600; }

/* ── Status badges ──────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.badge-green  { background: #052e16; color: var(--green);  border: 1px solid #14532d; }
.badge-amber  { background: #1c1003; color: var(--amber);  border: 1px solid #78350f; }
.badge-teal   { background: #022c34; color: var(--teal);   border: 1px solid #155e75; }
.badge-red    { background: #1c0505; color: var(--red);    border: 1px solid #7f1d1d; }
.badge-muted  { background: var(--navy-3); color: var(--muted); border: 1px solid var(--border); }

/* ── Sidebar nav ────────────────────────────────────────────── */
.sidebar-logo {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--amber);
    padding: 0.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sidebar-divider { border-top: 1px solid var(--border); margin: 0.8rem 0; }

/* ── Selectbox / slider theme ───────────────────────────────── */
.stSelectbox div[data-baseweb="select"] { background: var(--navy-3) !important; border: 1px solid var(--border) !important; }
.stSlider .stMarkdown { color: var(--muted) !important; }

/* ── Plotly container ───────────────────────────────────────── */
.js-plotly-plot { border-radius: 4px; }

/* ── Tables ─────────────────────────────────────────────────── */
.stDataFrame { background: var(--navy-2) !important; }
.stDataFrame thead { background: var(--navy-3) !important; }

/* ── Section divider ────────────────────────────────────────── */
.section-title {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--teal);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
    margin-top: 1.4rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly theme helper
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1526",
    plot_bgcolor="#080d1a",
    font=dict(family="IBM Plex Mono, monospace", color="#f0f4ff", size=11),
    xaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a", tickfont=dict(size=10)),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d4a"),
)

COLORS = dict(
    amber="#f59e0b",
    teal="#06b6d4",
    green="#10b981",
    red="#ef4444",
    muted="#6b7fa3",
    white="#f0f4ff",
    navy="#080d1a",
    border="#1e2d4a",
)
STRATEGY_COLORS = {
    "PPO": COLORS["amber"],
    "Equal Weight": COLORS["teal"],
    "Buy & Hold": COLORS["muted"],
    "Benchmark": "#8b5cf6",
}


# ─────────────────────────────────────────────────────────────────────────────
# Real-data pipeline loaders  (cached across Streamlit reruns)
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_fred_key() -> None:
    """If no FRED API key is set but a macro cache exists, use a placeholder
    so MacroDataIngester.__init__ passes validation and falls back to cache."""
    if not os.environ.get("FRED_API_KEY"):
        macro_cache = (
            _PROJECT_ROOT / "data" / "cache" / "macro" / "macro_signals.parquet"
        )
        if macro_cache.exists():
            os.environ["FRED_API_KEY"] = "cache_only"


@st.cache_resource(show_spinner="Fetching market & macro data…")
def _load_data_pipeline(tickers_csv: str, start_date: str):
    _ensure_fred_key()
    from data.config import DataConfig, DateRangeConfig, UniverseConfig
    from data.pipeline import DataPipeline

    tickers = [t.strip() for t in tickers_csv.split(",") if t.strip()]
    cfg = DataConfig(
        universe=UniverseConfig(tickers=tickers),
        dates=DateRangeConfig(start_date=start_date),
    )
    dp = DataPipeline(cfg, cache_root_dir=_CACHE_ROOT, log_level=logging.WARNING)
    dp.run(skip_sec=False, use_cache=True, force_refresh=True)
    return dp


@st.cache_resource(show_spinner="Running GARCH / HMM / Fama-French on latest fold…")
def _load_last_forecast_bundle(tickers_csv: str, _dp):
    from forecasting.config import ForecastConfig
    from forecasting.pipeline import ForecastingPipeline

    fp = ForecastingPipeline(ForecastConfig())
    fp.load_factors(use_cache=True)
    last_fold = _dp.get_fold(_dp.n_folds - 1)
    return fp.run_fold(last_fold)


@st.cache_resource(show_spinner="Running agent analysis…")
def _load_agent_bundle(tickers_csv: str, _dp):
    from agents import HuggingFaceConfig
    from agents.config import AgentConfig
    from agents.pipeline import AgentPipeline

    hf_cfg = HuggingFaceConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    llm_backend = (
        "huggingface" if bool(os.environ.get("HUGGINGFACE_MODEL")) else "claude"
    )
    mock = not bool(os.environ.get("ANTHROPIC_API_KEY")) or llm_backend != "huggingface"
    ap = AgentPipeline(
        AgentConfig(
            mock_mode=mock,
            enable_web_search=True,
            temperature=0.0,
            max_tokens=2048,
            llm_backend=llm_backend,
            huggingface=hf_cfg,
        ),
        cache_dir=str(_PROJECT_ROOT / "data" / "cache" / "agent_briefs"),
    )
    fold = _dp.get_fold(_dp.n_folds - 1)
    return ap.run_fold(fold, sec_metadata=None)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy return computation from actual prices
# ─────────────────────────────────────────────────────────────────────────────


def _compute_portfolio_returns(
    all_prices: pd.DataFrame,
    tickers: list[str],
    test_dates: "pd.DatetimeIndex",
    train_end: "pd.Timestamp",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute equal-weight, buy-and-hold, and SPY returns over test_dates."""
    prices = all_prices.reindex(columns=tickers).ffill()
    n = len(tickers)
    bh_weights = np.ones(n) / n

    ew_ret: list[float] = []
    bh_ret: list[float] = []
    prev = train_end

    for date in test_dates:
        p0 = prices.loc[:prev].iloc[-1]
        p1 = prices.loc[:date].iloc[-1]
        asset_ret = ((p1 / p0) - 1.0).fillna(0.0).values
        ew_ret.append(float(np.dot(np.ones(n) / n, asset_ret)))
        bh_ret.append(float(np.dot(bh_weights, asset_ret)))
        bh_weights = bh_weights * (1.0 + asset_ret)
        s = bh_weights.sum()
        if s > 0:
            bh_weights /= s
        prev = date

    # SPY benchmark from raw market cache
    bm_ret = np.zeros(len(test_dates))
    spy_cache = _PROJECT_ROOT / "data" / "cache" / "market" / "adj_close.parquet"
    if spy_cache.exists():
        try:
            spy = pd.read_parquet(spy_cache)
            if "SPY" in spy.columns:
                spy_s = spy["SPY"].ffill()
                prev = train_end
                for i, date in enumerate(test_dates):
                    p0 = spy_s.loc[:prev].iloc[-1]
                    p1 = spy_s.loc[:date].iloc[-1]
                    if p0 > 0:
                        bm_ret[i] = p1 / p0 - 1.0
                    prev = date
        except Exception:
            pass

    return np.array(ew_ret), np.array(bh_ret), bm_ret


def _fold_metrics(r: np.ndarray, bm: np.ndarray) -> dict:
    if len(r) == 0:
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "alpha": 0.0}
    cum = float(np.prod(1.0 + r) - 1.0)
    sh = float(r.mean() / (r.std(ddof=1) + 1e-9) * np.sqrt(4))
    cv = np.cumprod(1.0 + r)
    pk = np.maximum.accumulate(cv)
    dd = float(((cv - pk) / (pk + 1e-9)).min())
    alpha = float((r - bm[: len(r)]).mean() * 4)
    return {"total_return": cum, "sharpe": sh, "max_drawdown": dd, "alpha": alpha}


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard data builder  (called once per session)
# ─────────────────────────────────────────────────────────────────────────────


def _build_dashboard_data(dp, last_bundle, agent_bundle) -> dict:
    from data.universe import TICKER_SECTOR_MAP

    tickers = dp.universe.valid_tickers
    sectors = {t: TICKER_SECTOR_MAP.get(t, "Unknown") for t in tickers}
    prices = dp.all_prices.reindex(columns=tickers)

    # Quarterly dates
    q_dates = dp.rebalance_dates

    # ── Macro signals (quarterly) ────────────────────────────────────────────
    macro_raw = dp.macro
    core_cols = [
        c
        for c in ["fed_funds_rate", "cpi_yoy", "vix", "yield_curve_10y2y", "hy_spread"]
        if c in macro_raw.columns
    ]
    macro_q = (
        macro_raw[core_cols]
        .resample("QE")
        .last()
        .reindex(q_dates, method="ffill")
        .dropna(how="all")
    )

    # ── Forecasting (last fold) ───────────────────────────────────────────────
    garch_vol = pd.concat([last_bundle.train_vol, last_bundle.test_vol]).sort_index()

    regime_df = pd.concat(
        [last_bundle.train_regime, last_bundle.test_regime]
    ).sort_index()

    # FF betas: last available quarter → ticker × factor matrix
    ff_betas = pd.DataFrame()
    if not last_bundle.test_betas.empty:
        try:
            last_row = last_bundle.test_betas.iloc[-1]
            ff_betas = last_row.unstack(level=0)
            keep = [
                c
                for c in ["alpha_ann", "beta_mkt", "beta_smb", "beta_hml"]
                if c in ff_betas.columns
            ]
            ff_betas = ff_betas[keep].dropna(how="all")
        except Exception:
            ff_betas = pd.DataFrame()

    # ── Agent market brief ───────────────────────────────────────────────────
    latest_date = sorted(agent_bundle.briefs.keys())[-1] if agent_bundle.briefs else ""
    mb_obj = agent_bundle.briefs.get(latest_date)
    if mb_obj is not None:
        market_brief = {
            "as_of_date": mb_obj.as_of_date,
            "macro_regime": mb_obj.macro_regime,
            "portfolio_stance": mb_obj.portfolio_stance,
            "conviction_score": mb_obj.conviction_score,
            "top_overweights": mb_obj.top_overweights or [],
            "top_underweights": mb_obj.top_underweights or [],
            "sector_tilts": mb_obj.sector_tilts or {},
            "key_themes": mb_obj.key_themes or [],
            "risk_flags": mb_obj.risk_flags or [],
            "executive_summary": mb_obj.executive_summary or "",
        }
    else:
        market_brief = {
            "as_of_date": str(q_dates[-1].date()) if len(q_dates) else "N/A",
            "macro_regime": "transitional",
            "portfolio_stance": "neutral",
            "conviction_score": 0.5,
            "top_overweights": tickers[:5],
            "top_underweights": tickers[-3:],
            "sector_tilts": {},
            "key_themes": ["Agent analysis unavailable"],
            "risk_flags": ["Agent analysis unavailable"],
            "executive_summary": "Agent analysis unavailable.",
        }

    # ── Equal-weight portfolio ────────────────────────────────────────────────
    n = len(tickers)
    weights = {t: 1.0 / n for t in tickers}

    # ── Backtest returns (last fold) ─────────────────────────────────────────
    last_fold = dp.get_fold(dp.n_folds - 1)
    all_prices_fold = pd.concat(
        [last_fold.train_prices, last_fold.test_prices]
    ).sort_index()
    test_dates = last_fold.test_dates
    ew_ret, hold_ret, bm_ret = _compute_portfolio_returns(
        all_prices_fold, last_fold.tickers, test_dates, last_fold.train_end
    )

    # ── Per-fold breakdown (last 3 folds for chart) ───────────────────────────
    n_show = min(3, dp.n_folds)
    folds_data = []
    for fi in range(dp.n_folds - n_show, dp.n_folds):
        fi_fold = dp.get_fold(fi)
        fi_all = pd.concat([fi_fold.train_prices, fi_fold.test_prices]).sort_index()
        fi_ew, fi_bh, fi_bm = _compute_portfolio_returns(
            fi_all, fi_fold.tickers, fi_fold.test_dates, fi_fold.train_end
        )
        folds_data.append(
            {
                "fold": fi,
                "ppo": _fold_metrics(fi_ew, fi_bm),  # EW used as PPO proxy
                "ew": _fold_metrics(fi_ew, fi_bm),
                "bm": _fold_metrics(fi_bm, fi_bm),
            }
        )

    return {
        "tickers": tickers,
        "sectors": sectors,
        "prices": prices,
        "q_dates": q_dates,
        "macro_df": macro_q,
        "garch_vol": garch_vol,
        "regime_df": regime_df,
        "ff_betas": ff_betas,
        "market_brief": market_brief,
        "weights": weights,
        "test_dates": test_dates,
        "ppo_ret": ew_ret,  # EW proxy — label updated in Backtest page
        "ew_ret": ew_ret,
        "hold_ret": hold_ret,
        "bm_ret": bm_ret,
        "folds": folds_data,
        # metadata
        "n_folds": dp.n_folds,
        "mock_agents": not bool(os.environ.get("ANTHROPIC_API_KEY")),
        "last_fold_test_start": str(last_fold.test_start.date()),
        "last_fold_test_end": str(last_fold.test_end.date()),
    }


D = None  # populated after pipelines load (see sidebar section below)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: GPU / system status
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_system_status() -> dict:
    status: dict[str, str] = {}
    # GPU
    try:
        import torch

        if torch.cuda.is_available():
            status["GPU"] = torch.cuda.get_device_name(0)
            status["VRAM"] = (
                f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        else:
            status["GPU"] = "CPU only"
            status["VRAM"] = "—"
    except ImportError:
        status["GPU"] = "PyTorch not installed"
        status["VRAM"] = "—"

    # CuPy
    try:
        import cupy  # noqa: F401

        status["CuPy"] = "✓ installed"
    except ImportError:
        status["CuPy"] = "not installed"

    # SB3
    try:
        import stable_baselines3 as sb3

        status["SB3"] = sb3.__version__
    except ImportError:
        status["SB3"] = "not installed"

    # Gymnasium
    try:
        import gymnasium

        status["Gymnasium"] = gymnasium.__version__
    except ImportError:
        status["Gymnasium"] = "not installed"

    # sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401

        status["SentenceTransformers"] = "✓ installed"
    except ImportError:
        status["SentenceTransformers"] = "not installed"

    return status


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sharpe(r: np.ndarray, ppy: int = 4) -> float:
    if r.std(ddof=1) < 1e-9:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ppy))


def _max_dd(r: np.ndarray) -> float:
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    return float(((cum - peak) / (peak + 1e-9)).min())


def _ann_ret(r: np.ndarray, ppy: int = 4) -> float:
    T = len(r)
    return float((np.prod(1 + r) ** (ppy / max(T, 1))) - 1)


def _sortino(r: np.ndarray, ppy: int = 4) -> float:
    neg = r[r < 0]
    if len(neg) < 2:
        return 0.0
    dd = float(neg.std(ddof=1))
    return float(r.mean() / dd * np.sqrt(ppy)) if dd > 1e-9 else 0.0


def _calmar(r: np.ndarray, ppy: int = 4) -> float:
    mdd = abs(_max_dd(r))
    return float(_ann_ret(r, ppy) / mdd) if mdd > 1e-9 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
    <div class='sidebar-logo'>
        <span>⬡</span><span>QuantAgent-RL</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Market Data",
            "Forecasting",
            "Agent Insights",
            "Portfolio",
            "Backtest",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    # ── Universe configuration ────────────────────────────────────────────
    st.markdown("**UNIVERSE**")
    from data.config import DEFAULT_UNIVERSE

    default_tickers_text = ", ".join(DEFAULT_UNIVERSE)
    tickers_input = st.text_area(
        "Tickers (comma-separated)",
        value=default_tickers_text,
        height=100,
        label_visibility="collapsed",
        help="Enter comma-separated ticker symbols. Changes trigger a data reload.",
    )
    start_date_input = st.text_input(
        "Start date",
        value="2015-01-02",
        label_visibility="collapsed",
    )

    # Normalise the tickers key for stable cache keying
    _tickers_csv = ",".join(
        sorted(t.strip().upper() for t in tickers_input.split(",") if t.strip())
    )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    # ── Load pipelines ────────────────────────────────────────────────────
    _pipeline_ok = False
    _forecast_ok = False
    _agent_ok = False
    try:
        _dp = _load_data_pipeline(_tickers_csv, start_date_input)
        _pipeline_ok = True
    except Exception as _e:
        st.error(f"Data pipeline error: {_e}")
        _dp = None

    if _dp is not None:
        try:
            _last_bundle = _load_last_forecast_bundle(_tickers_csv, _dp)
            _forecast_ok = True
        except Exception as _e:
            st.warning(f"Forecast pipeline error: {_e}")
            _last_bundle = None

        try:
            _agent_bundle = _load_agent_bundle(_tickers_csv, _dp)
            _agent_ok = True
        except Exception as _e:
            st.warning(f"Agent pipeline error: {_e}")
            _agent_bundle = None

        if _last_bundle is not None and _agent_bundle is not None:
            if (
                "D" not in st.session_state
                or st.session_state.get("_tickers_key") != _tickers_csv
            ):
                with st.spinner("Building dashboard data…"):
                    st.session_state["D"] = _build_dashboard_data(
                        _dp, _last_bundle, _agent_bundle
                    )
                    st.session_state["_tickers_key"] = _tickers_csv

    D = st.session_state.get("D")

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    # System status
    status = get_system_status()
    st.markdown("**SYSTEM STATUS**")
    for k, v in status.items():
        ok = "✓" in v or any(c.isdigit() for c in v[:3])
        badge_cls = "badge-green" if ok else "badge-muted"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"margin-bottom:4px;font-size:0.72rem;font-family:IBM Plex Mono,monospace'>"
            f"<span style='color:#6b7fa3'>{k}</span>"
            f"<span class='badge {badge_cls}'>{v}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    # ── Data source status ────────────────────────────────────────────────
    _agent_mode = "Live LLM" if os.environ.get("ANTHROPIC_API_KEY") else "Mock mode"
    _fred_mode = (
        "Live FRED"
        if (
            os.environ.get("FRED_API_KEY")
            and os.environ.get("FRED_API_KEY") != "cache_only"
        )
        else "Cached"
    )
    st.markdown(
        f"<p style='font-size:0.65rem;color:#6b7fa3;font-family:IBM Plex Mono,monospace'>"
        f"Market data: yfinance<br>"
        f"Macro data: {_fred_mode}<br>"
        f"FF factors: Ken French library<br>"
        f"Agents: {_agent_mode}<br>"
        f"RL/PPO: not trained (EW proxy)</p>",
        unsafe_allow_html=True,
    )

if D is None:
    st.warning("⚠ Data is still loading — please wait or check the sidebar for errors.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: styled section title
# ─────────────────────────────────────────────────────────────────────────────
def section(text: str):
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)


def badge(text: str, kind: str = "teal") -> str:
    return f"<span class='badge badge-{kind}'>{text}</span>"


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown("## QuantAgent-RL")
    st.markdown(
        "Tax-Aware Portfolio Rebalancing via "
        + badge("Multi-Agent LLM", "teal")
        + " + "
        + badge("Reinforcement Learning", "amber")
        + " + "
        + badge("GPU Acceleration", "green"),
        unsafe_allow_html=True,
    )

    # Top metrics — actual values from the pipeline
    n_assets = len(D["tickers"])
    n_folds = D["n_folds"]
    last_ew_sharpe = _sharpe(D["ew_ret"])
    last_bm_sharpe = _sharpe(D["bm_ret"])
    sharpe_diff = last_ew_sharpe - last_bm_sharpe
    test_period = f"{D['last_fold_test_start']} → {D['last_fold_test_end']}"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Universe", f"{n_assets} assets", "S&P 100 subset")
    with col2:
        st.metric("Walk-Forward Folds", str(n_folds), "Expanding window")
    with col3:
        st.metric(
            "EW Sharpe (OOS)",
            f"{last_ew_sharpe:.2f}",
            f"{sharpe_diff:+.2f} vs Benchmark",
        )
    with col4:
        st.metric("Test Period", "", test_period)
    with col5:
        # State vector: quant features + forecast + agent embed + weights
        # quant features ≈ 10 per ticker × n_assets, forecast ≈ (4+3)×n_assets,
        # agent embed = 384 (MiniLM), weights = n_assets
        quant_dim = 10 * n_assets
        fcst_dim = 7 * n_assets
        embed_dim = 384
        state_dim = quant_dim + fcst_dim + embed_dim + n_assets
        st.metric("Obs. Vector Dim", str(state_dim), "quant+fcst+embed+w")

    section("SYSTEM ARCHITECTURE")

    # Architecture diagram using Plotly
    fig_arch = go.Figure()

    # Module boxes
    modules = [
        (
            "data/",
            0.10,
            0.80,
            COLORS["teal"],
            "yfinance · FRED\nEDGAR XBRL\ncuDF · cuML",
        ),
        (
            "forecasting/",
            0.30,
            0.80,
            COLORS["amber"],
            "GARCH(1,1)\nHMM Regime\nFama-French",
        ),
        (
            "agents/",
            0.50,
            0.80,
            "#8b5cf6",
            "MacroAgent\nSectorAgent\nCompanyAgent\nOrchestrator",
        ),
        ("rl/", 0.70, 0.80, COLORS["green"], "PortfolioEnv\nPPO (CUDA)\nDiff. Sharpe"),
        ("backtest/", 0.90, 0.80, "#f43f5e", "Metrics\nTear Sheet\nFold Report"),
    ]

    for name, x, y, color, detail in modules:
        fig_arch.add_shape(
            type="rect",
            x0=x - 0.085,
            y0=y - 0.14,
            x1=x + 0.085,
            y1=y + 0.08,
            fillcolor=color + "22",
            line=dict(color=color, width=1.5),
        )
        fig_arch.add_annotation(
            x=x,
            y=y + 0.02,
            text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(size=12, color=color, family="IBM Plex Mono"),
            align="center",
        )
        fig_arch.add_annotation(
            x=x,
            y=y - 0.07,
            text=detail,
            showarrow=False,
            font=dict(size=8.5, color="#f0f4ff88"),
            align="center",
        )

    # Arrows
    for i in range(len(modules) - 1):
        x0 = modules[i][1] + 0.085
        x1 = modules[i + 1][1] - 0.085
        y = modules[i][2]
        fig_arch.add_annotation(
            x=x1,
            y=y,
            ax=x0,
            ay=y,
            axref="x",
            ayref="y",
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=1.5,
            arrowcolor="#1e2d4a",
        )

    # State vector assembly bar
    state_items = [
        ("Quant Features", 300, COLORS["teal"]),
        ("Forecast Features", 60, COLORS["amber"]),
        ("Agent Embedding", 28, "#8b5cf6"),
        ("Weights", 15, COLORS["green"]),
    ]
    total_dim = sum(d for _, d, _ in state_items)
    x_start = 0.02
    bar_y = 0.28
    for label, dim, color in state_items:
        width = dim / total_dim * 0.96
        fig_arch.add_shape(
            type="rect",
            x0=x_start,
            y0=bar_y - 0.04,
            x1=x_start + width,
            y1=bar_y + 0.04,
            fillcolor=color + "44",
            line=dict(color=color, width=1),
        )
        if dim > 20:
            fig_arch.add_annotation(
                x=x_start + width / 2,
                y=bar_y,
                text=f"<b>{label}</b><br><span style='font-size:9px'>{dim} dims</span>",
                showarrow=False,
                font=dict(size=8.5, color=color, family="IBM Plex Mono"),
            )
        x_start += width

    fig_arch.add_annotation(
        x=0.0,
        y=bar_y,
        text="RL State Vector →",
        showarrow=False,
        xanchor="right",
        font=dict(size=9, color=COLORS["muted"], family="IBM Plex Mono"),
    )
    fig_arch.add_annotation(
        x=0.5,
        y=0.10,
        text="<b>GPU Acceleration Stack</b>   cuDF · cuML · CuPy · PyTorch CUDA · sentence-transformers",
        showarrow=False,
        font=dict(size=9.5, color=COLORS["muted"], family="IBM Plex Mono"),
    )

    fig_arch.update_layout(
        **PLOTLY_LAYOUT,
        height=340,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0.04, 1.0]),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_arch, use_container_width=True)

    section("GPU ACCELERATION LAYERS")
    gpu_table = {
        "Layer": [
            "Feature engineering",
            "GARCH batch recursion",
            "HMM Viterbi",
            "Factor rolling OLS",
            "Reward (Diff. Sharpe)",
            "RL policy network",
            "Brief embedding",
            "Backtest rolling metrics",
        ],
        "Technology": [
            "cuDF / cuML",
            "CuPy vectorized loop",
            "CuPy log-emission + DP",
            "CuPy einsum + solve",
            "CuPy EMA loop",
            "PyTorch CUDA (SB3)",
            "PyTorch CUDA (SentenceTransformers)",
            "CuPy prefix sums",
        ],
        "Est. Speedup": [
            "10–20×",
            "8–20×",
            "5–15×",
            "8–15×",
            "5–15×",
            "3–8×",
            "10–25×",
            "8–15×",
        ],
        "Fallback": [
            "pandas / sklearn",
            "NumPy",
            "NumPy",
            "NumPy",
            "NumPy",
            "CPU",
            "TF-IDF",
            "NumPy",
        ],
    }
    df_gpu = pd.DataFrame(gpu_table)
    st.dataframe(
        df_gpu.style.applymap(
            lambda v: "color: #f59e0b; font-family: IBM Plex Mono"
            if "×" in str(v)
            else "color: #06b6d4; font-family: IBM Plex Mono",
            subset=["Technology", "Est. Speedup"],
        ),
        hide_index=True,
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Market Data
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Market Data":
    st.markdown("## Market Data & Universe")

    col_ctrl, col_main = st.columns([1, 3])
    with col_ctrl:
        selected_ticker = st.selectbox("Ticker", D["tickers"])
        lookback = st.select_slider(
            "Lookback", ["1Y", "3Y", "5Y", "10Y", "Full"], value="5Y"
        )

    lb_map = {
        "1Y": 252,
        "3Y": 756,
        "5Y": 1260,
        "10Y": 2520,
        "Full": len(D["prices"]),
    }
    n = lb_map[lookback]
    p = D["prices"][selected_ticker].dropna().iloc[-n:]

    with col_main:
        section(f"PRICE HISTORY — {selected_ticker}")
        fig_price = go.Figure()
        fig_price.add_trace(
            go.Scatter(
                x=p.index,
                y=p.values,
                mode="lines",
                name=selected_ticker,
                line=dict(color=COLORS["amber"], width=1.5),
                fill="tozeroy",
                fillcolor=COLORS["amber"] + "18",
            )
        )
        fig_price.update_layout(
            **PLOTLY_LAYOUT, height=240, yaxis_title="Price (USD)", showlegend=False
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # Universe grid
    section("UNIVERSE OVERVIEW")
    cols = st.columns(5)
    for i, t in enumerate(D["tickers"]):
        px_series = D["prices"][t].dropna()
        if len(px_series) >= 252:
            chg = (px_series.iloc[-1] / px_series.iloc[-252] - 1) * 100
        elif len(px_series) >= 2:
            chg = (px_series.iloc[-1] / px_series.iloc[0] - 1) * 100
        else:
            chg = 0.0
        color = COLORS["green"] if chg >= 0 else COLORS["red"]
        sign = "+" if chg >= 0 else ""
        sector_short = D["sectors"].get(t, "Unknown")[:10]
        with cols[i % 5]:
            st.markdown(
                f"""
            <div class='card' style='text-align:center;padding:0.6rem'>
              <div style='font-family:IBM Plex Mono;font-size:0.85rem;
                   font-weight:600;color:{COLORS["amber"]}'>{t}</div>
              <div style='font-size:0.65rem;color:{COLORS["muted"]};
                   margin-bottom:0.2rem'>{sector_short}</div>
              <div style='font-family:IBM Plex Mono;font-size:0.8rem;color:{color}'>
                {sign}{chg:.1f}%</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Sector allocation
    section("SECTOR BREAKDOWN")
    c1, c2 = st.columns(2)
    with c1:
        sector_counts = {}
        for t in D["tickers"]:
            s = D["sectors"][t]
            sector_counts[s] = sector_counts.get(s, 0) + 1
        fig_sector = go.Figure(
            go.Pie(
                labels=list(sector_counts.keys()),
                values=list(sector_counts.values()),
                hole=0.55,
                marker=dict(
                    colors=[
                        COLORS["amber"],
                        COLORS["teal"],
                        COLORS["green"],
                        "#8b5cf6",
                        "#f43f5e",
                        "#f97316",
                        "#84cc16",
                        "#06b6d4",
                        "#a855f7",
                    ]
                ),
                textinfo="label+percent",
                textfont=dict(family="IBM Plex Mono", size=9),
            )
        )
        fig_sector.update_layout(
            **PLOTLY_LAYOUT,
            height=280,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_sector, use_container_width=True)

    with c2:
        # Macro signals
        section("MACRO SIGNALS (LATEST)")
        macro_latest = D["macro_df"].dropna(how="all").iloc[-1]
        _macro_label_map = {
            "fed_funds_rate": ("Fed Funds Rate", lambda v: f"{v:.2f}%"),
            "cpi_yoy": ("CPI YoY", lambda v: f"{v:.2f}%"),
            "vix": ("VIX", lambda v: f"{v:.1f}"),
            "yield_curve_10y2y": ("10Y-2Y Spread", lambda v: f"{v:.2f}%"),
            "hy_spread": ("HY OAS Spread", lambda v: f"{v:.0f} bps"),
        }
        for col, (label, fmt) in _macro_label_map.items():
            if col in macro_latest.index and not pd.isna(macro_latest[col]):
                val_str = fmt(macro_latest[col])
                muted_c = COLORS["muted"]
                amber_c = COLORS["amber"]
                border_c = COLORS["border"]
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"border-bottom:1px solid {border_c};padding:0.35rem 0;"
                    f"font-family:IBM Plex Mono;font-size:0.8rem'>"
                    f"<span style='color:{muted_c}'>{label}</span>"
                    f"<span style='color:{amber_c};font-weight:600'>{val_str}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Forecasting
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Forecasting":
    st.markdown("## Forecasting Module")
    st.markdown(
        badge("GARCH(1,1) Volatility", "amber")
        + "  "
        + badge("HMM Regime Detection", "teal")
        + "  "
        + badge("Fama-French FF3", "green"),
        unsafe_allow_html=True,
    )

    # ── GARCH vol ──
    section("GARCH VOLATILITY FORECASTS — QUARTERLY")
    garch_plot = D["garch_vol"].dropna(how="all")
    # Select up to 8 tickers that are present in the garch DataFrame
    plot_tickers = [t for t in D["tickers"][:8] if t in garch_plot.columns]
    recent_vol = garch_plot.iloc[-20:]
    fig_garch = go.Figure()
    for t in plot_tickers:
        fig_garch.add_trace(
            go.Scatter(
                x=recent_vol.index,
                y=recent_vol[t] * 100,
                mode="lines",
                name=t,
                line=dict(width=1.2),
            )
        )
    fig_garch.update_layout(
        **PLOTLY_LAYOUT,
        height=260,
        yaxis_title="Annualized Vol (%)",
        yaxis_ticksuffix="%",
    )
    st.plotly_chart(fig_garch, use_container_width=True)

    c1, c2 = st.columns(2)

    # ── HMM regime timeline ──
    with c1:
        section("HMM REGIME SEQUENCE")
        regime_recent = D["regime_df"].dropna(how="all").iloc[-24:]

        fig_regime = go.Figure()
        for state, label_name in [
            ("p_bear", "Bear"),
            ("p_sideways", "Sideways"),
            ("p_bull", "Bull"),
        ]:
            if state not in regime_recent.columns:
                continue
            color = {
                "p_bear": COLORS["red"],
                "p_sideways": COLORS["amber"],
                "p_bull": COLORS["green"],
            }[state]
            fig_regime.add_trace(
                go.Bar(
                    x=regime_recent.index,
                    y=regime_recent[state],
                    name=label_name,
                    marker_color=color + "99",
                    marker_line_width=0,
                )
            )
        fig_regime.update_layout(
            **PLOTLY_LAYOUT,
            barmode="stack",
            height=230,
            yaxis_title="Posterior Probability",
            yaxis=dict(**PLOTLY_LAYOUT["yaxis"], tickformat=".0%"),
            legend=dict(orientation="h", y=-0.25, x=0),
        )
        st.plotly_chart(fig_regime, use_container_width=True)

    # ── FF betas heatmap ──
    with c2:
        section("FAMA-FRENCH FACTOR EXPOSURES")
        ff = D["ff_betas"]
        if ff.empty:
            st.info("Factor exposures unavailable for this universe/period.")
        else:
            # Rename columns for display
            _col_display = {
                "alpha_ann": "α_ann",
                "beta_mkt": "β_MKT",
                "beta_smb": "β_SMB",
                "beta_hml": "β_HML",
            }
            ff_disp = ff.rename(columns=_col_display)
            disp_cols = [
                c for c in ["β_MKT", "β_SMB", "β_HML", "α_ann"] if c in ff_disp.columns
            ]
            ff_disp = ff_disp[disp_cols]
            fig_ff = go.Figure(
                go.Heatmap(
                    z=ff_disp.values,
                    x=disp_cols,
                    y=ff_disp.index.tolist(),
                    colorscale=[
                        [0.0, COLORS["red"] + "cc"],
                        [0.5, "#1e2d4a"],
                        [1.0, COLORS["green"] + "cc"],
                    ],
                    zmid=0,
                    text=[[f"{v:.2f}" for v in row] for row in ff_disp.values],
                    texttemplate="%{text}",
                    textfont=dict(size=9, family="IBM Plex Mono"),
                    showscale=True,
                    colorbar=dict(thickness=10, tickfont=dict(size=9)),
                )
            )
            fig_ff.update_layout(
                **PLOTLY_LAYOUT,
                height=max(260, len(ff_disp) * 18 + 60),
                xaxis=dict(**PLOTLY_LAYOUT["xaxis"], side="top"),
                margin=dict(l=60, r=20, t=30, b=10),
            )
            st.plotly_chart(fig_ff, use_container_width=True)

    # ── Vol vol surface ──
    section("CROSS-SECTIONAL VOLATILITY DISTRIBUTION OVER TIME")
    _garch_clean = D["garch_vol"].dropna(how="all")
    if not _garch_clean.empty:
        vol_quartiles = (
            _garch_clean.quantile([0.1, 0.25, 0.5, 0.75, 0.9], axis=1).T * 100
        )
        fig_vol = go.Figure()
        fills = [(0.1, 0.9, COLORS["teal"] + "18"), (0.25, 0.75, COLORS["teal"] + "30")]
        for lo, hi, fill in fills:
            fig_vol.add_trace(
                go.Scatter(
                    x=list(vol_quartiles.index) + list(vol_quartiles.index[::-1]),
                    y=list(vol_quartiles[lo]) + list(vol_quartiles[hi][::-1]),
                    fill="toself",
                    fillcolor=fill,
                    line=dict(width=0),
                    showlegend=False,
                )
            )
        fig_vol.add_trace(
            go.Scatter(
                x=vol_quartiles.index,
                y=vol_quartiles[0.5],
                name="Median",
                line=dict(color=COLORS["teal"], width=1.5),
            )
        )
        fig_vol.update_layout(
            **PLOTLY_LAYOUT, height=200, yaxis_title="Ann. Vol (%)", showlegend=False
        )
        st.plotly_chart(fig_vol, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Agent Insights
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Agent Insights":
    st.markdown("## Agent Intelligence")
    _mode_badge = (
        badge("Mock Mode", "amber") if D["mock_agents"] else badge("Live LLM", "green")
    )
    st.markdown(
        _mode_badge
        + "  MarketBrief from OrchestratorAgent — as of "
        + badge(D["market_brief"]["as_of_date"], "teal"),
        unsafe_allow_html=True,
    )

    mb = D["market_brief"]

    # Header cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Macro Regime", mb["macro_regime"].upper())
    with c2:
        st.metric("Portfolio Stance", mb["portfolio_stance"].upper())
    with c3:
        st.metric("Conviction Score", f"{mb['conviction_score']:.0%}")
    with c4:
        regime_color = {"risk_on": "green", "risk_off": "red", "transitional": "amber"}
        st.metric(
            "Signal",
            badge(mb["macro_regime"], regime_color.get(mb["macro_regime"], "teal")),
            label_visibility="hidden",
        )

    c_left, c_right = st.columns([1.5, 1])

    with c_left:
        # Sector tilts
        section("SECTOR TILTS")
        tilts = mb["sector_tilts"]
        sectors = list(tilts.keys())
        vals = list(tilts.values())
        colors = [COLORS["green"] if v > 0 else COLORS["red"] for v in vals]
        srt = sorted(zip(vals, sectors, colors))
        vals_s, secs_s, cols_s = zip(*srt)

        fig_tilt = go.Figure(
            go.Bar(
                x=list(vals_s),
                y=list(secs_s),
                orientation="h",
                marker=dict(color=list(cols_s), opacity=0.85),
                text=[f"{v:+.2f}" for v in vals_s],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=10),
            )
        )
        fig_tilt.add_vline(x=0, line=dict(color="#1e2d4a", width=1))
        fig_tilt.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            xaxis_title="Tilt (−1 = max underweight, +1 = max overweight)",
            showlegend=False,
            margin=dict(l=100, r=60, t=10, b=30),
        )
        st.plotly_chart(fig_tilt, use_container_width=True)

        # Key themes
        section("KEY INVESTMENT THEMES")
        for theme in mb["key_themes"]:
            tl = COLORS["teal"]
            br = COLORS["border"]
            st.markdown(
                f"<div style='padding:0.3rem 0;border-bottom:1px solid {br};"
                f"font-size:0.82rem;font-family:IBM Plex Sans'>"
                f"<span style='color:{tl};margin-right:0.5rem'>&#9656;</span>"
                f"{theme}</div>",
                unsafe_allow_html=True,
            )

    with c_right:
        # Over/under weights
        section("TOP POSITIONS")
        for t in mb["top_overweights"]:
            amber_c = COLORS["amber"]
            border_c = COLORS["border"]
            st.markdown(
                "<div style='display:flex;justify-content:space-between;"
                "padding:0.25rem 0;border-bottom:1px solid " + border_c + "'>"
                "<span style='font-family:IBM Plex Mono;font-weight:600;"
                "color:" + amber_c + "'>" + t + "</span>"
                "<span class='badge badge-green'>OVERWEIGHT</span></div>",
                unsafe_allow_html=True,
            )
        for t in mb["top_underweights"]:
            muted_c = COLORS["muted"]
            border_c = COLORS["border"]
            st.markdown(
                "<div style='display:flex;justify-content:space-between;"
                "padding:0.25rem 0;border-bottom:1px solid " + border_c + "'>"
                "<span style='font-family:IBM Plex Mono;font-weight:600;"
                "color:" + muted_c + "'>" + t + "</span>"
                "<span class='badge badge-red'>UNDERWEIGHT</span></div>",
                unsafe_allow_html=True,
            )
        section("RISK FLAGS")
        for flag in mb["risk_flags"]:
            red_c = COLORS["red"]
            border_c = COLORS["border"]
            st.markdown(
                "<div style='padding:0.25rem 0;border-bottom:1px solid "
                + border_c
                + ";"
                "font-size:0.8rem'>"
                "<span style='color:"
                + red_c
                + ";margin-right:0.5rem'>&#9888;</span>"
                + flag
                + "</div>",
                unsafe_allow_html=True,
            )
    # Executive summary
    section("EXECUTIVE SUMMARY")
    st.markdown(
        f"<div style='background:{COLORS['navy-2'] if False else '#0d1526'};"
        f"border:1px solid {COLORS['border']};border-left:3px solid {COLORS['amber']};"
        f"border-radius:4px;padding:1rem 1.2rem;"
        f"font-size:0.88rem;line-height:1.65;color:#d0d8f0'>"
        f"{mb['executive_summary']}</div>",
        unsafe_allow_html=True,
    )

    # Agent topology
    section("AGENT GRAPH TOPOLOGY (LangGraph)")
    fig_graph = go.Figure()
    nodes = [
        ("MacroAgent", 0.15, 0.75, COLORS["teal"]),
        ("SectorAgent×N", 0.50, 0.75, COLORS["amber"]),
        ("CompanyAgent×M", 0.50, 0.40, COLORS["green"]),
        ("OrchestratorAgent", 0.82, 0.57, "#8b5cf6"),
    ]
    start = ("START", 0.02, 0.57, COLORS["muted"])
    end = ("MarketBrief", 0.98, 0.57, COLORS["amber"])

    for name, x, y, color in [start] + nodes + [end]:
        is_node = name not in ("START", "MarketBrief")
        shape = "rect" if is_node else "circle"
        fig_graph.add_shape(
            type="rect",
            x0=x - 0.10,
            y0=y - 0.10,
            x1=x + 0.10,
            y1=y + 0.10,
            fillcolor=color + "22",
            line=dict(color=color, width=1.5),
        )
        fig_graph.add_annotation(
            x=x,
            y=y,
            text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(size=10, color=color, family="IBM Plex Mono"),
        )

    edges = [
        (0.02, 0.57, 0.15 - 0.10, 0.75, True),
        (0.02, 0.57, 0.50 - 0.10, 0.75, True),
        (0.15 + 0.10, 0.75, 0.50 - 0.10, 0.40, False),
        (0.50 + 0.10, 0.75, 0.50 + 0.10, 0.40, False),
        (0.50 + 0.10, 0.40, 0.82 - 0.10, 0.57, True),
        (0.82 + 0.10, 0.57, 0.98 - 0.10, 0.57, True),
    ]
    for x0, y0, x1, y1, arrow in edges:
        fig_graph.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            axref="x",
            ayref="y",
            arrowhead=2 if arrow else 0,
            arrowsize=1.2,
            arrowwidth=1.2,
            arrowcolor="#1e2d4a",
        )

    fig_graph.add_annotation(
        x=0.50,
        y=0.10,
        text="⟲  Macro + Sector nodes run in <b>parallel</b> · "
        "CompanyAgent × M runs after both · Orchestrator synthesizes all",
        showarrow=False,
        font=dict(size=9, color=COLORS["muted"], family="IBM Plex Mono"),
    )

    fig_graph.update_layout(
        **PLOTLY_LAYOUT,
        height=260,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_graph, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Portfolio
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Portfolio":
    st.markdown("## Portfolio Rebalancer")
    st.markdown(
        badge("Quarterly MDP", "teal")
        + "  "
        + badge("PPO Policy", "amber")
        + "  "
        + badge("Tax-Aware FIFO", "green"),
        unsafe_allow_html=True,
    )

    # Current weights: equal-weight baseline (PPO not trained)
    weights = D["weights"]  # already normalized equal weights

    c1, c2 = st.columns(2)

    with c1:
        section("CURRENT PORTFOLIO WEIGHTS")
        sorted_w = sorted(weights.items(), key=lambda x: -x[1])
        tickers_ = [t for t, _ in sorted_w]
        wvals_ = [w for _, w in sorted_w]
        colors_ = [
            COLORS["amber"]
            if t in D["market_brief"]["top_overweights"]
            else COLORS["red"]
            if t in D["market_brief"]["top_underweights"]
            else COLORS["teal"]
            for t in tickers_
        ]

        fig_w = go.Figure(
            go.Bar(
                x=tickers_,
                y=wvals_,
                marker=dict(color=colors_, opacity=0.85),
                text=[f"{w:.1%}" for w in wvals_],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=9),
            )
        )
        fig_w.add_hline(
            y=1 / len(tickers_),
            line=dict(color=COLORS["muted"], width=1, dash="dot"),
            annotation_text="1/N",
            annotation_font_size=9,
        )
        fig_w.update_layout(
            **PLOTLY_LAYOUT,
            height=260,
            yaxis_title="Weight",
            yaxis_tickformat=".0%",
            showlegend=False,
        )
        st.plotly_chart(fig_w, use_container_width=True)

    with c2:
        section("PORTFOLIO TREEMAP")
        fig_tree = go.Figure(
            go.Treemap(
                labels=list(weights.keys()),
                parents=["" for _ in weights],
                values=list(weights.values()),
                texttemplate="<b>%{label}</b><br>%{percentRoot:.1%}",
                textfont=dict(family="IBM Plex Mono", size=11),
                marker=dict(
                    colors=[weights[t] for t in weights],
                    colorscale=[
                        [0, "#0d1526"],
                        [0.5, COLORS["teal"] + "88"],
                        [1, COLORS["amber"]],
                    ],
                    line=dict(color=COLORS["border"], width=1),
                ),
            )
        )
        fig_tree.update_layout(
            **PLOTLY_LAYOUT, height=260, margin=dict(l=5, r=5, t=5, b=5)
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    # Differential Sharpe reward demo
    section("DIFFERENTIAL SHARPE REWARD — EPISODE SIMULATION")
    rng_ep = np.random.default_rng(77)
    ep_returns = np.concatenate(
        [
            rng_ep.normal(0.025, 0.040, 10),
            rng_ep.normal(-0.055, 0.110, 4),
            rng_ep.normal(0.030, 0.050, 10),
        ]
    )
    eta = 0.05
    A, B = 0.0, 0.0
    dsr_seq = []
    for r in ep_returns:
        A_p, B_p = A, B
        A = A_p + eta * (r - A_p)
        B = B_p + eta * (r**2 - B_p)
        denom = max(B_p - A_p**2, 1e-9) ** 1.5
        dsr_seq.append((B_p * (A - A_p) - 0.5 * A_p * (B - B_p)) / denom)

    ep_dates = pd.date_range("2022-01-01", periods=len(ep_returns), freq="QE")

    fig_ep = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08
    )
    fig_ep.add_trace(
        go.Bar(
            x=ep_dates,
            y=ep_returns * 100,
            name="Portfolio Return",
            marker=dict(
                color=[COLORS["green"] if r > 0 else COLORS["red"] for r in ep_returns],
                opacity=0.85,
            ),
        ),
        row=1,
        col=1,
    )
    fig_ep.add_trace(
        go.Scatter(
            x=ep_dates,
            y=dsr_seq,
            name="DSR Increment D_t",
            line=dict(color=COLORS["amber"], width=2),
            fill="tozeroy",
            fillcolor=COLORS["amber"] + "22",
        ),
        row=2,
        col=1,
    )
    fig_ep.add_vrect(
        x0=ep_dates[10],
        x1=ep_dates[13],
        fillcolor=COLORS["red"] + "18",
        line_width=0,
        annotation_text="Crash",
        annotation_font_size=9,
        annotation_position="top left",
    )
    fig_ep.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0),
    )
    fig_ep.update_yaxes(
        title_text="Return (%)",
        row=1,
        col=1,
        ticksuffix="%",
        **{k: v for k, v in PLOTLY_LAYOUT["yaxis"].items()},
    )
    fig_ep.update_yaxes(
        title_text="DSR D_t",
        row=2,
        col=1,
        **{k: v for k, v in PLOTLY_LAYOUT["yaxis"].items()},
    )
    st.plotly_chart(fig_ep, use_container_width=True)

    # Tax model
    section("TAX MODEL — FIFO LOT SIMULATION")
    c3, c4 = st.columns(2)
    with c3:
        hold_periods = list(range(1, 9))
        lt_rate, st_rate = 0.15, 0.37
        rates = [st_rate if h <= 4 else lt_rate for h in hold_periods]
        fig_tax = go.Figure(
            go.Bar(
                x=[f"Q{h}" for h in hold_periods],
                y=[r * 100 for r in rates],
                marker=dict(
                    color=[
                        COLORS["red"] + "cc" if r == st_rate else COLORS["green"] + "cc"
                        for r in rates
                    ],
                    line=dict(width=0),
                ),
                text=[f"{r * 100:.0f}%" for r in rates],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=10),
            )
        )
        fig_tax.add_hline(
            y=37,
            line=dict(color=COLORS["red"], width=1, dash="dot"),
            annotation_text="Short-term 37%",
            annotation_font_size=8,
        )
        fig_tax.add_hline(
            y=15,
            line=dict(color=COLORS["green"], width=1, dash="dot"),
            annotation_text="Long-term 15%",
            annotation_font_size=8,
        )
        fig_tax.add_vline(
            x=3.5,
            line=dict(color=COLORS["muted"], width=1, dash="dash"),
            annotation_text="LT threshold",
            annotation_font_size=8,
        )
        fig_tax.update_layout(
            **PLOTLY_LAYOUT,
            height=220,
            yaxis_title="Capital Gains Tax Rate (%)",
            xaxis_title="Holding Period",
            showlegend=False,
        )
        st.plotly_chart(fig_tax, use_container_width=True)

    with c4:
        section("REWARD COMPONENT WEIGHTS")
        lambda_vals = {
            "DSR base": 1.0,
            "λ_tax": -0.5,
            "λ_tlh": 0.3,
            "λ_turnover": -0.1,
        }
        fig_lam = go.Figure(
            go.Bar(
                x=list(lambda_vals.keys()),
                y=list(lambda_vals.values()),
                marker=dict(
                    color=[
                        COLORS["amber"] if v > 0 else COLORS["red"] + "bb"
                        for v in lambda_vals.values()
                    ],
                    opacity=0.9,
                ),
                text=[f"{v:+.1f}" for v in lambda_vals.values()],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=11),
            )
        )
        fig_lam.add_hline(y=0, line=dict(color="#1e2d4a", width=1))
        fig_lam.update_layout(
            **PLOTLY_LAYOUT, height=220, yaxis_title="Weight", showlegend=False
        )
        st.plotly_chart(fig_lam, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Backtest
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Backtest":
    st.markdown("## Backtest Report")
    st.markdown(
        badge("Walk-Forward OOS", "teal")
        + "  "
        + badge("3 Strategies", "amber")
        + "  "
        + badge("Tax-Adjusted", "green")
        + "  "
        + badge(f"{len(D['test_dates'])} Test Quarters", "muted"),
        unsafe_allow_html=True,
    )

    st.info(
        "ℹ PPO agent has not been trained — showing Equal Weight as the active strategy. "
        "Run `rl/pipeline.py` to train the PPO agent and replace EW with trained returns.",
        icon=None,
    )

    ew = D["ew_ret"]
    hold = D["hold_ret"]
    bm = D["bm_ret"]
    td = D["test_dates"]

    # Use EW as the primary strategy (PPO not trained)
    strategies = {
        "Equal Weight": ew,
        "Buy & Hold": hold,
        "Benchmark (SPY)": bm,
    }
    _STRATEGY_COLORS_ACTUAL = {
        "Equal Weight": COLORS["amber"],
        "Buy & Hold": COLORS["teal"],
        "Benchmark (SPY)": COLORS["muted"],
    }

    # Summary metrics
    section("AGGREGATE PERFORMANCE SUMMARY")
    mc = {
        name: {
            "Ann. Return": f"{_ann_ret(r) * 100:.2f}%",
            "Volatility": f"{r.std(ddof=1) * 2 * 100:.2f}%",
            "Sharpe": f"{_sharpe(r):.3f}",
            "Sortino": f"{_sortino(r):.3f}",
            "Calmar": f"{_calmar(r):.3f}",
            "Max Drawdown": f"{_max_dd(r) * 100:.2f}%",
            "Alpha (ann.)": f"{(r - bm).mean() * 4 * 100:.2f}%"
            if name != "Benchmark (SPY)"
            else "—",
            "Hit Rate": f"{(r > 0).mean() * 100:.0f}%",
        }
        for name, r in strategies.items()
    }

    df_summary = pd.DataFrame(mc).T
    st.dataframe(
        df_summary.style.apply(
            lambda col: [
                f"color: {COLORS['amber']}; font-family: IBM Plex Mono"
                if col.name in ("Sharpe", "Sortino", "Calmar", "Ann. Return")
                else f"color: {COLORS['white']}; font-family: IBM Plex Mono"
                for _ in col
            ],
            axis=0,
        ),
        use_container_width=True,
    )

    # Cumulative return chart
    section("CUMULATIVE RETURN — TEST PERIOD")
    fig_cum = go.Figure()
    for name, r in strategies.items():
        cum = np.cumprod(1 + r)
        color = _STRATEGY_COLORS_ACTUAL.get(name, COLORS["muted"])
        fig_cum.add_trace(
            go.Scatter(
                x=td,
                y=cum,
                name=name,
                line=dict(color=color, width=2.0 if name == "Equal Weight" else 1.4),
            )
        )
    fig_cum.add_hline(y=1.0, line=dict(color=COLORS["border"], width=1, dash="dot"))
    fig_cum.update_layout(
        **PLOTLY_LAYOUT, height=280, yaxis_title="Portfolio Value (1 = start)"
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Drawdown + rolling Sharpe
    c1, c2 = st.columns(2)

    with c1:
        section("DRAWDOWN PATHS")
        fig_dd = go.Figure()
        for name, r in strategies.items():
            cum = np.cumprod(1 + r)
            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / (peak + 1e-9)
            color = _STRATEGY_COLORS_ACTUAL.get(name, COLORS["muted"])
            fig_dd.add_trace(
                go.Scatter(
                    x=td,
                    y=dd * 100,
                    name=name,
                    fill="tozeroy" if name == "Equal Weight" else None,
                    fillcolor=color + "18",
                    line=dict(
                        color=color,
                        width=1.5 if name == "Equal Weight" else 1.0,
                    ),
                )
            )
        fig_dd.update_layout(
            **PLOTLY_LAYOUT,
            height=240,
            yaxis_title="Drawdown (%)",
            yaxis_ticksuffix="%",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    with c2:
        section("ROLLING SHARPE RATIO (W=8 QUARTERS)")
        W = 8
        fig_rs = go.Figure()
        for name, r in strategies.items():
            rs = np.full(len(r), np.nan)
            for t_i in range(W - 1, len(r)):
                w_sl = r[t_i - W + 1 : t_i + 1]
                rs[t_i] = w_sl.mean() / (w_sl.std(ddof=1) + 1e-9) * np.sqrt(4)
            color = _STRATEGY_COLORS_ACTUAL.get(name, COLORS["muted"])
            fig_rs.add_trace(
                go.Scatter(
                    x=td,
                    y=rs,
                    name=name,
                    line=dict(
                        color=color,
                        width=1.5 if name == "Equal Weight" else 1.0,
                    ),
                )
            )
        fig_rs.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
        fig_rs.add_hline(
            y=1,
            line=dict(color=COLORS["green"] + "55", width=1, dash="dot"),
            annotation_text="Sharpe=1",
            annotation_font_size=8,
        )
        fig_rs.update_layout(**PLOTLY_LAYOUT, height=240, yaxis_title="Sharpe")
        st.plotly_chart(fig_rs, use_container_width=True)

    # Per-fold breakdown
    section("WALK-FORWARD FOLD BREAKDOWN (LAST 3 FOLDS)")
    fold_rows = []
    for fd in D["folds"]:
        for strat, key in [("Equal Weight", "ew"), ("Benchmark (SPY)", "bm")]:
            fold_rows.append(
                {
                    "Fold": f"Fold {fd['fold']}",
                    "Strategy": strat,
                    "Total Return": f"{fd[key]['total_return'] * 100:.2f}%",
                    "Sharpe": f"{fd[key]['sharpe']:.3f}",
                    "Max Drawdown": f"{fd[key]['max_drawdown'] * 100:.2f}%",
                    "Alpha (ann.)": f"{fd[key]['alpha'] * 100:.2f}%"
                    if key != "bm"
                    else "—",
                }
            )
    fold_df = pd.DataFrame(fold_rows)

    c3, c4 = st.columns(2)
    with c3:
        fig_fold = go.Figure()
        for strat, color in [
            ("Equal Weight", COLORS["amber"]),
            ("Benchmark (SPY)", COLORS["muted"]),
        ]:
            mask = fold_df["Strategy"] == strat
            fdata = fold_df[mask]
            sharpes = [float(s) for s in fdata["Sharpe"]]
            fig_fold.add_trace(
                go.Bar(
                    name=strat,
                    x=fdata["Fold"].tolist(),
                    y=sharpes,
                    marker=dict(color=color, opacity=0.85),
                )
            )
        fig_fold.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
        fig_fold.update_layout(
            **PLOTLY_LAYOUT,
            barmode="group",
            height=230,
            yaxis_title="Sharpe Ratio",
            legend=dict(orientation="h", y=-0.3, x=0),
        )
        st.plotly_chart(fig_fold, use_container_width=True)

    with c4:
        st.dataframe(
            fold_df.style.applymap(
                lambda v: f"color:{COLORS['amber']};font-family:IBM Plex Mono"
                if isinstance(v, str)
                and v.replace("%", "")
                .replace(".", "")
                .replace("-", "")
                .replace("+", "")
                .lstrip()
                .isnumeric()
                else "font-family:IBM Plex Mono"
            ),
            height=230,
            hide_index=True,
            use_container_width=True,
        )

    # Tax drag
    section("TAX DRAG ANALYSIS")
    # Estimate tax cost: proportional to turnover × avg gain × short-term rate
    # For EW quarterly rebalancing: avg turnover ≈ 20%, gain rate ≈ return/2
    tc_ew = np.abs(ew) * 0.20 * 0.37  # 20% turnover, short-term rate
    tc_hold = np.zeros(len(hold))  # buy-and-hold: minimal turnover

    gross_ret = {
        k: _ann_ret(v) for k, v in {"Equal Weight": ew, "Buy & Hold": hold}.items()
    }
    drag = {"Equal Weight": tc_ew.sum() / 4, "Buy & Hold": 0.0}
    after_tax = {k: gross_ret[k] - drag[k] for k in gross_ret}

    fig_tax2 = go.Figure()
    x_strats = list(gross_ret.keys())
    fig_tax2.add_trace(
        go.Bar(
            name="Gross Return",
            x=x_strats,
            y=[v * 100 for v in gross_ret.values()],
            marker=dict(color=COLORS["amber"] + "99", opacity=0.9),
        )
    )
    fig_tax2.add_trace(
        go.Bar(
            name="After-Tax Return",
            x=x_strats,
            y=[v * 100 for v in after_tax.values()],
            marker=dict(color=COLORS["green"] + "99", opacity=0.9),
        )
    )
    fig_tax2.add_trace(
        go.Bar(
            name="Tax Drag",
            x=x_strats,
            y=[v * 100 for v in drag.values()],
            marker=dict(color=COLORS["red"] + "99", opacity=0.9),
        )
    )
    fig_tax2.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        height=230,
        yaxis_title="Annualized (%)",
        yaxis_ticksuffix="%",
        legend=dict(orientation="h", y=-0.3, x=0),
    )
    st.plotly_chart(fig_tax2, use_container_width=True)

    # Export
    section("EXPORT")
    c5, c6, c7 = st.columns(3)
    with c5:
        csv_data = df_summary.to_csv().encode()
        st.download_button(
            "⬇ Summary CSV", csv_data, "backtest_summary.csv", "text/csv"
        )
    with c6:
        json_data = json.dumps(
            {
                "strategies": list(strategies.keys()),
                "test_period": {
                    "start": str(td[0].date()) if len(td) else "",
                    "end": str(td[-1].date()) if len(td) else "",
                    "n_quarters": len(td),
                },
                "aggregate_metrics": {
                    n: {
                        "sharpe": round(_sharpe(r), 4),
                        "ann_return": round(_ann_ret(r), 4),
                        "max_dd": round(_max_dd(r), 4),
                    }
                    for n, r in strategies.items()
                },
            },
            indent=2,
        ).encode()
        st.download_button(
            "⬇ Report JSON", json_data, "backtest_report.json", "application/json"
        )
    with c7:
        muted_c = COLORS["muted"]
        amber_c = COLORS["amber"]
        green_c = COLORS["green"]
        ew_sharpe = _sharpe(ew)
        bm_sharpe = _sharpe(bm)
        sharpe_diff_val = ew_sharpe - bm_sharpe
        st.markdown(
            "<div style='font-size:0.75rem;color:" + muted_c + ";"
            "font-family:IBM Plex Mono;padding-top:0.5rem'>"
            "Best strategy: <b style='color:" + amber_c + "'>Equal Weight</b><br>"
            "Sharpe vs Benchmark: <b style='color:" + green_c + "'>"
            f"{sharpe_diff_val:+.3f}</b></div>",
            unsafe_allow_html=True,
        )
