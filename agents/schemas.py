"""
agents/schemas.py
=================
Structured output schemas for the QuantAgent-RL agents module.

Implemented as Python dataclasses with __post_init__ validation so the
module works without pydantic installed (though pydantic is listed in
requirements.txt for richer validation in production). Every field
carries a docstring-level description so downstream consumers know
exactly what each signal means.

Walk-Forward Safety
-------------------
Every schema carries ``as_of_date`` so consumers can verify that the
brief was produced with the correct information cutoff.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_range(name: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


# ---------------------------------------------------------------------------
# MacroBrief
# ---------------------------------------------------------------------------


@dataclass
class MacroBrief:
    """Macro-economic environment assessment produced by MacroAgent.

    Fields
    ------
    rate_environment : str
        Direction of monetary policy: 'tightening', 'neutral', or 'easing'.
    inflation_regime : str
        Inflation characterization: 'high', 'elevated', 'moderate', or 'low'.
    recession_risk : float
        Probability of US recession within 12 months. Range [0, 1].
    yield_curve_signal : str
        Shape of the yield curve: 'inverted', 'flat', or 'normal'.
    credit_stress : float
        High-yield spread stress level. Range [0, 1].
    overall_sentiment : float
        Aggregate macro sentiment. Range [-1, 1] where -1 = strongly bearish.
    key_risks : list[str]
        Up to 5 concise macro risk factors.
    tailwinds : list[str]
        Up to 5 concise macro tailwinds.
    analyst_summary : str
        Two to three sentence narrative synthesis.
    """

    rate_environment: str
    inflation_regime: str
    recession_risk: float
    yield_curve_signal: str
    credit_stress: float
    overall_sentiment: float
    key_risks: list[str] = field(default_factory=list)
    tailwinds: list[str] = field(default_factory=list)
    analyst_summary: str = ""

    def __post_init__(self) -> None:
        _check_choice(
            "rate_environment",
            self.rate_environment,
            ("tightening", "neutral", "easing"),
        )
        _check_choice(
            "inflation_regime",
            self.inflation_regime,
            ("high", "elevated", "moderate", "low"),
        )
        _check_choice(
            "yield_curve_signal",
            self.yield_curve_signal,
            ("inverted", "flat", "normal"),
        )
        _check_range("recession_risk", self.recession_risk, 0.0, 1.0)
        _check_range("credit_stress", self.credit_stress, 0.0, 1.0)
        _check_range("overall_sentiment", self.overall_sentiment, -1.0, 1.0)
        self.key_risks = list(self.key_risks)[:5]
        self.tailwinds = list(self.tailwinds)[:5]

    @classmethod
    def neutral(cls) -> "MacroBrief":
        """Return a neutral placeholder brief (used in mock mode or on failure)."""
        return cls(
            rate_environment="neutral",
            inflation_regime="moderate",
            recession_risk=0.25,
            yield_curve_signal="flat",
            credit_stress=0.3,
            overall_sentiment=0.0,
            key_risks=["Data unavailable"],
            tailwinds=["Data unavailable"],
            analyst_summary="Macro assessment unavailable — using neutral defaults.",
        )

    def to_dict(self) -> dict:
        return {
            "rate_environment": self.rate_environment,
            "inflation_regime": self.inflation_regime,
            "recession_risk": self.recession_risk,
            "yield_curve_signal": self.yield_curve_signal,
            "credit_stress": self.credit_stress,
            "overall_sentiment": self.overall_sentiment,
            "key_risks": self.key_risks,
            "tailwinds": self.tailwinds,
            "analyst_summary": self.analyst_summary,
        }


# ---------------------------------------------------------------------------
# SectorBrief
# ---------------------------------------------------------------------------


@dataclass
class SectorBrief:
    """Per-sector outlook produced by SectorAgent.

    Fields
    ------
    sector : str
        GICS sector name (e.g., 'Information Technology').
    momentum_score : float
        Near-term sector momentum. Range [-1, 1].
    earnings_revision_trend : str
        Direction of analyst estimate revisions: 'upgrades', 'neutral', 'downgrades'.
    valuation_signal : str
        Valuation characterization: 'stretched', 'fair', or 'cheap'.
    key_themes : list[str]
        Up to 5 concise investment themes driving the sector.
    risks : list[str]
        Up to 3 concise sector-specific risks.
    analyst_summary : str
        Two to three sentence sector narrative.
    """

    sector: str
    momentum_score: float
    earnings_revision_trend: str
    valuation_signal: str
    key_themes: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    analyst_summary: str = ""

    def __post_init__(self) -> None:
        _check_choice(
            "earnings_revision_trend",
            self.earnings_revision_trend,
            ("upgrades", "neutral", "downgrades"),
        )
        _check_choice(
            "valuation_signal", self.valuation_signal, ("stretched", "fair", "cheap")
        )
        _check_range("momentum_score", self.momentum_score, -1.0, 1.0)
        self.key_themes = list(self.key_themes)[:5]
        self.risks = list(self.risks)[:3]

    @classmethod
    def neutral(cls, sector: str) -> "SectorBrief":
        return cls(
            sector=sector,
            momentum_score=0.0,
            earnings_revision_trend="neutral",
            valuation_signal="fair",
            key_themes=["Data unavailable"],
            risks=["Data unavailable"],
            analyst_summary=f"{sector} assessment unavailable — using neutral defaults.",
        )

    def to_dict(self) -> dict:
        return {
            "sector": self.sector,
            "momentum_score": self.momentum_score,
            "earnings_revision_trend": self.earnings_revision_trend,
            "valuation_signal": self.valuation_signal,
            "key_themes": self.key_themes,
            "risks": self.risks,
            "analyst_summary": self.analyst_summary,
        }


# ---------------------------------------------------------------------------
# CompanyBrief
# ---------------------------------------------------------------------------


@dataclass
class CompanyBrief:
    """Per-company fundamental assessment produced by CompanyAgent.

    Fields
    ------
    ticker : str
        Stock ticker symbol.
    revenue_growth_trend : str
        Revenue trajectory: 'accelerating', 'stable', 'decelerating', or 'negative'.
    margin_trend : str
        Profitability trend: 'expanding', 'stable', or 'compressing'.
    balance_sheet_quality : str
        Financial health: 'strong', 'adequate', or 'stretched'.
    earnings_quality : str
        Quality of reported earnings: 'high', 'medium', or 'low'.
    fundamental_score : float
        Aggregate fundamental signal. Range [-1, 1].
    key_risks : list[str]
        Up to 4 concise company-specific risks.
    key_catalysts : list[str]
        Up to 4 concise near-term catalysts.
    analyst_summary : str
        Two to three sentence company narrative.
    """

    ticker: str
    revenue_growth_trend: str
    margin_trend: str
    balance_sheet_quality: str
    earnings_quality: str
    fundamental_score: float
    key_risks: list[str] = field(default_factory=list)
    key_catalysts: list[str] = field(default_factory=list)
    analyst_summary: str = ""

    def __post_init__(self) -> None:
        _check_choice(
            "revenue_growth_trend",
            self.revenue_growth_trend,
            ("accelerating", "stable", "decelerating", "negative"),
        )
        _check_choice(
            "margin_trend", self.margin_trend, ("expanding", "stable", "compressing")
        )
        _check_choice(
            "balance_sheet_quality",
            self.balance_sheet_quality,
            ("strong", "adequate", "stretched"),
        )
        _check_choice(
            "earnings_quality", self.earnings_quality, ("high", "medium", "low")
        )
        _check_range("fundamental_score", self.fundamental_score, -1.0, 1.0)
        self.key_risks = list(self.key_risks)[:4]
        self.key_catalysts = list(self.key_catalysts)[:4]

    @classmethod
    def neutral(cls, ticker: str) -> "CompanyBrief":
        return cls(
            ticker=ticker,
            revenue_growth_trend="stable",
            margin_trend="stable",
            balance_sheet_quality="adequate",
            earnings_quality="medium",
            fundamental_score=0.0,
            key_risks=["Data unavailable"],
            key_catalysts=["Data unavailable"],
            analyst_summary=f"{ticker} assessment unavailable — using neutral defaults.",
        )

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "revenue_growth_trend": self.revenue_growth_trend,
            "margin_trend": self.margin_trend,
            "balance_sheet_quality": self.balance_sheet_quality,
            "earnings_quality": self.earnings_quality,
            "fundamental_score": self.fundamental_score,
            "key_risks": self.key_risks,
            "key_catalysts": self.key_catalysts,
            "analyst_summary": self.analyst_summary,
        }


# ---------------------------------------------------------------------------
# MarketBrief
# ---------------------------------------------------------------------------


@dataclass
class MarketBrief:
    """Unified market brief produced by OrchestratorAgent.

    This is the primary output of the agents module and the qualitative
    component of the RL state vector. Embedded into a dense vector by
    ``agents.embedder.MarketBriefEmbedder``.

    Fields
    ------
    as_of_date : str
        Date the brief was produced (YYYY-MM-DD).
    macro_regime : str
        High-level risk environment: 'risk_on', 'risk_off', or 'transitional'.
    portfolio_stance : str
        Suggested portfolio posture: 'aggressive', 'neutral', or 'defensive'.
    conviction_score : float
        Confidence in the brief's signals. Range [0, 1].
    top_overweights : list[str]
        Tickers with the strongest positive signals (up to 5).
    top_underweights : list[str]
        Tickers with the strongest negative signals (up to 5).
    sector_tilts : dict[str, float]
        Per-sector allocation tilt. Values in [-1, 1].
    key_themes : list[str]
        Up to 6 cross-asset investment themes.
    risk_flags : list[str]
        Up to 4 portfolio-level risk warnings.
    executive_summary : str
        Three to five sentence investment narrative.
    macro_brief : MacroBrief | None
        Source macro brief (auditability).
    sector_briefs : dict[str, SectorBrief]
        Source sector briefs (auditability).
    company_briefs : dict[str, CompanyBrief]
        Source company briefs (auditability).
    """

    as_of_date: str
    macro_regime: str
    portfolio_stance: str
    conviction_score: float
    top_overweights: list[str] = field(default_factory=list)
    top_underweights: list[str] = field(default_factory=list)
    sector_tilts: dict[str, float] = field(default_factory=dict)
    key_themes: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    executive_summary: str = ""
    macro_brief: MacroBrief | None = None
    sector_briefs: dict[str, SectorBrief] = field(default_factory=dict)
    company_briefs: dict[str, CompanyBrief] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_choice(
            "macro_regime", self.macro_regime, ("risk_on", "risk_off", "transitional")
        )
        _check_choice(
            "portfolio_stance",
            self.portfolio_stance,
            ("aggressive", "neutral", "defensive"),
        )
        _check_range("conviction_score", self.conviction_score, 0.0, 1.0)
        # Clamp sector tilts to [-1, 1]
        self.sector_tilts = {
            k: max(-1.0, min(1.0, v)) for k, v in self.sector_tilts.items()
        }
        self.top_overweights = list(self.top_overweights)[:5]
        self.top_underweights = list(self.top_underweights)[:5]
        self.key_themes = list(self.key_themes)[:6]
        self.risk_flags = list(self.risk_flags)[:4]

    @classmethod
    def neutral(
        cls, as_of_date: str, tickers: list[str] | None = None
    ) -> "MarketBrief":
        return cls(
            as_of_date=as_of_date,
            macro_regime="transitional",
            portfolio_stance="neutral",
            conviction_score=0.0,
            executive_summary="Market brief unavailable — all signals neutral.",
        )

    def to_text(self) -> str:
        """Serialize to a prose string optimized for sentence embedding."""
        lines = [
            f"Market brief as of {self.as_of_date}.",
            f"Macro regime: {self.macro_regime}. Portfolio stance: {self.portfolio_stance}.",
            f"Conviction: {self.conviction_score:.2f}.",
        ]
        if self.key_themes:
            lines.append("Key themes: " + "; ".join(self.key_themes) + ".")
        if self.risk_flags:
            lines.append("Risk flags: " + "; ".join(self.risk_flags) + ".")
        if self.top_overweights:
            lines.append("Top overweights: " + ", ".join(self.top_overweights) + ".")
        if self.top_underweights:
            lines.append("Top underweights: " + ", ".join(self.top_underweights) + ".")
        if self.sector_tilts:
            tilt_strs = [f"{s}: {v:+.2f}" for s, v in self.sector_tilts.items()]
            lines.append("Sector tilts: " + "; ".join(tilt_strs) + ".")
        if self.executive_summary:
            lines.append(self.executive_summary)
        if self.macro_brief:
            lines.append(
                f"Macro: rate={self.macro_brief.rate_environment}, "
                f"inflation={self.macro_brief.inflation_regime}, "
                f"recession_risk={self.macro_brief.recession_risk:.2f}."
            )
        return " ".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (suitable for JSON caching)."""
        return {
            "as_of_date": self.as_of_date,
            "macro_regime": self.macro_regime,
            "portfolio_stance": self.portfolio_stance,
            "conviction_score": self.conviction_score,
            "top_overweights": self.top_overweights,
            "top_underweights": self.top_underweights,
            "sector_tilts": self.sector_tilts,
            "key_themes": self.key_themes,
            "risk_flags": self.risk_flags,
            "executive_summary": self.executive_summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MarketBrief":
        """Deserialize from a plain dict (e.g., loaded from JSON cache)."""
        return cls(
            as_of_date=data.get("as_of_date", ""),
            macro_regime=data.get("macro_regime", "transitional"),
            portfolio_stance=data.get("portfolio_stance", "neutral"),
            conviction_score=float(data.get("conviction_score", 0.0)),
            top_overweights=data.get("top_overweights", []),
            top_underweights=data.get("top_underweights", []),
            sector_tilts=data.get("sector_tilts", {}),
            key_themes=data.get("key_themes", []),
            risk_flags=data.get("risk_flags", []),
            executive_summary=data.get("executive_summary", ""),
        )
