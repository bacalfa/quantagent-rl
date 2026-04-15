"""
backtest/config.py
==================
Configuration for the QuantAgent-RL backtest module.
"""

from dataclasses import dataclass, field


@dataclass
class BacktestConfig:
    """Master configuration for the walk-forward backtest engine.

    Parameters
    ----------
    benchmark_ticker : str
        Ticker used as the market benchmark for alpha, beta, and
        information ratio calculations (e.g. 'SPY').
    periods_per_year : int
        Number of rebalancing periods per calendar year.
        4 = quarterly (the default), 12 = monthly, 252 = daily.
    rolling_window : int
        Number of rebalancing periods in the rolling Sharpe / alpha /
        drawdown windows.  12 = 3 years of quarterly data.
    risk_free_rate : float
        Annualized risk-free rate used for excess-return calculations.
        Applied as ``risk_free_rate / periods_per_year`` per period.
    min_periods_rolling : int
        Minimum number of non-null observations required before a
        rolling metric is emitted (avoids NaN-heavy early windows).
    use_gpu : bool | None
        True  → force CuPy (raises if unavailable).
        False → force NumPy.
        None  → auto-detect (CuPy if available, else NumPy).
    strategies : list[str]
        Strategy labels to include in the report.  Must match the keys
        used when constructing ``StrategyReturns`` objects.
        An empty list means all available strategies are included.
    include_tax_metrics : bool
        If True, compute effective tax drag and after-tax return metrics.
    include_attribution : bool
        If True, compute rolling period-level return attribution
        (sector tilts contribution, not full Brinson attribution).
    """

    benchmark_ticker: str = "SPY"
    periods_per_year: int = 4
    rolling_window: int = 12
    risk_free_rate: float = 0.04
    min_periods_rolling: int = 6
    use_gpu: bool | None = None
    strategies: list[str] = field(default_factory=list)
    include_tax_metrics: bool = True
    include_attribution: bool = False

    @property
    def rf_per_period(self) -> float:
        """Per-period risk-free rate (annualized rate / periods_per_year)."""
        return self.risk_free_rate / self.periods_per_year
