"""
backtest
========
QuantAgent-RL walk-forward backtest module.

Public API
----------
BacktestEngine      : run() / run_from_fold_results() → BacktestReport
BacktestReport      : aggregate + fold-level metrics, tearsheets, export
TearsheetData       : time-series data for one strategy tear sheet
StrategyReturns     : input container (return series + metadata)
MetricsCalculator   : GPU-accelerated scalar + rolling metric computation
BacktestConfig      : configuration dataclass

Quick start
-----------
From RLFoldResult objects (typical workflow):

>>> from backtest import BacktestEngine, BacktestConfig
>>> engine = BacktestEngine(BacktestConfig(), prices=prices_df)
>>> report = engine.run_from_fold_results(rl_fold_results)
>>> print(report.summary())
>>> report.to_csv('results/backtest_summary.csv')

From raw return arrays:

>>> from backtest import BacktestEngine, StrategyReturns
>>> strategies = [
...     StrategyReturns('ppo',   ppo_returns,   dates),
...     StrategyReturns('bench', bench_returns, dates),
... ]
>>> report = BacktestEngine().run(strategies)
"""

from backtest.config import BacktestConfig
from backtest.engine import BacktestEngine
from backtest.metrics import MetricsCalculator
from backtest.report import BacktestReport, StrategyReturns, TearsheetData

__all__ = [
    "BacktestEngine",
    "BacktestReport",
    "TearsheetData",
    "StrategyReturns",
    "MetricsCalculator",
    "BacktestConfig",
]
