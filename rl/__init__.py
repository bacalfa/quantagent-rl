"""
rl
==
QuantAgent-RL reinforcement learning module.

Public API
----------
RLPipeline          : walk-forward training + evaluation orchestrator
RLFoldResult        : dataclass holding per-fold train/test metrics
FoldMetrics         : per-strategy performance metrics dataclass

Core classes
------------
PortfolioEnv        : custom Gymnasium quarterly rebalancing environment
PPOAgent            : SB3 PPO wrapper with GPU policy and warm-start
StateBuilder        : assembles + normalizes the RL observation vector
RewardCalculator    : differential Sharpe / Sortino reward (GPU EMA)
LotTracker          : FIFO cost-basis tracker for tax computation
RecencyWeightedSampler : recency-biased episode sampling for training

Configuration
-------------
RLConfig            : master config grouping all hyperparameter sub-configs
RewardConfig        : reward shaping hyperparameters (η, λ_tax, λ_tlh, ...)
PortfolioConstraints: weight / turnover / sector constraints
PPOConfig           : SB3 PPO architecture and training hyperparameters
WalkForwardConfig   : walk-forward split and evaluation settings

Quick start
-----------
>>> from rl import RLPipeline, RLConfig
>>> pipeline = RLPipeline(RLConfig())
>>> # Requires a WalkForwardFold from the data module:
>>> result = pipeline.run_fold(fold)
>>> print(result.ppo_metrics.sharpe)
"""

from rl.agent import PPOAgent, RecencyWeightedSampler, RLCallback
from rl.config import (
    PortfolioConstraints,
    PPOConfig,
    RewardConfig,
    RLConfig,
    WalkForwardConfig,
)
from rl.env import Lot, LotTracker, PortfolioEnv
from rl.pipeline import FoldMetrics, RLFoldResult, RLPipeline
from rl.reward import DifferentialSharpeState, RewardCalculator
from rl.state import StateBuilder

__all__ = [
    # Pipeline
    "RLPipeline",
    "RLFoldResult",
    "FoldMetrics",
    # Config
    "RLConfig",
    "RewardConfig",
    "PortfolioConstraints",
    "PPOConfig",
    "WalkForwardConfig",
    # Environment
    "PortfolioEnv",
    "LotTracker",
    "Lot",
    # Agent
    "PPOAgent",
    "RecencyWeightedSampler",
    "RLCallback",
    # Reward
    "RewardCalculator",
    "DifferentialSharpeState",
    # State
    "StateBuilder",
]
