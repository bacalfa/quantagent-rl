"""
rl/config.py
============
Configuration dataclasses for the QuantAgent-RL reinforcement learning module.

All hyperparameters — reward shaping, action constraints, PPO network
architecture, walk-forward settings, and GPU preferences — live here so that
experiments can be reproduced by saving a single config object.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Reward Config
# ---------------------------------------------------------------------------


@dataclass
class RewardConfig:
    """Hyperparameters controlling the step-level reward signal.

    The base reward is the **Differential Sharpe Ratio** (Moody & Saffell,
    1998), which is an incremental, step-level approximation of the Sharpe
    ratio that is Markovian and suitable for RL.

    ``R_t = D_t(η) − λ_tax·tax_cost_t + λ_tlh·tlh_benefit_t − λ_turnover·|Δw|_t``

    where ``D_t`` is the differential Sharpe increment and the penalty terms
    discourage excessive trading and reward tax-loss harvesting.

    Parameters
    ----------
    use_sortino : bool
        If True, replace the symmetric variance denominator in the differential
        Sharpe with a downside-only (negative-return) variance — i.e. the
        Differential Sortino Ratio.  Penalizes downside risk only.
    eta : float
        EMA adaptation rate for the A_t (mean) and B_t (second-moment)
        running statistics used in the Differential Sharpe computation.
        Smaller values (e.g. 0.01) produce a slower-moving, more stable
        baseline; larger values (e.g. 0.1) react faster to recent returns.
    lambda_tax : float
        Weight on the realized capital gains tax cost term.  Higher values
        make the agent more reluctant to realize short-term gains.
    lambda_tlh : float
        Weight on the tax-loss harvesting benefit term.  A positive weight
        incentivizes selling positions at a loss to offset other gains.
    lambda_turnover : float
        Weight on the L1 turnover penalty ``|Δw|``.  Discourages excessive
        trading and implicit transaction costs.
    transaction_cost_fixed : float
        Fixed commission per trade as a fraction of traded notional (e.g.
        0.001 = 10 bps).  Applied to each non-zero weight change.
    transaction_cost_impact : float
        Linear market-impact coefficient.  Total cost per asset i is:
        ``c_fixed·|Δw_i| + c_impact·Δw_i²``.
    reward_scale : float
        Global scalar applied to the final reward before it is returned to
        the RL algorithm.  Useful for keeping rewards in a numerically
        well-conditioned range (≈ [−1, 1]).
    """

    use_sortino: bool = False
    eta: float = 0.05
    lambda_tax: float = 0.5
    lambda_tlh: float = 0.3
    lambda_turnover: float = 0.1
    transaction_cost_fixed: float = 0.001  # 10 bps
    transaction_cost_impact: float = 0.002  # 20 bps per unit Δw²
    reward_scale: float = 10.0


# ---------------------------------------------------------------------------
# Action / Portfolio Constraints
# ---------------------------------------------------------------------------


@dataclass
class PortfolioConstraints:
    """Constraints on portfolio weights and rebalancing actions.

    Parameters
    ----------
    max_position : float
        Maximum weight for any single asset (e.g. 0.20 = 20 % cap).
    min_position : float
        Minimum weight for any single asset; set to 0.0 for long-only.
    max_sector_weight : float
        Maximum combined weight for any single GICS sector.
    max_turnover : float
        Maximum total L1 turnover per rebalancing period.  The action is
        clipped before execution when this limit would be exceeded.
    no_trade_threshold : float
        Weight changes smaller than this value (in absolute terms) are
        rounded to zero to simulate a minimum trade size and discourage
        micro-rebalancing.
    """

    max_position: float = 0.20
    min_position: float = 0.00  # long-only
    max_sector_weight: float = 0.40
    max_turnover: float = 0.50  # 50 % max one-way turnover per quarter
    no_trade_threshold: float = 0.005  # 50 bps minimum trade size


# ---------------------------------------------------------------------------
# PPO / Network Config
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """Hyperparameters for the PPO agent (Stable-Baselines3).

    Parameters
    ----------
    policy : str
        SB3 policy class.  'MlpPolicy' (two-layer MLP) is the standard
        choice for tabular / flat-vector observations.
    net_arch : list[int]
        Hidden layer sizes for both the policy and value networks.
        [256, 256] is a reasonable default for state vectors of ≈ 500 dims.
    learning_rate : float
        Adam optimizer learning rate.
    n_steps : int
        Number of environment steps collected per PPO update (rollout buffer
        size).  Must be a multiple of ``batch_size``.
    batch_size : int
        Mini-batch size for the PPO gradient updates.
    n_epochs : int
        Number of gradient epochs per PPO update.
    gamma : float
        Discount factor.  Close to 1.0 for quarterly horizon (little
        discounting over the 3–5 year episode lengths).
    gae_lambda : float
        Generalized Advantage Estimation (GAE) lambda.
    clip_range : float
        PPO clipping parameter ε.
    ent_coef : float
        Entropy regularization coefficient.  A small positive value
        encourages exploration and prevents premature convergence.
    vf_coef : float
        Value function loss coefficient.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    total_timesteps : int
        Total environment interaction steps for the initial training fold.
        Subsequent folds use ``warmstart_timesteps`` (fewer steps since the
        agent already has a good starting policy).
    warmstart_timesteps : int
        Additional training steps when fine-tuning on an expanded fold.
    device : str
        PyTorch device for the policy and value networks.
        'auto' selects CUDA if available, else CPU.
    verbose : int
        SB3 verbosity: 0 = silent, 1 = training progress, 2 = debug.
    seed : int
        Random seed for reproducibility.
    """

    policy: str = "MlpPolicy"
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 50_000
    warmstart_timesteps: int = 10_000
    device: str = "auto"
    verbose: int = 0
    seed: int = 42


# ---------------------------------------------------------------------------
# Walk-Forward Config
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardConfig:
    """Walk-forward training and evaluation settings.

    Parameters
    ----------
    min_train_quarters : int
        Minimum number of quarterly steps in the first training window.
    test_quarters : int
        Number of quarterly steps in each test window.
    recency_weight_decay : float
        Exponential decay rate applied to episode sampling probabilities.
        A value of 0.1 means episodes from 1 year ago are sampled at
        ``exp(-0.1)`` ≈ 90 % of the rate of the most recent episode.
        Set to 0.0 for uniform (non-recency-weighted) sampling.
    n_eval_episodes : int
        Number of evaluation episodes used to estimate out-of-sample
        performance at each fold boundary.
    """

    min_train_quarters: int = 12  # 3 years minimum training
    test_quarters: int = 4  # 1-year test windows
    recency_weight_decay: float = 0.05
    n_eval_episodes: int = 5


# ---------------------------------------------------------------------------
# Master RL Config
# ---------------------------------------------------------------------------


@dataclass
class RLConfig:
    """Master configuration object for the QuantAgent-RL reinforcement learning module.

    Usage
    -----
    >>> cfg = RLConfig()                               # all defaults
    >>> cfg = RLConfig(ppo=PPOConfig(device='cuda'))   # force GPU policy
    """

    reward: RewardConfig = field(default_factory=RewardConfig)
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
