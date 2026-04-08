"""
rl/agent.py
===========
PPOAgent — Stable-Baselines3 PPO wrapper for QuantAgent-RL.

Design decisions
----------------
* **GPU policy network**: SB3's ``MlpPolicy`` uses PyTorch internally.
  When ``device='cuda'`` (or ``'auto'`` with a GPU available), the policy
  and value networks run on-device, giving a ~3–8× speedup over CPU for
  the gradient update step.

* **Warm-start across folds**: rather than training from scratch at each
  walk-forward fold, the agent's weights are preserved and fine-tuned on
  the expanded training window.  This implements continual learning under
  non-stationary market conditions and dramatically reduces total compute.

* **Recency-weighted episode sampling**: when building the training rollout
  buffer, more recent episodes are sampled with higher probability using an
  exponential decay scheme.  This respects the non-stationarity of financial
  markets without discarding old regime-transition data.

* **Callback-based diagnostics**: a lightweight ``RLCallback`` logs mean
  reward, portfolio value, and Sharpe ratio at the end of each rollout so
  training progress is visible without verbose SB3 output.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rl.config import RLConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training callback
# ---------------------------------------------------------------------------


class RLCallback:
    """Lightweight SB3-compatible callback for training diagnostics.

    Logs episode-level metrics at the end of each rollout buffer collection.

    Parameters
    ----------
    fold_idx : int
        Walk-forward fold index (for log labeling).
    log_interval : int
        Log every N rollout collections.
    """

    def __init__(self, fold_idx: int = 0, log_interval: int = 5) -> None:
        self.fold_idx = fold_idx
        self.log_interval = log_interval
        self._call_count = 0
        self.reward_history: list[float] = []
        self.value_history: list[float] = []

    def _on_rollout_end(self, locals_dict: dict, globals_dict: dict) -> bool:
        """Called by SB3 at the end of each rollout collection."""
        self._call_count += 1
        infos = locals_dict.get("infos", [])
        if infos:
            rewards = [
                i.get("reward_scaled", 0.0) for i in infos if "reward_scaled" in i
            ]
            values = [
                i.get("portfolio_value", 1.0) for i in infos if "portfolio_value" in i
            ]
            if rewards:
                self.reward_history.append(float(np.mean(rewards)))
            if values:
                self.value_history.append(float(np.mean(values)))

        if self._call_count % self.log_interval == 0 and self.reward_history:
            recent_r = np.mean(self.reward_history[-self.log_interval :])
            logger.info(
                f"[PPOAgent:fold={self.fold_idx}] "
                f"rollout={self._call_count:4d} | "
                f"mean_reward={recent_r:+.4f}"
            )
        return True


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------


class PPOAgent:
    """PPO agent backed by Stable-Baselines3 with GPU acceleration.

    Parameters
    ----------
    obs_dim : int
        Observation space dimensionality.
    n_assets : int
        Number of assets (= action space dimensionality).
    config : RLConfig
    fold_idx : int
        Current walk-forward fold index (used for logging).

    Examples
    --------
    >>> agent = PPOAgent(obs_dim=512, n_assets=20, config=RLConfig())
    >>> agent.train(env, timesteps=50_000)
    >>> weights = agent.predict(observation)
    """

    def __init__(
        self,
        obs_dim: int,
        n_assets: int,
        config: RLConfig | None = None,
        fold_idx: int = 0,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_assets = n_assets
        self.cfg = config or RLConfig()
        self.fold_idx = fold_idx
        self._model = None  # SB3 PPO model (lazy init)
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self, env: object) -> None:
        """Instantiate the SB3 PPO model on the configured device."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.torch_layers import FlattenExtractor
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required: pip install stable-baselines3"
            ) from exc

        ppo = self.cfg.ppo
        policy_kwargs = {
            "net_arch": ppo.net_arch,
            "features_extractor_class": FlattenExtractor,
        }

        self._model = PPO(
            policy=ppo.policy,
            env=env,
            learning_rate=ppo.learning_rate,
            n_steps=ppo.n_steps,
            batch_size=ppo.batch_size,
            n_epochs=ppo.n_epochs,
            gamma=ppo.gamma,
            gae_lambda=ppo.gae_lambda,
            clip_range=ppo.clip_range,
            ent_coef=ppo.ent_coef,
            vf_coef=ppo.vf_coef,
            max_grad_norm=ppo.max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=ppo.device,
            verbose=ppo.verbose,
            seed=ppo.seed,
        )

        device_used = str(self._model.device)
        logger.info(
            f"[PPOAgent:fold={self.fold_idx}] PPO built: "
            f"obs_dim={self.obs_dim}, n_assets={self.n_assets}, "
            f"device={device_used}, net_arch={ppo.net_arch}"
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        env: object,
        timesteps: int | None = None,
        warm_start: bool = False,
    ) -> "PPOAgent":
        """Train the PPO agent on the provided environment.

        Parameters
        ----------
        env : PortfolioEnv (or any gym.Env)
            Training environment.
        timesteps : int | None
            Total environment interaction steps.  Defaults to
            ``cfg.ppo.warmstart_timesteps`` when ``warm_start=True``,
            else ``cfg.ppo.total_timesteps``.
        warm_start : bool
            If True and the model already exists, fine-tune rather than
            retrain from scratch.  The policy weights are preserved and
            only the environment is updated (e.g. after fold expansion).

        Returns
        -------
        self
        """
        if timesteps is None:
            timesteps = (
                self.cfg.ppo.warmstart_timesteps
                if warm_start
                else self.cfg.ppo.total_timesteps
            )

        if self._model is None or not warm_start:
            self._build_model(env)
        else:
            # Warm-start: update the environment reference without
            # reinitializing the policy weights.
            self._model.set_env(env)
            logger.info(
                f"[PPOAgent:fold={self.fold_idx}] Warm-starting from previous fold "
                f"({timesteps:,} additional steps)."
            )

        cb = RLCallback(fold_idx=self.fold_idx)

        self._model.learn(
            total_timesteps=timesteps,
            callback=self._wrap_callback(cb),
            progress_bar=False,
        )
        self._is_fitted = True
        logger.info(
            f"[PPOAgent:fold={self.fold_idx}] Training complete ({timesteps:,} steps)."
        )
        return self

    @staticmethod
    def _wrap_callback(cb: RLCallback) -> object:
        """Wrap a RLCallback in an SB3-compatible callback object."""
        try:
            from stable_baselines3.common.callbacks import BaseCallback

            class _SB3Callback(BaseCallback):
                def __init__(self, inner: RLCallback) -> None:
                    super().__init__(verbose=0)
                    self.inner = inner

                def _on_rollout_end(self) -> None:
                    self.inner._on_rollout_end(self.locals, self.globals)

                def _on_step(self) -> bool:
                    return True

            return _SB3Callback(cb)
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Produce a delta-weight action from an observation.

        Parameters
        ----------
        observation : np.ndarray, shape (obs_dim,)
        deterministic : bool
            If True, use the policy mean (no sampling).

        Returns
        -------
        np.ndarray, shape (n_assets,)
            Raw delta-weight action (before environment clipping).
        """
        self._check_fitted()
        action, _ = self._model.predict(observation, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    def predict_weights(
        self,
        observation: np.ndarray,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """Predict new portfolio weights from an observation.

        Applies the delta-weight action to ``current_weights`` and clips
        to [0, 1] before normalizing.

        Parameters
        ----------
        observation : np.ndarray
        current_weights : np.ndarray, shape (n_assets,)

        Returns
        -------
        np.ndarray, shape (n_assets,)
            New portfolio weights (sum = 1).
        """
        self._check_fitted()
        delta = self.predict(observation)
        max_pos = self.cfg.constraints.max_position
        min_pos = self.cfg.constraints.min_position
        new_w = np.clip(current_weights + delta, min_pos, max_pos)
        total = new_w.sum()
        return (
            (new_w / total).astype(np.float32)
            if total > 1e-6
            else current_weights.copy()
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the SB3 PPO model to disk.

        Parameters
        ----------
        path : str or Path
            File path (SB3 appends ``.zip`` if not present).
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))
        logger.info(f"[PPOAgent:fold={self.fold_idx}] Saved to {path}.zip")

    def load(self, path: str | Path, env: object) -> "PPOAgent":
        """Load a previously saved PPO model.

        Parameters
        ----------
        path : str or Path
        env : gym.Env
            Environment instance (needed by SB3 for action/obs space info).

        Returns
        -------
        self
        """
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required: pip install stable-baselines3"
            ) from exc

        path = Path(path)
        if not str(path).endswith(".zip"):
            path = Path(str(path) + ".zip")

        self._model = PPO.load(str(path), env=env, device=self.cfg.ppo.device)
        self._is_fitted = True
        logger.info(f"[PPOAgent:fold={self.fold_idx}] Loaded from {path}")
        return self

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def policy_device(self) -> str:
        """Return the device string where the policy network lives."""
        if self._model is None:
            return "uninitialized"
        return str(self._model.device)

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError(
                "PPOAgent has not been trained yet. Call agent.train(env) first."
            )


# ---------------------------------------------------------------------------
# Recency-weighted episode sampler
# ---------------------------------------------------------------------------


class RecencyWeightedSampler:
    """Assigns sampling probabilities to walk-forward episodes.

    More recent episodes are sampled more frequently using an exponential
    decay scheme.  During PPO rollout collection, the environment is reset
    to randomly selected episode start dates drawn with these probabilities.

    Parameters
    ----------
    dates : list[pd.Timestamp]
        Ordered list of episode start dates (one per training quarter).
    decay : float
        Exponential decay rate.  0.0 = uniform; larger = more recency bias.

    Examples
    --------
    >>> sampler = RecencyWeightedSampler(train_dates, decay=0.05)
    >>> start_date = sampler.sample()
    """

    def __init__(
        self,
        dates: list[pd.Timestamp],
        decay: float = 0.05,
    ) -> None:
        self.dates = dates
        n = len(dates)
        if n == 0:
            self._probs = np.array([], dtype=np.float64)
            return

        # Compute recency weights: episode at index i gets weight exp(decay·i)
        indices = np.arange(n, dtype=np.float64)
        raw_weights = np.exp(decay * indices)
        self._probs = raw_weights / raw_weights.sum()

    def sample(self, rng: np.random.Generator | None = None) -> pd.Timestamp:
        """Sample one episode start date according to recency weights.

        Parameters
        ----------
        rng : np.random.Generator | None
            Optional random generator for reproducibility.

        Returns
        -------
        pd.Timestamp
        """
        if not self.dates:
            raise ValueError("No dates to sample from.")
        rng = rng or np.random.default_rng()
        idx = int(rng.choice(len(self.dates), p=self._probs))
        return self.dates[idx]

    @property
    def probabilities(self) -> np.ndarray:
        """Sampling probability for each date (read-only)."""
        return self._probs.copy()
