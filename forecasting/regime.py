"""
forecasting/regime.py
=====================
Hidden Markov Model (HMM) regime detector for QuantAgent-RL.

Identifies the latent market regime (bull, bear, sideways/transition)
from the joint distribution of cross-sectional return moments and realized
volatility. The regime signal feeds into the RL state as a soft-assignment
probability vector, enabling the agent to condition its rebalancing decisions
on the macro risk environment.

Design
------
- **Model**: Gaussian HMM (``hmmlearn.hmm.GaussianHMM``) with diagonal
  covariance. The emission distribution is Gaussian over a small feature
  vector derived from the market-level return series.
- **Input features**: Equally-weighted portfolio log return and cross-
  sectional realized volatility. Using market-level inputs (rather than
  per-asset) makes the regime signal universal and stable.
- **State labeling**: After fitting, states are sorted by their mean
  market return so that State 0 = bear, State 1 = sideways, State 2 = bull.
  This ordering is deterministic and interpretable regardless of HMM
  initialization.
- **GPU acceleration**: Viterbi decoding and forward-backward (smoothed
  posterior) computations are re-implemented in CuPy for the inference
  pass. The EM fitting step stays on CPU (``hmmlearn`` uses Cython/NumPy
  internally).
- **Walk-forward safety**: ``RegimeDetector.fit()`` uses only in-sample
  data. At inference time, ``decode()`` and ``predict_proba()`` operate on
  a rolling lookback window of recent observations.

Outputs (per quarter-end date)
-------------------------------
- ``regime_label``  : str — 'bull', 'bear', or 'sideways'
- ``regime_proba``  : np.ndarray shape (n_states,) — posterior probabilities
- ``regime_index``  : int — 0-based state index
"""

import logging

import numpy as np
import pandas as pd

from forecasting.config import RegimeConfig

logger = logging.getLogger(__name__)

# State label assignments after sorting by mean return (low → high)
_DEFAULT_STATE_LABELS = {0: "bear", 1: "sideways", 2: "bull"}


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def _get_cupy() -> tuple[object, bool]:
    try:
        import cupy as cp

        return cp, True
    except ImportError:
        return np, False


# ---------------------------------------------------------------------------
# Lightweight NumPy Gaussian HMM (fallback when hmmlearn is not installed)
# ---------------------------------------------------------------------------


class _NumpyGaussianHMM:
    """Minimal Gaussian HMM with diagonal covariance via Baum-Welch EM.

    This is a dependency-free fallback. It implements the core Baum-Welch
    algorithm using NumPy so the regime detector remains functional without
    ``hmmlearn``. For production use, install hmmlearn for better numerical
    stability and more configuration options.
    """

    def __init__(
        self, n_states: int = 3, n_iter: int = 100, random_state: int = 42
    ) -> None:
        self.n_components = n_states
        self.n_iter = n_iter
        self._rng = np.random.default_rng(random_state)

        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None  # shape (S, F), diagonal
        self.transmat_: np.ndarray | None = None
        self.startprob_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "_NumpyGaussianHMM":
        """Baum-Welch EM with numerically stable gamma and vectorized xi."""
        T, F = X.shape
        S = self.n_components

        # K-means initialization
        from sklearn.cluster import KMeans

        km = KMeans(
            n_clusters=S, random_state=int(self._rng.integers(0, 1000)), n_init=5
        )
        labels = km.fit_predict(X)
        self.means_ = km.cluster_centers_.copy()  # (S, F)
        self.covars_ = np.array(
            [
                np.maximum(X[labels == s].var(axis=0), 1e-4)
                if (labels == s).sum() > 1
                else np.ones(F) * 0.01
                for s in range(S)
            ]
        )
        self.transmat_ = np.full((S, S), 1.0 / S)
        self.startprob_ = np.full(S, 1.0 / S)

        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            # E-step: forward-backward
            log_emit = self._log_emission(X)  # (T, S)
            alpha, beta, log_px = self._forward_backward(log_emit)

            # Robust gamma: fall back to alpha where alpha*beta underflows
            gamma = alpha * beta
            row_sums = gamma.sum(axis=1, keepdims=True)
            bad = row_sums.squeeze() < 1e-300
            if np.any(bad):
                gamma[bad] = alpha[bad]
                row_sums[bad] = alpha[bad].sum(axis=1, keepdims=True).clip(min=1e-300)
            gamma /= row_sums.clip(min=1e-300)

            # Vectorized xi: xi[t,i,j] = alpha[t,i] * A[i,j] * emit[t+1,j] * beta[t+1,j]
            # shape: (T-1, S, S)
            emit_next = np.exp(log_emit[1:])  # (T-1, S)
            xi_unnorm = (
                alpha[:-1, :, None]  # (T-1, S, 1)
                * self.transmat_[None, :, :]  # (1,   S, S)
                * emit_next[:, None, :]  # (T-1, 1, S)
                * beta[1:, None, :]  # (T-1, 1, S)
            )  # (T-1, S, S)
            xi_sum = xi_unnorm.sum(axis=0)  # (S, S)
            xi_row_sum = xi_sum.sum(axis=1, keepdims=True).clip(min=1e-300)
            xi_norm = xi_sum / xi_row_sum

            # M-step
            self.startprob_ = np.clip(gamma[0], 1e-9, None)
            self.startprob_ /= self.startprob_.sum()

            self.transmat_ = np.clip(xi_norm, 1e-9, None)
            self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

            gamma_sum = gamma.sum(axis=0)  # (S,)
            self.means_ = (gamma.T @ X) / gamma_sum[:, None].clip(min=1e-9)

            for s in range(S):
                diff = X - self.means_[s]  # (T, F)
                self.covars_[s] = (gamma[:, s, None] * diff**2).sum(axis=0) / max(
                    gamma_sum[s], 1e-9
                )
                self.covars_[s] = np.maximum(self.covars_[s], 1e-4)

            # Convergence check
            if abs(log_px - prev_ll) < 1e-4:
                break
            prev_ll = log_px

        return self

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        T, F = X.shape
        S = self.n_components
        log_emit = np.zeros((T, S))
        for s in range(S):
            diff = X - self.means_[s]
            log_emit[:, s] = -0.5 * (
                np.sum(diff**2 / (self.covars_[s] + 1e-9), axis=1)
                + np.sum(np.log(2 * np.pi * self.covars_[s] + 1e-9))
            )
        return log_emit

    def _forward_backward(
        self, log_emit: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Numerically stable scaled forward-backward algorithm."""
        T, S = log_emit.shape
        alpha = np.zeros((T, S))
        beta = np.ones((T, S))
        log_scales = np.zeros(T)

        # Forward pass — shift each emission by its max before exponentiation
        shift0 = log_emit[0].max()
        a0 = self.startprob_ * np.exp(log_emit[0] - shift0)
        s0 = max(a0.sum(), 1e-300)
        alpha[0] = a0 / s0
        log_scales[0] = shift0 + np.log(s0)

        for t in range(1, T):
            shift_t = log_emit[t].max()
            emit_t = np.exp(log_emit[t] - shift_t)
            a_t = (alpha[t - 1] @ self.transmat_) * emit_t
            s_t = max(a_t.sum(), 1e-300)
            alpha[t] = a_t / s_t
            log_scales[t] = shift_t + np.log(s_t)

        log_px = log_scales.sum()

        # Backward pass — separate scaling keeps beta well-conditioned
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            shift_next = log_emit[t + 1].max()
            emit_next = np.exp(log_emit[t + 1] - shift_next)
            b_t = self.transmat_ @ (emit_next * beta[t + 1])
            s_t = max(b_t.sum(), 1e-300)
            beta[t] = b_t / s_t

        return alpha, beta, log_px

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Smoothed posterior probabilities via forward-backward.

        Falls back to forward-only (alpha) posteriors if the combined
        alpha*beta product underflows to zero for any time step.
        """
        log_emit = self._log_emission(X)
        alpha, beta, _ = self._forward_backward(log_emit)
        gamma = alpha * beta
        row_sums = gamma.sum(axis=1, keepdims=True)

        # Rows that underflowed: replace with alpha (forward-only posterior)
        bad = row_sums.squeeze() < 1e-300
        if np.any(bad):
            gamma[bad] = alpha[bad]
            row_sums[bad] = alpha[bad].sum(axis=1, keepdims=True).clip(min=1e-300)

        gamma /= row_sums.clip(min=1e-300)
        return gamma

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding — returns most likely state sequence."""
        log_emit = self._log_emission(X)
        T, S = log_emit.shape
        log_trans = np.log(self.transmat_ + 1e-300)
        delta = np.zeros((T, S))
        psi = np.zeros((T, S), dtype=int)

        delta[0] = np.log(self.startprob_ + 1e-300) + log_emit[0]
        for t in range(1, T):
            trans_scores = delta[t - 1][:, None] + log_trans  # (S, S)
            psi[t] = trans_scores.argmax(axis=0)
            delta[t] = trans_scores.max(axis=0) + log_emit[t]

        states = np.zeros(T, dtype=int)
        states[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------


class RegimeDetector:
    """Gaussian HMM regime detector over market return features.

    Parameters
    ----------
    config : RegimeConfig
        Hyperparameters for the HMM.

    Examples
    --------
    >>> rd = RegimeDetector(RegimeConfig(n_states=3))
    >>> rd.fit(train_log_returns)
    >>> label, proba = rd.decode_latest(log_returns)
    >>> quarterly_df = rd.forecast_quarterly(log_returns, rebalance_dates)
    """

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.cfg = config or RegimeConfig()
        self._model = None
        self._state_order: list[int] = []  # maps sorted index → raw HMM state
        self._label_map: dict[int, str] = {}
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

        cp, self._gpu = _get_cupy()
        self._cp = cp
        if self.cfg.use_gpu is True and not self._gpu:
            raise RuntimeError(
                "use_gpu=True but CuPy is not installed. "
                "Install via: pip install cupy-cuda12x"
            )
        if self.cfg.use_gpu is False:
            self._gpu = False

        backend = "GPU (CuPy)" if self._gpu else "CPU (NumPy)"
        logger.info(f"[Regime] Backend: {backend}")

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _build_features(self, log_returns: pd.DataFrame) -> np.ndarray:
        """Construct the HMM observation sequence from log returns.

        Uses two market-level features to keep the observation space small
        and the regime signal stable across different universes:

        1. Equally-weighted portfolio log return (market direction)
        2. Cross-sectional standard deviation of returns (dispersion / risk)

        Parameters
        ----------
        log_returns : pd.DataFrame
            Daily log returns for all assets (columns = tickers).

        Returns
        -------
        np.ndarray, shape (T, 2)
            Observation matrix for the HMM.
        """
        ew_return = log_returns.mean(axis=1)  # market return proxy
        dispersion = log_returns.std(axis=1)  # cross-sectional vol

        features = np.column_stack(
            [
                ew_return.values,
                dispersion.values,
            ]
        )
        return np.nan_to_num(features, nan=0.0)

    def _normalize_features(
        self, features: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        """Z-score normalize the feature matrix.

        Parameters
        ----------
        features : np.ndarray
        fit : bool
            If True, compute and store mean/std from this data.
            If False, apply previously stored parameters (inference path).
        """
        if fit or self._feature_mean is None:
            self._feature_mean = features.mean(axis=0)
            self._feature_std = features.std(axis=0) + 1e-9

        return (features - self._feature_mean) / self._feature_std

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, log_returns: pd.DataFrame) -> "RegimeDetector":
        """Estimate HMM parameters from in-sample log returns.

        Uses ``hmmlearn.hmm.GaussianHMM`` (preferred). Falls back to a
        lightweight NumPy-based Gaussian HMM implementation (k-means
        initialization + EM) when ``hmmlearn`` is not installed.

        Parameters
        ----------
        log_returns : pd.DataFrame
            Daily log returns. Must contain only in-sample data.

        Returns
        -------
        self
        """
        logger.info(
            f"[Regime] Fitting HMM ({self.cfg.n_states} states) on "
            f"{len(log_returns)} obs "
            f"({log_returns.index[0].date()} → {log_returns.index[-1].date()})"
        )

        features = self._build_features(log_returns)
        features_norm = self._normalize_features(features, fit=True)

        try:
            from hmmlearn.hmm import GaussianHMM

            self._model = GaussianHMM(
                n_components=self.cfg.n_states,
                covariance_type=self.cfg.covariance_type,
                n_iter=self.cfg.n_iter,
                random_state=self.cfg.random_state,
            )
            self._model.fit(features_norm)
            monitor = self._model.monitor_
            converged = monitor.converged
            ll = monitor.history[-1]
            logger.info(
                f"[Regime] hmmlearn fit complete. "
                f"Converged: {converged}, Log-likelihood: {ll:.2f}"
            )

        except ImportError:
            logger.info(
                "[Regime] hmmlearn not installed — using NumPy fallback HMM. "
                "Install for better convergence: pip install hmmlearn"
            )
            self._model = _NumpyGaussianHMM(
                n_states=self.cfg.n_states,
                n_iter=self.cfg.n_iter,
                random_state=self.cfg.random_state,
            )
            self._model.fit(features_norm)

        # Sort states by mean market-return feature (low → high = bear → bull)
        means = self._model.means_[:, 0]
        self._state_order = list(np.argsort(means))

        sorted_labels = _DEFAULT_STATE_LABELS.copy()
        if self.cfg.n_states != 3:
            sorted_labels = {i: f"state_{i}" for i in range(self.cfg.n_states)}
        self._label_map = {
            sorted_idx: sorted_labels.get(sorted_idx, f"state_{sorted_idx}")
            for sorted_idx in range(self.cfg.n_states)
        }
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def decode_latest(
        self,
        log_returns: pd.DataFrame,
        lookback: int | None = None,
    ) -> tuple[str, np.ndarray, int]:
        """Decode the most recent regime from recent return history.

        Uses the forward-backward algorithm (smoothed posteriors) rather
        than Viterbi so that the regime assignment reflects uncertainty.

        Parameters
        ----------
        log_returns : pd.DataFrame
            Daily log returns (full history up to the current date).
        lookback : int | None
            Number of recent trading days to use. Defaults to
            ``cfg.lookback_window``.

        Returns
        -------
        label : str
            'bull', 'bear', or 'sideways'.
        proba : np.ndarray, shape (n_states,)
            Posterior state probabilities (sorted: bear=0, sideways=1, bull=2).
        state_index : int
            Most likely sorted state index.
        """
        self._check_fitted()
        w = lookback or self.cfg.lookback_window
        recent = log_returns.iloc[-w:]

        features = self._build_features(recent)
        features_norm = self._normalize_features(features, fit=False)

        # Forward-backward posteriors (CPU path via hmmlearn)
        posteriors = self._model.predict_proba(features_norm)  # shape (T, S)
        last_posterior = posteriors[-1]  # shape (S,)

        # Reorder to sorted state indexing (bear=0, sideways=1, bull=2)
        sorted_posterior = self._reorder_posterior(last_posterior)

        state_idx = int(np.argmax(sorted_posterior))
        label = self._label_map[state_idx]
        return label, sorted_posterior, state_idx

    def decode_sequence(
        self,
        log_returns: pd.DataFrame,
        lookback: int | None = None,
    ) -> np.ndarray:
        """Decode the full regime sequence using Viterbi (or GPU variant).

        Parameters
        ----------
        log_returns : pd.DataFrame
        lookback : int | None
            If set, only the most recent ``lookback`` observations are decoded.

        Returns
        -------
        np.ndarray, shape (T,)
            Most likely regime state index (sorted: 0=bear, 1=sideways, 2=bull)
            for each time step.
        """
        self._check_fitted()
        w = lookback or len(log_returns)
        recent = log_returns.iloc[-w:]

        features = self._build_features(recent)
        features_norm = self._normalize_features(features, fit=False)

        if self._gpu:
            states_raw = self._viterbi_gpu(features_norm)
        else:
            states_raw = self._model.predict(features_norm)

        # Remap raw state indices to sorted indices
        raw_to_sorted = {
            raw: sorted_idx for sorted_idx, raw in enumerate(self._state_order)
        }
        return np.array([raw_to_sorted[s] for s in states_raw])

    def _reorder_posterior(self, posterior: np.ndarray) -> np.ndarray:
        """Reorder raw HMM posterior probabilities to sorted state indexing."""
        reordered = np.zeros(self.cfg.n_states)
        for sorted_idx, raw_idx in enumerate(self._state_order):
            reordered[sorted_idx] = posterior[raw_idx]
        return reordered

    # ------------------------------------------------------------------
    # GPU Viterbi
    # ------------------------------------------------------------------

    def _viterbi_gpu(self, features_norm: np.ndarray) -> np.ndarray:
        """CuPy-accelerated Viterbi decoding.

        Reimplements the standard log-domain Viterbi algorithm using
        CuPy arrays so that the trellis forward pass runs on-device.
        Falls back to CPU if any CuPy operation fails.

        Parameters
        ----------
        features_norm : np.ndarray, shape (T, n_features)

        Returns
        -------
        np.ndarray, shape (T,)
            Most likely raw state sequence (NOT sorted).
        """
        cp = self._cp
        try:
            T, _ = features_norm.shape
            S = self.cfg.n_states

            # Log transition matrix and log initial probabilities
            log_trans = cp.asarray(np.log(self._model.transmat_ + 1e-300))
            log_init = cp.asarray(np.log(self._model.startprob_ + 1e-300))

            # Compute log emission probabilities on GPU
            log_emit = self._log_emission_proba_gpu(features_norm)  # (T, S)

            # Viterbi forward pass
            delta = cp.zeros((T, S))
            psi = cp.zeros((T, S), dtype=cp.int32)

            delta[0] = log_init + log_emit[0]

            for t in range(1, T):
                trans_scores = delta[t - 1].reshape(-1, 1) + log_trans  # (S, S)
                psi[t] = cp.argmax(trans_scores, axis=0)
                delta[t] = cp.max(trans_scores, axis=0) + log_emit[t]

            # Backtrack
            states_gpu = cp.zeros(T, dtype=cp.int32)
            states_gpu[T - 1] = cp.argmax(delta[T - 1])
            for t in range(T - 2, -1, -1):
                states_gpu[t] = psi[t + 1, states_gpu[t + 1]]

            return states_gpu.get().astype(int)

        except Exception as exc:
            logger.warning(
                f"[Regime] GPU Viterbi failed ({exc}) — falling back to CPU."
            )
            return self._model.predict(features_norm)

    def _log_emission_proba_gpu(self, features_norm: np.ndarray) -> object:
        """Compute log Gaussian emission probabilities on GPU.

        For each time step t and state s, compute:
            log p(x_t | state=s) = -0.5 · Σ_k [(x_{t,k} - μ_{s,k})² / σ²_{s,k}]
                                   - 0.5 · Σ_k log(2π σ²_{s,k})

        Parameters
        ----------
        features_norm : np.ndarray, shape (T, n_features)

        Returns
        -------
        CuPy array, shape (T, n_states)
        """
        cp = self._cp
        x = cp.asarray(features_norm)  # (T, F)
        mu = cp.asarray(self._model.means_)  # (S, F)

        # Diagonal covariance: covars_ shape is (S, F) for 'diag'
        if self.cfg.covariance_type == "diag":
            var = cp.asarray(self._model.covars_)  # (S, F)
        else:
            # Full covariance — take diagonal as approximation on GPU
            covars = self._model.covars_
            var = cp.asarray(
                np.array([np.diag(covars[s]) for s in range(self.cfg.n_states)])
            )

        # Broadcast: (T, 1, F) - (1, S, F) → (T, S, F)
        diff_sq = (x[:, None, :] - mu[None, :, :]) ** 2  # (T, S, F)
        log_prob = -0.5 * cp.sum(diff_sq / (var[None, :, :] + 1e-9), axis=2)
        log_prob -= 0.5 * cp.sum(cp.log(2 * np.pi * var + 1e-9), axis=1)[None, :]
        return log_prob  # (T, S)

    # ------------------------------------------------------------------
    # Walk-forward quarterly output
    # ------------------------------------------------------------------

    def forecast_quarterly(
        self,
        log_returns: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
        lookback: int | None = None,
    ) -> pd.DataFrame:
        """Produce regime labels and probabilities at every rebalance date.

        The model must be fitted before calling this method. This method
        only runs the inference pass — no refitting occurs here. Refitting
        across walk-forward folds is handled by the ForecastingPipeline.

        Parameters
        ----------
        log_returns : pd.DataFrame
            Full daily log return history.
        rebalance_dates : pd.DatetimeIndex
            Quarter-end dates at which regime inference is performed.
        lookback : int | None
            Override the configured lookback window.

        Returns
        -------
        pd.DataFrame
            Index = rebalance_dates. Columns:
              - 'regime_label'  : str
              - 'regime_index'  : int (0=bear, 1=sideways, 2=bull)
              - 'p_bear', 'p_sideways', 'p_bull' (or generic p_state_i)
        """
        self._check_fitted()
        n = self.cfg.n_states
        prob_cols = [f"p_{self._label_map[i]}" for i in range(n)]
        records = []

        for date in rebalance_dates:
            history = log_returns[log_returns.index <= date]
            if len(history) < 20:
                records.append(
                    {
                        "regime_label": "unknown",
                        "regime_index": -1,
                        **{col: 1.0 / n for col in prob_cols},
                    }
                )
                continue

            label, proba, state_idx = self.decode_latest(history, lookback=lookback)
            record: dict = {
                "regime_label": label,
                "regime_index": state_idx,
            }
            for col, p in zip(prob_cols, proba):
                record[col] = float(p)
            records.append(record)

        df = pd.DataFrame(records, index=rebalance_dates)
        df.index.name = "date"
        return df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def transition_matrix(self) -> pd.DataFrame:
        """Return the HMM transition matrix with sorted state labels.

        Returns
        -------
        pd.DataFrame
            Rows = from-state, columns = to-state.
            Values = transition probabilities (row-stochastic).
        """
        self._check_fitted()
        raw = self._model.transmat_

        labels = [self._label_map[i] for i in range(self.cfg.n_states)]
        # Reorder rows and columns to sorted state indexing
        idx = self._state_order
        reordered = raw[np.ix_(idx, idx)]
        return pd.DataFrame(reordered, index=labels, columns=labels)

    def state_statistics(self, log_returns: pd.DataFrame) -> pd.DataFrame:
        """Compute mean return and volatility per regime state.

        Parameters
        ----------
        log_returns : pd.DataFrame
            Daily log returns used for computing in-state statistics.

        Returns
        -------
        pd.DataFrame
            Index = regime labels. Columns: mean_return_ann, vol_ann,
            sharpe_ann, frequency.
        """
        self._check_fitted()
        ew_returns = log_returns.mean(axis=1)
        states_sorted = self.decode_sequence(log_returns)

        rows = {}
        for sorted_idx in range(self.cfg.n_states):
            label = self._label_map[sorted_idx]
            mask = states_sorted == sorted_idx
            r_in = ew_returns.values[mask]
            if len(r_in) < 2:
                continue
            mean_ann = r_in.mean() * 252
            vol_ann = r_in.std() * np.sqrt(252)
            sharpe = mean_ann / (vol_ann + 1e-9)
            rows[label] = {
                "mean_return_ann": mean_ann,
                "vol_ann": vol_ann,
                "sharpe_ann": sharpe,
                "frequency": mask.mean(),
            }
        return pd.DataFrame(rows).T

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Call fit() before calling inference methods.")
