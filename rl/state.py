"""
rl/state.py
===========
StateBuilder — assembles the full RL observation vector from the outputs of
the three upstream modules.

The RL state at quarter-end date t is the concatenation of three blocks:

    s_t = [ quant_features_t | forecast_features_t | agent_embedding_t ]

where:
    quant_features_t   : normalized technical/macro features from data module
                         (shape: n_assets × n_feature_groups, flattened)
    forecast_features_t: GARCH vol, HMM regime probs, FF factor betas
                         from forecasting module (from ForecastBundle)
    agent_embedding_t  : sentence-transformer embedding of MarketBrief
                         from agents module (shape: embedding_dim)

All three blocks are L2-normalized at the column level using in-sample
statistics (fit on the training window, applied to test data).  The full
state is then clipped to [−5, 5] to protect against outlier steps.

This module also carries the current portfolio weights and per-lot cost-basis
information as auxiliary state (not part of the RL observation vector, but
available to the environment for reward computation).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StateBuilder
# ---------------------------------------------------------------------------


class StateBuilder:
    """Constructs and normalizes RL observation vectors.

    Parameters
    ----------
    tickers : list[str]
        Asset tickers in the portfolio universe.

    Examples
    --------
    >>> sb = StateBuilder(tickers)
    >>> sb.fit(train_quant, train_forecast, train_embed)
    >>> obs = sb.build(date, quant_df, forecast_df, embed_df)
    """

    def __init__(self, tickers: list[str]) -> None:
        self.tickers = tickers
        self.n_assets = len(tickers)

        # Scaler params fitted on training data (walk-forward safe)
        self._quant_mean: pd.Series | None = None
        self._quant_std: pd.Series | None = None
        self._forecast_mean: pd.Series | None = None
        self._forecast_std: pd.Series | None = None
        self._embed_mean: np.ndarray | None = None
        self._embed_std: np.ndarray | None = None

        # Cached observation dimension (computed on first build)
        self._obs_dim: int | None = None

    # ------------------------------------------------------------------
    # Fit (in-sample only)
    # ------------------------------------------------------------------

    def fit(
        self,
        quant_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        embed_df: pd.DataFrame,
    ) -> "StateBuilder":
        """Fit column-level z-score scalers on in-sample data.

        Must be called with training-period data only.  Applying this scaler
        to out-of-sample data is safe because the statistics come exclusively
        from the training window.

        Parameters
        ----------
        quant_df : pd.DataFrame
            Quantitative feature matrix from the data module.
            Shape: (n_train_quarters, n_quant_features).
        forecast_df : pd.DataFrame
            Forecasting feature matrix (from ``ForecastBundle.rl_state_extension``).
            Shape: (n_train_quarters, n_forecast_features).
        embed_df : pd.DataFrame
            Agent embedding matrix.
            Shape: (n_train_quarters, embedding_dim).

        Returns
        -------
        self
        """
        self._quant_mean, self._quant_std = self._fit_scaler(quant_df)
        self._forecast_mean, self._forecast_std = self._fit_scaler(forecast_df)

        # Embedding: fit on numpy array directly
        if not embed_df.empty:
            mat = embed_df.values.astype(np.float32)
            self._embed_mean = mat.mean(axis=0)
            self._embed_std = np.maximum(mat.std(axis=0), 1e-9)
        else:
            self._embed_mean = None
            self._embed_std = None

        logger.info(
            f"[StateBuilder] Fitted scalers: "
            f"quant={quant_df.shape[1]}, "
            f"forecast={forecast_df.shape[1]}, "
            f"embed={embed_df.shape[1] if not embed_df.empty else 0} dims."
        )
        return self

    @staticmethod
    def _fit_scaler(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Return (mean, std) Series fitted on df, with std floored at 1e-9."""
        mu = df.mean()
        sigma = df.std().clip(lower=1e-9)
        return mu, sigma

    # ------------------------------------------------------------------
    # Build observation vector
    # ------------------------------------------------------------------

    def build(
        self,
        date: pd.Timestamp | str,
        quant_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        embed_df: pd.DataFrame,
        portfolio_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build and return the normalized observation vector for ``date``.

        Parameters
        ----------
        date : pd.Timestamp or str
            The quarter-end date for which to build the observation.
        quant_df : pd.DataFrame
            Full quantitative feature matrix (all dates); the method picks
            the row for ``date`` using ``asof`` (safe with quarter-end dates).
        forecast_df : pd.DataFrame
            Full forecasting feature matrix.
        embed_df : pd.DataFrame
            Full embedding matrix.
        portfolio_weights : np.ndarray | None
            Current portfolio weights, shape (n_assets,).  When provided,
            appended to the observation so the agent can condition on
            current allocations.

        Returns
        -------
        np.ndarray, shape (obs_dim,), dtype float32
            Clipped, normalized observation vector ready for the policy network.
        """
        if self._quant_mean is None:
            raise RuntimeError("Call StateBuilder.fit() before build().")

        ts = pd.Timestamp(date)

        # ── Quantitative block ──────────────────────────────────────
        quant_row = self._asof_row(quant_df, ts)
        quant_vec = self._normalize(quant_row, self._quant_mean, self._quant_std)

        # ── Forecasting block ───────────────────────────────────────
        fc_row = self._asof_row(forecast_df, ts)
        fc_vec = self._normalize(fc_row, self._forecast_mean, self._forecast_std)

        # ── Agent embedding block ───────────────────────────────────
        if not embed_df.empty and self._embed_mean is not None:
            embed_row = self._asof_row(embed_df, ts)
            emb_vec = (
                embed_row.values.astype(np.float32) - self._embed_mean
            ) / self._embed_std
        else:
            emb_vec = np.zeros(0, dtype=np.float32)

        # ── Current portfolio weights (optional) ────────────────────
        weight_vec = (
            portfolio_weights.astype(np.float32)
            if portfolio_weights is not None
            else np.zeros(self.n_assets, dtype=np.float32)
        )

        # ── Concatenate and clip ────────────────────────────────────
        obs = np.concatenate([quant_vec, fc_vec, emb_vec, weight_vec]).astype(
            np.float32
        )
        obs = np.clip(obs, -5.0, 5.0)

        if self._obs_dim is None:
            self._obs_dim = len(obs)
            logger.info(f"[StateBuilder] Observation dim: {self._obs_dim}")

        return obs

    # ------------------------------------------------------------------
    # Observation space dimension
    # ------------------------------------------------------------------

    def obs_dim(
        self,
        quant_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        embed_df: pd.DataFrame,
        include_weights: bool = True,
    ) -> int:
        """Return the observation vector dimension without building it.

        Parameters
        ----------
        quant_df, forecast_df, embed_df : pd.DataFrame
            Feature DataFrames (used only to read column counts).
        include_weights : bool
            Whether the current portfolio weights are appended.

        Returns
        -------
        int
        """
        if self._obs_dim is not None:
            return self._obs_dim
        dim = quant_df.shape[1] + forecast_df.shape[1]
        if not embed_df.empty:
            dim += embed_df.shape[1]
        if include_weights:
            dim += self.n_assets
        return dim

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _asof_row(df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
        """Return the most recent row of df on or before ts."""
        if df.empty:
            return pd.Series(dtype=np.float32)
        # Align index
        idx = df.index
        candidates = idx[idx <= ts]
        if len(candidates) == 0:
            return df.iloc[0]
        return df.loc[candidates[-1]]

    @staticmethod
    def _normalize(
        row: pd.Series,
        mean: pd.Series,
        std: pd.Series,
    ) -> np.ndarray:
        """Z-score normalize a row using pre-fitted mean/std."""
        if row.empty:
            return np.zeros(len(mean), dtype=np.float32)
        aligned = row.reindex(mean.index).fillna(0.0)
        return ((aligned - mean) / std).values.astype(np.float32)

    # ------------------------------------------------------------------
    # Scaler persistence
    # ------------------------------------------------------------------

    def scaler_params(self) -> dict:
        """Return scaler parameters as a dict for serialization."""
        return {
            "quant_mean": self._quant_mean,
            "quant_std": self._quant_std,
            "forecast_mean": self._forecast_mean,
            "forecast_std": self._forecast_std,
            "embed_mean": self._embed_mean,
            "embed_std": self._embed_std,
        }

    def load_scaler_params(self, params: dict) -> "StateBuilder":
        """Restore scaler parameters from a previously saved dict."""
        self._quant_mean = params["quant_mean"]
        self._quant_std = params["quant_std"]
        self._forecast_mean = params["forecast_mean"]
        self._forecast_std = params["forecast_std"]
        self._embed_mean = params["embed_mean"]
        self._embed_std = params["embed_std"]
        return self
