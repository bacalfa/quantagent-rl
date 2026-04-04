"""
agents/embedder.py
==================
GPU-accelerated MarketBrief embedder for QuantAgent-RL.

Converts the qualitative ``MarketBrief`` output of the agents module into a
dense numeric vector that can be concatenated with the quantitative feature
matrix produced by the ``data`` and ``forecasting`` modules to form the
complete RL state vector.

Embedding Strategy
------------------
1. **Primary path** (GPU if available): ``sentence-transformers`` encodes the
   ``MarketBrief.to_text()`` string using a pre-trained transformer model.
   The model runs on CUDA via PyTorch when a GPU is detected, falling back to
   CPU. Output is a 384-dimensional vector (``all-MiniLM-L6-v2``).

2. **Fallback path** (no sentence-transformers): A deterministic TF-IDF bag-of-
   words representation is extracted from the text. A fixed vocabulary of
   finance-domain terms is used so the vector dimension is stable. Output is a
   200-dimensional vector.

Both paths produce L2-normalized vectors so the magnitude is comparable
across different quarters.

GPU Acceleration
----------------
``sentence-transformers`` uses PyTorch internally. When the target device is
``'cuda'``, the transformer forward pass runs on-device. For a single brief
(one string), the GPU is lightly loaded, but in batch mode (encoding many
quarters at once) the speedup is substantial: ~10–25× vs. CPU for batches
of 40+ strings.

Usage
-----
>>> embedder = MarketBriefEmbedder()                  # auto-detect device
>>> vec = embedder.encode(market_brief)               # np.ndarray (384,)
>>> batch = embedder.encode_batch(market_briefs)      # np.ndarray (N, 384)
>>> dim = embedder.embedding_dim
"""

import logging
import re

import numpy as np
import pandas as pd

from agents.schemas import MarketBrief

logger = logging.getLogger(__name__)

# Vocabulary of finance-domain terms for the TF-IDF fallback
# Covers the key signals that appear in MarketBrief.to_text() output
_FINANCE_VOCAB: list[str] = [
    # Macro regime terms
    "tightening",
    "easing",
    "neutral",
    "risk_on",
    "risk_off",
    "transitional",
    "aggressive",
    "defensive",
    # Rate / inflation terms
    "inflation",
    "high",
    "elevated",
    "moderate",
    "low",
    "inverted",
    "flat",
    "normal",
    "recession",
    "credit",
    "stress",
    # Sector names
    "technology",
    "healthcare",
    "financials",
    "energy",
    "industrials",
    "consumer",
    "materials",
    "utilities",
    "estate",
    "communication",
    # Fundamental terms
    "accelerating",
    "stable",
    "decelerating",
    "negative",
    "expanding",
    "compressing",
    "strong",
    "adequate",
    "stretched",
    # Sentiment / signal terms
    "bullish",
    "bearish",
    "overweight",
    "underweight",
    "upgrade",
    "downgrade",
    "conviction",
    "momentum",
    "valuation",
    "cheap",
    "fair",
    # Action words
    "rising",
    "falling",
    "higher",
    "lower",
    "growth",
    "decline",
    # Risk terms
    "risk",
    "flag",
    "uncertainty",
    "geopolitical",
    "supply",
    "demand",
    # Common tickers in universe
    "aapl",
    "msft",
    "nvda",
    "googl",
    "meta",
    "amzn",
    "tsla",
    "jpm",
    "bac",
    "gs",
    "jnj",
    "lly",
    "unh",
    "xom",
    "cvx",
    "rtx",
    "hon",
    "cat",
    "pg",
    "ko",
    "wmt",
    "nee",
    "pld",
    "lin",
]


class MarketBriefEmbedder:
    """Encodes ``MarketBrief`` objects as fixed-size dense vectors.

    Parameters
    ----------
    model_name : str
        ``sentence-transformers`` model identifier.
        Default: ``'all-MiniLM-L6-v2'`` (384-dim, 22 MB, fast inference).
    device : str | None
        Torch device string: ``'cuda'``, ``'cpu'``, or ``None`` (auto-detect).
    normalize : bool
        If True, L2-normalize all output vectors. Recommended: True.

    Examples
    --------
    >>> embedder = MarketBriefEmbedder(device='cuda')
    >>> vec = embedder.encode(brief)    # shape (384,)
    >>> mat = embedder.encode_batch([b1, b2, b3])    # shape (3, 384)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self._model = None
        self._backend = "uninitialized"

        # Resolve device
        self._device = device or self._auto_detect_device()
        logger.info(f"[Embedder] Target device: {self._device}")

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the model on first use (lazy initialization)."""
        if self._model is not None:
            return

        # Primary: sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self._device)
            self._backend = "sentence_transformers"
            logger.info(
                f"[Embedder] Loaded '{self.model_name}' via sentence-transformers "
                f"on {self._device}."
            )
            return
        except ImportError:
            logger.info(
                "[Embedder] sentence-transformers not installed — using TF-IDF fallback. "
                "Install for better embeddings: pip install sentence-transformers"
            )
        except Exception as exc:
            logger.warning(f"[Embedder] sentence-transformers load failed: {exc}")

        # Fallback: TF-IDF with fixed finance vocabulary
        self._model = _TFIDFEmbedder(_FINANCE_VOCAB)
        self._backend = "tfidf_fallback"
        logger.info(f"[Embedder] Using TF-IDF fallback ({len(_FINANCE_VOCAB)}-dim).")

    @staticmethod
    def _auto_detect_device() -> str:
        """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"[Embedder] CUDA GPU detected: {gpu_name}")
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vector."""
        self._ensure_loaded()
        if self._backend == "sentence_transformers":
            return self._model.get_sentence_embedding_dimension()
        return len(_FINANCE_VOCAB)

    def encode(self, brief: MarketBrief) -> np.ndarray:
        """Encode a single MarketBrief to a dense vector.

        Parameters
        ----------
        brief : MarketBrief
            The brief to encode. Must have a ``to_text()`` method.

        Returns
        -------
        np.ndarray, shape (embedding_dim,)
            L2-normalized embedding vector (if normalize=True).
        """
        self._ensure_loaded()
        text = brief.to_text() if hasattr(brief, "to_text") else str(brief)
        return self._encode_text(text)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a raw string to a dense vector.

        Parameters
        ----------
        text : str

        Returns
        -------
        np.ndarray, shape (embedding_dim,)
        """
        self._ensure_loaded()
        return self._encode_text(text)

    def encode_batch(self, briefs: list[MarketBrief]) -> np.ndarray:
        """Encode a list of MarketBriefs in a single batched forward pass.

        Batch encoding is significantly faster than encoding one at a time
        when using a GPU, as it amortizes kernel launch overhead.

        Parameters
        ----------
        briefs : list[MarketBrief]

        Returns
        -------
        np.ndarray, shape (N, embedding_dim)
        """
        self._ensure_loaded()
        texts = [b.to_text() if hasattr(b, "to_text") else str(b) for b in briefs]
        return self._encode_batch_texts(texts)

    def encode_quarterly(self, briefs_by_date: dict[str, MarketBrief]) -> pd.DataFrame:
        """Encode a time series of MarketBriefs indexed by quarter-end date.

        Parameters
        ----------
        briefs_by_date : dict[str, MarketBrief]
            Keys = YYYY-MM-DD date strings, values = MarketBrief objects.

        Returns
        -------
        pd.DataFrame
            Index = DatetimeIndex (quarter-end dates).
            Columns = 'embed_0', 'embed_1', ..., 'embed_{D-1}'.
        """

        if not briefs_by_date:
            return pd.DataFrame()

        dates = sorted(briefs_by_date.keys())
        briefs = [briefs_by_date[d] for d in dates]
        mat = self.encode_batch(briefs)

        cols = [f"embed_{i}" for i in range(mat.shape[1])]
        return pd.DataFrame(mat, index=pd.to_datetime(dates), columns=cols)

    # ------------------------------------------------------------------
    # Internal encoding
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> np.ndarray:
        if self._backend == "sentence_transformers":
            vec = self._model.encode(
                text,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return np.asarray(vec, dtype=np.float32)
        else:
            return self._model.encode(text, normalize=self.normalize)

    def _encode_batch_texts(self, texts: list[str]) -> np.ndarray:
        if self._backend == "sentence_transformers":
            mat = self._model.encode(
                texts,
                normalize_embeddings=self.normalize,
                batch_size=32,
                show_progress_bar=False,
            )
            return np.asarray(mat, dtype=np.float32)
        else:
            vecs = [self._model.encode(t, normalize=self.normalize) for t in texts]
            return np.stack(vecs, axis=0)

    # ------------------------------------------------------------------
    # Benchmark utility
    # ------------------------------------------------------------------

    def benchmark(self, n_samples: int = 50, text_length: int = 500) -> dict:
        """Measure encoding throughput on the current device.

        Parameters
        ----------
        n_samples : int
            Number of synthetic samples to encode.
        text_length : int
            Approximate character length of each synthetic text.

        Returns
        -------
        dict
            Keys: 'backend', 'device', 'n_samples', 'total_sec', 'samples_per_sec',
            'embedding_dim'.
        """
        import time

        self._ensure_loaded()

        # Generate synthetic brief texts
        sample_words = [
            "market",
            "risk",
            "tightening",
            "inflation",
            "AI",
            "energy",
            "earnings",
            "valuation",
            "neutral",
            "growth",
            "recession",
        ]
        texts = []
        rng = np.random.default_rng(42)
        for _ in range(n_samples):
            words = rng.choice(sample_words, size=text_length // 8, replace=True)
            texts.append(" ".join(words))

        start = time.perf_counter()
        self._encode_batch_texts(texts)
        elapsed = time.perf_counter() - start

        return {
            "backend": self._backend,
            "device": self._device,
            "n_samples": n_samples,
            "total_sec": round(elapsed, 4),
            "samples_per_sec": round(n_samples / elapsed, 1),
            "embedding_dim": self.embedding_dim,
        }


# ---------------------------------------------------------------------------
# TF-IDF fallback embedder
# ---------------------------------------------------------------------------


class _TFIDFEmbedder:
    """Lightweight TF-IDF embedder over a fixed finance vocabulary.

    Produces a deterministic, fixed-dimension bag-of-words vector from
    a text string. Not as expressive as a transformer, but has zero
    additional dependencies and is fully reproducible.
    """

    def __init__(self, vocab: list[str]) -> None:
        self._vocab = [w.lower() for w in vocab]
        self._v2idx = {w: i for i, w in enumerate(self._vocab)}
        self._dim = len(vocab)

        # IDF weights: pre-computed from a small corpus of finance terms
        # (uniform here — could be refined from actual brief history)
        self._idf = np.ones(self._dim, dtype=np.float32)

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Convert text to a TF-IDF vector over the fixed vocabulary."""
        tokens = re.findall(r"[a-z_]+", text.lower())
        tf = np.zeros(self._dim, dtype=np.float32)
        for tok in tokens:
            if tok in self._v2idx:
                tf[self._v2idx[tok]] += 1.0
        if len(tokens) > 0:
            tf /= len(tokens)

        vec = tf * self._idf
        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
        return vec
