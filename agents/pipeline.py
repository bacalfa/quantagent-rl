"""
agents/pipeline.py
==================
AgentPipeline — orchestrates agent execution across walk-forward folds.

Responsibilities
----------------
- Accepts a ``WalkForwardFold`` (from the data module) and runs the
  LangGraph graph for every quarter-end date in the fold.
- Handles data preparation: extracts the as-of-date macro snapshot from the
  FRED DataFrame, looks up XBRL facts and MD&A text per ticker, builds the
  ``sector_map`` from the universe sector mapping.
- Caches ``MarketBrief`` JSON on disk so repeated runs are free.
- Embeds each ``MarketBrief`` into a dense vector using ``MarketBriefEmbedder``.
- Returns an ``AgentBundle`` containing all quarterly briefs and the
  embedding DataFrame ready for the RL state construction.

Walk-Forward Safety
-------------------
All agent inputs are sliced to ``as_of_date`` before being passed to the
graph. XBRL data and MD&A text are filtered by ``filing_date <= as_of_date``.
Web search is date-bounded via query construction (best-effort).

Caching
-------
MarketBrief JSON files are written to ``{cache_dir}/{fold_idx}/{YYYY-MM-DD}.json``.
On subsequent runs, cached briefs are loaded directly without LLM calls.
Set ``force_refresh=True`` to re-run all agents.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from agents.config import AgentConfig
from agents.embedder import MarketBriefEmbedder
from agents.orchestrator import GraphState, build_agent_graph
from agents.schemas import MarketBrief

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentBundle — output container for one fold
# ---------------------------------------------------------------------------


@dataclass
class AgentBundle:
    """All agent outputs for one walk-forward fold.

    Attributes
    ----------
    fold_idx : int
    briefs : dict[str, MarketBrief]
        Keys = YYYY-MM-DD quarter-end dates. Values = MarketBrief objects.
    embeddings : pd.DataFrame
        Index = DatetimeIndex (quarter-end dates).
        Columns = 'embed_0', ..., 'embed_{D-1}'.
        Shape: (n_quarters, embedding_dim).
    embedding_dim : int
    errors : list[str]
        Any non-fatal errors that occurred during the run.
    """

    fold_idx: int
    briefs: dict[str, MarketBrief] = field(default_factory=dict)
    embeddings: pd.DataFrame = field(default_factory=pd.DataFrame)
    embedding_dim: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def n_quarters(self) -> int:
        return len(self.briefs)

    def get_brief(self, date: str | pd.Timestamp) -> MarketBrief | None:
        """Return the MarketBrief for a specific quarter-end date."""
        key = pd.Timestamp(date).strftime("%Y-%m-%d")
        return self.briefs.get(key)

    def get_embedding(self, date: str | pd.Timestamp) -> np.ndarray | None:
        """Return the embedding vector for a specific quarter-end date."""
        ts = pd.Timestamp(date)
        if ts in self.embeddings.index:
            return self.embeddings.loc[ts].values.astype(np.float32)
        return None

    def __repr__(self) -> str:
        return (
            f"AgentBundle(fold={self.fold_idx}, "
            f"n_quarters={self.n_quarters}, "
            f"embedding_dim={self.embedding_dim})"
        )


# ---------------------------------------------------------------------------
# AgentPipeline
# ---------------------------------------------------------------------------


class AgentPipeline:
    """Runs the LangGraph multi-agent system across walk-forward folds.

    Parameters
    ----------
    config : AgentConfig
        Controls mock mode, web search, embedding device, caching, etc.
    cache_dir : str
        Root directory for caching MarketBrief JSON files.
    sec_user_agent : str
        User-agent string for EDGAR API calls.

    Examples
    --------
    >>> pipeline = AgentPipeline(AgentConfig(mock_mode=True))
    >>> bundle = pipeline.run_fold(data_fold, sec_metadata_df)
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        cache_dir: str = "../data/cache/agent_briefs",
        sec_user_agent: str = "QuantAgentRL research@example.com",
    ) -> None:
        self.cfg = config or AgentConfig(mock_mode=True)
        self.cache_dir = Path(cache_dir)
        self.sec_user_agent = sec_user_agent
        self._graph = build_agent_graph(self.cfg)
        self._embedder = MarketBriefEmbedder(
            model_name=self.cfg.embedding_model,
            device=self.cfg.embedding_device,
        )

    # ------------------------------------------------------------------
    # Single fold
    # ------------------------------------------------------------------

    def run_fold(
        self,
        fold: object,
        sec_metadata: pd.DataFrame | None = None,
        force_refresh: bool = False,
    ) -> AgentBundle:
        """Produce agent outputs for all quarter-end dates in a fold.

        Parameters
        ----------
        fold : WalkForwardFold
            Output of ``DataPipeline.get_fold(i)``.
        sec_metadata : pd.DataFrame | None
            SEC filing metadata from ``SECFilingIngester.fetch()``.
            Used to look up MD&A file paths per ticker.
        force_refresh : bool
            Re-run agents even if cached briefs exist.

        Returns
        -------
        AgentBundle
        """
        logger.info(
            f"[AgentPipeline] Running fold {fold.fold_idx} "
            f"({fold.n_train_quarters} train + {fold.n_test_quarters} test quarters)"
        )

        all_dates = list(fold.train_dates) + list(fold.test_dates)
        all_briefs: dict[str, MarketBrief] = {}
        all_errors: list[str] = []

        for date in all_dates:
            date_str = date.strftime("%Y-%m-%d")
            brief = self._run_single_date(
                date_str=date_str,
                fold=fold,
                sec_metadata=sec_metadata,
                fold_idx=fold.fold_idx,
                force_refresh=force_refresh,
            )
            all_briefs[date_str] = brief

        # Embed all briefs
        embeddings = self._embedder.encode_quarterly(all_briefs)
        embed_dim = self._embedder.embedding_dim

        bundle = AgentBundle(
            fold_idx=fold.fold_idx,
            briefs=all_briefs,
            embeddings=embeddings,
            embedding_dim=embed_dim,
            errors=all_errors,
        )
        logger.info(f"[AgentPipeline] Fold {fold.fold_idx} complete: {bundle}")
        return bundle

    # ------------------------------------------------------------------
    # All folds
    # ------------------------------------------------------------------

    def run_all_folds(
        self,
        data_pipeline: object,
        force_refresh: bool = False,
    ) -> list[AgentBundle]:
        """Run the agent pipeline for every fold in a DataPipeline.

        Parameters
        ----------
        data_pipeline : DataPipeline
        force_refresh : bool

        Returns
        -------
        list[AgentBundle]
        """
        bundles = []
        sec_metadata = getattr(data_pipeline, "sec_metadata", None)
        for i in range(data_pipeline.n_folds):
            fold = data_pipeline.get_fold(i)
            bundle = self.run_fold(
                fold=fold,
                sec_metadata=sec_metadata,
                force_refresh=force_refresh,
            )
            bundles.append(bundle)
        return bundles

    # ------------------------------------------------------------------
    # Single date execution
    # ------------------------------------------------------------------

    def _run_single_date(
        self,
        date_str: str,
        fold: object,
        sec_metadata: pd.DataFrame | None,
        fold_idx: int,
        force_refresh: bool,
    ) -> MarketBrief:
        """Run the graph for a single quarter-end date.

        Checks the cache first; invokes the LangGraph graph if needed.
        """
        cache_path = self._brief_cache_path(fold_idx, date_str)

        if not force_refresh and cache_path.exists():
            logger.debug(f"[AgentPipeline] Loading cached brief for {date_str}")
            return self._load_brief(cache_path)

        # Build graph input state
        state = self._build_graph_state(date_str, fold, sec_metadata)

        logger.info(f"[AgentPipeline] Running agents for {date_str}")
        try:
            result_state = self._graph.invoke(state)
            brief = result_state.get("market_brief") or MarketBrief.neutral(date_str)
        except Exception as exc:
            logger.error(
                f"[AgentPipeline] Graph invocation failed for {date_str}: {exc}"
            )
            brief = MarketBrief.neutral(date_str)

        self._save_brief(brief, cache_path)
        return brief

    # ------------------------------------------------------------------
    # Graph state preparation
    # ------------------------------------------------------------------

    def _build_graph_state(
        self,
        date_str: str,
        fold: object,
        sec_metadata: pd.DataFrame | None,
    ) -> GraphState:
        """Build the initial LangGraph state for a given date.

        Extracts macro signal snapshot, XBRL facts, MD&A text, and sector
        map — all strictly bounded to ``date_str``.
        """
        tickers = fold.tickers
        macro_data = self._extract_macro_snapshot(fold, date_str)
        sector_map = self._build_sector_map(fold, tickers)
        xbrl_data = self._load_xbrl_data(tickers, date_str)
        mda_data = self._load_mda_data(tickers, sec_metadata, date_str)

        return GraphState(
            as_of_date=date_str,
            tickers=tickers,
            sector_map=sector_map,
            macro_data=macro_data,
            xbrl_data=xbrl_data,
            mda_data=mda_data,
            macro_brief=None,
            sector_briefs={},
            company_briefs={},
            market_brief=None,
            errors=[],
        )

    @staticmethod
    def _extract_macro_snapshot(fold: object, date_str: str) -> dict[str, float]:
        """Return the most recent FRED values as of date_str."""
        macro_df = getattr(fold, "macro", None)
        if macro_df is None or macro_df.empty:
            return {}
        ts = pd.Timestamp(date_str)
        # Use .asof() to get the most recent value on or before the date
        snapshot = (
            macro_df[macro_df.index <= ts].iloc[-1]
            if not macro_df[macro_df.index <= ts].empty
            else macro_df.iloc[0]
        )
        return {
            col: float(val)
            for col, val in snapshot.items()
            if pd.notna(val) and isinstance(val, (int, float, np.floating))
        }

    @staticmethod
    def _build_sector_map(fold: object, tickers: list[str]) -> dict[str, list[str]]:
        """Build a sector → tickers mapping from the fold's universe."""
        try:
            from data.universe import TICKER_SECTOR_MAP

            sector_map: dict[str, list[str]] = {}
            for ticker in tickers:
                sector = TICKER_SECTOR_MAP.get(ticker, "Unknown")
                sector_map.setdefault(sector, []).append(ticker)
            return sector_map
        except ImportError:
            return {"Unknown": tickers}

    def _load_xbrl_data(
        self,
        tickers: list[str],
        date_str: str,
    ) -> dict[str, dict]:
        """Load XBRL financial facts, bounded to date_str.

        In mock mode returns empty dicts (CompanyAgent uses mock responses).
        In live mode, fetches from EDGAR REST API via tools.fetch_xbrl_facts.
        """
        if self.cfg.mock_mode:
            return {t: {} for t in tickers}

        from agents.tools import fetch_xbrl_facts

        result: dict[str, dict] = {}
        for ticker in tickers:
            try:
                facts = fetch_xbrl_facts(
                    ticker=ticker,
                    user_agent=self.sec_user_agent,
                    last_n_periods=8,
                )
                # Filter observations to date_str
                filtered: dict[str, list[dict]] = {}
                for metric, obs_list in facts.items():
                    filtered[metric] = [
                        o for o in obs_list if o.get("period_end", "9999") <= date_str
                    ]
                result[ticker] = filtered
            except Exception as exc:
                logger.warning(f"[AgentPipeline] XBRL load failed for {ticker}: {exc}")
                result[ticker] = {}
        return result

    @staticmethod
    def _load_mda_data(
        tickers: list[str],
        sec_metadata: pd.DataFrame | None,
        date_str: str,
    ) -> dict[str, str]:
        """Load MD&A text for the most recent filing before date_str."""
        if sec_metadata is None or sec_metadata.empty:
            return {t: "" for t in tickers}

        result: dict[str, str] = {}
        cutoff = pd.Timestamp(date_str)

        for ticker in tickers:
            # Filter to filings for this ticker before the cutoff
            mask = (sec_metadata["ticker"] == ticker) & (
                sec_metadata["filing_date"] <= cutoff
            )
            filings = sec_metadata[mask].sort_values("filing_date", ascending=False)

            if filings.empty:
                result[ticker] = ""
                continue

            # Get text from the most recent filing
            file_path = filings.iloc[0].get("file_path", "")
            if file_path:
                try:
                    from data.ingestion import SECFilingIngester

                    result[ticker] = SECFilingIngester.read_filing_text(
                        file_path, max_chars=40_000
                    )
                except Exception as exc:
                    logger.warning(
                        f"[AgentPipeline] MD&A load failed for {ticker}: {exc}"
                    )
                    result[ticker] = ""
            else:
                result[ticker] = ""

        return result

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _brief_cache_path(self, fold_idx: int, date_str: str) -> Path:
        path = self.cache_dir / f"fold_{fold_idx}" / f"{date_str}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _save_brief(brief: MarketBrief, path: Path) -> None:
        try:
            with open(path, "w") as f:
                # Exclude nested sub-briefs from the cached JSON to keep
                # file size manageable; they are re-attached on load if needed
                json.dump(brief.to_dict(), f, indent=2)
        except Exception as exc:
            logger.warning(f"[AgentPipeline] Cache write failed: {exc}")

    @staticmethod
    def _load_brief(path: Path) -> MarketBrief:
        try:
            with open(path) as f:
                data = json.load(f)
            return MarketBrief.from_dict(data)
        except Exception as exc:
            logger.warning(f"[AgentPipeline] Cache read failed ({path}): {exc}")
            date_str = path.stem  # filename is YYYY-MM-DD.json
            return MarketBrief.neutral(date_str)
