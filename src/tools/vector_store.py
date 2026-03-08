"""
VectorStoreTool – manages the rare-disease vector store (ChromaDB or FAISS).

Provides:
  • build()     – create a new collection from a list of Documents
  • load()      – load an existing persisted collection
  • add()       – incrementally add documents
  • search()    – similarity search returning (Document, score) pairs
  • delete()    – remove documents by metadata filter
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from ..utils.logger import get_logger

logger = get_logger(__name__)

VectorStoreBackend = Chroma | FAISS


class VectorStoreTool:
    """
    Unified vector store interface supporting ChromaDB and FAISS.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        vs_cfg = self.config.get("vector_store", {})
        self.provider: str = vs_cfg.get("provider", "chromadb")
        self.persist_dir: str = vs_cfg.get("persist_directory", "./data/vector_store")
        self.collection_name: str = vs_cfg.get("collection_name", "rare_disease_docs")
        self.top_k: int = vs_cfg.get("top_k", 5)

        self.embedder = OpenAIEmbeddings(
            model=self.config.get("embedding", {}).get("model", "text-embedding-3-small")
        )
        self._store: Optional[VectorStoreBackend] = None

    # ──────────────────────────────────────────────────────────
    # Build / Load
    # ──────────────────────────────────────────────────────────

    def build(self, documents: List[Document]) -> VectorStoreBackend:
        """Create a new vector store from *documents* and persist it."""
        logger.info(
            "VectorStoreTool: building %s store with %d documents.",
            self.provider,
            len(documents),
        )
        if self.provider == "faiss":
            self._store = FAISS.from_documents(documents, self.embedder)
            faiss_path = Path(self.persist_dir) / "faiss_index"
            faiss_path.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(faiss_path))
            logger.info("VectorStoreTool: FAISS index saved to '%s'.", faiss_path)
        else:  # chromadb
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._store = Chroma.from_documents(
                documents,
                self.embedder,
                collection_name=self.collection_name,
                persist_directory=self.persist_dir,
            )
            logger.info(
                "VectorStoreTool: ChromaDB collection '%s' built and persisted.",
                self.collection_name,
            )
        return self._store

    def load(self) -> Optional[VectorStoreBackend]:
        """Load a persisted vector store.  Returns None if not found."""
        try:
            if self.provider == "faiss":
                faiss_path = str(Path(self.persist_dir) / "faiss_index")
                self._store = FAISS.load_local(
                    faiss_path,
                    self.embedder,
                    allow_dangerous_deserialization=True,
                )
            else:
                self._store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedder,
                    persist_directory=self.persist_dir,
                )
            logger.info("VectorStoreTool: loaded existing %s store.", self.provider)
            return self._store
        except Exception as exc:
            logger.warning("VectorStoreTool: could not load store (%s).", exc)
            return None

    # ──────────────────────────────────────────────────────────
    # CRUD operations
    # ──────────────────────────────────────────────────────────

    def add(self, documents: List[Document]) -> None:
        """Add documents to an existing store."""
        if self._store is None:
            raise RuntimeError("Vector store not initialised. Call build() or load() first.")
        self._store.add_documents(documents)
        logger.info("VectorStoreTool: added %d documents.", len(documents))

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Similarity search returning (Document, score) pairs sorted by relevance.

        Parameters
        ----------
        query : str
        top_k : int, optional  – overrides config top_k
        filter_metadata : dict, optional  – ChromaDB metadata filter
        """
        if self._store is None:
            raise RuntimeError("Vector store not initialised.")
        k = top_k or self.top_k
        kwargs: Dict[str, Any] = {}
        if filter_metadata and self.provider == "chromadb":
            kwargs["filter"] = filter_metadata

        return self._store.similarity_search_with_relevance_scores(query, k=k, **kwargs)

    @property
    def store(self) -> Optional[VectorStoreBackend]:
        return self._store
