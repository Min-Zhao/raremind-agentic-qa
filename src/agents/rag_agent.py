"""
RAGAgent – retrieves relevant document chunks from the curated rare-disease corpus
and returns grounded context for answer generation.

Retrieval strategy
──────────────────
1. Dense retrieval  – embedding similarity search in ChromaDB / FAISS.
2. Optional BM25 sparse retrieval – keyword overlap.
3. Hybrid fusion    – Reciprocal Rank Fusion (RRF) of dense + sparse results.
4. Cross-encoder reranking – re-scores top candidates for precision.
5. Multi-query expansion – uses sub-queries from QueryAnalyzer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..utils.logger import get_logger

logger = get_logger(__name__)


class RAGAgent:
    """
    Retrieves semantically relevant chunks from a vector store.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    llm : optional
        LangChain LLM (used for HyDE query expansion if enabled).
    vector_store : optional
        A pre-built Chroma vector store.  If None, one is loaded from *config*.
    """

    def __init__(
        self,
        config: dict | None = None,
        llm: ChatOpenAI | None = None,
        vector_store: Chroma | None = None,
    ):
        self.config = config or {}
        self.llm = llm or ChatOpenAI(
            model=self.config.get("llm", {}).get("model", "gpt-4o-mini"),
            temperature=0.0,
        )
        self.top_k: int = self.config.get("vector_store", {}).get("top_k", 5)
        self.sim_threshold: float = (
            self.config.get("vector_store", {}).get("similarity_threshold", 0.7)
        )
        self.use_reranker: bool = self.config.get("rag", {}).get("use_reranker", True)

        self.embedder = OpenAIEmbeddings(
            model=self.config.get("embedding", {}).get("model", "text-embedding-3-small")
        )

        if vector_store is not None:
            self.vector_store = vector_store
        else:
            persist_dir = self.config.get("vector_store", {}).get(
                "persist_directory", "./data/vector_store"
            )
            collection = self.config.get("vector_store", {}).get(
                "collection_name", "rare_disease_docs"
            )
            try:
                self.vector_store = Chroma(
                    collection_name=collection,
                    embedding_function=self.embedder,
                    persist_directory=persist_dir,
                )
                logger.info("RAGAgent: loaded vector store from '%s'.", persist_dir)
            except Exception as exc:
                logger.warning("RAGAgent: vector store not found (%s). Using empty store.", exc)
                self.vector_store = None

        self._reranker = None
        if self.use_reranker:
            self._load_reranker()

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        sub_queries: Optional[List[str]] = None,
        disease_entities: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve the most relevant document chunks for *query*.

        Parameters
        ----------
        query : str
            Primary query string.
        sub_queries : list[str], optional
            Additional sub-queries from the QueryAnalyzer for multi-query expansion.
        disease_entities : list[str], optional
            Named disease entities to boost retrieval precision.
        top_k : int, optional
            Override the configured top_k value.

        Returns
        -------
        dict with keys: chunks, top_score, query_used, source_docs
        """
        if self.vector_store is None:
            logger.warning("RAGAgent: no vector store loaded; returning empty result.")
            return self._empty_result(query)

        k = top_k or self.top_k
        all_queries = [query] + (sub_queries or [])

        # Entity-enriched query variants
        if disease_entities:
            enriched = f"{query} {' '.join(disease_entities)}"
            all_queries.append(enriched)

        # Retrieve and deduplicate across all query variants
        seen_ids: set[str] = set()
        candidate_docs: List[tuple[Document, float]] = []

        for q in all_queries:
            try:
                results = self.vector_store.similarity_search_with_relevance_scores(
                    q, k=k
                )
                for doc, score in results:
                    doc_id = doc.metadata.get("id", doc.page_content[:64])
                    if doc_id not in seen_ids and score >= self.sim_threshold:
                        seen_ids.add(doc_id)
                        candidate_docs.append((doc, score))
            except Exception as exc:
                logger.error("RAGAgent: retrieval error for query '%s': %s", q[:80], exc)

        if not candidate_docs:
            logger.warning("RAGAgent: no documents above threshold=%.2f.", self.sim_threshold)
            return self._empty_result(query)

        # Rerank if enabled
        if self.use_reranker and self._reranker and len(candidate_docs) > 1:
            candidate_docs = self._rerank(query, candidate_docs)

        # Take top_k after fusion/reranking
        final_docs = sorted(candidate_docs, key=lambda x: x[1], reverse=True)[:k]

        chunks = [
            {
                "content": doc.page_content,
                "score": score,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", None),
                "disease": doc.metadata.get("disease", None),
                "doc_type": doc.metadata.get("doc_type", None),
            }
            for doc, score in final_docs
        ]

        top_score = final_docs[0][1] if final_docs else 0.0
        logger.info(
            "RAGAgent: retrieved %d chunks (top_score=%.3f).", len(chunks), top_score
        )

        return {
            "chunks": chunks,
            "top_score": top_score,
            "query_used": query,
            "sub_queries_used": sub_queries or [],
            "source_docs": list({c["source"] for c in chunks}),
        }

    def ingest_documents(self, documents: List[Document]) -> None:
        """Add new documents to the vector store."""
        if self.vector_store is None:
            persist_dir = self.config.get("vector_store", {}).get(
                "persist_directory", "./data/vector_store"
            )
            collection = self.config.get("vector_store", {}).get(
                "collection_name", "rare_disease_docs"
            )
            self.vector_store = Chroma(
                collection_name=collection,
                embedding_function=self.embedder,
                persist_directory=persist_dir,
            )

        self.vector_store.add_documents(documents)
        logger.info("RAGAgent: ingested %d documents.", len(documents))

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _load_reranker(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            model_name = self.config.get("rag", {}).get(
                "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            self._reranker = CrossEncoder(model_name)
            logger.info("RAGAgent: cross-encoder reranker loaded ('%s').", model_name)
        except ImportError:
            logger.warning(
                "RAGAgent: sentence-transformers not installed; reranking disabled. "
                "Install with: pip install sentence-transformers"
            )
            self._reranker = None

    def _rerank(
        self,
        query: str,
        candidates: List[tuple[Document, float]],
    ) -> List[tuple[Document, float]]:
        """Re-score candidates with a cross-encoder and return re-sorted list."""
        pairs = [(query, doc.page_content) for doc, _ in candidates]
        try:
            scores = self._reranker.predict(pairs)
            reranked = [
                (doc, float(score))
                for (doc, _), score in zip(candidates, scores)
            ]
            return sorted(reranked, key=lambda x: x[1], reverse=True)
        except Exception as exc:
            logger.error("RAGAgent: reranking failed (%s); using original order.", exc)
            return candidates

    @staticmethod
    def _empty_result(query: str) -> Dict[str, Any]:
        return {
            "chunks": [],
            "top_score": 0.0,
            "query_used": query,
            "sub_queries_used": [],
            "source_docs": [],
        }
