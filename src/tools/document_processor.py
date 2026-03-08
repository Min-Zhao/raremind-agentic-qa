"""
DocumentProcessor – loads, chunks, and prepares rare-disease documents for indexing.

Supported input formats: PDF, plain text (.txt), JSON, Markdown.
Chunking strategy: recursive character splitter with configurable overlap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Load and chunk documents from various sources.

    Parameters
    ----------
    config : dict
        Pipeline configuration (uses rag.chunk_size and rag.chunk_overlap).
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        rag_cfg = self.config.get("rag", {})
        self.chunk_size: int = rag_cfg.get("chunk_size", 512)
        self.chunk_overlap: int = rag_cfg.get("chunk_overlap", 64)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def load_and_chunk(
        self,
        paths: List[str | Path],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Load documents from file paths and split them into chunks.

        Parameters
        ----------
        paths : list of str or Path
            File paths to load.  Supported: .pdf, .txt, .md, .json
        extra_metadata : dict, optional
            Additional metadata to attach to every chunk.

        Returns
        -------
        List of LangChain Document objects.
        """
        all_chunks: List[Document] = []
        for path in paths:
            p = Path(path)
            if not p.exists():
                logger.warning("DocumentProcessor: file not found '%s'. Skipping.", p)
                continue
            try:
                docs = self._load_file(p)
                chunks = self.splitter.split_documents(docs)
                # Enrich metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update(
                        {
                            "source": p.name,
                            "chunk_id": i,
                            "file_path": str(p),
                            **(extra_metadata or {}),
                        }
                    )
                all_chunks.extend(chunks)
                logger.info(
                    "DocumentProcessor: '%s' → %d chunks.", p.name, len(chunks)
                )
            except Exception as exc:
                logger.error("DocumentProcessor: failed to process '%s': %s", p, exc)

        logger.info("DocumentProcessor: total chunks produced: %d.", len(all_chunks))
        return all_chunks

    def load_from_json(
        self,
        json_path: str | Path,
        text_key: str = "content",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Load documents from a JSON file (list of dicts).

        Parameters
        ----------
        json_path : str or Path
        text_key : str  – key in each dict that holds the text content.
        extra_metadata : dict, optional  – merged into each document's metadata.
        """
        p = Path(json_path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        docs: List[Document] = []
        for item in data:
            content = item.get(text_key, "")
            if not content:
                continue
            metadata = {k: v for k, v in item.items() if k != text_key}
            metadata["source"] = p.name
            if extra_metadata:
                metadata.update(extra_metadata)
            docs.append(Document(page_content=content, metadata=metadata))

        chunks = self.splitter.split_documents(docs)
        logger.info(
            "DocumentProcessor: loaded %d records → %d chunks from '%s'.",
            len(docs),
            len(chunks),
            p.name,
        )
        return chunks

    def texts_to_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Document]:
        """Convert plain text strings directly into chunked Documents."""
        docs = [
            Document(page_content=t, metadata=(metadatas[i] if metadatas else {}))
            for i, t in enumerate(texts)
        ]
        return self.splitter.split_documents(docs)

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _load_file(self, path: Path) -> List[Document]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            return loader.load()
        elif suffix in (".txt", ".md"):
            loader = TextLoader(str(path), encoding="utf-8")
            return loader.load()
        elif suffix == ".json":
            return self.load_from_json(path)
        else:
            logger.warning(
                "DocumentProcessor: unsupported file type '%s'. Treating as text.", suffix
            )
            loader = TextLoader(str(path), encoding="utf-8")
            return loader.load()
