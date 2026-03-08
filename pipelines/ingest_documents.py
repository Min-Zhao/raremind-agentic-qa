"""
ingest_documents.py – builds the vector store from the pseudo dataset (or real documents).

Usage:
    python pipelines/ingest_documents.py
    python pipelines/ingest_documents.py --docs_path path/to/your/documents.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.document_processor import DocumentProcessor
from src.tools.vector_store import VectorStoreTool
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument(
        "--docs_path",
        type=str,
        default="data/pseudo_dataset/rare_disease_docs.json",
        help="Path to the JSON document file.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    args = parser.parse_args()

    config = load_config(args.config)

    # Generate dataset if not present
    docs_path = Path(args.docs_path)
    if not docs_path.exists():
        logger.info("Dataset not found; generating pseudo dataset…")
        from data.pseudo_dataset.generate_dataset import main as gen_main
        gen_main()

    logger.info("Processing documents from '%s'…", docs_path)
    processor = DocumentProcessor(config=config)
    chunks = processor.load_from_json(
        docs_path,
        text_key="content",
        extra_metadata={"pipeline": "agentic_framework"},
    )
    logger.info("Produced %d chunks.", len(chunks))

    vs = VectorStoreTool(config=config)
    vs.build(chunks)
    logger.info("Vector store built successfully.")
    print(f"\n✓ Ingested {len(chunks)} chunks into the vector store.")
    print(f"  Persist directory: {config.get('vector_store', {}).get('persist_directory', './data/vector_store')}")


if __name__ == "__main__":
    main()
