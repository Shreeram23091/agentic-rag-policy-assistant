#!/usr/bin/env python3
"""
scripts/build_index.py — One-time (or re-run) script to build/rebuild the FAISS vector index.

Usage:
    python scripts/build_index.py [--force]

Options:
    --force   Rebuild even if an index already exists on disk.

The script loads all documents from the path defined in DOCUMENTS_PATH,
chunks them, generates embeddings via the configured OpenAI/Azure OpenAI model,
and saves the FAISS index to FAISS_INDEX_PATH.
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

import structlog

from app.config import get_settings
from app.rag.retriever import RAGRetriever

logger = structlog.get_logger(__name__)


async def main(force: bool) -> None:
    cfg = get_settings()
    logger.info(
        "Building FAISS index",
        documents_path=cfg.documents_path,
        index_path=cfg.faiss_index_path,
        force=force,
    )

    retriever = RAGRetriever()
    await retriever.initialise(force_rebuild=force)

    if retriever.is_ready:
        logger.info("Index build complete")
    else:
        logger.error("Index build FAILED — check logs above")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the FAISS document index.")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index exists.")
    args = parser.parse_args()

    asyncio.run(main(force=args.force))
