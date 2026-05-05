"""
src/ingestion/data_loader.py
────────────────────────────────────────────────────────────────────────────
Orchestrates the full ingestion pipeline for .txt judgment files:

  .txt file
     │
     ▼  JudgmentTXTParser
  raw text + metadata
     │
     ├──▶ LegalEntityExtractor  ──▶  Neo4j Knowledge Graph
     │
     └──▶ LegalVectorStore      ──▶  ChromaDB

Run from project root:
  python -m src.ingestion.data_loader --input ./data/judgments/
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from tqdm import tqdm

from config import config
from src.ingestion.txt_parser      import JudgmentTXTParser
from src.ingestion.entity_extractor import LegalEntityExtractor
from src.graph.graph_builder        import LegalGraphBuilder
from src.vector.vector_store        import LegalVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end ingestion: parse → extract → graph + vector store."""

    def __init__(self):
        self.parser    = JudgmentTXTParser(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.extractor = LegalEntityExtractor()
        self.graph     = LegalGraphBuilder()
        self.vector    = LegalVectorStore()

    # ── Directory ─────────────────────────────────────────────────────────────

    def ingest_directory(self, directory: str) -> dict:
        """Ingest every .txt file found (recursively) in *directory*."""
        paths = list(Path(directory).rglob("*.txt"))
        if not paths:
            log.warning(f"No .txt files found in {directory}")
            return {"total": 0, "success": 0, "failed": 0}

        log.info(f"Found {len(paths)} judgment files in '{directory}'")
        stats = {"total": len(paths), "success": 0, "failed": 0, "files": []}

        for path in tqdm(paths, desc="Ingesting"):
            try:
                r = self._ingest_one(str(path))
                stats["success"] += 1
                stats["files"].append({"path": str(path), "status": "ok", **r})
            except Exception as exc:
                log.error(f"  ✗ {path.name}: {exc}")
                stats["failed"] += 1
                stats["files"].append({"path": str(path), "status": "error", "error": str(exc)})

        log.info(
            f"Done — {stats['success']} succeeded, {stats['failed']} failed"
        )
        return stats

    # ── Single file ───────────────────────────────────────────────────────────

    def _ingest_one(self, file_path: str) -> dict:
        # 1. Parse
        metadata, chunks = self.parser.parse(file_path)
        log.info(f"  Parsed  {Path(file_path).name}: "
                 f"'{metadata.case_name}' → {len(chunks)} chunks")

        # 2. Extract entities and update Graph DB
        try:
            seed_text = chunks[0].text if chunks else metadata.raw_text[:3500]
            entities  = self.extractor.extract(seed_text, metadata.case_number)
            log.info(
                f"  Entities: {len(entities.citations)} citations, "
                f"{len(entities.statutes)} statutes, "
                f"outcome={entities.outcome or 'n/a'}"
            )
            self.graph.add_judgment(metadata, entities)
        except Exception as e:
            log.warning(f"  Graph extraction/insertion failed (skipping graph for this case): {e}")

        # 4. Index chunks in vector store
        self.vector.add_chunks(chunks)

        return {
            "case_name":   metadata.case_name,
            "case_number": metadata.case_number,
            "chunks":      len(chunks),
        }

    def close(self):
        self.graph.close()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Ingest Supreme Court .txt judgments")
    ap.add_argument("--input", default=config.JUDGMENTS_DATA_DIR,
                    help="Directory containing .txt judgment files")
    args = ap.parse_args()

    pipeline = IngestionPipeline()
    try:
        stats = pipeline.ingest_directory(args.input)
        print(f"\n✅  Ingested {stats['success']} / {stats['total']} judgments")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
