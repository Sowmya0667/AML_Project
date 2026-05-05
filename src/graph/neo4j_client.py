"""
src/graph/neo4j_client.py
────────────────────────────────────────────────────────────────────────────
Neo4j driver wrapper with:
  - Singleton connection pool
  - Automatic schema / constraint creation on first use
  - Simple run() and run_write() helpers
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver

from config import config

log = logging.getLogger(__name__)

# ─── Schema ──────────────────────────────────────────────────────────────────
# Run once when the DB is first connected.

_SCHEMA = [
    # Uniqueness constraints (idempotent)
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Case)         REQUIRE c.case_number IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Judge)        REQUIRE j.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Statute)      REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (lc:LegalConcept) REQUIRE lc.name IS UNIQUE",
    # Full-text index for keyword search
    "CREATE FULLTEXT INDEX case_fulltext IF NOT EXISTS "
    "FOR (c:Case) ON EACH [c.case_name, c.subject_matter, c.outcome, c.holdings]",
]


class Neo4jClient:
    _instance: Optional["Neo4jClient"] = None

    def __init__(self):
        self._driver: Driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
        )
        self._apply_schema()
        log.info("Neo4j connected and schema verified.")

    # ── Singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> "Neo4jClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Schema ────────────────────────────────────────────────────────────────

    def _apply_schema(self):
        try:
            with self._driver.session() as s:
                for q in _SCHEMA:
                    try:
                        s.run(q)
                    except Exception as e:
                        log.debug(f"Schema (skipped): {e}")
        except Exception as e:
            log.warning(f"Could not connect to Neo4j to apply schema (DB might be paused): {e}")

    # ── Query helpers ─────────────────────────────────────────────────────────

    def run(self, query: str, params: Dict = None) -> List[Dict]:
        try:
            with self._driver.session() as s:
                return [dict(r) for r in s.run(query, params or {})]
        except Exception as e:
            log.warning(f"Neo4j query failed (DB might be paused): {e}")
            return []

    def run_write(self, query: str, params: Dict = None) -> None:
        try:
            with self._driver.session() as s:
                s.execute_write(lambda tx: tx.run(query, params or {}))
        except Exception as e:
            log.warning(f"Neo4j write failed (DB might be paused): {e}")

    def close(self):
        self._driver.close()
        Neo4jClient._instance = None
