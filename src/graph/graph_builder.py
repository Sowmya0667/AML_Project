"""
src/graph/graph_builder.py
────────────────────────────────────────────────────────────────────────────
Builds the Neo4j knowledge graph from parsed judgment data.

Graph Schema
────────────
Nodes
  (:Case)          case_number, case_name, date, petitioner, respondent,
                   subject_matter, outcome, holdings, file_path
  (:Judge)         name
  (:Statute)       name
  (:LegalConcept)  name

Relationships
  (:Case)-[:DECIDED_BY]->(:Judge)
  (:Case)-[:APPLIES]   ->(:Statute)
  (:Case)-[:INVOLVES]  ->(:LegalConcept)
  (:Case)-[:CITES]     ->(:Case)
"""
from __future__ import annotations

import logging

from src.graph.neo4j_client          import Neo4jClient
from src.ingestion.txt_parser        import JudgmentMetadata
from src.ingestion.entity_extractor  import ExtractedEntities

log = logging.getLogger(__name__)


class LegalGraphBuilder:

    def __init__(self):
        self.db = Neo4jClient.get()

    # ── Public ────────────────────────────────────────────────────────────────

    def add_judgment(self, meta: JudgmentMetadata, ents: ExtractedEntities):
        """Upsert a Case node and all its relationships into Neo4j."""
        self._case(meta, ents)
        self._judges(meta, ents)
        self._statutes(meta, ents)
        self._concepts(meta, ents)
        self._citations(meta, ents)

    # ── Case ──────────────────────────────────────────────────────────────────

    def _case(self, m: JudgmentMetadata, e: ExtractedEntities):
        self.db.run_write(
            """
            MERGE (c:Case {case_number: $cn})
            SET c.case_name     = $case_name,
                c.date          = $date,
                c.petitioner    = $petitioner,
                c.respondent    = $respondent,
                c.subject_matter= $subject_matter,
                c.outcome       = $outcome,
                c.holdings      = $holdings,
                c.file_path     = $file_path
            """,
            {
                "cn":            m.case_number,
                "case_name":     m.case_name,
                "date":          m.date,
                "petitioner":    m.petitioner,
                "respondent":    m.respondent,
                "subject_matter":e.subject_matter,
                "outcome":       e.outcome or m.outcome,
                "holdings":      " | ".join(e.holdings),
                "file_path":     m.file_path,
            },
        )

    # ── Judges ────────────────────────────────────────────────────────────────

    def _judges(self, m: JudgmentMetadata, e: ExtractedEntities):
        judges = e.judges or m.bench
        for name in judges:
            name = name.strip()
            if not name:
                continue
            self.db.run_write(
                """
                MERGE (j:Judge {name: $name})
                WITH j
                MATCH (c:Case {case_number: $cn})
                MERGE (c)-[:DECIDED_BY]->(j)
                """,
                {"name": name, "cn": m.case_number},
            )

    # ── Statutes ──────────────────────────────────────────────────────────────

    def _statutes(self, m: JudgmentMetadata, e: ExtractedEntities):
        for s in e.statutes:
            s = s.strip()
            if not s:
                continue
            self.db.run_write(
                """
                MERGE (s:Statute {name: $name})
                WITH s
                MATCH (c:Case {case_number: $cn})
                MERGE (c)-[:APPLIES]->(s)
                """,
                {"name": s, "cn": m.case_number},
            )

    # ── Legal Concepts ────────────────────────────────────────────────────────

    def _concepts(self, m: JudgmentMetadata, e: ExtractedEntities):
        for lc in e.legal_concepts:
            lc = lc.strip()
            if not lc:
                continue
            self.db.run_write(
                """
                MERGE (lc:LegalConcept {name: $name})
                WITH lc
                MATCH (c:Case {case_number: $cn})
                MERGE (c)-[:INVOLVES]->(lc)
                """,
                {"name": lc, "cn": m.case_number},
            )

    # ── Citations ─────────────────────────────────────────────────────────────

    def _citations(self, m: JudgmentMetadata, e: ExtractedEntities):
        """
        CITES relationships.
        Cited cases may not yet exist → MERGE them as stub nodes.
        """
        for cited in e.citations:
            cited = cited.strip()
            if not cited or cited == m.case_number:
                continue
            self.db.run_write(
                """
                MATCH (src:Case {case_number: $src_cn})
                MERGE (tgt:Case {case_number: $tgt_cn})
                ON CREATE SET tgt.case_name = $tgt_cn
                MERGE (src)-[:CITES]->(tgt)
                """,
                {"src_cn": m.case_number, "tgt_cn": cited},
            )

    def close(self):
        self.db.close()
