"""
src/graph/graph_queries.py
────────────────────────────────────────────────────────────────────────────
All Cypher queries used during retrieval.
Each method returns a list of dicts (one per Case node found).
"""
from __future__ import annotations

from typing import Dict, List

from src.graph.neo4j_client import Neo4jClient


class GraphQueryEngine:

    def __init__(self):
        self.db = Neo4jClient.get()

    # ── By statute ────────────────────────────────────────────────────────────

    def by_statute(self, statute: str, limit: int = 10) -> List[Dict]:
        return self.db.run(
            """
            MATCH (c:Case)-[:APPLIES]->(s:Statute)
            WHERE toLower(s.name) CONTAINS toLower($q)
            RETURN c.case_number     AS case_number,
                   c.case_name       AS case_name,
                   c.date            AS date,
                   c.outcome         AS outcome,
                   c.subject_matter  AS subject_matter,
                   c.holdings        AS holdings
            LIMIT $limit
            """,
            {"q": statute, "limit": limit},
        )

    # ── By legal concept ──────────────────────────────────────────────────────

    def by_concept(self, concept: str, limit: int = 10) -> List[Dict]:
        return self.db.run(
            """
            MATCH (c:Case)-[:INVOLVES]->(lc:LegalConcept)
            WHERE toLower(lc.name) CONTAINS toLower($q)
            RETURN c.case_number     AS case_number,
                   c.case_name       AS case_name,
                   c.date            AS date,
                   c.outcome         AS outcome,
                   c.subject_matter  AS subject_matter,
                   c.holdings        AS holdings
            LIMIT $limit
            """,
            {"q": concept, "limit": limit},
        )

    # ── By judge ──────────────────────────────────────────────────────────────

    def by_judge(self, judge: str, limit: int = 10) -> List[Dict]:
        return self.db.run(
            """
            MATCH (c:Case)-[:DECIDED_BY]->(j:Judge)
            WHERE toLower(j.name) CONTAINS toLower($q)
            RETURN c.case_number     AS case_number,
                   c.case_name       AS case_name,
                   c.date            AS date,
                   c.outcome         AS outcome,
                   c.subject_matter  AS subject_matter,
                   c.holdings        AS holdings,
                   j.name            AS judge
            LIMIT $limit
            """,
            {"q": judge, "limit": limit},
        )

    # ── Citation network ──────────────────────────────────────────────────────

    def citing_cases(self, case_number: str, limit: int = 10) -> List[Dict]:
        """Cases that cite the given case."""
        return self.db.run(
            """
            MATCH (citing:Case)-[:CITES]->(target:Case {case_number: $cn})
            RETURN citing.case_number    AS case_number,
                   citing.case_name      AS case_name,
                   citing.date           AS date,
                   citing.outcome        AS outcome,
                   citing.subject_matter AS subject_matter,
                   citing.holdings       AS holdings
            LIMIT $limit
            """,
            {"cn": case_number, "limit": limit},
        )

    def cited_cases(self, case_number: str, limit: int = 10) -> List[Dict]:
        """Cases cited within the given case."""
        return self.db.run(
            """
            MATCH (src:Case {case_number: $cn})-[:CITES]->(cited:Case)
            RETURN cited.case_number    AS case_number,
                   cited.case_name      AS case_name,
                   cited.date           AS date,
                   cited.holdings       AS holdings
            LIMIT $limit
            """,
            {"cn": case_number, "limit": limit},
        )

    # ── Full-text search ──────────────────────────────────────────────────────

    def fulltext(self, query: str, limit: int = 10) -> List[Dict]:
        """Keyword search across Case.case_name + subject_matter + holdings."""
        try:
            return self.db.run(
                """
                CALL db.index.fulltext.queryNodes("case_fulltext", $q)
                YIELD node, score
                RETURN node.case_number     AS case_number,
                       node.case_name       AS case_name,
                       node.date            AS date,
                       node.outcome         AS outcome,
                       node.subject_matter  AS subject_matter,
                       node.holdings        AS holdings,
                       score
                ORDER BY score DESC
                LIMIT $limit
                """,
                {"q": query, "limit": limit},
            )
        except Exception:
            # Full-text index not available in all Neo4j editions
            return []

    # ── Related via shared entities (1-hop neighbourhood) ────────────────────

    def related(self, case_number: str, limit: int = 10) -> List[Dict]:
        """
        Cases related to the given case through shared judges,
        statutes, or legal concepts (1-hop).
        """
        return self.db.run(
            """
            MATCH (c:Case {case_number: $cn})-[*1..2]-(related:Case)
            WHERE related.case_number <> $cn
            RETURN DISTINCT
                   related.case_number    AS case_number,
                   related.case_name      AS case_name,
                   related.date           AS date,
                   related.outcome        AS outcome,
                   related.subject_matter AS subject_matter,
                   related.holdings       AS holdings
            LIMIT $limit
            """,
            {"cn": case_number, "limit": limit},
        )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        rows = self.db.run(
            """
            MATCH (c:Case)         WITH count(c)  AS cases
            MATCH (j:Judge)        WITH cases, count(j)  AS judges
            MATCH (s:Statute)      WITH cases, judges, count(s) AS statutes
            MATCH (lc:LegalConcept) WITH cases, judges, statutes, count(lc) AS concepts
            RETURN cases, judges, statutes, concepts
            """
        )
        return rows[0] if rows else {}
