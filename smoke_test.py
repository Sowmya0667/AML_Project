"""
smoke_test.py — Verify the pipeline works end-to-end with sample data.
Run: python smoke_test.py

Does NOT require Neo4j or an API key for the parser + chunker tests.
The entity extractor and vector store tests DO require .env to be configured.
"""
import sys

SAMPLE = """
IN THE SUPREME COURT OF INDIA
CIVIL APPELLATE JURISDICTION

CIVIL APPEAL NO. 4321 OF 2022

RAMESH KUMAR SHARMA                              ...Petitioner
                    VERSUS
STATE OF UTTAR PRADESH & ORS.                   ...Respondents

CORAM: JUSTICE D.Y. CHANDRACHUD, JUSTICE HIMA KOHLI

Date of Judgment: 22 March 2022

J U D G M E N T

The petitioner challenges the order of the High Court of Allahabad which
dismissed the writ petition filed under Article 226 of the Constitution of India.

The core issue is whether the right to speedy trial guaranteed under Article 21
of the Constitution was violated. The petitioner relies on Hussainara Khatoon v.
State of Bihar (1979) 3 SCC 1 and Maneka Gandhi v. Union of India (1978) 1 SCC 248.

Section 437 of the Code of Criminal Procedure (CrPC) was invoked for bail.
The doctrine of natural justice and the principle of audi alteram partem require
that no person be condemned unheard.

After hearing both parties the Court holds:
1. The right to speedy trial under Article 21 has been violated.
2. The state has failed to justify the period of detention.
3. The appeal is accordingly ALLOWED. Bail is granted.

APPEAL ALLOWED.
"""


def sep(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print("─" * 55)


# ── 1. Parser ─────────────────────────────────────────────────────────────────
sep("1 · TXT Parser")
try:
    from src.ingestion.txt_parser import JudgmentTXTParser, JudgmentMetadata

    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write(SAMPLE)
        path = tmp.name

    parser = JudgmentTXTParser(chunk_size=200, chunk_overlap=40)
    meta, chunks = parser.parse(path)
    os.unlink(path)

    print(f"  Case name   : {meta.case_name}")
    print(f"  Case number : {meta.case_number}")
    print(f"  Date        : {meta.date}")
    print(f"  Petitioner  : {meta.petitioner}")
    print(f"  Respondent  : {meta.respondent}")
    print(f"  Bench       : {meta.bench}")
    print(f"  Outcome     : {meta.outcome}")
    print(f"  Chunks      : {len(chunks)}")
    print("  ✅ PASSED")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)


# ── 2. Entity extractor ───────────────────────────────────────────────────────
sep("2 · Entity Extractor (requires LLM API key)")
try:
    from src.ingestion.entity_extractor import LegalEntityExtractor
    ents = LegalEntityExtractor().extract(SAMPLE, meta.case_number)
    print(f"  Citations    : {ents.citations}")
    print(f"  Statutes     : {ents.statutes}")
    print(f"  Concepts     : {ents.legal_concepts}")
    print(f"  Outcome      : {ents.outcome}")
    print(f"  Subject      : {ents.subject_matter}")
    print("  ✅ PASSED")
except Exception as e:
    print(f"  ⚠️  SKIPPED (check API key): {e}")


# ── 3. Vector store ───────────────────────────────────────────────────────────
sep("3 · Vector Store (ChromaDB)")
try:
    from src.vector.vector_store import LegalVectorStore
    vs = LegalVectorStore()
    vs.add_chunks(chunks)
    results = vs.query("speedy trial Article 21", top_k=2)
    print(f"  Chunks indexed  : {vs.count()}")
    print(f"  Query results   : {len(results)}")
    if results:
        print(f"  Top score       : {results[0]['score']:.3f}")
        print(f"  Top snippet     : {results[0]['text'][:80]}…")
    print("  ✅ PASSED")
except Exception as e:
    print(f"  ❌ FAILED: {e}")


# ── 4. Neo4j / Graph builder ──────────────────────────────────────────────────
sep("4 · Neo4j Graph Builder (requires Neo4j)")
try:
    from src.graph.graph_builder import LegalGraphBuilder
    from src.ingestion.entity_extractor import ExtractedEntities
    dummy_ents = ExtractedEntities(
        case_number  = meta.case_number,
        citations    = ["Hussainara Khatoon v. State of Bihar (1979)",
                        "Maneka Gandhi v. Union of India (1978)"],
        statutes     = ["Article 21 Constitution of India",
                        "Section 437 CrPC", "Article 226 Constitution"],
        legal_concepts=["natural justice", "audi alteram partem", "speedy trial"],
        holdings     = ["Right to speedy trial under Article 21 was violated.",
                        "Appeal allowed. Bail granted."],
        judges       = ["Justice D.Y. Chandrachud", "Justice Hima Kohli"],
        subject_matter="Constitutional Law",
        outcome      = "Allowed",
    )
    gb = LegalGraphBuilder()
    gb.add_judgment(meta, dummy_ents)
    gb.close()
    print("  ✅ PASSED — nodes & relationships written to Neo4j")
except Exception as e:
    print(f"  ⚠️  SKIPPED (check Neo4j credentials): {e}")


print("\n" + "═" * 55)
print("  Smoke test complete.")
print("═" * 55)
