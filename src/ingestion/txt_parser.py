"""
src/ingestion/txt_parser.py
────────────────────────────────────────────────────────────────────────────
Parses plain-text Indian Supreme Court judgment files from the Kaggle dataset:
  https://www.kaggle.com/datasets/vxrunsonii/supreme-court-judgments-txt

Each .txt file contains a full judgment. This module:
  1. Reads the raw text
  2. Extracts metadata (case number, date, bench, parties, outcome)
  3. Splits text into overlapping chunks for embedding
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class JudgmentMetadata:
    """All structured metadata extracted from one judgment file."""
    file_path: str
    case_number: str     = ""
    case_name: str       = ""
    date: str            = ""
    petitioner: str      = ""
    respondent: str      = ""
    bench: List[str]     = field(default_factory=list)
    outcome: str         = ""
    raw_text: str        = ""


@dataclass
class TextChunk:
    """One sliding-window chunk from a judgment, ready for embedding."""
    chunk_id: str
    case_number: str
    case_name: str
    text: str
    chunk_index: int
    total_chunks: int
    date: str            = ""
    petitioner: str      = ""
    respondent: str      = ""
    file_path: str       = ""


# ─── Regex Patterns ───────────────────────────────────────────────────────────

_CASE_NO_RE = re.compile(
    r"(Civil Appeal|Criminal Appeal|Writ Petition|SLP|Special Leave Petition|"
    r"Transfer Petition|Original Suit|Review Petition|Curative Petition)"
    r"[\s\(]*(?:No\.?|Nos\.?)?[\s]*(\d[\d\-\/]*)\s+(?:of|OF)\s+(\d{4})",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"(?:decided on|date of judgment|judgment dated|order dated)[:\s]+"
    r"(\d{1,2}[\s\-\.\/]+"
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|\d{1,2})[\s\-\.\/]+\d{4})",
    re.IGNORECASE,
)
_BENCH_RE = re.compile(
    r"(?:CORAM|BENCH|Before)[:\s]+([A-Z][A-Za-z\.\s,]+?(?:J\.|JJ\.|CJI|C\.J\.))",
    re.IGNORECASE,
)
_VS_RE = re.compile(r"\s+(?:Vs?\.?|VERSUS)\s+", re.IGNORECASE)
_OUTCOME_RE = re.compile(
    r"\b(appeal\s+(?:is\s+)?allowed|appeal\s+(?:is\s+)?dismissed|"
    r"petition\s+(?:is\s+)?allowed|petition\s+(?:is\s+)?dismissed|"
    r"partly\s+allowed|remanded|writ\s+(?:is\s+)?issued|modified)\b",
    re.IGNORECASE,
)


# ─── Parser ───────────────────────────────────────────────────────────────────

class JudgmentTXTParser:
    """
    Parses a single .txt judgment file.

    Usage:
        parser = JudgmentTXTParser()
        metadata, chunks = parser.parse("/path/to/judgment.txt")
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public ────────────────────────────────────────────────────────────────

    def parse(self, file_path: str):
        """Return (JudgmentMetadata, List[TextChunk])."""
        raw_text = self._read(file_path)
        metadata = self._extract_metadata(file_path, raw_text)
        chunks   = self._chunk(metadata)
        return metadata, chunks

    # ── Read ──────────────────────────────────────────────────────────────────

    def _read(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _extract_metadata(self, file_path: str, text: str) -> JudgmentMetadata:
        m = JudgmentMetadata(file_path=file_path, raw_text=text)

        # Case number
        hit = _CASE_NO_RE.search(text)
        if hit:
            m.case_number = hit.group(0).strip()

        # Date
        hit = _DATE_RE.search(text)
        if hit:
            m.date = hit.group(1).strip()

        # Bench / CORAM
        hit = _BENCH_RE.search(text)
        if hit:
            bench_str = hit.group(1).strip()
            m.bench = [
                j.strip()
                for j in re.split(r",\s*|\band\b", bench_str, flags=re.I)
                if j.strip()
            ]

        # Case name — look for "Party A  v/vs  Party B" in first 40 lines
        first_lines = [ln.strip() for ln in text.split("\n")[:40] if ln.strip()]
        for line in first_lines:
            if _VS_RE.search(line) and 5 < len(line) < 300:
                m.case_name = line
                parts = _VS_RE.split(line, maxsplit=1)
                if len(parts) == 2:
                    m.petitioner = parts[0].strip()
                    m.respondent = parts[1].strip()
                break

        # Fallback — use filename as case name
        if not m.case_name:
            m.case_name = Path(file_path).stem.replace("_", " ").replace("-", " ")

        # Fallback — use case name as case number if missing
        if not m.case_number:
            m.case_number = m.case_name

        # Outcome
        hit = _OUTCOME_RE.search(text[-3000:])   # search last 3000 chars
        if hit:
            m.outcome = hit.group(0).strip().title()

        return m

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _chunk(self, meta: JudgmentMetadata) -> List[TextChunk]:
        """Sliding-window word-level chunking with overlap."""
        words  = meta.raw_text.split()
        step   = max(1, self.chunk_size - self.chunk_overlap)
        starts = list(range(0, len(words), step))
        total  = len(starts)
        chunks = []

        for idx, start in enumerate(starts):
            end        = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            if not chunk_text:
                continue

            chunks.append(TextChunk(
                chunk_id    = f"{_safe_id(meta.case_number)}_c{idx}",
                case_number = meta.case_number,
                case_name   = meta.case_name,
                text        = chunk_text,
                chunk_index = idx,
                total_chunks= total,
                date        = meta.date,
                petitioner  = meta.petitioner,
                respondent  = meta.respondent,
                file_path   = meta.file_path,
            ))
            if end == len(words):
                break

        return chunks


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _safe_id(s: str) -> str:
    """Strip characters not allowed in ChromaDB IDs."""
    return re.sub(r"[^\w\-]", "_", s)[:80]
