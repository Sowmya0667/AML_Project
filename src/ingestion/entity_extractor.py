"""
src/ingestion/entity_extractor.py
────────────────────────────────────────────────────────────────────────────
Uses an LLM to extract structured legal entities from judgment text:
  - Case citations       (cases referred to inside this judgment)
  - Statutes / Sections  (Acts, Articles, Sections cited)
  - Legal concepts       (doctrines, principles)
  - Key holdings         (ratio decidendi)
  - Judges               (bench members)
  - Subject matter       (area of law)
  - Outcome              (Allowed / Dismissed / etc.)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List

from config import config


# ─── Model ───────────────────────────────────────────────────────────────────

@dataclass
class ExtractedEntities:
    case_number: str
    citations: List[str]      = field(default_factory=list)
    statutes: List[str]       = field(default_factory=list)
    legal_concepts: List[str] = field(default_factory=list)
    holdings: List[str]       = field(default_factory=list)
    judges: List[str]         = field(default_factory=list)
    subject_matter: str       = ""
    outcome: str              = ""


# ─── Prompts ─────────────────────────────────────────────────────────────────

_SYS = (
    "You are a legal information extractor for Indian Supreme Court judgments. "
    "Return ONLY valid JSON — no markdown fences, no extra text."
)

_USER = """Extract structured entities from this Indian Supreme Court judgment excerpt.

TEXT:
{text}

Return exactly this JSON structure (empty list/string if not found):
{{
  "citations":      ["other case names or numbers cited in this text"],
  "statutes":       ["Acts, Sections, Articles mentioned e.g. 'Section 302 IPC', 'Article 21 Constitution of India'"],
  "legal_concepts": ["legal doctrines or principles e.g. 'res judicata', 'natural justice', 'due process'"],
  "holdings":       ["1-2 sentence summary of what was decided"],
  "judges":         ["names of judges on the bench"],
  "subject_matter": "primary area of law e.g. 'Criminal Law' or 'Constitutional Law'",
  "outcome":        "one of: Allowed / Dismissed / Partly Allowed / Remanded / Modified / Not determined"
}}"""


# ─── Extractor ───────────────────────────────────────────────────────────────

class LegalEntityExtractor:
    """
    Calls the configured LLM to extract legal entities.
    Provider is chosen via LLM_PROVIDER in .env.
    """

    def __init__(self):
        self._client = _build_client()

    def extract(self, text: str, case_number: str, max_chars: int = 3500) -> ExtractedEntities:
        """Extract entities from up to max_chars of text (to control API cost)."""
        prompt = _USER.format(text=text[:max_chars])
        raw    = self._call(prompt)
        return _parse(raw, case_number)

    def _call(self, prompt: str) -> str:
        provider = config.LLM_PROVIDER

        if provider == "anthropic":
            r = self._client.messages.create(
                model=config.LLM_MODEL, max_tokens=800,
                system=_SYS,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.content[0].text

        if provider in ("openai", "groq", "ollama"):
            r = self._client.chat.completions.create(
                model=config.LLM_MODEL, max_tokens=800, temperature=0,
                messages=[
                    {"role": "system", "content": _SYS},
                    {"role": "user",   "content": prompt},
                ],
            )
            return r.choices[0].message.content

        return "{}"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_client():
    p = config.LLM_PROVIDER
    if p == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    if p == "openai":
        from openai import OpenAI
        return OpenAI(api_key=config.OPENAI_API_KEY)
    if p == "ollama":
        from openai import OpenAI
        return OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
    if p == "groq":
        from groq import Groq
        return Groq(api_key=config.GROQ_API_KEY)
    raise ValueError(f"Unknown LLM_PROVIDER: {p}")


def _parse(raw: str, case_number: str) -> ExtractedEntities:
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        d = {}
    return ExtractedEntities(
        case_number   = case_number,
        citations     = d.get("citations",      []),
        statutes      = d.get("statutes",        []),
        legal_concepts= d.get("legal_concepts",  []),
        holdings      = d.get("holdings",        []),
        judges        = d.get("judges",          []),
        subject_matter= d.get("subject_matter",  ""),
        outcome       = d.get("outcome",         ""),
    )
