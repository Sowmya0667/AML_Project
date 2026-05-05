"""
src/generation/llm_chain.py
────────────────────────────────────────────────────────────────────────────
RAG answer generation chain.

  query
    │
    ▼  HybridRetriever
  RetrievalResult  (graph + vector chunks)
    │
    ▼  _format_context()
  formatted context string
    │
    ▼  LLM (Anthropic / OpenAI / Groq)
  LegalResearchResponse
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator, List

from config import config
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult

log = logging.getLogger(__name__)


# ─── Prompts ─────────────────────────────────────────────────────────────────

_SYSTEM = """You are an expert Indian Supreme Court legal research assistant with
deep knowledge of constitutional law, criminal law, civil procedure, and all major
areas of Indian jurisprudence.

Rules:
- GUARDRAIL: If the user asks questions outside the legal/judgment context (e.g., asking for passwords, security information, general knowledge, or unrelated topics), you MUST reject the query and respond EXACTLY with: "This is outside my context. Please ask me a relevant legal question."
- GUARDRAIL: If the user simply greets you (e.g., "hi", "hello", "good morning"), you MUST respond EXACTLY with: "Hi. I am your legal Assistant . tell me how may i help you"
- Base answers strictly on the provided context.
- Cite specific case numbers and names.
- Distinguish ratio decidendi (binding) from obiter dicta.
- Note the date of judgments when relevant.
- If the context is insufficient, clearly say so rather than guessing.
- Use precise legal terminology appropriate for legal professionals."""

_RAG_TEMPLATE = """\
RETRIEVED JUDGMENTS
═══════════════════
{context}

LEGAL RESEARCH QUERY
════════════════════
{query}

INSTRUCTIONS
════════════
1. Synthesise the key holdings from the retrieved cases.
2. Identify applicable legal principles.
3. Cite case names and numbers.
4. Note any conflicting judgments or evolution of legal position.
5. Give a clear, structured legal analysis.

Answer:"""


# ─── Response ────────────────────────────────────────────────────────────────

@dataclass
class LegalResearchResponse:
    query:       str
    answer:      str
    sources:     List[dict] = field(default_factory=list)
    graph_hits:  int = 0
    vector_hits: int = 0
    model_used:  str = ""


# ─── Chain ───────────────────────────────────────────────────────────────────

class LegalRAGChain:

    def __init__(self):
        self.retriever = HybridRetriever()
        self._llm      = _build_llm()

    # ── Public ────────────────────────────────────────────────────────────────

    def query(self, user_query: str, history: List[dict] = None, image: str = None) -> LegalResearchResponse:
        import string
        clean_q = user_query.strip().lower().translate(str.maketrans('', '', string.punctuation))
        if clean_q in {"hi", "hello", "hey", "good morning", "good evening", "good afternoon", "greetings"}:
            self.last_result = RetrievalResult(chunks=[], graph_hits=0, vector_hits=0)
            return LegalResearchResponse(
                query=user_query,
                answer="Hi. I am your legal Assistant . tell me how may i help you",
                sources=[],
                graph_hits=0,
                vector_hits=0,
                model_used=config.LLM_MODEL
            )

        result  = self.retriever.retrieve(user_query)
        self.last_result = result
        context = _format_context(result)
        prompt  = _RAG_TEMPLATE.format(context=context, query=user_query)
        
        answer  = self._generate(prompt, history, image=image)

        return LegalResearchResponse(
            query       = user_query,
            answer      = answer,
            sources     = [
                {
                    "case_number": c.case_number,
                    "case_name":   c.case_name,
                    "date":        c.date,
                    "source":      c.source,
                    "score":       round(c.score, 3),
                }
                for c in result.chunks
            ],
            graph_hits  = result.graph_hits,
            vector_hits = result.vector_hits,
            model_used  = config.LLM_MODEL,
        )

    def stream_query(self, user_query: str, history: List[dict] = None, image: str = None) -> Generator[str, None, None]:
        import string
        clean_q = user_query.strip().lower().translate(str.maketrans('', '', string.punctuation))
        if clean_q in {"hi", "hello", "hey", "good morning", "good evening", "good afternoon", "greetings"}:
            self.last_result = RetrievalResult(chunks=[], graph_hits=0, vector_hits=0)
            yield "Hi. I am your legal Assistant . tell me how may i help you"
            return

        result  = self.retriever.retrieve(user_query)
        self.last_result = result
        context = _format_context(result)
        prompt  = _RAG_TEMPLATE.format(context=context, query=user_query)
        yield from self._stream(prompt, history, image=image)

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def _generate(self, prompt: str, history: List[dict] = None, image: str = None) -> str:
        if history is None: history = []
        p = config.LLM_PROVIDER
        model = config.LLM_MODEL
        llm_client = self._llm

        if image:
            # Route vision request to Groq explicitly
            p = "groq"
            model = "llama-3.2-11b-vision-preview"
            from groq import Groq
            llm_client = Groq(api_key=config.GROQ_API_KEY)
            
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
            ]
        else:
            user_content = prompt

        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
        messages.append({"role": "user", "content": user_content})

        if p == "anthropic":
            r = llm_client.messages.create(
                model=model, max_tokens=2048,
                system=_SYSTEM,
                messages=messages,
            )
            return r.content[0].text

        if p in ("openai", "groq", "ollama"):
            full_messages = [{"role": "system", "content": _SYSTEM}] + messages
            r = llm_client.chat.completions.create(
                model=model, max_tokens=2048, temperature=0.1,
                messages=full_messages,
            )
            return r.choices[0].message.content

        return "Error: LLM provider not configured."

    def _stream(self, prompt: str, history: List[dict] = None, image: str = None) -> Generator[str, None, None]:
        if history is None: history = []
        p = config.LLM_PROVIDER
        model = config.LLM_MODEL
        llm_client = self._llm

        if image:
            # Route vision request to Groq explicitly
            p = "groq"
            model = "llama-3.2-11b-vision-preview"
            from groq import Groq
            llm_client = Groq(api_key=config.GROQ_API_KEY)
            
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
            ]
        else:
            user_content = prompt

        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
        messages.append({"role": "user", "content": user_content})

        if p == "anthropic":
            with llm_client.messages.stream(
                model=model, max_tokens=2048,
                system=_SYSTEM,
                messages=messages,
            ) as s:
                yield from s.text_stream

        elif p in ("openai", "groq", "ollama"):
            full_messages = [{"role": "system", "content": _SYSTEM}] + messages
            stream = llm_client.chat.completions.create(
                model=model, max_tokens=2048, stream=True,
                messages=full_messages,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_llm():
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


def _format_context(result: RetrievalResult) -> str:
    if not result.chunks:
        return "No relevant judgments found in the knowledge base."
    parts = []
    for i, c in enumerate(result.chunks, 1):
        parts.append(
            f"[{i}] {c.case_name} ({c.case_number})\n"
            f"    Date: {c.date}  |  Source: {c.source.upper()}\n"
            f"    {c.text[:700]}"
        )
    return "\n\n".join(parts)
