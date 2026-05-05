"""
config.py
Central configuration loaded from .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Neo4j ─────────────────────────────────────────────
    NEO4J_URI: str       = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    NEO4J_USERNAME: str  = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str  = os.getenv("NEO4J_PASSWORD", "password")

    # ── LLM ───────────────────────────────────────────────
    LLM_PROVIDER: str    = os.getenv("LLM_PROVIDER",   "groq")
    LLM_MODEL: str       = os.getenv("LLM_MODEL",      "llama-3.1-8b-instant")
    GROQ_API_KEY: str    = os.getenv("GROQ_API_KEY",   "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str  = os.getenv("OPENAI_API_KEY", "")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    # ── Embeddings ────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Data Paths ────────────────────────────────────────
    JUDGMENTS_DATA_DIR: str  = os.getenv("JUDGMENTS_DATA_DIR", "./data/judgments")
    CHROMA_PERSIST_DIR: str  = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION: str   = os.getenv("CHROMA_COLLECTION",  "legal_judgments")

    # ── Chunking ──────────────────────────────────────────
    CHUNK_SIZE: int    = int(os.getenv("CHUNK_SIZE",    "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # ── Retrieval ─────────────────────────────────────────
    GRAPH_TOP_K: int  = int(os.getenv("GRAPH_TOP_K",  "5"))
    VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", "5"))
    FINAL_TOP_K: int  = int(os.getenv("FINAL_TOP_K",  "6"))


config = Config()
