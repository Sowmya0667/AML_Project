# ⚖️ Vidhi — The Justice Engine

> **A hybrid Legal GraphRAG system combining a Neo4j Knowledge Graph and ChromaDB semantic search to enable intelligent, multimodal legal research over Indian Supreme Court judgments.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Neo4j](https://img.shields.io/badge/Neo4j-Knowledge%20Graph-green?style=flat-square&logo=neo4j)](https://neo4j.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=flat-square)](https://www.trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-LLM-purple?style=flat-square)](https://groq.com)

---

> *"Constitution is not a mere lawyers document, it is a vehicle of Life, and its spirit is always the spirit of Age."*
> — **Dr. B.R. Ambedkar**

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Knowledge Graph Schema](#-knowledge-graph-schema)
5. [Project Structure](#-project-structure)
6. [Tech Stack](#-tech-stack)
7. [Requirements](#-requirements)
8. [Setup & Installation](#-setup--installation)
9. [Data — Kaggle Dataset](#-data--kaggle-dataset)
10. [Running the System](#-running-the-system)
11. [How It Works](#-how-it-works)
12. [API Reference](#-api-reference)
13. [Evaluation (RAGAS)](#-evaluation-ragas)
14. [Example Queries](#-example-queries)

---

## 🏛️ Project Overview

Legal research over Indian Supreme Court judgments is challenging because legal knowledge is both **relational** (cases cite each other, judges interpret statutes, doctrines evolve over time) and **semantic** (meaning matters as much as keywords).

**Vidhi** solves this by combining two complementary retrieval strategies into one unified pipeline:

| Strategy | Technology | Strength |
|---|---|---|
| **Graph Traversal** | Neo4j (Cypher) | Explicit relationships: citations, statutes, judges, doctrines |
| **Semantic Search** | ChromaDB + Sentence-Transformers | Meaning-based similarity across judgment text |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) | Best of both worlds in a single re-ranked result set |

The fused top-K results are passed to an LLM (Groq / Anthropic / OpenAI) which synthesises a grounded, cited legal research answer — streamed live in the chat UI.

---

## ✨ Key Features

- 🔍 **Hybrid GraphRAG** — Neo4j graph traversal + ChromaDB vector search fused via RRF
- 🧠 **Intent-Aware Retrieval** — automatically detects if query is about a statute, judge, legal concept, or general topic
- 💬 **Streaming Chat UI** — token-by-token streamed answers via Streamlit
- 🎙️ **Voice Input** — speak your legal query via Groq Whisper (falls back to Google STT)
- 👁️ **Vision / Multimodal** — attach a legal document image; analysed by Groq's vision LLM
- 📚 **Source Transparency** — every answer shows exactly which cases were retrieved and their relevance scores
- 🔗 **Citation Network** — cases are linked by `CITES` edges so the system understands legal precedent chains
- 📊 **RAGAS Evaluation** — Faithfulness, Answer Relevancy, Context Recall, Context Precision measured on 40 test questions
- ⚙️ **Multi-LLM Support** — Groq, OpenAI, Anthropic Claude, or local Ollama models

---

## 🗺️ System Architecture

### Full Pipeline

```
╔══════════════════════════════════════════════════════════════════╗
║                  OFFLINE — Ingestion (run once)                  ║
║                                                                  ║
║   .txt judgment files                                            ║
║        │                                                         ║
║        ▼  txt_parser.py  (regex metadata + sliding-window)       ║
║   JudgmentMetadata + TextChunks (800 words, 150 overlap)         ║
║        │                                                         ║
║        ├──▶ entity_extractor.py (LLM → JSON)                     ║
║        │         citations, statutes, concepts, holdings          ║
║        │              ▼  graph_builder.py                         ║
║        │         Neo4j Knowledge Graph                           ║
║        │                                                         ║
║        └──▶ embeddings.py (all-MiniLM-L6-v2, local, 384-dim)    ║
║                   ▼  vector_store.py                              ║
║             ChromaDB (cosine similarity, persisted on disk)      ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║                    ONLINE — Per Query                            ║
║                                                                  ║
║   User Input (text / voice / image)                              ║
║        │                                                         ║
║        ▼  hybrid_retriever.py                                    ║
║   ┌────────────────┐     ┌──────────────────┐                    ║
║   │  Graph Leg     │     │   Vector Leg      │                    ║
║   │  (Neo4j)       │     │   (ChromaDB)      │                    ║
║   │                │     │                   │                    ║
║   │ • by_statute() │     │ embed query →     │                    ║
║   │ • by_judge()   │     │ cosine search     │                    ║
║   │ • by_concept() │     │                   │                    ║
║   │ • fulltext()   │     │                   │                    ║
║   └───────┬────────┘     └─────────┬─────────┘                    ║
║           └───────────┬────────────┘                              ║
║                       ▼  RRF Fusion  (score = Σ 1/(60+rank))      ║
║                Top-K RetrievedChunks                             ║
║                       │                                           ║
║                       ▼  llm_chain.py                             ║
║               LLM Answer (streamed tokens)                       ║
║                       │                                           ║
║                       ▼  app.py (Streamlit)                       ║
║            Chat response + Source expander                       ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 🕸️ Knowledge Graph Schema

### Nodes

| Label | Key Properties |
|---|---|
| `:Case` | `case_number` (unique), `case_name`, `date`, `petitioner`, `respondent`, `subject_matter`, `outcome`, `holdings`, `file_path` |
| `:Judge` | `name` (unique) |
| `:Statute` | `name` (unique) — e.g. `"Section 302 IPC"`, `"Article 21 Constitution of India"` |
| `:LegalConcept` | `name` (unique) — e.g. `"res judicata"`, `"natural justice"`, `"habeas corpus"` |

### Relationships

| Relationship | Direction | Meaning |
|---|---|---|
| `DECIDED_BY` | Case → Judge | Judge sat on the bench for this case |
| `APPLIES` | Case → Statute | Case invokes or interprets this statute/article |
| `INVOLVES` | Case → LegalConcept | Case discusses this legal doctrine or principle |
| `CITES` | Case → Case | This case cites another case as precedent |

### Example Cypher Queries

```cypher
-- Find all cases decided by Justice Chandrachud applying Article 21
MATCH (c:Case)-[:DECIDED_BY]->(j:Judge),
      (c)-[:APPLIES]->(s:Statute)
WHERE j.name CONTAINS "Chandrachud"
  AND s.name CONTAINS "Article 21"
RETURN c.case_name, c.date, c.outcome
ORDER BY c.date DESC
LIMIT 10

-- Traverse citation network 2 hops from a landmark case
MATCH (c:Case {case_number: "Writ Petition No. 231 of 1978"})
      -[:CITES*1..2]->(cited:Case)
RETURN c.case_name, cited.case_name, cited.date

-- Find cases related via shared statutes or concepts (1-hop neighbourhood)
MATCH (c:Case {case_number: $cn})-[*1..2]-(related:Case)
WHERE related.case_number <> $cn
RETURN DISTINCT related.case_name, related.outcome
LIMIT 10
```

---

## 📁 Project Structure

```
AML_Project-main/
│
├── README.md                           ← This file
├── requirements.txt                    ← All Python dependencies
├── config.py                           ← Central settings (loaded from .env)
├── .env                                ← Your API keys (never committed)
├── .gitignore                          ← Excludes .env, chroma_db, .venv, __pycache__
│
├── app.py                              ← Streamlit web UI (4 pages + chat)
├── api.py                              ← FastAPI REST backend
├── main.py                             ← Minimal entry point
├── smoke_test.py                       ← End-to-end sanity checks
│
├── src/
│   ├── ingestion/
│   │   ├── data_loader.py              ← Orchestrates full ingestion pipeline
│   │   ├── txt_parser.py               ← Regex parsing + sliding-window chunking
│   │   └── entity_extractor.py         ← LLM-based entity extraction → JSON
│   │
│   ├── graph/
│   │   ├── neo4j_client.py             ← Singleton Neo4j driver + schema setup
│   │   ├── graph_builder.py            ← MERGE Case/Judge/Statute/Concept nodes
│   │   └── graph_queries.py            ← All Cypher retrieval queries
│   │
│   ├── vector/
│   │   ├── embeddings.py               ← sentence-transformers wrapper (local)
│   │   └── vector_store.py             ← ChromaDB index + query + stats
│   │
│   ├── retrieval/
│   │   └── hybrid_retriever.py         ← Intent detection + RRF fusion
│   │
│   ├── generation/
│   │   ├── llm_chain.py                ← RAG prompt + LLM call (stream + batch)
│   │   └── stt.py                      ← Speech-to-text (Groq Whisper + Google fallback)
│   │
│   └── evaluation/
│       ├── generate_eval_data.py       ← Generate Q&A pairs from judgments
│       ├── run_ragas_eval.py           ← RAGAS evaluation runner (40 questions)
│       ├── retry_failed_evals.py       ← Retry NaN entries with extra API keys
│       ├── cleanup.py                  ← Clean and filter raw eval results
│       ├── eval_queries_final_2020.csv ← Test question set
│       └── test_metrics_40_samples.csv ← Final RAGAS metric scores
│
└── data/
    └── judgements/                     ← Raw .txt judgment files (200+ cases)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Knowledge Graph** | [Neo4j AuraDB](https://neo4j.com/cloud/aura/) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) (local, persisted) |
| **Embeddings** | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — 384-dim, runs locally |
| **Primary LLM** | [Groq](https://groq.com/) — `llama-3.1-8b-instant` |
| **Vision LLM** | [Groq](https://groq.com/) — `llama-3.2-11b-vision-preview` |
| **STT** | [Groq Whisper](https://groq.com/) (`whisper-large-v3`) + Google STT fallback |
| **Alt LLMs** | Anthropic Claude, OpenAI GPT, Ollama (local) |
| **Web UI** | [Streamlit](https://streamlit.io/) |
| **REST API** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| **Evaluation** | [RAGAS](https://docs.ragas.io/) + OpenRouter + HuggingFace Embeddings |
| **Dataset** | [Kaggle: vxrunsonii/supreme-court-judgments-txt](https://www.kaggle.com/datasets/vxrunsonii/supreme-court-judgments-txt) |
| **Language** | Python 3.10+ |

---

## 📦 Requirements

### Python Version

Python **3.10 or later** is required.

### External Services

| Service | Purpose | Cost |
|---|---|---|
| **Neo4j AuraDB** | Knowledge graph database | Free tier available |
| **Groq** | LLM inference + Whisper STT + Vision | Free tier available |
| Anthropic Claude | Alternative LLM | Paid |
| OpenAI GPT | Alternative LLM / Evaluation judge | Paid |
| OpenRouter | LLM routing for RAGAS evaluation | Free credits available |

> ✅ Sentence-transformer embeddings run **entirely locally** — no API key required.

---

## ⚙️ Setup & Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/Sowmya0667/AML_Project.git
cd AML_Project
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> First run will automatically download the `all-MiniLM-L6-v2` model (~22 MB) from HuggingFace.

### Step 4 — Set up Neo4j AuraDB (Free)

1. Go to [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/) and sign up
2. Click **New Instance → AuraDB Free**
3. Save the **Connection URI**, **Username**, and **Password** shown after creation
4. Wait ~2 minutes for the instance to become active

### Step 5 — Get a Groq API Key (Free)

1. Go to [console.groq.com](https://console.groq.com) and sign up
2. Navigate to **API Keys** → **Create API Key**
3. Copy the key starting with `gsk_...`

### Step 6 — Configure `.env`

Create a `.env` file in the project root:

```env
# Neo4j (required)
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_aura_password

# LLM — choose one provider
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx

# Optional alternative LLMs
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4o-mini
# OPENAI_API_KEY=sk-...

# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-3-haiku-20240307
# ANTHROPIC_API_KEY=sk-ant-...

# Embeddings (default, runs locally — no key needed)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Paths (defaults are fine)
JUDGMENTS_DATA_DIR=./data/judgements
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION=legal_judgments

# Chunking
CHUNK_SIZE=800
CHUNK_OVERLAP=150

# Retrieval
GRAPH_TOP_K=5
VECTOR_TOP_K=5
FINAL_TOP_K=6

# For RAGAS evaluation (optional)
OPENROUTER_API_KEY_1=sk-or-v1-...
OPENROUTER_API_KEY_2=sk-or-v1-...
```

### Step 7 — Verify Setup

```bash
python smoke_test.py
```

All tests should show ✅. A ⚠️ means a service is not yet configured (non-fatal).

---

## 📂 Data — Kaggle Dataset

This project uses:
**[LEGAL-Text-Supreme Court Judgments (India)](https://www.kaggle.com/datasets/vxrunsonii/supreme-court-judgments-txt)**

Plain-text `.txt` files of Indian Supreme Court judgments — no PDF parsing needed.

### Download via Kaggle CLI (fastest)

```bash
pip install kaggle

# Place your kaggle.json at ~/.kaggle/kaggle.json
# Get it: kaggle.com → Account → Settings → API → Create New Token

kaggle datasets download -d vxrunsonii/supreme-court-judgments-txt
unzip supreme-court-judgments-txt.zip -d ./data/judgements/
```

### Download via Browser

1. Visit the dataset page and click **Download**
2. Extract the `.zip`
3. Move all `.txt` files into `./data/judgements/`

---

## 🚀 Running the System

### 1 — Ingest Judgment Files (Offline, one-time)

```bash
python -m src.ingestion.data_loader --input ./data/judgements/
```

This will:
- Parse every `.txt` file (regex metadata extraction)
- Call the LLM to extract citations, statutes, legal concepts, judges, holdings
- Build the **Neo4j knowledge graph** (Case, Judge, Statute, LegalConcept nodes + relationships)
- **Embed** all text chunks and store them in **ChromaDB**

Progress is shown with a `tqdm` progress bar. Failed files are logged but don't crash the run.

### 2 — Launch the Streamlit Chat UI

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

**UI Pages:**
| Page | Description |
|---|---|
| 🏠 Home | Landing page with project introduction |
| ℹ️ About Vidhi | Explains the 4 pillars: Graph, Semantic, Hybrid Fusion, Vision |
| 📖 How to Use | Step-by-step usage guide |
| 💬 The Chatbot | Main chat interface with streaming answers |

**Chatbot Features:**
- 💬 Type a legal research query in the chat box
- 🎙️ Click the **microphone** (bottom right) to speak your query via voice
- 📎 Click the **paperclip** to attach an image of a legal document for visual analysis
- 📊 Click **Refresh** in the sidebar to see live graph stats (Cases, Judges, Statutes, Concepts, Chunks)
- ⚙️ Adjust **Graph Top-K**, **Vector Top-K**, **Final Top-K** sliders in the sidebar
- 📚 Expand **View Retrieved Sources** to see which cases were retrieved and their relevance scores

### 3 — Launch the REST API (Optional)

```bash
uvicorn api:app --reload --port 8000
```

Interactive API docs: **http://localhost:8000/docs**

---

## 🔬 How It Works

### Stage 1 — Parsing (`txt_parser.py`)

Each `.txt` file is parsed with **regex patterns** to extract:

| Field | Strategy |
|---|---|
| `case_number` | Matches "Civil Appeal No. 1234 of 2020", "SLP", "Writ Petition", etc. |
| `date` | Matches "Decided on 15 January 2020", "Judgment dated..." |
| `bench` | Matches "CORAM:", "Before J." patterns |
| `case_name` | Scans first 40 lines for "Party A vs Party B" |
| `outcome` | Scans last 3000 characters for "appeal allowed/dismissed", "partly allowed", etc. |

The text is then split into **overlapping 800-word chunks** (150-word overlap, step = 650):

```
Window 1: words[0:800]
Window 2: words[650:1450]
Window 3: words[1300:2100]
...
```

Overlap ensures context spanning a chunk boundary is never lost.

---

### Stage 2 — Entity Extraction (`entity_extractor.py`)

The LLM is prompted on the first 3,500 characters of each judgment to return structured JSON:

```json
{
  "citations":      ["Hussainara Khatoon v. State of Bihar (1979)"],
  "statutes":       ["Article 21 Constitution of India", "Section 437 CrPC"],
  "legal_concepts": ["right to speedy trial", "natural justice"],
  "holdings":       ["Detention beyond reasonable time violates Article 21."],
  "judges":         ["Justice P.N. Bhagwati"],
  "subject_matter": "Constitutional Law",
  "outcome":        "Allowed"
}
```

This populates the Neo4j graph. All writes use `MERGE` (upsert) — safe to re-run.

---

### Stage 3 — Hybrid Retrieval (`hybrid_retriever.py`)

**Intent Detection** (regex-based):

| Pattern | Graph Strategy |
|---|---|
| `Section 302`, `Article 21`, `IPC`, `CrPC`, `Act 2024` | `by_statute()` |
| `Justice Chandrachud`, `Justice Bhat` | `by_judge()` |
| `res judicata`, `natural justice`, `habeas corpus` | `by_concept()` |
| Any query (always runs) | `fulltext()` Neo4j index + ChromaDB semantic search |

**Reciprocal Rank Fusion (RRF):**

```
RRF score(item) = Σ  1 / (60 + rank_in_list)
                 for each retrieval list the item appears in
```

Items ranking highly in **both** graph and vector results get the highest combined scores. Deduplication is done by `hash(chunk.text)` — not `case_number` — so multiple relevant paragraphs from the same case can all appear.

---

### Stage 4 — LLM Generation (`llm_chain.py`)

The formatted context + user query are injected into the RAG prompt template and sent to the LLM with a system prompt instructing it to:

- Answer only from the provided context
- Cite specific case numbers and names  
- Distinguish *ratio decidendi* from *obiter dicta*
- Reject off-topic queries with a guardrail response
- Support **streaming** (token-by-token) and **batch** modes

**Vision routing:** If an image is attached, the call is automatically routed to Groq's `llama-3.2-11b-vision-preview` model with the image as a base64-encoded `image_url`.

---

### Stage 5 — Voice Input (`stt.py`)

Speech is captured via the `audio_recorder_streamlit` widget and transcribed in two tiers:

1. **Groq Whisper API** (`whisper-large-v3`) — primary, fast and accurate
2. **Google Web Speech API** — free fallback via `SpeechRecognition` library

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Full RAG pipeline: retrieve + generate |
| `POST` | `/query/stream` | Streaming SSE response (token-by-token) |
| `POST` | `/retrieve` | Return retrieved chunks without LLM generation |
| `POST` | `/ingest/text` | Ingest a judgment provided as raw text (async background task) |
| `GET` | `/graph/stats` | Count of Cases, Judges, Statutes, Concepts in Neo4j |
| `GET` | `/graph/case/{case_number}` | Case detail: cites, cited_by, related cases |
| `POST` | `/graph/search` | Targeted graph search (statute/judge/concept/fulltext) |
| `GET` | `/vector/stats` | ChromaDB chunk count + list of indexed cases |

### Example: POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the law on right to speedy trial under Article 21?",
    "graph_top_k": 5,
    "vector_top_k": 5,
    "final_top_k": 6
  }'
```

**Response:**
```json
{
  "query": "What is the law on right to speedy trial under Article 21?",
  "answer": "The right to speedy trial is an implicit fundamental right under Article 21...",
  "sources": [
    {
      "case_number": "Writ Petition No. 57 of 1979",
      "case_name": "Hussainara Khatoon vs State of Bihar",
      "date": "9 March 1979",
      "source": "hybrid",
      "score": 0.923
    }
  ],
  "graph_hits": 3,
  "vector_hits": 5,
  "model_used": "llama-3.1-8b-instant"
}
```

---

## 📊 Evaluation (RAGAS)

The system is evaluated using the **RAGAS** framework on 40 hand-crafted test questions about 2020 Indian Supreme Court judgments.

### Metrics

| Metric | What It Measures |
|---|---|
| **Faithfulness** | Is the answer factually grounded in retrieved context? (no hallucination) |
| **Answer Relevancy** | Does the answer actually address the question? |
| **Context Recall** | Did retrieval find all relevant information vs. ground truth? |
| **Context Precision** | Are retrieved chunks actually relevant? (signal-to-noise ratio) |

### Running the Evaluation

```bash
# Generate evaluation Q&A pairs from judgments
python src/evaluation/generate_eval_data.py

# Run RAGAS evaluation (uses OpenRouter API keys from .env)
python src/evaluation/run_ragas_eval.py

# Retry any NaN entries with additional API keys
python src/evaluation/retry_failed_evals.py
```

Results are saved to `src/evaluation/test_metrics_40_samples.csv`.

---

## 💬 Example Queries

```
# Constitutional Law
"What are the landmark judgments on Article 21 right to life and personal liberty?"
"How has the Supreme Court interpreted the right to privacy?"
"What is the basic structure doctrine from Kesavananda Bharati?"

# Criminal Law
"What is the test for awarding death penalty under Section 302 IPC?"
"Cases on bail under Section 437 and 438 CrPC"
"What is anticipatory bail and when can it be refused?"

# Legal Doctrines
"Explain the doctrine of res judicata as applied by the Supreme Court"
"Cases on natural justice and the audi alteram partem principle"
"What is promissory estoppel in Indian contract law?"

# Judge-Specific
"Important judgments delivered by Justice D.Y. Chandrachud"
"Cases decided by Justice Indu Malhotra on gender equality"

# Citation Network
"Which cases have cited Maneka Gandhi v. Union of India?"
"Cases related to Hussainara Khatoon on the right to speedy trial"

# Voice / Vision Input
(click the 🎙️ microphone and speak)
(click the 📎 paperclip and attach an image of a legal notice)
```

---

## 📄 License

This project is built for **educational and research purposes** as part of an Academic Machine Learning project.

The Kaggle dataset ([vxrunsonii/supreme-court-judgments-txt](https://www.kaggle.com/datasets/vxrunsonii/supreme-court-judgments-txt)) is subject to its own license — please review before any commercial use.

---

<p align="center">
  Built with ❤️ for Indian Legal Research
</p>
