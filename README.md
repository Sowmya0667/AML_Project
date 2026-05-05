# ⚖️ Legal GraphRAG

> **A hybrid GraphRAG system that combines Neo4j knowledge graph traversal with
> semantic vector search to enable intelligent legal research over Indian Supreme
> Court judgments.**

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Knowledge Graph Schema](#knowledge-graph-schema)
4. [Project Structure](#project-structure)
5. [Requirements](#requirements)
6. [Setup & Installation](#setup--installation)
7. [Data — Kaggle Dataset](#data--kaggle-dataset)
8. [Running the System](#running-the-system)
9. [How It Works](#how-it-works)
10. [API Reference](#api-reference)
11. [Example Queries](#example-queries)
12. [Tech Stack](#tech-stack)

---

## Project Overview

Legal research over Indian Supreme Court judgments is challenging because legal
knowledge is both **relational** (cases cite each other, judges interpret statutes,
doctrines evolve) and **semantic** (meaning matters as much as keywords).

This system solves that by combining two complementary retrieval strategies:

| Strategy | Technology | Strength |
|----------|-----------|---------|
| **Graph Traversal** | Neo4j (Cypher) | Explicit relationships: citations, statutes, judges, doctrines |
| **Semantic Search** | ChromaDB + Sentence-Transformers | Meaning-based similarity across judgment text |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) | Best of both worlds in one ranked result set |

The fused results are passed to an LLM (Groq / Anthropic / OpenAI) which
synthesises a grounded, cited legal research answer.

---

## System Architecture

```
                        ┌──────────────────────────────────┐
                        │          User Query               │
                        └──────────────┬───────────────────┘
                                       │
                        ┌──────────────▼───────────────────┐
                        │         Intent Detection          │
                        │  (statute? judge? concept? free)  │
                        └──────┬───────────────┬───────────┘
                               │               │
               ┌───────────────▼──┐     ┌──────▼────────────────┐
               │  Graph Retrieval  │     │   Vector Retrieval     │
               │  (Neo4j Cypher)   │     │   (ChromaDB cosine)    │
               │                   │     │                        │
               │  • by_statute()   │     │  sentence-transformers │
               │  • by_judge()     │     │  all-MiniLM-L6-v2      │
               │  • by_concept()   │     │  (runs locally)        │
               │  • fulltext()     │     │                        │
               └───────────────┬──┘     └──────┬────────────────┘
                               │               │
                        ┌──────▼───────────────▼───────────┐
                        │    Reciprocal Rank Fusion (RRF)   │
                        │   score = Σ 1 / (60 + rank_i)     │
                        └──────────────┬───────────────────┘
                                       │
                        ┌──────────────▼───────────────────┐
                        │    LLM Answer Generation          │
                        │    (Groq / Anthropic / OpenAI)    │
                        └──────────────┬───────────────────┘
                                       │
                        ┌──────────────▼───────────────────┐
                        │  LegalResearchResponse            │
                        │  • answer (cited, grounded)       │
                        │  • sources (case name, date)      │
                        │  • graph_hits / vector_hits       │
                        └──────────────────────────────────┘
```

### Ingestion Pipeline

```
 .txt file  (Kaggle dataset)
      │
      ▼  JudgmentTXTParser
  raw text  +  metadata
  (case number, date, bench, parties, outcome extracted with regex)
      │
      ├──▶  LegalEntityExtractor  (LLM)
      │         citations, statutes, concepts, holdings, judges
      │              │
      │              ▼
      │         LegalGraphBuilder
      │         MERGE nodes & relationships → Neo4j
      │
      └──▶  sliding-window chunks  (800 words, 150 overlap)
                    │
                    ▼  EmbeddingModel (local)
               float32 vectors  (384-dim)
                    │
                    ▼  LegalVectorStore
               ChromaDB  (persisted on disk)
```

---

## Knowledge Graph Schema

### Nodes

| Label | Key Properties |
|-------|---------------|
| `Case` | `case_number` (unique), `case_name`, `date`, `petitioner`, `respondent`, `subject_matter`, `outcome`, `holdings` |
| `Judge` | `name` (unique) |
| `Statute` | `name` (unique) — e.g. `"Section 302 IPC"`, `"Article 21 Constitution"` |
| `LegalConcept` | `name` (unique) — e.g. `"res judicata"`, `"natural justice"` |

### Relationships

| Relationship | From → To | Meaning |
|-------------|-----------|---------|
| `DECIDED_BY` | Case → Judge | Judge sat on the bench |
| `APPLIES` | Case → Statute | Case invokes this statute/article |
| `INVOLVES` | Case → LegalConcept | Case discusses this doctrine |
| `CITES` | Case → Case | Citation network |

### Example Graph (Cypher)

```cypher
// Find all cases decided by Justice Chandrachud that applied Article 21
MATCH (c:Case)-[:DECIDED_BY]->(j:Judge),
      (c)-[:APPLIES]->(s:Statute)
WHERE j.name CONTAINS "Chandrachud"
  AND s.name CONTAINS "Article 21"
RETURN c.case_name, c.date, c.outcome
ORDER BY c.date DESC
LIMIT 10

// Traverse citation network 2 hops from a landmark case
MATCH (c:Case {case_number: "Civil Appeal No. 4321 of 2022"})
      -[:CITES*1..2]->(cited:Case)
RETURN c.case_name, cited.case_name, cited.date
```

---

## Project Structure

```
LegalGraphRAG/
│
├── README.md                      ← This file
├── requirements.txt               ← Python dependencies
├── .env.example                   ← Copy to .env and fill in keys
├── config.py                      ← Central settings (loaded from .env)
│
├── app.py                         ← Streamlit web UI
├── api.py                         ← FastAPI REST backend
├── smoke_test.py                  ← End-to-end sanity check
│
├── src/
│   ├── ingestion/
│   │   ├── txt_parser.py          ← Parse .txt files → metadata + chunks
│   │   ├── entity_extractor.py    ← LLM-based entity extraction (JSON)
│   │   └── data_loader.py         ← Orchestrate full ingestion pipeline
│   │
│   ├── graph/
│   │   ├── neo4j_client.py        ← Driver + schema + helpers
│   │   ├── graph_builder.py       ← MERGE nodes & relationships
│   │   └── graph_queries.py       ← All Cypher retrieval queries
│   │
│   ├── vector/
│   │   ├── embeddings.py          ← sentence-transformers (local)
│   │   └── vector_store.py        ← ChromaDB index + query
│   │
│   ├── retrieval/
│   │   └── hybrid_retriever.py    ← RRF fusion of graph + vector results
│   │
│   └── generation/
│       └── llm_chain.py           ← RAG prompt + LLM call + response model
│
└── data/
    └── judgments/                 ← Drop .txt files here
```

---

## Requirements

### Python
Python **3.10 or later** is required.

### External Services

| Service | Purpose | Cost |
|---------|---------|------|
| **Neo4j AuraDB** | Knowledge graph database | Free tier available |
| **Groq** | LLM inference (recommended) | Free tier available |
| Anthropic Claude | Alternative LLM | Paid |
| OpenAI GPT | Alternative LLM | Paid |

> Sentence-transformer embeddings run **entirely locally** — no API key needed.

---

## Setup & Installation

### Step 1 — Clone / unzip the project

```bash
cd LegalGraphRAG
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the `all-MiniLM-L6-v2` model (~22 MB). This happens
> automatically from HuggingFace.

### Step 4 — Create a free Neo4j AuraDB instance

1. Go to [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/) and sign up
2. Click **New Instance → AuraDB Free**
3. Save the **URI**, **username**, and **password** shown after creation
4. Wait ~2 minutes for the instance to start

### Step 5 — Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com) and sign up
2. Create an API key

### Step 6 — Configure `.env`

```bash
cp .env.example .env
```

Edit `.env`:

```env
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

LLM_PROVIDER=groq
LLM_MODEL=llama3-8b-8192
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
```

### Step 7 — Verify the setup

```bash
python smoke_test.py
```

All four tests should show ✅ (or ⚠️ skipped for services not yet configured).

---

## Data — Kaggle Dataset

This project uses:
**[LEGAL-Text-Supreme Court Judgments (India)](https://www.kaggle.com/datasets/vxrunsonii/supreme-court-judgments-txt)**

Plain-text `.txt` files of Indian Supreme Court judgments. No PDF parsing needed.

### Download (Kaggle CLI — fastest)

```bash
pip install kaggle

# Place your kaggle.json token at ~/.kaggle/kaggle.json
# Get it from: kaggle.com → Account → Settings → API → Create New Token

kaggle datasets download -d vxrunsonii/supreme-court-judgments-txt
unzip supreme-court-judgments-txt.zip -d ./data/judgments/
```

### Download (browser)

1. Visit the dataset page and click **Download**
2. Extract the `.zip`
3. Move all `.txt` files into `./data/judgments/`

---

## Running the System

### 1 — Ingest judgment files

```bash
python -m src.ingestion.data_loader --input ./data/judgments/
```

This will:
- Parse every `.txt` file (extract metadata with regex)
- Call the LLM to extract citations, statutes, legal concepts, holdings
- Build the Neo4j knowledge graph (Case, Judge, Statute, LegalConcept nodes)
- Embed all text chunks and store them in ChromaDB

Progress is shown with a `tqdm` progress bar.

### 2 — Launch the Streamlit UI

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

**UI features:**
- 🔍 Natural language legal research query
- 📂 Upload `.txt` files directly from the sidebar
- 📊 Live graph stats (cases, judges, statutes, concepts, chunks)
- 📝 Streamed LLM answer with citations
- 📚 Retrieved sources with graph/vector/hybrid badge
- 🔗 Example Cypher queries shown for each search

### 3 — Launch the REST API (optional)

```bash
uvicorn api:app --reload --port 8000
```

Interactive docs: **http://localhost:8000/docs**

---

## How It Works

### Hybrid Retrieval in Detail

When a query arrives, the system detects its **intent** using lightweight regex:

| Pattern detected | Graph strategy |
|-----------------|---------------|
| `Section 302`, `Article 21`, `IPC`, `CrPC` | `by_statute()` |
| `Justice Chandrachud`, `Justice Bhat` | `by_judge()` |
| `res judicata`, `natural justice`, `habeas corpus` | `by_concept()` |
| Any query | `fulltext()` index on Neo4j + semantic vector search |

Results from both legs are merged using **Reciprocal Rank Fusion**:

```
RRF score = Σ  1 / (60 + rank_i)
```

This gives higher scores to items that rank well in *both* retrieval methods,
without requiring score normalisation.

### Entity Extraction

For each new judgment, the LLM is prompted to return a structured JSON object:

```json
{
  "citations":      ["Hussainara Khatoon v. State of Bihar (1979)"],
  "statutes":       ["Article 21 Constitution", "Section 437 CrPC"],
  "legal_concepts": ["right to speedy trial", "natural justice"],
  "holdings":       ["Appeal allowed. Detention was unconstitutional."],
  "judges":         ["Justice D.Y. Chandrachud"],
  "subject_matter": "Constitutional Law",
  "outcome":        "Allowed"
}
```

This JSON is used to create graph nodes and relationships in Neo4j. Extraction
runs only once per file (on the first 3500 characters to control API cost).

### Chunking Strategy

Each judgment is split into overlapping word-level windows:

```
chunk_size    = 800 words
chunk_overlap = 150 words
step          = 650 words

Window 1: words[0:800]
Window 2: words[650:1450]
Window 3: words[1300:2100]
...
```

Overlap ensures that context spanning a chunk boundary is not lost.

---

## API Reference

| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Full RAG query (retrieve + generate) |
| `POST` | `/query/stream` | Streaming SSE response |
| `POST` | `/retrieve` | Retrieve chunks without generation |
| `POST` | `/ingest/text` | Ingest raw text judgment |
| `GET` | `/graph/stats` | Node count statistics |
| `GET` | `/graph/case/{id}` | Case detail + citations |
| `POST` | `/graph/search` | Direct graph search |
| `GET` | `/vector/stats` | Vector store statistics |

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

Response:
```json
{
  "query": "What is the law on right to speedy trial ...",
  "answer": "The right to speedy trial is a fundamental right under Article 21 ...",
  "sources": [
    {"case_number": "...", "case_name": "Hussainara Khatoon ...", "source": "graph", "score": 0.91}
  ],
  "graph_hits": 3,
  "vector_hits": 5,
  "model_used": "llama3-8b-8192"
}
```

---

## Example Queries

```
# Constitutional law
"What are the landmark judgments on Article 21 right to life and personal liberty?"
"How has the Supreme Court interpreted the right to privacy?"
"What is the basic structure doctrine established in Kesavananda Bharati?"

# Criminal law
"What is the test for awarding death penalty under Section 302 IPC?"
"Cases on bail under Section 437 and 438 CrPC"
"What is anticipatory bail and when can it be refused?"

# Legal doctrines
"Explain the doctrine of res judicata as applied by the Supreme Court"
"Cases on natural justice and the audi alteram partem principle"
"What is promissory estoppel in Indian contract law?"

# Judge-specific
"Important judgments delivered by Justice D.Y. Chandrachud"
"Cases decided by Justice Indu Malhotra on gender equality"

# Citation network
"Which cases have cited Maneka Gandhi v. Union of India?"
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Knowledge Graph | [Neo4j AuraDB](https://neo4j.com/cloud/aura/) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) (local) |
| Embeddings | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| LLM | [Groq](https://groq.com/) / Anthropic / OpenAI |
| Web UI | [Streamlit](https://streamlit.io/) |
| REST API | [FastAPI](https://fastapi.tiangolo.com/) |
| Dataset | [Kaggle: vxrunsonii/supreme-court-judgments-txt](https://www.kaggle.com/datasets/vxrunsonii/supreme-court-judgments-txt) |
| Language | Python 3.10+ |

---

## License

This project is for educational and research purposes.
The Kaggle dataset is subject to its own license — check the dataset page before
commercial use.
