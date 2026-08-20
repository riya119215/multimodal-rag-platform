# 🧠 Enterprise Multi-Modal Grounded RAG Platform

A production-grade, modular **Retrieval-Augmented Generation (RAG) System** supporting multi-format document ingestion (PDF, Word DOCX, Text, Markdown, CSV) and multi-language audio/video tutorial transcripts (OpenAI Whisper). 

Featuring **Hybrid Search (Dense FAISS + Sparse BM25)**, **Two-Stage Cross-Encoder Re-Ranking**, **Grounding & Hallucination Guardrails**, **Explainable Confidence Scoring**, **Conversational Memory & Query Rewriting**, a **FastAPI REST API**, and a modern **Streamlit Web Dashboard** with live Matplotlib code execution and synchronized audio snippet playback.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Highlights & Standout Features](#-key-highlights--standout-features)
3. [System Architecture](#-system-architecture)
4. [Project Directory Structure](#-project-directory-structure)
5. [How the System Works (Step-by-Step)](#-how-the-system-works-step-by-step)
6. [File-to-File Call Graphs & Connections](#-file-to-file-call-graphs--connections)
7. [Quick Start Guide (VS Code / Windows)](#-quick-start-guide-vs-code--windows)
8. [Running the Application](#-running-the-application)
9. [REST API Documentation](#-rest-api-documentation)
10. [Automated Testing Suite](#-automated-testing-suite)
11. [Configuration Reference](#-configuration-reference)

---

## 🌟 Project Overview

Traditional RAG prototypes suffer from three major vulnerabilities:
1. **Keyword Blindness**: Pure dense vector search frequently fails on exact product names, error codes, function signatures, or numbers.
2. **Hallucination & Overconfidence**: When context is absent or weak, naive LLMs invent plausible-sounding answers.
3. **Conversational Amnesia**: Follow-up questions containing pronouns (*"explain its parameters"*) fail during vector retrieval.

This platform resolves these limitations with an enterprise-grade pipeline combining:
- **Multi-Format Ingestion**: PDF, DOCX, TXT, MD, CSV, JSON, MP3, WAV.
- **Recursive Boundary Chunking**: Splits along semantic paragraph and sentence breaks with overlap.
- **Hybrid Retrieval & RRF**: Dense Cosine Similarity (`all-MiniLM-L6-v2`) + Lexical Keyword Matching (`BM25Okapi`) fused via **Reciprocal Rank Fusion (RRF)**.
- **Two-Stage Re-Ranking**: Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to re-score candidate chunks for maximum precision.
- **Strict Grounding & Confidence Scoring**: Multi-signal composite confidence calculation and hallucination prevention pre-checks.
- **Conversational Memory & Query Rewriting**: Multi-turn dialogue context reformulation before retrieval.
- **Interactive Execution**: Live in-browser Python / Matplotlib rendering and audio playback synced to timestamped citations.

---

## 🚀 Key Highlights & Standout Features

| Feature | Description |
| :--- | :--- |
| **Hybrid Search (FAISS + BM25)** | Combines semantic vector similarity with exact keyword/token matching using Reciprocal Rank Fusion (RRF). |
| **Cross-Encoder Re-Ranking** | Refines candidate chunk ranking using cross-attention scoring (`ms-marco-MiniLM-L-6-v2`). |
| **Hallucination Guardrails** | Validates context sufficiency before calling LLM. Declines off-topic or unsupported queries gracefully. |
| **Explainable Confidence Score** | Transparent score based on vector similarity ($40\%$), re-rank score ($25\%$), source quantity ($20\%$), and query-context overlap ($15\%$). |
| **Conversational Memory** | Multi-turn buffer and query rewriting resolving pronouns (*"it"*, *"they"*, *"explain more"*) into standalone queries. |
| **Live Matplotlib Sandbox** | Detects generated Python visualization code and renders interactive charts directly inside the web UI. |
| **Timestamped Audio Citations** | Synchronizes audio player playback directly to the exact start second of retrieved transcript segments. |
| **Production FastAPI Backend** | Async OpenAPI endpoints for `/query`, `/chat`, `/upload`, `/documents`, `/reindex`, and `/health`. |

---

## 🏛️ System Architecture

```text
                                  USER QUERY
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Conversational Memory   │
                        │  & Query Rewriter (LLM)   │
                        └─────────────┬─────────────┘
                                      │ (Standalone Search Query)
                                      ▼
                        ┌───────────────────────────┐
                        │   Dual-Stage Retrieval    │
                        ├─────────────┬─────────────┤
                        │ Dense Search│Sparse Search│
                        │ (FAISS IP)  │(BM25 Okapi) │
                        └──────┬──────┴──────┬──────┘
                               │             │
                               ▼             ▼
                        ┌───────────────────────────┐
                        │ Reciprocal Rank Fusion    │
                        │        (RRF Fusion)       │
                        └─────────────┬─────────────┘
                                      │ (Top Candidates)
                                      ▼
                        ┌───────────────────────────┐
                        │ Cross-Encoder Re-Ranker   │
                        │(ms-marco-MiniLM-L-6-v2)   │
                        └─────────────┬─────────────┘
                                      │ (Top-K High-Precision Chunks)
                                      ▼
                        ┌───────────────────────────┐
                        │ Grounding & Sufficiency   │
                        │ Pre-Check / Confidence    │
                        └─────────────┬─────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │ (Context Sufficient)              │ (Context Insufficient)
                    ▼                                   ▼
        ┌───────────────────────────┐       ┌───────────────────────────┐
        │ Context Builder & Prompt  │       │ Grounded Refusal Response │
        │ Synthesis via Gemini LLM  │       │ ("Sufficient info not     │
        └─────────────┬─────────────┘       │  found in documents")     │
                      │                     └─────────────┬─────────────┘
                      ▼                                   │
        ┌───────────────────────────┐                     │
        │ Grounded Answer + Citations│                    │
        │ Confidence + Live Code    │                     │
        └─────────────┬─────────────┘                     │
                      │                                   │
                      └─────────────────┬─────────────────┘
                                        ▼
                                  FINAL RESPONSE
```

---

## 📂 Project Directory Structure

```text
RAG SYSTEM/
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Pinned Python package dependencies
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git exclusions (vectorstore, cache, env)
├── app.py                         # Root Streamlit app entrypoint
├── rag.py                         # Root backward-compatible RAG CLI adapter
├── app/
│   ├── __init__.py                # Package declaration
│   ├── main.py                    # Unified entrypoint (CLI & FastAPI server launcher)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic Settings & dynamic environment config
│   │   └── logging_config.py      # Colorized structured logging
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── metadata.py            # Document and DocumentChunk data schemas
│   │   ├── loaders.py             # Multi-format loaders (PDF, DOCX, TXT, MD, CSV, Audio)
│   │   ├── chunking.py            # Recursive character & sentence boundary chunker
│   │   └── pipeline.py            # Unified Ingestion Pipeline orchestrator
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py        # FAISS IndexFlatIP dense cosine retriever
│   │   ├── bm25_store.py          # BM25 Okapi lexical keyword index
│   │   ├── reranker.py            # Cross-Encoder Re-ranker with score normalization
│   │   └── hybrid_search.py       # Hybrid Search with Reciprocal Rank Fusion (RRF)
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompts.py             # System prompts and context formatters
│   │   ├── llm.py                 # Google Gemini client with model fallbacks
│   │   ├── memory.py              # Conversational memory buffer & query rewriter
│   │   ├── grounding.py           # Grounding check & transparent confidence calculator
│   │   └── answer_generator.py    # End-to-end RAG orchestrator
│   ├── api/
│   │   ├── __init__.py
│   │   ├── schemas.py             # Pydantic request & response models
│   │   └── routes.py              # FastAPI endpoints (/health, /upload, /query, /chat)
│   ├── frontend/
│   │   ├── __init__.py
│   │   └── streamlit_app.py       # Modern multi-tab Streamlit dashboard
│   └── utils/
│       ├── __init__.py
│       └── helpers.py             # Headless Matplotlib execution, time formatters, cleaners
├── data/
│   ├── documents/                 # PDF, DOCX, TXT, MD, CSV documents
│   ├── audios/                    # MP3, WAV audio tutorial files
│   └── processed/                 # Timestamped JSON transcript chunks
├── vectorstore/
│   ├── faiss_index.bin            # FAISS dense index file
│   ├── metadata.pkl               # Pickled chunk metadata dictionary
│   └── bm25_index.pkl             # BM25 Okapi search index
└── tests/
    ├── __init__.py
    ├── test_ingestion.py          # Unit tests for loaders & chunking
    ├── test_retrieval.py          # Unit tests for FAISS, BM25 & Hybrid RRF
    ├── test_generation.py         # Unit tests for prompts, grounding & memory
    └── test_api.py                # Integration tests for FastAPI endpoints
```

---

## 🔍 How the System Works (Step-by-Step)

1. **Document Loading**: When a document (PDF, Word DOCX, TXT, MD, CSV) or audio transcript (JSON/Whisper) is ingested, [`loaders.py`](file:///app/ingestion/loaders.py) extracts clean text and metadata (page number, audio timestamp, file name).
2. **Text Chunking**: [`chunking.py`](file:///app/ingestion/chunking.py) splits documents into manageable semantic units (default 500 characters with 100 character overlap) while preserving natural paragraph and sentence boundaries.
3. **Dual Indexing**:
   - Dense Embeddings: [`vector_store.py`](file:///app/retrieval/vector_store.py) passes chunk texts through `SentenceTransformer("all-MiniLM-L6-v2")`, normalizes vectors, and stores them in a **FAISS IndexFlatIP** index for exact cosine similarity.
   - Sparse Indexing: [`bm25_store.py`](file:///app/retrieval/bm25_store.py) tokenizes words and builds an **Okapi BM25** inverted index.
4. **User Query & Context Rewriting**: When the user enters a question in the UI or API, [`memory.py`](file:///app/generation/memory.py) checks if previous turns exist. If the question contains relative pronouns (*"how do I customize it?"*), it rewrites the query into a self-contained search query.
5. **Hybrid Retrieval & RRF**: [`hybrid_search.py`](file:///app/retrieval/hybrid_search.py) queries both FAISS and BM25, combining results with Reciprocal Rank Fusion:
   $$RRF(d) = \frac{W_{dense}}{60 + rank_{dense}(d)} + \frac{W_{sparse}}{60 + rank_{sparse}(d)}$$
6. **Cross-Encoder Re-Ranking**: [`reranker.py`](file:///app/retrieval/reranker.py) scores the top fused candidate pairs `(query, chunk_text)` using `cross-encoder/ms-marco-MiniLM-L-6-v2` to evaluate semantic relevance with cross-attention.
7. **Grounding & Sufficiency Pre-Check**: [`grounding.py`](file:///app/generation/grounding.py) evaluates keyword overlap and similarity signals. If retrieved context is insufficient or irrelevant, it returns a grounded refusal instead of letting the model hallucinate.
8. **LLM Generation**: [`answer_generator.py`](file:///app/generation/answer_generator.py) formats the validated context chunks and invokes Google Gemini (`gemini-3.6-flash` or fallback) with strict grounding instructions.
9. **Citation & Interactive Post-Processing**: The system returns the generated answer, exact source citations (page numbers or timestamps), transparent confidence breakdown, and extracted Python/Matplotlib code blocks ready for live in-browser execution.

---

## 🔗 File-to-File Call Graphs & Connections

### 1. Ingestion Pipeline Connection
```text
app/main.py (--reindex)
  └── app/ingestion/pipeline.py (IngestionPipeline.run)
        ├── app/ingestion/loaders.py (DocumentLoaderFactory.load_document)
        ├── app/ingestion/chunking.py (RecursiveCharacterChunker.chunk_documents)
        ├── app/retrieval/vector_store.py (VectorStore.build_index -> faiss_index.bin)
        └── app/retrieval/bm25_store.py (BM25Index.build_index -> bm25_index.pkl)
```

### 2. Query & Generation Connection
```text
User Request (FastAPI / Streamlit UI / CLI)
  └── app/generation/answer_generator.py (RAGAnswerGenerator.generate_answer)
        ├── app/generation/memory.py (QueryRewriter.rewrite)
        ├── app/retrieval/hybrid_search.py (HybridRetriever.retrieve)
        │     ├── app/retrieval/vector_store.py (VectorStore.search)
        │     ├── app/retrieval/bm25_store.py (BM25Index.search)
        │     └── app/retrieval/reranker.py (ReRanker.rerank)
        ├── app/generation/grounding.py (GroundingChecker.check_context_sufficiency)
        ├── app/generation/prompts.py (format_context_for_llm)
        ├── app/generation/llm.py (GeminiLLM.generate)
        └── app/utils/helpers.py (extract_python_code, execute_matplotlib_code)
```

---

## ⚡ Quick Start Guide (VS Code / Windows)

Follow this simple sequence to set up and run the system:

```text
1. Open folder 'RAG SYSTEM' in VS Code
2. Open terminal (Ctrl + `)
3. Create virtual environment: python -m venv .venv
4. Activate virtual environment: .venv\Scripts\activate
5. Install dependencies: pip install -r requirements.txt
6. Copy environment template: copy .env.example .env
7. Open .env and add your GEMINI_API_KEY
8. Start Streamlit UI: streamlit run app.py
9. Ask questions, upload documents, and explore answers!
```

---

## 🖥️ Running the Application

### 1. Launch the Streamlit Web Application (Recommended)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Launch the FastAPI REST Server
```bash
python -m app.main --server --port 8000
```
- Interactive OpenAPI Docs: `http://localhost:8000/docs`
- Redoc Documentation: `http://localhost:8000/redoc`

### 3. Run via Interactive Command-Line Interface (CLI)
```bash
python -m app.main --interactive
```
Or run a single query directly:
```bash
python -m app.main --query "How do I create a bar chart in Matplotlib?"
```

### 4. Rebuild Search Indices
```bash
python -m app.main --reindex
```

---

## 📡 REST API Documentation

### `GET /api/v1/health`
Check server health, model configuration, and vector index size.
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "faiss_indexed_chunks": 874,
  "bm25_indexed_chunks": 874,
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_model": "gemini-3.6-flash"
}
```

### `POST /api/v1/query`
Stateless grounded query answering.

**Request Body:**
```json
{
  "query": "How do I create subplots in Matplotlib?",
  "top_k": 4,
  "retrieval_mode": "hybrid",
  "min_score": 0.10
}
```

**Response Body:**
```json
{
  "query": "How do I create subplots in Matplotlib?",
  "rewritten_query": "How do I create subplots in Matplotlib?",
  "answer": "To create subplots in Matplotlib, use `plt.subplots(nrows, ncols)` ...",
  "confidence": {
    "score": 0.882,
    "percentage": "88.2%",
    "level": "High",
    "is_grounded": true,
    "breakdown": {
      "dense_similarity": 0.854,
      "rerank_score": 0.912,
      "keyword_overlap": 0.75,
      "num_sources": 4
    }
  },
  "sources": [...],
  "code_blocks": ["import matplotlib.pyplot as plt\nfig, ax = plt.subplots(2, 2)"]
}
```

### `POST /api/v1/chat`
Conversational multi-turn RAG with session memory.
```json
{
  "session_id": "session_user_1",
  "message": "Explain its parameters.",
  "top_k": 5,
  "retrieval_mode": "hybrid"
}
```

### `POST /api/v1/upload`
Upload multiple files (`multipart/form-data`) with auto-reindexing.

### `GET /api/v1/documents`
List all ingested raw documents and transcript files.

---

## 🧪 Automated Testing Suite

The repository includes a test suite covering ingestion loaders, chunkers, FAISS vector search, BM25 keyword search, hybrid RRF, conversational memory, grounding, and FastAPI endpoints.

Run all tests:
```bash
pytest tests -v
```

---

## ⚙️ Configuration Reference

All settings can be customized via environment variables in `.env`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | *(None)* | Your Google Gemini API Key. |
| `LLM_MODEL` | `gemini-3.6-flash` | Gemini model name (`gemini-flash-latest`, `gemini-2.5-flash`). |
| `EMBEDDING_MODEL`| `all-MiniLM-L6-v2` | Dense sentence-transformer embedding model. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder re-ranker model. |
| `ENABLE_RERANKER`| `true` | Enable/disable second-stage re-ranking. |
| `CHUNK_SIZE` | `500` | Target character size per text chunk. |
| `CHUNK_OVERLAP` | `100` | Character overlap between consecutive chunks. |
| `TOP_K` | `5` | Number of context chunks retrieved for prompt. |
| `RETRIEVAL_MODE` | `hybrid` | Search mode: `hybrid`, `dense`, or `sparse`. |
| `MIN_SIMILARITY_SCORE` | `0.10` | Minimum cosine similarity threshold. |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant $k$. |
| `DENSE_WEIGHT` | `0.6` | Weight assigned to dense vector score. |
| `SPARSE_WEIGHT` | `0.4` | Weight assigned to sparse BM25 score. |
