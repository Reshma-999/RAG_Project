# RAG Pipeline

**End-to-end Retrieval-Augmented Generation on 10K+ documents**

| Metric | Value |
|---|---|
| Answer Faithfulness | ~85% (RAGAS) |
| LLM Response Time (cache miss) | ~3 s |
| LLM Response Time (cache hit) | < 1 s |
| Redundant API Call Reduction | ~38% |

---

## Architecture

```
User Query
    │
    ▼
SemanticCache (Redis + RediSearch)   ──HIT──►  Cached Answer  (<1s)
    │
  MISS
    │
    ▼
Retriever (ChromaDB + MMR)
    │
    ▼
OpenAI GPT-4o                                                  (~3s)
    │
    ▼
Cache Store  ──►  RAGResponse
```

**Key components:**

- **Ingestion** (`src/ingest.py`) — loads PDF, DOCX, TXT, HTML, Markdown; splits into 512-token chunks; embeds and upserts into ChromaDB in batches of 500.
- **Vector Store** (`src/vector_store.py`) — ChromaDB with OpenAI `text-embedding-3-small`; supports MMR-based retrieval.
- **Semantic Cache** (`src/redis_cache.py`) — Redis Stack (RediSearch + HNSW) caches query embeddings and answers; cosine similarity threshold of 0.92 prevents stale results.
- **Retriever** (`src/retriever.py`) — Maximal Marginal Relevance selection for diverse, non-redundant context chunks.
- **RAG Pipeline** (`src/rag_pipeline.py`) — orchestrates retrieval → prompt assembly → generation → caching.
- **Evaluator** (`src/evaluator.py`) — RAGAS harness measuring faithfulness, answer relevancy, context precision, context recall, and citation accuracy.
- **API** (`api.py`) — FastAPI server with `/query`, `/ingest`, `/stats`, and `/cache` endpoints.

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/Reshma-999/RAG_Project.git
cd RAG_Project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (at minimum)
```

### 3. Start Redis Stack (required for semantic caching)

```bash
docker-compose up redis -d
```

Or run Redis Stack directly:

```bash
docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

---

## Usage

### Ingest documents

Place your documents in `./data/raw/` (PDF, DOCX, TXT, HTML, MD), then:

```bash
python main.py ingest --data-dir ./data/raw
```

### Ask a question

```bash
python main.py query "What are the main findings of the Q3 report?"
```

### Run evaluation harness

```bash
# First generate an eval dataset from your indexed documents
python scripts/generate_eval_dataset.py --n 100

# Then run evaluation
python main.py eval --dataset ./data/eval_dataset.json
```

### Start the API server

```bash
python main.py serve
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Check stats

```bash
python main.py stats
```

---

## API Reference

### `POST /query`

```json
{
  "question": "What is the main conclusion?",
  "use_cache": true,
  "top_k": 5
}
```

Response:

```json
{
  "question": "What is the main conclusion?",
  "answer": "The main conclusion is… [1]",
  "sources": [{"rank": 1, "filename": "report.pdf", "score": 0.91, "snippet": "…"}],
  "latency_s": 0.42,
  "from_cache": true
}
```

### `GET /stats`

Returns cache hit rate, API call reduction, and vector store document count.

### `DELETE /cache`

Flushes all entries from the semantic cache.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer Relevancy** | Is the answer relevant to the question? |
| **Context Precision** | Are retrieved docs relevant to the question? |
| **Context Recall** | Does the context contain all needed information? |
| **Citation Accuracy** | Do inline [N] citations point to real retrieved sources? |

Run `python main.py eval` to produce a CSV report in `./data/eval_results/`.

---

## Docker (full stack)

```bash
docker-compose up --build
```

This starts both the Redis Stack and the RAG API on port 8000.

---

## Project Structure

```
RAG_Project/
├── src/
│   ├── ingest.py          # Document loading & chunking
│   ├── vector_store.py    # ChromaDB wrapper
│   ├── redis_cache.py     # Semantic Redis cache
│   ├── retriever.py       # MMR retrieval + citations
│   ├── rag_pipeline.py    # End-to-end orchestration
│   ├── evaluator.py       # RAGAS eval harness
│   └── utils.py           # Shared utilities
├── tests/
│   └── test_pipeline.py   # Unit + integration tests
├── scripts/
│   └── generate_eval_dataset.py
├── data/
│   ├── raw/               # Place source documents here
│   ├── chroma_db/         # ChromaDB persistence (auto-created)
│   └── eval_results/      # Evaluation output CSVs
├── api.py                 # FastAPI server
├── main.py                # CLI entry point
├── config.py              # Pydantic settings
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## License

MIT
