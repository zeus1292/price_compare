# Retail Right — Price Compare

A multi-agent product price comparison tool. Users search by text, URL, or image; the system returns matching products from a local dataset and live web results.

---

## Architecture Overview

```
User Query (text / URL / image)
        │
        ▼
  FastAPI API  (src/api/)
        │
        ▼
  LangGraph Orchestrator  (src/agents/orchestrator.py)
     ├── PropertyExtractorAgent   → extract structured product info from query
     ├── ProductMatcherAgent      → hybrid search against local DB
     └── LiveSearcherAgent        → web search via Tavily (supplements DB results)
        │
        ▼
  Result Synthesis → sorted, deduplicated, confidence-scored results
```

---

## Key Files

| Path | Purpose |
|------|---------|
| `src/api/main.py` | FastAPI app factory, lifespan (DB init/teardown), static files, CORS |
| `src/api/routes/search.py` | POST `/api/v1/search`, image search, metrics, stats, debug |
| `src/api/routes/auth.py` | Cookie-based session auth (signup/login/logout) |
| `src/api/routes/products.py` | Product CRUD endpoints |
| `src/api/routes/dataset.py` | Dataset ingestion trigger endpoints |
| `src/api/routes/feedback.py` | User feedback on search results |
| `src/api/middleware/tracing.py` | Request tracing middleware |
| `src/config/settings.py` | Pydantic `Settings` — loads all env vars with `SecretStr` for API keys |
| `src/agents/orchestrator.py` | LangGraph `StateGraph` — wires extraction → matching → live search → synthesis |
| `src/agents/property_extractor.py` | Extracts product name/brand/price/GTIN from text (LLM), URL (fetch+LLM), or image (CLIP → GPT-4o fallback) |
| `src/agents/product_matcher.py` | Calls `HybridSearchService`; wraps results for orchestrator state |
| `src/agents/live_searcher.py` | Tavily web search with in-memory cache; fast (regex) or slow (LLM) result parsing |
| `src/agents/base_agent.py` | Shared `BaseAgent` ABC — state validation, LLM call delegation |
| `src/services/search_service.py` | `HybridSearchService`: GTIN exact match → SQL pre-filter → ChromaDB vector search → RRF fusion → confidence scoring |
| `src/services/embedding_service.py` | OpenAI `text-embedding-3-small` embeddings (sync + async) |
| `src/services/llm_service.py` | Unified LLM client for OpenAI and Anthropic |
| `src/services/clip_service.py` | CLIP visual embeddings via `sentence-transformers` (for image search) |
| `src/services/encryption_service.py` | Fernet encryption for cached search results stored in SQLite |
| `src/services/metrics_service.py` | In-memory search quality metrics (precision/recall proxies) |
| `src/services/category_service.py` | Product category helpers |
| `src/database/sqlite_manager.py` | SQLite via SQLAlchemy — products, search history, users, cache |
| `src/database/chroma_manager.py` | ChromaDB client — two collections: product names and descriptions |
| `src/database/models.py` | SQLAlchemy ORM models |
| `src/pipeline/klarna_parser.py` | Parses Klarna product page dataset JSON files |
| `src/pipeline/batch_processor.py` | Ingests parsed products into SQLite + ChromaDB in batches |
| `src/pipeline/embeddings.py` | Embedding generation helpers used during ingestion |
| `src/web/templates/index.html` | Single-page web UI ("Retail Right") |
| `src/web/static/css/styles.css` | UI styles |
| `src/web/static/js/app.js` | Frontend JS — search form, results rendering, auth |

---

## Databases

### SQLite (`./data/db/products.db`)
Stores: products, search history, user accounts, encrypted search result cache.

### ChromaDB (`./data/db/chroma`)
Two collections for vector similarity search:
- `product_names` — embeddings of product name + brand
- `product_descriptions` — embeddings of product descriptions

Both are populated by the ingestion pipeline from the Klarna dataset.

---

## Search Flow (detailed)

1. **Property Extraction** — LLM (or CLIP for images) converts the raw query into structured `{name, brand, category, price, gtin, merchant, ...}`.
2. **Hybrid DB Search**
   - GTIN exact match (instant, confidence=1.0)
   - SQL pre-filter (name pattern, merchant, price range, category)
   - ChromaDB vector search on the SQL-filtered subset
   - Reciprocal Rank Fusion (RRF, k=60) combines both ranking lists
   - `ConfidenceScorer` weights: name similarity (35%), vector similarity (20%), price (20%), merchant (15%), attributes (10%)
3. **Live Search** — Always runs when Tavily is configured, to supplement DB results. Fast mode uses regex parsing; slow mode calls LLM per result in parallel.
4. **Synthesis** — Deduplicates by URL, sorts by confidence, caps at `limit`.

---

## Environment Variables

Copy `example.env` to `.env` and fill in:

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | **Yes** | Embeddings + GPT-4o (extraction, vision) |
| `TAVILY_API_KEY` | Recommended | Live web search |
| `ANTHROPIC_API_KEY` | Optional | Claude-based reasoning |
| `LANGSMITH_API_KEY` | Optional | LangSmith observability |
| `ENCRYPTION_KEY` | Optional | Fernet key for encrypting cached results |
| `SQLITE_PATH` | Optional | Default `./data/db/products.db` |
| `CHROMA_PATH` | Optional | Default `./data/db/chroma` |
| `CONFIDENCE_THRESHOLD` | Optional | Default `0.5` |
| `PORT` | Optional | Default `8000` (used by cloud platforms) |

---

## Running Locally

```bash
# Create virtualenv and install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp example.env .env
# edit .env with your API keys

# Start the server
python -m src.api.main
# App available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

---

## Data Ingestion (Klarna Dataset)

The `data/klarna/` directory holds the Klarna product page dataset. Trigger ingestion via the API:

```bash
POST /api/v1/dataset/ingest
```

Or run the pipeline directly. Products are stored in SQLite and embedded into ChromaDB.

---

## Tech Stack

- **API**: FastAPI + Uvicorn
- **Agent Orchestration**: LangGraph (StateGraph)
- **LLMs**: OpenAI GPT-4o / GPT-4o-mini, Anthropic Claude
- **Embeddings**: OpenAI `text-embedding-3-small`; CLIP (`sentence-transformers`) for images
- **Vector DB**: ChromaDB (local persistence)
- **SQL DB**: SQLite via SQLAlchemy + aiosqlite
- **Web Search**: Tavily
- **Observability**: LangSmith
- **Security**: Pydantic `SecretStr`, Fernet encryption for cache
- **Frontend**: Vanilla HTML/CSS/JS (no framework)

---

## Deployment

### Docker
```bash
docker build -t retail-right .
docker run -p 8000:8000 --env-file .env retail-right
```

### Render.com (pre-configured)
A `render.yaml` blueprint is included. Connect the repo on render.com → New Blueprint. Set secret env vars in the Render dashboard. A persistent disk (1 GB) is mounted at `/app/data` for SQLite and ChromaDB.

---

## Important Notes for AI Assistants

- All API keys are loaded via `src/config/settings.py` using Pydantic `SecretStr` — never log or print `.get_secret_value()` output.
- The `data/` directory contains the Klarna dataset and local databases — do not commit `.env` or `data/db/`.
- ChromaDB and SQLite both write to `./data/db/` — any deployment platform must provide persistent disk storage at this path.
- The Orchestrator is a singleton (`get_orchestrator()`). The LangGraph graph is compiled once on first call.
- Live search is always triggered when `TAVILY_API_KEY` is set, regardless of DB confidence (by design).
