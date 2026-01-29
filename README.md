# Price Compare

A multi-agent product matching tool that allows users to search for products via text, URL, or image input. Built with LangGraph for agent orchestration, hybrid search (SQL + vector similarity), and LangSmith for observability.

## Features

- **Unified Smart Search**: Single input that auto-detects text queries, URLs, or pasted/dropped images
- **CLIP Image Search**: Fast visual similarity search using CLIP embeddings (512-dim vectors)
- **Hybrid Search Engine**: Combines SQL filtering with vector similarity using Reciprocal Rank Fusion
- **Multi-Agent Architecture**: LangGraph-powered agents for property extraction, database matching, and live web search
- **Live Search with Caching**: Tavily-powered web search with intelligent result caching (30-min TTL)
- **User Feedback System**: Thumbs up/down ratings on results for quality tracking
- **Precision/Recall Metrics**: Built-in search quality analytics and reporting
- **Real-time Observability**: Full tracing with LangSmith for debugging and optimization
- **Modern Dark Theme UI**: Amazon-style interface with collapsible search and confidence scores

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Web UI (Unified Input)                        │
│              Auto-detects: Text | URL | Image                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Orchestrator Agent                          │
│                    (LangGraph Supervisor)                        │
└─────────────────────────────────────────────────────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐    ┌───────────────────┐    ┌─────────────────┐
│   Property    │    │     Product       │    │   Live Search   │
│   Extractor   │    │     Matcher       │    │     Agent       │
│    Agent      │    │      Agent        │    │   (Tavily)      │
└───────────────┘    └───────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐    ┌───────────────────┐    ┌─────────────────┐
│  CLIP / LLM   │    │  Hybrid Search    │    │  Result Cache   │
│  (Image/Text) │    │  SQL + ChromaDB   │    │  (30-min TTL)   │
└───────────────┘    └───────────────────┘    └─────────────────┘
```

## Tech Stack

- **Backend**: Python 3.9+, FastAPI, SQLAlchemy
- **Agent Framework**: LangChain, LangGraph
- **Databases**: SQLite (structured data), ChromaDB (vector embeddings)
- **LLMs**: OpenAI GPT-4o-mini (default), Anthropic Claude (optional)
- **Image Embeddings**: CLIP ViT-B-32 via sentence-transformers
- **Text Embeddings**: OpenAI text-embedding-3-small
- **Search**: Tavily API for live web search (with caching)
- **Observability**: LangSmith

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/price-compare.git
cd price-compare

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp example.env .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

Required API keys:
- `OPENAI_API_KEY` - For embeddings and LLM calls
- `TAVILY_API_KEY` - For live web search

Optional:
- `ANTHROPIC_API_KEY` - For Claude-based reasoning
- `LANGSMITH_API_KEY` - For observability (recommended)

### 3. Run the Application

```bash
python -m src.api.main
```

Open http://localhost:8000 in your browser.

## Project Structure

```
Price_Compare/
├── src/
│   ├── agents/                    # LangGraph agents
│   │   ├── orchestrator.py        # Central supervisor
│   │   ├── property_extractor.py  # Extract properties (text/URL/image)
│   │   ├── product_matcher.py     # Hybrid + CLIP search
│   │   └── live_searcher.py       # Tavily search with caching
│   ├── api/                       # FastAPI application
│   │   ├── main.py
│   │   └── routes/
│   │       ├── search.py          # Search endpoints
│   │       ├── products.py        # Product CRUD
│   │       ├── feedback.py        # User feedback
│   │       └── dataset.py         # Data ingestion
│   ├── database/                  # Data layer
│   │   ├── models.py              # SQLAlchemy models
│   │   ├── sqlite_manager.py      # SQLite operations
│   │   └── chroma_manager.py      # Vector store (3 collections)
│   ├── services/                  # Business logic
│   │   ├── llm_service.py         # Multi-provider LLM client
│   │   ├── embedding_service.py   # OpenAI embeddings
│   │   ├── clip_service.py        # CLIP image embeddings
│   │   ├── search_service.py      # Hybrid search engine
│   │   └── metrics_service.py     # Precision/recall tracking
│   ├── pipeline/                  # Data ingestion
│   │   ├── klarna_parser.py
│   │   └── batch_processor.py
│   └── web/                       # Frontend
│       ├── templates/
│       └── static/
├── data/                          # Database storage
├── requirements.txt
└── README.md
```

## API Endpoints

### Search

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/search` | POST | Search products (text/URL/image) |
| `/api/v1/search/image` | POST | Search by image upload |
| `/api/v1/search/stats` | GET | Database & cache statistics |
| `/api/v1/search/metrics` | GET | Precision/recall metrics |
| `/api/v1/search/debug` | GET | Debug LLM connectivity |

### Products

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/products` | GET | List products (paginated) |
| `/api/v1/products/{id}` | GET | Get product details |
| `/api/v1/products/contribute` | POST | Add a product |

### Feedback

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/feedback` | POST | Submit thumbs up/down |
| `/api/v1/feedback` | GET | Query feedback (with filters) |
| `/api/v1/feedback/stats` | GET | Satisfaction rate & analytics |
| `/api/v1/feedback/export` | GET | Export as JSON or CSV |

### Other

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/v1/dataset/ingest` | POST | Ingest product dataset |
| `/health` | GET | Health check |
| `/health/detailed` | GET | Component health status |
| `/docs` | GET | API documentation (Swagger) |

## Search Features

### Unified Input
The search input automatically detects:
- **Text**: Product names, descriptions, keywords
- **URL**: Paste any product page URL (detected via regex)
- **Image**: Drag & drop, paste from clipboard, or click to upload

### CLIP Image Search
When searching with an image:
1. CLIP generates a 512-dimensional embedding (~50ms)
2. Embedding queries ChromaDB's image collection directly
3. Returns visually similar products without text extraction
4. Falls back to GPT-4o vision if CLIP unavailable

### Confidence Threshold
- Set minimum confidence (0-100%) in the UI
- Only results meeting the threshold are returned
- Prevents low-quality matches from cluttering results

### Live Search Caching
- Tavily results cached for 30 minutes
- Reduces API calls and improves response time
- Cache stats available via `/api/v1/search/stats`

## Metrics & Analytics

### Search Quality Metrics
```bash
curl http://localhost:8000/api/v1/search/metrics
```

Returns:
- Aggregate precision/recall proxies
- Results by confidence tier (high/medium/low)
- Recent search performance
- Processing time statistics

### User Feedback
```bash
# Get satisfaction statistics
curl http://localhost:8000/api/v1/feedback/stats

# Export negative feedback for analysis
curl "http://localhost:8000/api/v1/feedback?rating=-1"

# Export all feedback as CSV
curl "http://localhost:8000/api/v1/feedback/export?format=csv" -o feedback.csv
```

Feedback includes:
- Rating (+1 thumbs up, -1 thumbs down)
- Query context (text, type, trace ID)
- Product context (name, merchant, confidence)
- Optional user comments

## Configuration

See `example.env` for all configuration options:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `TAVILY_API_KEY` | Tavily API key for live search | - |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional) | - |
| `LANGSMITH_API_KEY` | LangSmith API key | - |
| `LANGSMITH_TRACING` | Enable LangSmith tracing | `false` |
| `CONFIDENCE_THRESHOLD` | Default minimum match confidence | `0.9` |
| `ENABLE_LIVE_SEARCH` | Enable web search fallback | `true` |
| `DEFAULT_SEARCH_LIMIT` | Max results per search | `10` |

## ChromaDB Collections

| Collection | Embedding | Dimensions | Purpose |
|------------|-----------|------------|---------|
| `product_names` | OpenAI | 1536 | Text similarity search |
| `product_descriptions` | OpenAI | 1536 | Full description search |
| `product_images` | CLIP | 512 | Visual similarity search |

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black src/

# Lint
ruff check src/
```

### Debug Mode

```bash
# Test LLM connectivity
curl http://localhost:8000/api/v1/search/debug

# Check component health
curl http://localhost:8000/health/detailed
```

## Data Ingestion

The tool supports ingesting product data from the [Klarna Product Page Dataset](https://zenodo.org/records/12605480):

```bash
# Download dataset from Zenodo
# Then ingest via API:
curl -X POST http://localhost:8000/api/v1/dataset/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_path": "./data/klarna/"}'
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `/` | Focus search input |
| `Escape` | Collapse search / clear image |
| `Ctrl+V` | Paste image from clipboard |

## Performance Optimizations

1. **CLIP for Images**: ~50ms vs ~2s for GPT-4o vision
2. **Tavily Caching**: Instant results for repeated queries
3. **Fast Mode for Live Search**: Heuristic parsing instead of LLM
4. **Parallel LLM Calls**: Concurrent result parsing with asyncio.gather()
5. **Confidence Filtering**: Early filtering reduces processing

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- [Klarna AI](https://github.com/nickmvincent/product-page-dataset) for the product dataset
- [LangChain](https://langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph) for the agent framework
- [Tavily](https://tavily.com/) for the search API
- [OpenAI CLIP](https://openai.com/research/clip) for visual embeddings
