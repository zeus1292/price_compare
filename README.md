# Price Compare

A multi-agent product matching tool that allows users to search for products via text, URL, or image input. Built with LangGraph for agent orchestration, hybrid search (SQL + vector similarity), and LangSmith for observability.

## Features

- **Multi-modal Search**: Search by text description, product URL, or image upload
- **Hybrid Search Engine**: Combines SQL filtering with vector similarity search using Reciprocal Rank Fusion
- **Multi-Agent Architecture**: LangGraph-powered agents for property extraction, database matching, and live web search
- **Live Search Fallback**: Automatically searches the web via Tavily when database confidence is low
- **Real-time Observability**: Full tracing with LangSmith for debugging and optimization
- **Modern Web UI**: Clean, responsive interface with loading indicators and confidence scores

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web UI / API                              │
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
        │                       │
        ▼                       ▼
┌───────────────┐    ┌───────────────────┐
│   GPT-4o-mini │    │  Hybrid Search    │
│   (LLM)       │    │  SQL + ChromaDB   │
└───────────────┘    └───────────────────┘
```

## Tech Stack

- **Backend**: Python 3.9+, FastAPI, SQLAlchemy
- **Agent Framework**: LangChain, LangGraph
- **Databases**: SQLite (structured data), ChromaDB (vector embeddings)
- **LLMs**: OpenAI GPT-4o-mini (default), Anthropic Claude Haiku (optional)
- **Search**: Tavily API for live web search
- **Embeddings**: OpenAI text-embedding-3-small
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
│   ├── agents/                 # LangGraph agents
│   │   ├── orchestrator.py     # Central supervisor
│   │   ├── property_extractor.py
│   │   ├── product_matcher.py
│   │   └── live_searcher.py
│   ├── api/                    # FastAPI application
│   │   ├── main.py
│   │   └── routes/
│   ├── database/               # Data layer
│   │   ├── models.py           # SQLAlchemy models
│   │   ├── sqlite_manager.py
│   │   └── chroma_manager.py
│   ├── services/               # Business logic
│   │   ├── llm_service.py
│   │   ├── embedding_service.py
│   │   └── search_service.py
│   ├── pipeline/               # Data ingestion
│   │   ├── klarna_parser.py
│   │   └── batch_processor.py
│   └── web/                    # Frontend
│       ├── templates/
│       └── static/
├── data/                       # Database storage
├── requirements.txt
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/v1/search` | POST | Search products (text/URL/image) |
| `/api/v1/search/image` | POST | Search by image upload |
| `/api/v1/products` | GET | List products |
| `/api/v1/products/{id}` | GET | Get product details |
| `/api/v1/dataset/ingest` | POST | Ingest product dataset |
| `/health` | GET | Health check |
| `/docs` | GET | API documentation (Swagger) |

## Configuration

See `example.env` for all configuration options:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `TAVILY_API_KEY` | Tavily API key for live search | - |
| `LANGSMITH_TRACING` | Enable LangSmith tracing | `false` |
| `CONFIDENCE_THRESHOLD` | Minimum match confidence | `0.9` |
| `ENABLE_LIVE_SEARCH` | Enable web search fallback | `true` |

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

## Data Ingestion

The tool supports ingesting product data from the [Klarna Product Page Dataset](https://zenodo.org/records/12605480):

```bash
# Download dataset from Zenodo
# Then ingest via API:
curl -X POST http://localhost:8000/api/v1/dataset/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_path": "./data/klarna/"}'
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- [Klarna AI](https://github.com/nickmvincent/product-page-dataset) for the product dataset
- [LangChain](https://langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph) for the agent framework
- [Tavily](https://tavily.com/) for the search API
