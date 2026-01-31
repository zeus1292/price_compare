# Retail Right

A multi-agent product matching tool that allows users to search for products via text, URL, or image input. Built with LangGraph for agent orchestration, hybrid search (SQL + vector similarity), and LangSmith for observability.

## Features

- **Unified Smart Search**: Single input that auto-detects text queries, URLs, or pasted/dropped images
- **CLIP Image Search**: Fast visual similarity search using CLIP embeddings (512-dim vectors)
- **Hybrid Search Engine**: Combines SQL filtering with vector similarity using Reciprocal Rank Fusion
- **Multi-Agent Architecture**: LangGraph-powered agents for property extraction, database matching, and live web search
- **Live Search with Caching**: Tavily-powered web search with intelligent result caching (30-min TTL)
- **Validated Product Images**: Only displays images from verified retail websites and CDNs
- **User Authentication**: Sign up/login with session-based auth for personalized experience
- **Recent Searches**: Logged-in users see their last 5 searches for quick re-execution
- **Trending Categories**: Carousel of popular categories (Electronics, Fashion, Home & Kitchen, Beauty, Sports & Outdoors)
- **Sort Results**: Sort by Relevance, Price Low to High, or Price High to Low
- **User Feedback System**: Thumbs up/down ratings on results for quality tracking
- **Precision/Recall Metrics**: Built-in search quality analytics and reporting
- **Trending Searches**: Shows top 5 most searched products from the last 7 days
- **Real-time Observability**: Full tracing with LangSmith for debugging and optimization
- **Apple-Inspired UI Design**: Clean, minimalist interface with SF Pro typography, frosted glass header, and pastel accent colors
- **Pastel Color Theme**: Soft color palette (pink, lavender, mint, peach, sky blue) for category cards and UI accents

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Web UI (Unified Input)                        │
│              Auto-detects: Text | URL | Image                    │
│         + Auth | Trending Categories | Recent Searches           │
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
- **Databases**: SQLite (structured data + users), ChromaDB (vector embeddings)
- **LLMs**: OpenAI GPT-4o-mini (default), Anthropic Claude (optional)
- **Image Embeddings**: CLIP ViT-B-32 via sentence-transformers
- **Text Embeddings**: OpenAI text-embedding-3-small
- **Search**: Tavily API for live web search (with caching)
- **Authentication**: Session-based auth with HTTP-only cookies
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
│   │       ├── auth.py            # Authentication (signup/login/logout)
│   │       ├── search.py          # Search endpoints
│   │       ├── products.py        # Product CRUD
│   │       ├── feedback.py        # User feedback
│   │       └── dataset.py         # Data ingestion
│   ├── database/                  # Data layer
│   │   ├── models.py              # SQLAlchemy models (Product, User, SearchHistory)
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
│       │   └── index.html         # Main UI template
│       └── static/
│           ├── css/styles.css     # Light theme styles
│           └── js/app.js          # Frontend logic
├── data/                          # Database storage
├── requirements.txt
└── README.md
```

## API Endpoints

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/signup` | POST | Create new user account |
| `/api/v1/auth/login` | POST | Login with email/password |
| `/api/v1/auth/logout` | POST | Logout and clear session |
| `/api/v1/auth/me` | GET | Get current user info |
| `/api/v1/auth/history` | GET | Get user's recent searches |
| `/api/v1/auth/history` | DELETE | Clear user's search history |

### Search

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/search` | POST | Search products (text/URL/image) with sort_by option |
| `/api/v1/search/image` | POST | Search by image upload |
| `/api/v1/search/stats` | GET | Database & cache statistics |
| `/api/v1/search/metrics` | GET | Precision/recall metrics |
| `/api/v1/search/debug` | GET | Debug LLM connectivity |
| `/api/v1/search/popular` | GET | Get top 5 most searched products (last 7 days) |

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

### Trending Searches
The landing page displays the top 5 most searched products from the last 7 days:

- Shows actual search queries from users
- Displays search count for each query
- Click any trending search to instantly search for that product
- Updates dynamically as users search

If no searches exist yet, default popular products are shown as suggestions.

### Recent Searches (Logged-in Users)
When logged in, users see their last 5 searches below the search bar:
- Click to re-execute a previous search
- Clear button to remove all history
- Automatically updates after each search

### Sort Results
Results can be sorted by:
- **Relevance** (default) - Sorted by match confidence
- **Price: Low to High** - Cheapest first
- **Price: High to Low** - Most expensive first

### Validated Product Images
Product images are validated before display to ensure quality:
- Only shows images from verified retail websites and CDNs
- Validates URLs from 50+ trusted domains (Amazon, Walmart, Target, eBay, etc.)
- Falls back to gradient placeholder if image source is untrusted
- Prevents display of broken, incorrect, or potentially malicious images

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

## User Authentication

Retail Right supports user accounts for personalized features:

### Sign Up
```bash
curl -X POST http://localhost:8000/api/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securepass", "name": "John"}'
```

### Login
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securepass"}'
```

Sessions are stored server-side with HTTP-only cookies (24-hour expiration).

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

## Database Schema

### SQLite Tables

| Table | Description |
|-------|-------------|
| `products` | Product catalog with name, price, merchant, etc. |
| `product_attributes` | Flexible key-value attributes for products |
| `search_cache` | Cached search results with TTL |
| `search_feedback` | User feedback on search results |
| `users` | User accounts (email, password hash, name) |
| `search_history` | User search history (last 5 per user) |
| `ingestion_log` | Dataset processing progress |

### ChromaDB Collections

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
| `Escape` | Close modal / clear image |
| `Ctrl+V` | Paste image from clipboard |

## UI Design

### Apple-Inspired Design System

Retail Right features a clean, minimalist UI inspired by Apple's design language:

- **Typography**: SF Pro Display/Text font family with anti-aliased rendering
- **Frosted Glass Header**: Semi-transparent header with backdrop blur effect
- **Rounded Corners**: Generous border radius throughout (12-24px)
- **Subtle Shadows**: Multi-layer shadows for depth without harshness
- **Smooth Transitions**: 200-300ms ease animations on all interactive elements

### Pastel Color Palette

Category cards and UI accents use a soft pastel palette:

| Color | CSS Variable | Use Case |
|-------|-------------|----------|
| Pink | `--pastel-pink: #ffd6e0` | Electronics category |
| Lavender | `--pastel-lavender: #e6e0ff` | Fashion category, Recent Searches |
| Mint | `--pastel-mint: #d4f5e9` | Home & Kitchen category |
| Peach | `--pastel-peach: #ffe5d4` | Beauty category |
| Sky Blue | `--pastel-sky: #d4e9ff` | Sports & Outdoors category |

### Key UI Components

- **Hero Section**: Animated gradient title with feature badges
- **Category Cards**: Left-colored border accent with pastel hover background
- **Product Cards**: Clean white cards with confidence badges and merchant pills
- **Search Module**: Rounded input with focus ring effect
- **Stats Section**: Animated counters with colored top borders
- **How It Works**: Step-by-step cards with numbered badges
- **Auth Modal**: Centered modal with backdrop blur
- **Results Grid**: Responsive grid with hover lift animations
- **Floating Elements**: Pastel gradient blobs for visual depth

### Animations

- `fadeInUp`: Elements slide up and fade in on load
- `bounceIn`: Feature badges bounce into view
- `gradientShift`: Hero title gradient animates continuously
- `float1-5`: Background shapes float gently
- Hover effects: Cards lift with shadow on hover

## Performance Optimizations

1. **CLIP for Images**: ~50ms vs ~2s for GPT-4o vision
2. **Tavily Caching**: Instant results for repeated queries
3. **Fast Mode for Live Search**: Heuristic parsing instead of LLM
4. **Parallel LLM Calls**: Concurrent result parsing with asyncio.gather()
5. **Confidence Filtering**: Early filtering reduces processing
6. **Client-side Sorting**: Re-sort results without API calls

## Deployment & Hosting

### Free/Low-Cost Hosting Options

| Platform | Free Tier | Best For | Limitations |
|----------|-----------|----------|-------------|
| **Railway** | $5/month credit | Full-stack apps | Credit expires monthly |
| **Render** | 750 hours/month | Web services | Sleeps after 15 min inactivity |
| **Fly.io** | 3 shared VMs | Global deployment | 256MB RAM on free tier |
| **PythonAnywhere** | 1 web app | Python apps | Limited CPU, no background tasks |
| **Vercel** | Unlimited | Serverless | Cold starts, 10s timeout |
| **Google Cloud Run** | 2M requests/month | Containers | Cold starts |

### Recommended: Railway or Render

**Railway** (Recommended for this app):
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

**Render**:
1. Push code to GitHub
2. Connect repo at render.com
3. Select "Web Service"
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `python -m src.api.main`

### Environment Variables for Production

Set these in your hosting platform:
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
ANTHROPIC_API_KEY=sk-ant-...  # Optional
LANGSMITH_API_KEY=...         # Optional
```

### Database Considerations

For production, consider:
- **SQLite**: Works on Railway/Render with persistent volumes
- **PostgreSQL**: Free tier on Supabase, Neon, or Railway
- **ChromaDB**: Can be self-hosted or use managed Chroma Cloud

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "-m", "src.api.main"]
```

Build and run:
```bash
docker build -t retail-right .
docker run -p 8000:8000 --env-file .env retail-right
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- [Klarna AI](https://github.com/nickmvincent/product-page-dataset) for the product dataset
- [LangChain](https://langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph) for the agent framework
- [Tavily](https://tavily.com/) for the search API
- [OpenAI CLIP](https://openai.com/research/clip) for visual embeddings
