# Redrob AI Ranker — Production Job Search Engine

**AI-powered candidate ranking engine** that scores candidates against job descriptions using hybrid scoring (semantic embeddings + BM25 + lexical overlap), behavioral analysis, coherence checking, and fraud detection — all running on CPU.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js)                       │
│  Dashboard │ Search & Rank │ Candidate Browser │ Job Manager   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ REST API
┌───────────────────────────▼─────────────────────────────────────┐
│                     BACKEND (FastAPI)                            │
│  /api/rank  │  /api/candidates  │  /api/jobs  │  /api/health   │
├─────────────────────────────────────────────────────────────────┤
│                    RANKING ENGINE (Python)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Skill    │ │ Career   │ │ Experience│ │ Location │          │
│  │ Match    │ │ Match    │ │ Fit      │ │ Fit      │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Education│ │ Coherence│ │ Behavioral│ │ Honeypot │          │
│  │ Fit      │ │ Scorer   │ │ Modifier │ │ Detector │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                     ARTIFACTS (Precomputed)                      │
│  embeddings.npy │ bm25_index.pkl │ coherence_scores.parquet    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: Docker (Recommended)

```bash
docker-compose up --build
```

Open [http://localhost:8000](http://localhost:8000)

### Option 2: Manual Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Generate sample data (if needed)
python scripts/generate_sample_data.py --n 2000 --output data/candidates.csv

# 3. Precompute embeddings (requires internet for first model download)
python scripts/precompute_embeddings.py --input data/candidates.csv --output artifacts/

# 4. Start API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 5. Start frontend (development mode)
cd frontend && npm install && npm run dev
```

- **API:** [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
- **Frontend:** [http://localhost:3000](http://localhost:3000)

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/stats` | Dashboard statistics |
| `POST` | `/api/rank` | Rank candidates against a JD |
| `GET` | `/api/candidates` | List candidates (paginated, filterable) |
| `GET` | `/api/candidates/{id}` | Candidate detail |
| `GET` | `/api/jobs` | List job descriptions |
| `POST` | `/api/jobs` | Create job description |
| `DELETE` | `/api/jobs/{id}` | Delete job description |

### Example: Rank Candidates

```bash
curl -X POST http://localhost:8000/api/rank \
  -H "Content-Type: application/json" \
  -d '{
    "job_title": "Senior AI Engineer",
    "required_skills": ["python", "machine learning", "pytorch"],
    "preferred_skills": ["docker", "kubernetes"],
    "top_k": 10
  }'
```

## Scoring Formula

```
base_score = 0.30 × career_match
           + 0.25 × skill_match
           + 0.15 × experience_fit
           + 0.10 × location_fit
           + 0.05 × education_fit
           + 0.15 × rule_adjustments

final_score = base_score × coherence_factor × behavioral_modifier × honeypot_gate
```

All weights are configurable in `config/weights.yaml`.

## Project Structure

```
JOB-SEARCH-ENGINE/
├── api/                    # FastAPI backend
│   ├── main.py             # Application entry point
│   ├── schemas.py          # Pydantic request/response models
│   ├── services.py         # Business logic
│   ├── dependencies.py     # Dependency injection
│   ├── middleware.py        # Logging, rate limiting
│   └── config.py           # Environment configuration
├── frontend/               # Next.js frontend
│   └── src/
│       ├── app/            # Pages (Dashboard, Search, Candidates, Jobs)
│       ├── components/     # Reusable UI components
│       └── lib/            # API client
├── src/                    # Core ranking engine
│   ├── data_loader.py      # Candidate data ingestion
│   ├── jd_parser.py        # Job description parsing
│   ├── ranker.py           # Ranking pipeline orchestrator
│   ├── scoring.py          # Score computation
│   ├── reasoning.py        # Reasoning generation
│   ├── honeypot_detector.py # Fraud detection
│   └── features/           # Feature scoring modules
├── config/                 # YAML configurations
├── data/                   # Candidate data (CSV)
├── artifacts/              # Precomputed embeddings & indexes
├── scripts/                # Data generation & preprocessing
├── tests/                  # Test suite
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Single-command deployment
└── requirements.txt        # Python dependencies
```

## Configuration

Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `8000` | API server port |
| `ARTIFACTS_DIR` | `./artifacts` | Path to precomputed artifacts |
| `DATA_DIR` | `./data` | Path to candidate data |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed CORS origins |
| `LOG_LEVEL` | `INFO` | Logging level |

## Testing

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Performance

| Metric | Value |
|--------|-------|
| Candidates | 2,000+ (scalable to 100K) |
| Ranking Time | ~30s for 100K candidates |
| RAM Usage | ~4 GB |
| GPU Required | No — CPU only |

## License

MIT
