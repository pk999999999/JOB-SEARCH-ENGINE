# Redrob AI Ranker

**Production-grade candidate ranking system for the Redrob AI Hackathon.**

Ranks 100,000 candidates against a Senior AI Engineer job description and outputs the top 100 with grounded reasoning — all on CPU, under 5 minutes.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    OFFLINE STAGE                         │
│  precompute_embeddings.py                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Load &   │→ │ Build    │→ │ Generate │→ │ Save    │ │
│  │ Validate │  │ Narratives│  │ Embeds   │  │ Artifacts│ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    RANKING STAGE                         │
│  rank.py  (< 5 min, CPU, no network)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Load     │→ │ Compute  │→ │ Apply    │→ │ Sort &  │ │
│  │ Artifacts│  │ Features │  │ Modifiers│  │ Export  │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Precompute Embeddings (requires internet for model download)

```bash
python scripts/precompute_embeddings.py --input data/candidates.csv --output artifacts/
```

### 3. Run Ranking (offline, CPU only)

```bash
python rank.py --artifacts artifacts/ --output output/top_100.csv
```

### 4. Launch Streamlit Demo

```bash
streamlit run sandbox/streamlit_app.py
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

## Features

| Feature | Method |
|---|---|
| Skill Match | Lexical overlap + BM25 + Embedding similarity |
| Career Match | Narrative embedding cosine similarity |
| Experience Fit | Gaussian decay around 5–9 year ideal |
| Location Fit | Tiered scoring (Pune/Noida → Mumbai/Hyd → India → Global) |
| Education Fit | Degree level + field relevance (tie-breaker) |
| Coherence | Cross-section embedding similarity |
| Behavioral | Availability & responsiveness multiplier |
| Honeypot | Multi-signal fraud detection gate |

## Testing

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Evaluation

```bash
python scripts/run_eval.py --predictions output/top_100.csv --gold eval/gold_labels.csv
```

Metrics: NDCG@10, NDCG@50, MAP, Precision@10

## Performance

| Metric | Target | Achieved |
|---|---|---|
| Candidates | 100,000 | ✅ |
| Ranking Time | < 5 min | ✅ (~30s) |
| RAM Usage | < 16 GB | ✅ (~4 GB) |
| GPU Required | No | ✅ CPU only |
| Network Required | No | ✅ Offline |

## License

MIT
