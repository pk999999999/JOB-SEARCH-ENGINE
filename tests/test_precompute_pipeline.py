from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.precompute_embeddings as pe  # noqa: E402


def _fake_generate_embeddings(texts, model_name="x", batch_size=256):
    """Deterministic stand-in for the real sentence-transformers call,
    so the rest of the offline pipeline can be tested without network
    access. Same text -> same vector; different text -> different vector.
    """
    d = 384
    out = np.zeros((len(texts), d), dtype=np.float32)
    for i, t in enumerate(texts):
        seed = abs(hash(t)) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.normal(size=d).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        out[i] = v
    return out


@pytest.fixture
def small_candidates_csv(tmp_path):
    import json

    import pandas as pd

    rows = []
    for i in range(15):
        rows.append({
            "candidate_id": f"PRE_{i:03d}",
            "headline": "ML Engineer",
            "summary": "Built ML pipelines at scale.",
            "current_title": "ML Engineer",
            "skills": json.dumps([{"name": "python", "endorsements": 5, "duration_months": 12}]),
            "career_history": json.dumps([{
                "title": "ML Engineer", "company": "Acme",
                "is_current": True, "duration_months": 24,
            }]),
            "education": json.dumps([{"degree": "B.Tech", "field_of_study": "CS"}]),
            "location": "Pune",
            "years_of_experience": 5.0,
            "behavioral_signals": json.dumps({"recruiter_response_rate": 0.5}),
        })
    path = tmp_path / "candidates.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_precompute_pipeline_produces_all_artifacts(monkeypatch, small_candidates_csv, tmp_path):
    monkeypatch.setattr(pe, "generate_embeddings", _fake_generate_embeddings)

    output_dir = tmp_path / "artifacts"
    monkeypatch.setattr(sys, "argv", [
        "precompute_embeddings.py",
        "--input", str(small_candidates_csv),
        "--output", str(output_dir),
        "--jd", str(ROOT / "config" / "jd_requirements.yaml"),
    ])

    exit_code = pe.main()
    assert exit_code == 0

    expected_files = [
        "embeddings.npy", "candidate_ids.npy", "skill_embeddings.npy",
        "jd_embedding.npy", "jd_skill_embedding.npy",
        "coherence_scores.parquet", "honeypot_flags.parquet", "bm25_index.pkl",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists(), f"Missing artifact: {filename}"


def test_precompute_artifacts_have_consistent_shapes(monkeypatch, small_candidates_csv, tmp_path):
    monkeypatch.setattr(pe, "generate_embeddings", _fake_generate_embeddings)

    output_dir = tmp_path / "artifacts"
    monkeypatch.setattr(sys, "argv", [
        "precompute_embeddings.py",
        "--input", str(small_candidates_csv),
        "--output", str(output_dir),
        "--jd", str(ROOT / "config" / "jd_requirements.yaml"),
    ])
    pe.main()

    embeddings = np.load(output_dir / "embeddings.npy")
    candidate_ids = np.load(output_dir / "candidate_ids.npy")
    skill_embeddings = np.load(output_dir / "skill_embeddings.npy")

    assert embeddings.shape == (15, 384)
    assert candidate_ids.shape == (15,)
    assert skill_embeddings.shape == (15, 384)


def test_precompute_then_rank_end_to_end(monkeypatch, small_candidates_csv, tmp_path):
    """Full offline -> online pipeline, mocking only the network call."""
    monkeypatch.setattr(pe, "generate_embeddings", _fake_generate_embeddings)

    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(sys, "argv", [
        "precompute_embeddings.py",
        "--input", str(small_candidates_csv),
        "--output", str(artifacts_dir),
        "--jd", str(ROOT / "config" / "jd_requirements.yaml"),
    ])
    assert pe.main() == 0

    from src.data_loader import CandidateLoader
    from src.jd_parser import JDParser
    from src.ranker import RankingPipeline

    candidates = CandidateLoader(small_candidates_csv).load()
    jd = JDParser(ROOT / "config" / "jd_requirements.yaml").parse()
    pipeline = RankingPipeline(
        candidates=candidates, jd=jd, artifacts_dir=artifacts_dir,
        weights_config=ROOT / "config" / "weights.yaml", top_k=10,
    )
    df = pipeline.rank()

    assert len(df) <= 10
    assert list(df.columns) == ["candidate_id", "rank", "score", "reasoning"]
    assert df["score"].is_monotonic_decreasing
