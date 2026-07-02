from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.jd_parser import JDParser, JobDescription  # noqa: E402


@pytest.fixture(scope="session")
def weights_cfg() -> dict:
    """Raw contents of config/weights.yaml."""
    with open(ROOT / "config" / "weights.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def jd() -> JobDescription:
    """Parsed job description from config/jd_requirements.yaml."""
    return JDParser(ROOT / "config" / "jd_requirements.yaml").parse()


def build_fake_artifacts(
    artifacts_dir: Path,
    candidate_ids: list[str],
    embedding_dim: int = 384,
    seed: int = 0,
) -> None:
    """Write a complete, internally-consistent set of fake precomputed
    artifacts to `artifacts_dir`, in the exact format RankingPipeline
    expects. This lets us test the ranking stage end-to-end without
    needing network access to download a real embedding model.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = len(candidate_ids)

    embeddings = rng.normal(size=(n, embedding_dim)).astype(np.float32)
    skill_embeddings = rng.normal(size=(n, embedding_dim)).astype(np.float32)
    jd_embedding = rng.normal(size=(embedding_dim,)).astype(np.float32)
    jd_skill_embedding = rng.normal(size=(embedding_dim,)).astype(np.float32)

    np.save(artifacts_dir / "embeddings.npy", embeddings)
    np.save(artifacts_dir / "candidate_ids.npy", np.array(candidate_ids))
    np.save(artifacts_dir / "skill_embeddings.npy", skill_embeddings)
    np.save(artifacts_dir / "jd_embedding.npy", jd_embedding)
    np.save(artifacts_dir / "jd_skill_embedding.npy", jd_skill_embedding)

    coherence_df = pd.DataFrame({
        "candidate_id": candidate_ids,
        "coherence_score": rng.uniform(0.3, 1.0, size=n),
    })
    coherence_df.to_parquet(artifacts_dir / "coherence_scores.parquet", index=False)

    honeypot_df = pd.DataFrame({
        "candidate_id": candidate_ids,
        "is_honeypot": [False] * n,
        "gate_value": [1.0] * n,
        "strong_count": [0] * n,
        "medium_count": [0] * n,
        "weak_count": [0] * n,
    })
    honeypot_df.to_parquet(artifacts_dir / "honeypot_flags.parquet", index=False)

    # No bm25_index.pkl written on purpose — RankingPipeline must fall back
    # gracefully to zeros when it's absent (see _compute_bm25_scores).


@pytest.fixture
def fake_artifacts_factory(tmp_path):
    """Returns a callable that builds a fake artifacts directory for a
    given list of candidate ids and returns its path.
    """

    def _build(candidate_ids: list[str], seed: int = 0) -> Path:
        artifacts_dir = tmp_path / "artifacts"
        build_fake_artifacts(artifacts_dir, candidate_ids, seed=seed)
        return artifacts_dir

    return _build
