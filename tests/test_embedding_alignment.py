"""
Tests for the embedding alignment fix in RankingPipeline._init_components().

The precomputed artifacts store embeddings in the order they were produced.
If the candidate list passed to RankingPipeline is in a different order
(e.g., the input CSV was filtered or sorted between precompute and rank),
each candidate must still receive the correct embedding — not whoever
happened to sit at the same list position during precompute.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import BehavioralSignals, Candidate, CareerEntry, SkillEntry
from src.jd_parser import JDParser
from src.ranker import RankingPipeline


def _make_candidate(candidate_id: str) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        current_title="ML Engineer",
        skills=[SkillEntry(name="python")],
        career_history=[CareerEntry(is_current=True, duration_months=24)],
        location="Pune",
        years_of_experience=5.0,
        behavioral_signals=BehavioralSignals(),
    )


def _build_artifacts(
    artifacts_dir: Path,
    candidate_ids: list[str],
    distinct_embeddings: bool = False,
    dim: int = 384,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Write artifacts for the given ids.

    When distinct_embeddings=True, each candidate gets a unique,
    deterministic embedding (seeded from its id) so we can verify
    the correct embedding was assigned to the correct candidate.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = len(candidate_ids)

    if distinct_embeddings:
        embeddings = np.zeros((n, dim), dtype=np.float32)
        for i, cid in enumerate(candidate_ids):
            v = np.random.default_rng(abs(hash(cid)) % 2**32).normal(size=dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            embeddings[i] = v
    else:
        embeddings = rng.normal(size=(n, dim)).astype(np.float32)

    skill_embeddings = rng.normal(size=(n, dim)).astype(np.float32)
    jd_embedding = rng.normal(size=(dim,)).astype(np.float32)
    jd_skill_embedding = rng.normal(size=(dim,)).astype(np.float32)

    np.save(artifacts_dir / "embeddings.npy", embeddings)
    np.save(artifacts_dir / "candidate_ids.npy", np.array(candidate_ids))
    np.save(artifacts_dir / "skill_embeddings.npy", skill_embeddings)
    np.save(artifacts_dir / "jd_embedding.npy", jd_embedding)
    np.save(artifacts_dir / "jd_skill_embedding.npy", jd_skill_embedding)

    pd.DataFrame({
        "candidate_id": candidate_ids,
        "coherence_score": rng.uniform(0.3, 1.0, size=n),
    }).to_parquet(artifacts_dir / "coherence_scores.parquet", index=False)

    pd.DataFrame({
        "candidate_id": candidate_ids,
        "is_honeypot": [False] * n,
        "gate_value": [1.0] * n,
        "strong_count": [0] * n,
        "medium_count": [0] * n,
        "weak_count": [0] * n,
    }).to_parquet(artifacts_dir / "honeypot_flags.parquet", index=False)

    return {cid: embeddings[i] for i, cid in enumerate(candidate_ids)}


class TestEmbeddingAlignment:
    @pytest.fixture
    def jd(self):
        return JDParser(ROOT / "config" / "jd_requirements.yaml").parse()

    def _make_pipeline(self, candidates, artifacts_dir, jd):
        return RankingPipeline(
            candidates=candidates,
            jd=jd,
            artifacts_dir=artifacts_dir,
            weights_config=ROOT / "config" / "weights.yaml",
            top_k=len(candidates),
        )

    def test_reversed_candidate_list_gets_correct_embeddings(self, tmp_path, jd):
        """If candidates are passed in reversed order relative to how they
        were precomputed, each candidate must still get its own embedding.
        """
        ids = [f"CAND_{i:03d}" for i in range(10)]
        artifacts_dir = tmp_path / "artifacts"
        id_to_embedding = _build_artifacts(artifacts_dir, ids, distinct_embeddings=True)

        # Reverse the candidate list — completely different order from precompute
        candidates_reversed = [_make_candidate(cid) for cid in reversed(ids)]
        pipeline = self._make_pipeline(candidates_reversed, artifacts_dir, jd)

        # pipeline.embeddings[i] must equal id_to_embedding[candidates_reversed[i].candidate_id]
        for i, cand in enumerate(candidates_reversed):
            expected = id_to_embedding[cand.candidate_id]
            actual = pipeline.embeddings[i]
            assert np.allclose(actual, expected, atol=1e-6), (
                f"Candidate {cand.candidate_id} at row {i} got wrong embedding after reversal."
            )

    def test_filtered_candidate_list_gets_correct_embeddings(self, tmp_path, jd):
        """A subset of the precomputed candidates must each get the right embedding."""
        all_ids = [f"CAND_{i:03d}" for i in range(20)]
        artifacts_dir = tmp_path / "artifacts"
        id_to_embedding = _build_artifacts(artifacts_dir, all_ids, distinct_embeddings=True)

        # Only rank the even-indexed candidates
        subset_ids = [f"CAND_{i:03d}" for i in range(0, 20, 2)]
        candidates_subset = [_make_candidate(cid) for cid in subset_ids]
        pipeline = self._make_pipeline(candidates_subset, artifacts_dir, jd)

        assert pipeline.embeddings.shape == (len(subset_ids), 384)
        for i, cand in enumerate(candidates_subset):
            expected = id_to_embedding[cand.candidate_id]
            actual = pipeline.embeddings[i]
            assert np.allclose(actual, expected, atol=1e-6), (
                f"Candidate {cand.candidate_id} at row {i} got wrong embedding after filtering."
            )

    def test_unknown_candidate_gets_zero_embedding(self, tmp_path, jd):
        """A candidate whose id is not in the precomputed artifacts should get
        a zero-vector embedding, not crash or silently steal another's embedding.
        """
        ids = [f"CAND_{i:03d}" for i in range(5)]
        artifacts_dir = tmp_path / "artifacts"
        _build_artifacts(artifacts_dir, ids, distinct_embeddings=True)

        # Mix a known candidate with an unknown one
        candidates = [
            _make_candidate("CAND_000"),
            _make_candidate("CAND_UNKNOWN"),
        ]
        pipeline = self._make_pipeline(candidates, artifacts_dir, jd)

        assert not np.allclose(pipeline.embeddings[0], 0.0), "Known candidate should have non-zero embedding."
        assert np.allclose(pipeline.embeddings[1], 0.0), "Unknown candidate must get a zero-vector."

    def test_alignment_does_not_change_scores(self, tmp_path, jd):
        """With aligned embeddings the pipeline must still rank without error
        and produce a valid output DataFrame.
        """
        ids = [f"CAND_{i:03d}" for i in range(30)]
        artifacts_dir = tmp_path / "artifacts"
        _build_artifacts(artifacts_dir, ids, seed=77)

        # Shuffle the candidate order
        shuffled = list(reversed(ids))
        candidates = [_make_candidate(cid) for cid in shuffled]
        df = self._make_pipeline(candidates, artifacts_dir, jd).rank()

        assert list(df.columns) == ["candidate_id", "rank", "score", "reasoning"]
        assert df["score"].is_monotonic_decreasing
        assert df["candidate_id"].nunique() == len(df)
        assert list(df["rank"]) == list(range(1, len(df) + 1))
