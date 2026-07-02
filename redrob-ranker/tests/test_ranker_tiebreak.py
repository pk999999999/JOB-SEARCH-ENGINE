
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.data_loader import Candidate
from src.features.experience_fit import ExperienceFitScorer
from src.scoring import FinalScorer


def load_weights() -> dict:
    with open("config/weights.yaml") as f:
        return yaml.safe_load(f)


def make_score_inputs(n: int, seed: int = 0) -> dict:
    """Generate random score arrays for n candidates."""
    rng = np.random.default_rng(seed)
    return {
        "career_scores": rng.uniform(0, 1, n).astype(np.float32),
        "skill_scores": rng.uniform(0, 1, n).astype(np.float32),
        "experience_scores": rng.uniform(0, 1, n).astype(np.float32),
        "location_scores": rng.uniform(0, 1, n).astype(np.float32),
        "education_scores": rng.uniform(0, 1, n).astype(np.float32),
        "rule_deltas": rng.uniform(-0.2, 0.2, n).astype(np.float32),
        "coherence_scores": rng.uniform(0.4, 1.0, n).astype(np.float32),
        "coherence_factors": rng.uniform(0.8, 1.1, n).astype(np.float32),
        "behavioral_modifiers": rng.uniform(0.55, 1.15, n).astype(np.float32),
        "honeypot_gates": np.ones(n, dtype=np.float32),
    }


class TestFinalScorer:
    def setup_method(self):
        self.scorer = FinalScorer(load_weights())

    def test_base_scores_in_range(self):
        inp = make_score_inputs(100)
        base = self.scorer.compute_base_scores(
            inp["career_scores"], inp["skill_scores"],
            inp["experience_scores"], inp["location_scores"],
            inp["education_scores"], inp["rule_deltas"],
        )
        assert base.shape == (100,)
        assert np.all(base >= 0.0) and np.all(base <= 1.0)

    def test_honeypot_zeroes_score(self):
        inp = make_score_inputs(5)
        base = self.scorer.compute_base_scores(
            inp["career_scores"], inp["skill_scores"],
            inp["experience_scores"], inp["location_scores"],
            inp["education_scores"], inp["rule_deltas"],
        )
        gates = np.array([1, 1, 0, 1, 0], dtype=np.float32)
        final = self.scorer.compute_final_scores(
            base, inp["coherence_factors"], inp["behavioral_modifiers"], gates
        )
        assert final[2] == pytest.approx(0.0)
        assert final[4] == pytest.approx(0.0)
        assert final[0] > 0.0

    def test_behavioral_multiplier_effect(self):
        base = np.array([0.5, 0.5], dtype=np.float32)
        coh = np.array([1.0, 1.0], dtype=np.float32)
        gate = np.array([1.0, 1.0], dtype=np.float32)
        beh_high = np.array([1.1, 0.6], dtype=np.float32)
        final = self.scorer.compute_final_scores(base, coh, beh_high, gate)
        assert final[0] > final[1]

    def test_build_score_records_length(self):
        n = 20
        ids = [f"C{i:03d}" for i in range(n)]
        inp = make_score_inputs(n)
        records = self.scorer.build_score_records(
            candidate_ids=ids,
            **inp,
            rule_flags_list=None,
            honeypot_results=None,
        )
        assert len(records) == n
        for r in records:
            assert 0.0 <= r.final_score <= 1.5  # coherence can push above 1 before clip


class TestTieBreaking:
    """Verify tie-breaking: same score → ascending candidate_id."""

    def test_tie_break_ascending_id(self):
        # Simulate two candidates with identical scores
        ids = np.array(["CAND_002", "CAND_001", "CAND_003"])
        scores = np.array([0.75, 0.75, 0.75], dtype=np.float32)
        sorted_idx = np.lexsort((ids, -scores))
        sorted_ids = ids[sorted_idx].tolist()
        assert sorted_ids == ["CAND_001", "CAND_002", "CAND_003"]

    def test_tie_break_mixed(self):
        ids = np.array(["CAND_003", "CAND_001", "CAND_002"])
        scores = np.array([0.90, 0.90, 0.80], dtype=np.float32)
        sorted_idx = np.lexsort((ids, -scores))
        sorted_ids = ids[sorted_idx].tolist()
        # CAND_001 and CAND_003 tie at 0.90; CAND_001 < CAND_003 alphabetically
        assert sorted_ids[0] == "CAND_001"
        assert sorted_ids[1] == "CAND_003"
        assert sorted_ids[2] == "CAND_002"

    def test_no_tie_respects_score_order(self):
        ids = np.array(["CAND_003", "CAND_001", "CAND_002"])
        scores = np.array([0.70, 0.90, 0.80], dtype=np.float32)
        sorted_idx = np.lexsort((ids, -scores))
        sorted_scores = scores[sorted_idx].tolist()
        assert sorted_scores == pytest.approx([0.90, 0.80, 0.70])

    def test_honeypot_excluded_never_in_top(self):
        """Excluded candidates (gate=0) should score 0 and sink to bottom."""
        n = 10
        ids = [f"CAND_{i:03d}" for i in range(n)]
        inp = make_score_inputs(n, seed=99)
        # Exclude first 3
        inp["honeypot_gates"] = np.array(
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32
        )
        scorer = FinalScorer(load_weights())
        base = scorer.compute_base_scores(
            inp["career_scores"], inp["skill_scores"],
            inp["experience_scores"], inp["location_scores"],
            inp["education_scores"], inp["rule_deltas"],
        )
        final = scorer.compute_final_scores(
            base, inp["coherence_factors"],
            inp["behavioral_modifiers"], inp["honeypot_gates"]
        )
        final_arr = np.array(final)
        ids_arr = np.array(ids)
        sorted_idx = np.lexsort((ids_arr, -final_arr))
        top3_ids = ids_arr[sorted_idx[:3]].tolist()
        # None of excluded (CAND_000, 001, 002) should be in top 3
        for excl in ["CAND_000", "CAND_001", "CAND_002"]:
            assert excl not in top3_ids


class TestExperienceScorerEdgeCases:
    def setup_method(self):
        self.scorer = ExperienceFitScorer(load_weights())

    def test_negative_experience_treated_as_zero(self):
        s = self.scorer.score(-5.0)
        assert s == self.scorer.score(0.0)

    def test_very_large_experience_near_zero(self):
        s = self.scorer.score(50.0)
        assert 0.0 <= s < 0.30

    def test_monotone_increase_to_ideal(self):
        """Score should be non-decreasing as we approach ideal range from below."""
        scores = [self.scorer.score(float(y)) for y in range(0, 6)]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1] + 0.01  # allow tiny floating point

    def test_vectorized_vs_scalar_consistency(self):
        yoe_values = np.array([0, 1, 3, 5, 7, 9, 12, 20], dtype=np.float32)
        batch_results = self.scorer.score_batch(yoe_values)
        for i, yoe in enumerate(yoe_values):
            scalar = self.scorer.score(float(yoe))
            assert batch_results[i] == pytest.approx(scalar, abs=1e-5)