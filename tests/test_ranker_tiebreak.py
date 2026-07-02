from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring import Scorer

ROOT = Path(__file__).parent.parent


def make_score_inputs(n: int, seed: int = 0) -> dict:
    """Generate random score arrays for n candidates."""
    rng = np.random.default_rng(seed)
    return {
        "career_match": rng.uniform(0, 1, n),
        "skill_match": rng.uniform(0, 1, n),
        "experience_fit": rng.uniform(0, 1, n),
        "location_fit": rng.uniform(0, 1, n),
        "education_fit": rng.uniform(0, 1, n),
        "rule_adjustments": rng.uniform(0, 1, n),
        "coherence_factors": rng.uniform(0.7, 1.1, n),
        "behavioral_modifiers": rng.uniform(0.55, 1.15, n),
        "honeypot_gates": np.ones(n, dtype=np.float64),
    }


class TestScorer:
    def setup_method(self):
        self.scorer = Scorer(config_path=ROOT / "config" / "weights.yaml")

    def test_weights_loaded_from_config(self):
        assert self.scorer.weights["career_match"] == pytest.approx(0.30)
        assert self.scorer.weights["skill_match"] == pytest.approx(0.25)
        assert sum(self.scorer.weights.values()) == pytest.approx(1.0)

    def test_unnormalized_weights_get_normalized(self):
        scorer = Scorer(weights={
            "career_match": 1, "skill_match": 1, "experience_fit": 1,
            "location_fit": 1, "education_fit": 1, "rule_adjustments": 1,
        })
        assert sum(scorer.weights.values()) == pytest.approx(1.0)
        assert scorer.weights["career_match"] == pytest.approx(1 / 6)

    def test_base_scores_in_valid_range(self):
        inp = make_score_inputs(100)
        base = self.scorer.compute_base_scores(
            inp["career_match"], inp["skill_match"], inp["experience_fit"],
            inp["location_fit"], inp["education_fit"], inp["rule_adjustments"],
        )
        assert base.shape == (100,)
        assert np.all(base >= 0.0) and np.all(base <= 1.0)

    def test_base_score_is_weighted_sum(self):
        career = np.array([1.0, 0.0])
        zeros = np.zeros(2)
        base = self.scorer.compute_base_scores(career, zeros, zeros, zeros, zeros, zeros)
        assert base[0] == pytest.approx(self.scorer.weights["career_match"])
        assert base[1] == pytest.approx(0.0)

    def test_honeypot_gate_zeroes_final_score(self):
        inp = make_score_inputs(5)
        base = self.scorer.compute_base_scores(
            inp["career_match"], inp["skill_match"], inp["experience_fit"],
            inp["location_fit"], inp["education_fit"], inp["rule_adjustments"],
        )
        gates = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
        final = self.scorer.compute_final_scores(
            base, inp["coherence_factors"], inp["behavioral_modifiers"], gates
        )
        assert final[2] == pytest.approx(0.0)
        assert final[4] == pytest.approx(0.0)
        assert final[0] > 0.0

    def test_behavioral_modifier_affects_final_score(self):
        base = np.array([0.5, 0.5])
        coh = np.array([1.0, 1.0])
        gate = np.array([1.0, 1.0])
        beh = np.array([1.1, 0.6])
        final = self.scorer.compute_final_scores(base, coh, beh, gate)
        assert final[0] > final[1]

    def test_get_score_breakdown_contains_all_components(self):
        inp = make_score_inputs(3)
        base = self.scorer.compute_base_scores(
            inp["career_match"], inp["skill_match"], inp["experience_fit"],
            inp["location_fit"], inp["education_fit"], inp["rule_adjustments"],
        )
        breakdown = self.scorer.get_score_breakdown(
            index=0,
            career_match=inp["career_match"], skill_match=inp["skill_match"],
            experience_fit=inp["experience_fit"], location_fit=inp["location_fit"],
            education_fit=inp["education_fit"], rule_adjustments=inp["rule_adjustments"],
            coherence_factors=inp["coherence_factors"],
            behavioral_modifiers=inp["behavioral_modifiers"],
            honeypot_gates=inp["honeypot_gates"],
        )
        expected_keys = {
            "career_match", "skill_match", "experience_fit", "location_fit",
            "education_fit", "rule_adjustments", "base_score",
            "coherence_factor", "behavioral_modifier", "honeypot_gate", "final_score",
        }
        assert expected_keys <= set(breakdown.keys())
        assert breakdown["final_score"] == pytest.approx(
            base[0] * inp["coherence_factors"][0]
            * inp["behavioral_modifiers"][0] * inp["honeypot_gates"][0]
        )


class TestTieBreaking:
    """Verify the ranking sort order: score descending, ties broken by
    candidate_id ascending — exactly what RankingPipeline.rank() does
    via np.lexsort((candidate_id_array, -final_scores)).
    """

    def test_tie_break_ascending_id(self):
        ids = np.array(["CAND_002", "CAND_001", "CAND_003"])
        scores = np.array([0.75, 0.75, 0.75])
        sorted_idx = np.lexsort((ids, -scores))
        assert ids[sorted_idx].tolist() == ["CAND_001", "CAND_002", "CAND_003"]

    def test_tie_break_mixed_with_clear_winner(self):
        ids = np.array(["CAND_003", "CAND_001", "CAND_002"])
        scores = np.array([0.90, 0.90, 0.80])
        sorted_idx = np.lexsort((ids, -scores))
        sorted_ids = ids[sorted_idx].tolist()
        assert sorted_ids == ["CAND_001", "CAND_003", "CAND_002"]

    def test_no_tie_respects_score_order(self):
        ids = np.array(["CAND_003", "CAND_001", "CAND_002"])
        scores = np.array([0.70, 0.90, 0.80])
        sorted_idx = np.lexsort((ids, -scores))
        sorted_scores = scores[sorted_idx].tolist()
        assert sorted_scores == pytest.approx([0.90, 0.80, 0.70])

    def test_honeypot_gated_candidates_sink_to_bottom(self):
        n = 10
        ids = np.array([f"CAND_{i:03d}" for i in range(n)])
        scorer = Scorer(config_path=ROOT / "config" / "weights.yaml")
        inp = make_score_inputs(n, seed=99)
        inp["honeypot_gates"] = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)

        base = scorer.compute_base_scores(
            inp["career_match"], inp["skill_match"], inp["experience_fit"],
            inp["location_fit"], inp["education_fit"], inp["rule_adjustments"],
        )
        final = scorer.compute_final_scores(
            base, inp["coherence_factors"], inp["behavioral_modifiers"], inp["honeypot_gates"]
        )
        sorted_idx = np.lexsort((ids, -final))
        top3_ids = ids[sorted_idx[:3]].tolist()

        for excluded_id in ["CAND_000", "CAND_001", "CAND_002"]:
            assert excluded_id not in top3_ids
