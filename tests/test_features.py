from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import Candidate, CareerEntry, EducationEntry, SkillEntry
from src.features.behavioral_modifier import BehavioralModifier
from src.features.coherence_score import CoherenceScorer
from src.features.disqualifiers import DisqualifierChecker
from src.features.education_fit import EducationFitter
from src.features.experience_fit import ExperienceFitter
from src.features.location_fit import LocationFitter
from src.features.skill_match import SkillMatcher
from src.features.title_career_match import CareerMatcher


# ---------------------------------------------------------------------------
# ExperienceFitter
# ---------------------------------------------------------------------------

class TestExperienceFitter:
    def setup_method(self):
        self.scorer = ExperienceFitter(ideal_min=5, ideal_max=9, sigma=3.0)

    def test_ideal_range_returns_one(self):
        for yoe in [5.0, 6.0, 7.0, 8.0, 9.0]:
            assert self.scorer.score_single(yoe) == pytest.approx(1.0, abs=1e-6)

    def test_zero_experience_decayed(self):
        # exp(-0.5 * ((0-5)/3)^2) ~= 0.2494
        assert self.scorer.score_single(0.0) == pytest.approx(0.2494, abs=1e-3)

    def test_decays_symmetrically_below_and_above(self):
        below = self.scorer.score_single(3.0)
        above = self.scorer.score_single(11.0)  # same distance from boundary as 3.0 is from 5.0
        assert below == pytest.approx(above, abs=1e-6)

    def test_monotonic_decay_above_ideal(self):
        s9 = self.scorer.score_single(9.0)
        s12 = self.scorer.score_single(12.0)
        s20 = self.scorer.score_single(20.0)
        assert s9 > s12 > s20

    def test_monotonic_decay_below_ideal(self):
        s0 = self.scorer.score_single(0.0)
        s3 = self.scorer.score_single(3.0)
        s5 = self.scorer.score_single(5.0)
        assert s0 < s3 < s5

    def test_batch_matches_single(self):
        yoe_values = np.array([0.0, 3.0, 5.0, 7.0, 9.0, 12.0, 20.0], dtype=np.float64)
        batch_results = self.scorer.score_batch(yoe_values)
        for i, yoe in enumerate(yoe_values):
            assert batch_results[i] == pytest.approx(
                self.scorer.score_single(float(yoe)), abs=1e-9
            )

    def test_scores_within_unit_interval(self):
        yoe_values = np.linspace(0, 40, 50)
        scores = self.scorer.score_batch(yoe_values)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


# ---------------------------------------------------------------------------
# LocationFitter
# ---------------------------------------------------------------------------

class TestLocationFitter:
    def setup_method(self):
        self.scorer = LocationFitter(
            preferred=["Pune", "Noida"],
            good=["Mumbai", "Hyderabad", "Delhi NCR", "Delhi", "New Delhi", "Gurgaon", "Gurugram"],
            acceptable_country="India",
        )

    def test_preferred_city_scores_one(self):
        assert self.scorer.score_single("Pune") == pytest.approx(1.0)
        assert self.scorer.score_single("Noida") == pytest.approx(1.0)

    def test_good_city_scores_point_eight(self):
        assert self.scorer.score_single("Mumbai") == pytest.approx(0.8)
        assert self.scorer.score_single("Hyderabad") == pytest.approx(0.8)

    def test_other_indian_city_scores_point_five(self):
        assert self.scorer.score_single("Bengaluru") == pytest.approx(0.5)
        assert self.scorer.score_single("Chennai") == pytest.approx(0.5)

    def test_outside_country_scores_point_two(self):
        assert self.scorer.score_single("San Francisco, USA") == pytest.approx(0.2)
        assert self.scorer.score_single("London, UK") == pytest.approx(0.2)

    def test_empty_location_treated_as_outside(self):
        assert self.scorer.score_single("") == pytest.approx(0.2)

    def test_case_insensitive(self):
        assert self.scorer.score_single("pune") == self.scorer.score_single("PUNE")

    def test_batch_length_matches_input(self):
        locs = ["Pune", "Mumbai", "Bengaluru", "", "San Francisco, USA"]
        result = self.scorer.score_batch(locs)
        assert len(result) == len(locs)
        assert result.tolist() == pytest.approx([1.0, 0.8, 0.5, 0.2, 0.2])


# ---------------------------------------------------------------------------
# SkillMatcher
# ---------------------------------------------------------------------------

class TestSkillMatcher:
    def setup_method(self):
        self.required = ["python", "machine learning", "deep learning", "nlp", "pytorch"]
        self.preferred = ["docker", "kubernetes", "aws"]
        self.matcher = SkillMatcher(
            required_skills=self.required,
            preferred_skills=self.preferred,
        )

    def test_full_required_match_scores_point_seven(self):
        assert self.matcher.lexical_overlap(self.required) == pytest.approx(0.7)

    def test_no_skills_scores_zero(self):
        assert self.matcher.lexical_overlap([]) == pytest.approx(0.0)

    def test_partial_required_match(self):
        score = self.matcher.lexical_overlap(["python", "machine learning"])
        assert score == pytest.approx(0.28)

    def test_required_weighted_more_than_preferred(self):
        required_only = self.matcher.lexical_overlap(self.required)
        preferred_only = self.matcher.lexical_overlap(self.preferred)
        assert required_only > preferred_only

    def test_trust_multiplier_full_trust(self):
        trust = self.matcher.compute_trust_multiplier(
            np.array([10.0]), np.array([12.0])
        )
        assert trust[0] == pytest.approx(1.0)

    def test_trust_multiplier_zero_trust(self):
        trust = self.matcher.compute_trust_multiplier(
            np.array([0.0]), np.array([0.0])
        )
        assert trust[0] == pytest.approx(0.0)

    def test_bm25_normalization_identical_scores(self):
        result = self.matcher.score_batch_bm25(np.array([5.0, 5.0, 5.0]))
        assert np.allclose(result, 0.5)

    def test_bm25_normalization_range(self):
        result = self.matcher.score_batch_bm25(np.array([0.0, 5.0, 10.0]))
        assert result.tolist() == pytest.approx([0.0, 0.5, 1.0])

    def test_bm25_all_zero_returns_zero(self):
        result = self.matcher.score_batch_bm25(np.array([0.0, 0.0]))
        assert np.allclose(result, 0.0)

    def test_embedding_score_with_no_jd_embedding_warns_and_zeros(self):
        matcher = SkillMatcher(required_skills=self.required, preferred_skills=self.preferred)
        emb = np.random.randn(3, 384).astype(np.float32)
        result = matcher.score_batch_embedding(emb)
        assert np.allclose(result, 0.0)

    def test_score_batch_lexical_shape(self):
        skill_lists = [["python", "pytorch"], [], self.required]
        result = self.matcher.score_batch_lexical(skill_lists)
        assert result.shape == (3,)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# EducationFitter
# ---------------------------------------------------------------------------

class TestEducationFitter:
    def setup_method(self):
        self.scorer = EducationFitter(
            preferred_fields=["computer science", "artificial intelligence"]
        )

    def test_phd_in_relevant_field_scores_highest(self):
        edu = [EducationEntry(degree="PhD", field_of_study="Computer Science")]
        assert self.scorer.score_candidate(edu) == pytest.approx(1.0)

    def test_mtech_scores_higher_than_btech(self):
        btech = [EducationEntry(degree="B.Tech", field_of_study="Computer Science")]
        mtech = [EducationEntry(degree="M.Tech", field_of_study="Computer Science")]
        assert self.scorer.score_candidate(btech) < self.scorer.score_candidate(mtech)

    def test_missing_education_returns_default_floor(self):
        assert self.scorer.score_candidate([]) == pytest.approx(0.3)

    def test_best_entry_is_used(self):
        edu = [
            EducationEntry(degree="Diploma", field_of_study="Unrelated"),
            EducationEntry(degree="PhD", field_of_study="Computer Science"),
        ]
        assert self.scorer.score_candidate(edu) == pytest.approx(1.0)

    def test_batch_shape_matches_input(self):
        batch = [[EducationEntry(degree="B.Tech")] for _ in range(7)]
        result = self.scorer.score_batch(batch)
        assert result.shape == (7,)


# ---------------------------------------------------------------------------
# BehavioralModifier
# ---------------------------------------------------------------------------

class TestBehavioralModifier:
    def setup_method(self):
        from datetime import datetime
        from src.data_loader import BehavioralSignals

        self.BehavioralSignals = BehavioralSignals
        self.reference_date = datetime(2026, 6, 22)
        self.modifier = BehavioralModifier(reference_date=self.reference_date)

    def test_strong_signals_score_near_max(self):
        signals = self.BehavioralSignals(
            last_active_date="2026-06-20",
            recruiter_response_rate=1.0,
            avg_response_time_hours=1.0,
            interview_completion_rate=1.0,
            offer_acceptance_rate=1.0,
            verified_email=True,
            verified_phone=True,
            linkedin_connected=True,
        )
        mult = self.modifier.score_single(signals)
        assert mult > 1.10
        assert mult <= self.modifier.max_multiplier

    def test_weak_signals_score_near_min(self):
        signals = self.BehavioralSignals(
            last_active_date="2020-01-01",
            recruiter_response_rate=0.0,
            avg_response_time_hours=300.0,
            interview_completion_rate=0.0,
            offer_acceptance_rate=0.0,
            verified_email=False,
            verified_phone=False,
            linkedin_connected=False,
        )
        mult = self.modifier.score_single(signals)
        assert mult < 0.60
        assert mult >= self.modifier.min_multiplier

    def test_default_signals_within_bounds(self):
        mult = self.modifier.score_single(self.BehavioralSignals())
        assert self.modifier.min_multiplier <= mult <= self.modifier.max_multiplier

    def test_strong_outranks_weak(self):
        strong = self.BehavioralSignals(
            last_active_date="2026-06-20", recruiter_response_rate=1.0,
            avg_response_time_hours=1.0,
        )
        weak = self.BehavioralSignals(
            last_active_date="2020-01-01", recruiter_response_rate=0.0,
            avg_response_time_hours=300.0,
        )
        assert self.modifier.score_single(strong) > self.modifier.score_single(weak)

    def test_batch_matches_single(self):
        signals_list = [self.BehavioralSignals() for _ in range(10)]
        batch = self.modifier.score_batch(signals_list)
        for i in range(10):
            assert batch[i] == pytest.approx(self.modifier.score_single(signals_list[i]))


# ---------------------------------------------------------------------------
# CoherenceScorer
# ---------------------------------------------------------------------------

class TestCoherenceScorer:
    def setup_method(self):
        self.scorer = CoherenceScorer(min_factor=0.70, max_factor=1.10)

    def test_identical_embeddings_have_similarity_one(self):
        v = np.random.RandomState(0).randn(1, 384).astype(np.float32)
        v = v / np.linalg.norm(v)
        sim = self.scorer.compute_pairwise_similarity(v, v)
        assert sim[0] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_embeddings_have_similarity_zero(self):
        a = np.zeros((1, 384), dtype=np.float32)
        a[0, 0] = 1.0
        b = np.zeros((1, 384), dtype=np.float32)
        b[0, 1] = 1.0
        sim = self.scorer.compute_pairwise_similarity(a, b)
        assert sim[0] == pytest.approx(0.0, abs=1e-5)

    def test_coherence_to_factor_bounds(self):
        assert self.scorer.coherence_to_factor(np.array([0.0]))[0] == pytest.approx(0.70)
        assert self.scorer.coherence_to_factor(np.array([1.0]))[0] == pytest.approx(1.10)
        assert self.scorer.coherence_to_factor(np.array([0.5]))[0] == pytest.approx(0.90)

    def test_score_batch_shape(self):
        n = 20
        narrative = np.random.randn(n, 384).astype(np.float32)
        career = np.random.randn(n, 384).astype(np.float32)
        skills = np.random.randn(n, 384).astype(np.float32)
        result = self.scorer.score_batch(narrative, career, skills)
        assert result.shape == (n,)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_score_from_precomputed_matches_coherence_to_factor(self):
        scores = np.array([0.2, 0.5, 0.9])
        assert np.allclose(
            self.scorer.score_from_precomputed(scores),
            self.scorer.coherence_to_factor(scores),
        )


# ---------------------------------------------------------------------------
# DisqualifierChecker
# ---------------------------------------------------------------------------

class TestDisqualifierChecker:
    def setup_method(self):
        self.required = ["python", "machine learning", "deep learning", "nlp", "pytorch"]
        self.checker = DisqualifierChecker(required_skills=self.required)

    def _make_candidate(self, **kwargs) -> Candidate:
        defaults = dict(candidate_id="X")
        defaults.update(kwargs)
        return Candidate(**defaults)

    def test_strong_candidate_scores_well(self):
        c = self._make_candidate(
            current_title="Senior AI Engineer",
            skills=[SkillEntry(name="python"), SkillEntry(name="machine learning"),
                    SkillEntry(name="deep learning")],
            career_history=[CareerEntry(title="AI Engineer", company="X", is_current=True, duration_months=24)],
        )
        assert self.checker.score_candidate(c) == pytest.approx(0.7)

    def test_unrelated_candidate_scores_zero(self):
        c = self._make_candidate(
            current_title="Sales Executive",
            skills=[SkillEntry(name="excel")],
            career_history=[],
        )
        assert self.checker.score_candidate(c) == pytest.approx(0.0)

    def test_missing_critical_skills_penalized(self):
        with_python = self._make_candidate(
            skills=[SkillEntry(name="python"), SkillEntry(name="machine learning")],
            career_history=[CareerEntry(is_current=True, duration_months=12)],
        )
        without_python = self._make_candidate(
            skills=[SkillEntry(name="nlp")],
            career_history=[CareerEntry(is_current=True, duration_months=12)],
        )
        assert self.checker.score_candidate(with_python) > self.checker.score_candidate(without_python)

    def test_relevant_title_gives_bonus(self):
        with_title = self._make_candidate(
            current_title="Machine Learning Engineer",
            career_history=[CareerEntry(is_current=True, duration_months=12)],
        )
        without_title = self._make_candidate(
            current_title="Office Coordinator",
            career_history=[CareerEntry(is_current=True, duration_months=12)],
        )
        assert self.checker.score_candidate(with_title) > self.checker.score_candidate(without_title)

    def test_no_career_history_penalized(self):
        shared_skills = [SkillEntry(name="python"), SkillEntry(name="machine learning")]
        no_history = self._make_candidate(skills=shared_skills, career_history=[])
        with_history = self._make_candidate(
            skills=shared_skills,
            career_history=[CareerEntry(is_current=True, duration_months=12)],
        )
        assert self.checker.score_candidate(no_history) < self.checker.score_candidate(with_history)

    def test_scores_clamped_to_unit_interval(self):
        c = self._make_candidate(
            current_title="Senior AI Engineer Machine Learning",
            skills=[SkillEntry(name=s) for s in self.required],
            career_history=[CareerEntry(is_current=True, duration_months=24)],
        )
        score = self.checker.score_candidate(c)
        assert 0.0 <= score <= 1.0

    def test_batch_shape_matches_input(self):
        candidates = [self._make_candidate(candidate_id=f"C{i}") for i in range(5)]
        result = self.checker.score_batch(candidates)
        assert result.shape == (5,)


# ---------------------------------------------------------------------------
# CareerMatcher
# ---------------------------------------------------------------------------

class TestCareerMatcher:
    def test_identical_embedding_scores_one(self):
        jd_emb = np.random.RandomState(1).randn(384).astype(np.float32)
        matcher = CareerMatcher(jd_embedding=jd_emb)
        score = matcher.score_single(jd_emb)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_no_jd_embedding_returns_zeros(self):
        matcher = CareerMatcher(jd_embedding=None)
        result = matcher.score_batch(np.random.randn(5, 384).astype(np.float32))
        assert np.allclose(result, 0.0)

    def test_negative_similarity_clamped_to_zero(self):
        jd_emb = np.zeros(384, dtype=np.float32)
        jd_emb[0] = 1.0
        opposite = np.zeros(384, dtype=np.float32)
        opposite[0] = -1.0
        matcher = CareerMatcher(jd_embedding=jd_emb)
        score = matcher.score_single(opposite)
        assert score == pytest.approx(0.0)

    def test_batch_shape(self):
        jd_emb = np.random.randn(384).astype(np.float32)
        matcher = CareerMatcher(jd_embedding=jd_emb)
        result = matcher.score_batch(np.random.randn(10, 384).astype(np.float32))
        assert result.shape == (10,)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
