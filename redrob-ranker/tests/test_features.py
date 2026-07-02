from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.features.behavioral_modifier import BehavioralModifier, _active_score, _response_time_score
from src.features.coherence_score import CoherenceScorer
from src.features.disqualifiers import (
    RuleAdjustmentScorer,
    _average_tenure_months,
    _has_long_gap,
    _parse_date,
)
from src.features.education_fit import EducationFitScorer
from src.features.experience_fit import ExperienceFitScorer, gaussian_experience_score
from src.features.location_fit import LocationFitScorer, _normalize_location
from src.features.skill_match import (
    SkillMatcher,
    lexical_skill_overlap,
    embedding_skill_score,
)
from src.features.title_career_match import CareerMatcher, _title_match_score
from src.jd_parser import JobDescription


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def load_weights() -> dict:
    with open("config/weights.yaml") as f:
        return yaml.safe_load(f)


def make_jd() -> JobDescription:
    from src.jd_parser import JDParser
    return JDParser("config/jd_requirements.yaml").parse()


def make_skills(names: list[str], endorsements: int = 5,
                duration_months: int = 12) -> list[dict]:
    return [
        {"name": n, "endorsements": endorsements, "duration_months": duration_months}
        for n in names
    ]


# ---------------------------------------------------------------------------
# ExperienceFit
# ---------------------------------------------------------------------------

class TestExperienceFit:
    def setup_method(self):
        self.scorer = ExperienceFitScorer(load_weights())

    def test_ideal_range_returns_one(self):
        for yoe in [5.0, 6.0, 7.0, 8.0, 9.0]:
            assert self.scorer.score(yoe) == pytest.approx(1.0, abs=1e-6)

    def test_zero_experience_heavily_penalized(self):
        assert self.scorer.score(0.0) < 0.20

    def test_very_senior_decays_gracefully(self):
        score_10 = self.scorer.score(10.0)
        score_15 = self.scorer.score(15.0)
        assert 0.0 < score_15 < score_10 < 1.0

    def test_below_ideal_decays(self):
        score_3 = self.scorer.score(3.0)
        score_5 = self.scorer.score(5.0)
        assert score_3 < score_5

    def test_batch_matches_single(self):
        yoe_vals = np.array([0.0, 3.0, 7.0, 12.0, 20.0], dtype=np.float32)
        batch = self.scorer.score_batch(yoe_vals)
        for i, yoe in enumerate(yoe_vals):
            assert batch[i] == pytest.approx(self.scorer.score(float(yoe)), abs=1e-5)

    def test_gaussian_decay_formula(self):
        s = gaussian_experience_score(5.0, 5, 9, sigma=2.5)
        assert s == pytest.approx(1.0)
        s_out = gaussian_experience_score(3.0, 5, 9, sigma=2.5)
        assert 0.0 < s_out < 1.0

    def test_batch_shape(self):
        arr = np.arange(0, 20, dtype=np.float32)
        result = self.scorer.score_batch(arr)
        assert result.shape == (20,)
        assert result.dtype == np.float32
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# LocationFit
# ---------------------------------------------------------------------------

class TestLocationFit:
    def setup_method(self):
        self.jd = make_jd()
        self.scorer = LocationFitScorer(self.jd, load_weights())

    def test_preferred_location(self):
        assert self.scorer.score("Pune") == pytest.approx(1.0)
        assert self.scorer.score("Noida") == pytest.approx(1.0)

    def test_good_location(self):
        s = self.scorer.score("Mumbai")
        assert 0.70 <= s <= 0.80

    def test_outside_india_penalized(self):
        assert self.scorer.score("San Francisco") < 0.20
        assert self.scorer.score("London, UK") < 0.20

    def test_empty_location(self):
        s = self.scorer.score("")
        assert 0.0 <= s <= 1.0

    def test_case_insensitive(self):
        assert self.scorer.score("pune") == self.scorer.score("PUNE")

    def test_normalize_location(self):
        assert _normalize_location("  Pune, MH ") == "pune  mh"

    def test_batch_length(self):
        locs = ["Pune", "Mumbai", "London", "", "Hyderabad"]
        result = self.scorer.score_batch(locs)
        assert len(result) == 5

    def test_bangalore_variants(self):
        s1 = self.scorer.score("Bangalore")
        s2 = self.scorer.score("Bengaluru")
        assert s1 > 0.0 and s2 > 0.0


# ---------------------------------------------------------------------------
# SkillMatch
# ---------------------------------------------------------------------------

class TestSkillMatch:
    def setup_method(self):
        self.jd = make_jd()
        self.matcher = SkillMatcher(
            jd=self.jd,
            weights=load_weights().get("skill_match", {}),
        )

    def test_perfect_required_match(self):
        skills = make_skills(self.jd.required_skills[:5])
        score = self.matcher.score(skills)
        assert score > 0.50

    def test_no_skills_returns_zero(self):
        score = self.matcher.score([])
        assert score == pytest.approx(0.0)

    def test_unrelated_skills_low_score(self):
        skills = make_skills(["Cooking", "Gardening", "Yoga"])
        score = self.matcher.score(skills)
        assert score < 0.20

    def test_lexical_with_trust(self):
        # High endorsements + duration should score higher
        high_trust = make_skills(["Python"], endorsements=10, duration_months=24)
        low_trust = make_skills(["Python"], endorsements=0, duration_months=0)
        s_high = lexical_skill_overlap(high_trust, self.jd, use_trust=True)
        s_low = lexical_skill_overlap(low_trust, self.jd, use_trust=True)
        # Both match Python but trust floors at 0.3 so low_trust still scores
        assert s_high >= s_low

    def test_embedding_similarity_same_vector(self):
        v = np.random.randn(384).astype(np.float32)
        v /= np.linalg.norm(v)
        score = embedding_skill_score(v, v)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_embedding_similarity_orthogonal(self):
        a = np.zeros(384, dtype=np.float32); a[0] = 1.0
        b = np.zeros(384, dtype=np.float32); b[1] = 1.0
        score = embedding_skill_score(a, b)
        assert score == pytest.approx(0.5, abs=1e-4)  # cos=0 → (0+1)/2=0.5

    def test_embedding_zero_vector(self):
        zero = np.zeros(384, dtype=np.float32)
        v = np.ones(384, dtype=np.float32)
        assert embedding_skill_score(zero, v) == pytest.approx(0.0)

    def test_batch_shape(self):
        batch = [make_skills(["Python", "PyTorch"]) for _ in range(10)]
        result = self.matcher.score_batch(batch)
        assert result.shape == (10,)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_bm25_fallback_when_no_index(self):
        skills = make_skills(["Python", "Machine Learning"])
        score = self.matcher.score(skills, bm25_index=None)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# CareerMatch
# ---------------------------------------------------------------------------

class TestCareerMatch:
    def setup_method(self):
        self.jd = make_jd()
        self.matcher = CareerMatcher(
            jd=self.jd,
            weights=load_weights().get("career_match", {}),
        )

    def test_ai_title_scores_high(self):
        s = _title_match_score("Senior AI Engineer")
        assert s >= 0.9

    def test_unrelated_title_scores_low(self):
        s = _title_match_score("Chef")
        assert s < 0.2

    def test_management_title_penalized(self):
        s = _title_match_score("VP of Engineering")
        assert s < 0.5

    def test_score_no_embedding(self):
        career = [{"title": "ML Engineer", "company": "Google",
                   "duration_months": 24, "start_date": "2022-01-01"}]
        s = self.matcher.score(career, "ML Engineer", narrative_embedding=None)
        assert 0.0 <= s <= 1.0

    def test_score_with_embedding(self):
        v = np.random.randn(384).astype(np.float32)
        v /= np.linalg.norm(v)
        jd_v = np.random.randn(384).astype(np.float32)
        jd_v /= np.linalg.norm(jd_v)
        matcher = CareerMatcher(
            self.jd, load_weights().get("career_match", {}),
            jd_narrative_embedding=jd_v
        )
        s = matcher.score([], "AI Engineer", narrative_embedding=v)
        assert 0.0 <= s <= 1.0

    def test_batch_vectorized(self):
        careers = [[{"title": "AI Engineer", "company": "OpenAI"}]] * 5
        titles = ["AI Engineer"] * 5
        embeddings = np.random.randn(5, 384).astype(np.float32)
        result = self.matcher.score_batch(careers, titles, embeddings)
        assert result.shape == (5,)


# ---------------------------------------------------------------------------
# EducationFit
# ---------------------------------------------------------------------------

class TestEducationFit:
    def setup_method(self):
        self.jd = make_jd()
        self.scorer = EducationFitScorer(self.jd, load_weights())

    def test_phd_highest(self):
        edu = [{"degree": "PhD", "field_of_study": "Computer Science",
                "institution": "IIT Bombay"}]
        s = self.scorer.score(edu)
        assert s >= 0.90

    def test_btech_lower_than_mtech(self):
        btech = [{"degree": "B.Tech", "field_of_study": "CS", "institution": "NIT"}]
        mtech = [{"degree": "M.Tech", "field_of_study": "CS", "institution": "NIT"}]
        assert self.scorer.score(btech) < self.scorer.score(mtech)

    def test_no_education_returns_floor(self):
        s = self.scorer.score([])
        assert s > 0.0  # floor is 0.25

    def test_preferred_institution_boost(self):
        with_iit = [{"degree": "B.Tech", "field_of_study": "CS",
                     "institution": "IIT Delhi"}]
        without = [{"degree": "B.Tech", "field_of_study": "CS",
                    "institution": "Unknown Univ"}]
        assert self.scorer.score(with_iit) > self.scorer.score(without)

    def test_batch_shape(self):
        batch = [[{"degree": "B.Tech"}] for _ in range(7)]
        result = self.scorer.score_batch(batch)
        assert len(result) == 7


# ---------------------------------------------------------------------------
# BehavioralModifier
# ---------------------------------------------------------------------------

class TestBehavioralModifier:
    def setup_method(self):
        self.modifier = BehavioralModifier(load_weights())

    def _make_signals(self, **kwargs) -> dict:
        base = {
            "last_active_date": "2024-01-01",
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12.0,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.7,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        }
        base.update(kwargs)
        return base

    def test_excellent_profile_near_max(self):
        from datetime import datetime
        today = datetime.today().strftime("%Y-%m-%d")
        signals = self._make_signals(
            last_active_date=today,
            recruiter_response_rate=1.0,
            avg_response_time_hours=1.0,
        )
        mult = self.modifier.compute(signals)
        assert mult >= 1.0

    def test_poor_profile_near_min(self):
        signals = self._make_signals(
            last_active_date="2020-01-01",
            recruiter_response_rate=0.1,
            avg_response_time_hours=200.0,
            interview_completion_rate=0.1,
            offer_acceptance_rate=0.1,
            verified_email=False,
            verified_phone=False,
            linkedin_connected=False,
        )
        mult = self.modifier.compute(signals)
        assert mult <= 0.75

    def test_multiplier_bounds(self):
        signals = self._make_signals()
        mult = self.modifier.compute(signals)
        assert self.modifier.min_mult <= mult <= self.modifier.max_mult

    def test_active_score_function(self):
        assert _active_score(0) == pytest.approx(1.0)
        assert _active_score(7) == pytest.approx(1.0)
        assert _active_score(365) == pytest.approx(0.10)

    def test_response_time_score(self):
        assert _response_time_score(1.0) == pytest.approx(1.0)
        assert _response_time_score(200.0) == pytest.approx(0.20)

    def test_batch_vectorized(self):
        signals_list = [self._make_signals() for _ in range(50)]
        result = self.modifier.compute_batch(signals_list)
        assert result.shape == (50,)
        assert np.all(result >= self.modifier.min_mult)
        assert np.all(result <= self.modifier.max_mult)

    def test_missing_fields_handled(self):
        # Partial signals — should not raise
        mult = self.modifier.compute({})
        assert 0.0 < mult


# ---------------------------------------------------------------------------
# CoherenceScorer
# ---------------------------------------------------------------------------

class TestCoherenceScorer:
    def setup_method(self):
        self.scorer = CoherenceScorer(load_weights())

    def test_identical_embeddings_high_coherence(self):
        v = np.random.randn(384).astype(np.float32)
        v /= np.linalg.norm(v)
        score = self.scorer.compute_from_embeddings(v, v, v)
        assert score > 0.90

    def test_orthogonal_embeddings_moderate_coherence(self):
        a = np.zeros(384, dtype=np.float32); a[0] = 1.0
        b = np.zeros(384, dtype=np.float32); b[1] = 1.0
        score = self.scorer.compute_from_embeddings(a, b)
        # cos=0 → (0+1)/2 = 0.5
        assert 0.45 <= score <= 0.55

    def test_factor_within_bounds(self):
        for raw in [0.0, 0.3, 0.5, 0.7, 1.0]:
            f = self.scorer.compute_factor(raw)
            assert self.scorer.floor <= f <= self.scorer.ceiling

    def test_batch_shape(self):
        n = 20
        hs = np.random.randn(n, 384).astype(np.float32)
        ca = np.random.randn(n, 384).astype(np.float32)
        sk = np.random.randn(n, 384).astype(np.float32)
        result = self.scorer.compute_batch(hs, ca, sk)
        assert result.shape == (n,)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_factors_from_scores(self):
        scores = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        factors = self.scorer.factors_from_scores(scores)
        assert factors[0] == pytest.approx(self.scorer.floor, abs=1e-4)
        assert factors[2] == pytest.approx(self.scorer.ceiling, abs=1e-4)


# ---------------------------------------------------------------------------
# DisqualifierScorer (Rule adjustments)
# ---------------------------------------------------------------------------

class TestRuleAdjustments:
    def setup_method(self):
        self.scorer = RuleAdjustmentScorer(load_weights())

    def _make_career(self, n_roles: int = 2, dur: int = 18) -> list[dict]:
        return [
            {
                "title": "ML Engineer",
                "company": f"Co{i}",
                "start_date": f"{2020+i}-01-01",
                "end_date": f"{2021+i}-12-31",
                "duration_months": dur,
            }
            for i in range(n_roles)
        ]

    def test_open_source_boost(self):
        summary = "Active open source contributor with merged PRs on GitHub."
        delta, flags = self.scorer.score(summary, self._make_career(), "ML Engineer")
        assert delta > 0.0
        assert any("open_source" in f for f in flags)

    def test_publication_boost(self):
        summary = "Published paper at NeurIPS 2023 on RAG systems."
        delta, flags = self.scorer.score(summary, self._make_career(), "ML Engineer")
        assert any("publication" in f for f in flags)

    def test_job_hopping_penalty(self):
        career = self._make_career(n_roles=4, dur=6)  # 6 months avg = job hopping
        delta, flags = self.scorer.score("", career, "ML Engineer")
        assert delta < 0.0
        assert any("job_hopping" in f for f in flags)

    def test_stable_career_no_penalty(self):
        career = self._make_career(n_roles=3, dur=24)  # 24 months = stable
        delta, flags = self.scorer.score("", career, "ML Engineer")
        hop_flags = [f for f in flags if "job_hopping" in f]
        assert len(hop_flags) == 0

    def test_delta_within_bounds(self):
        for _ in range(20):
            career = self._make_career()
            delta, _ = self.scorer.score("", career, "SWE")
            assert -0.35 <= delta <= 0.30

    def test_parse_date_formats(self):
        assert _parse_date("2023-06-15") is not None
        assert _parse_date("2023-06") is not None
        assert _parse_date("2023") is not None
        assert _parse_date(None) is None
        assert _parse_date("not-a-date") is None

    def test_average_tenure(self):
        career = [{"duration_months": 12}, {"duration_months": 24}]
        avg = _average_tenure_months(career)
        assert avg == pytest.approx(18.0)

    def test_empty_career_no_crash(self):
        delta, flags = self.scorer.score("", [], "Engineer")
        assert isinstance(delta, float)