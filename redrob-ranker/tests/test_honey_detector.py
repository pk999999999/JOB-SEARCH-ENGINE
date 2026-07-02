from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.data_loader import Candidate
from src.honeypot_detector import (
    HoneypotDetector,
    _check_expert_skill_short_duration,
    _check_impossible_chronology,
    _check_multiple_current_jobs,
    _check_overlapping_jobs,
    _check_salary_inversion,
    _check_yoe_mismatch,
)


def load_weights() -> dict:
    with open("config/weights.yaml") as f:
        return yaml.safe_load(f)


def make_candidate(**kwargs) -> Candidate:
    defaults = dict(
        candidate_id="TEST_001",
        headline="ML Engineer",
        summary="Experienced ML engineer.",
        current_title="ML Engineer",
        skills=[{"name": "Python", "endorsements": 5, "duration_months": 24}],
        career_history=[
            {
                "title": "ML Engineer",
                "company": "Acme",
                "start_date": "2019-01-01",
                "end_date": "2023-12-31",
                "duration_months": 60,
            }
        ],
        education=[{"degree": "B.Tech", "field_of_study": "CS"}],
        location="Pune",
        years_of_experience=5.0,
        behavioral_signals={},
        salary_min=1_000_000.0,
        salary_max=2_000_000.0,
    )
    defaults.update(kwargs)
    return Candidate(**defaults)


class TestHoneypotSignalFunctions:

    # --- Strong: expert skill with short duration ---
    def test_expert_skill_long_duration_ok(self):
        skills = [{"name": "PyTorch", "proficiency": "expert", "duration_months": 24}]
        assert _check_expert_skill_short_duration(skills) == []

    def test_expert_skill_short_duration_flagged(self):
        skills = [{"name": "PyTorch", "proficiency": "expert", "duration_months": 2}]
        signals = _check_expert_skill_short_duration(skills)
        assert len(signals) == 1
        assert "expert_skill_short_duration" in signals[0]

    def test_non_expert_short_duration_ok(self):
        skills = [{"name": "Python", "proficiency": "beginner", "duration_months": 1}]
        assert _check_expert_skill_short_duration(skills) == []

    def test_multiple_expert_short_duration(self):
        skills = [
            {"name": "PyTorch", "proficiency": "expert", "duration_months": 1},
            {"name": "TensorFlow", "proficiency": "advanced", "duration_months": 2},
        ]
        signals = _check_expert_skill_short_duration(skills)
        assert len(signals) == 2

    # --- Strong: YOE mismatch ---
    def test_yoe_match_ok(self):
        career = [{"duration_months": 60}]
        signals = _check_yoe_mismatch(5.0, career)
        assert signals == []

    def test_yoe_mismatch_flagged(self):
        career = [{"duration_months": 12}]  # computed=1y, declared=10y
        signals = _check_yoe_mismatch(10.0, career)
        assert len(signals) == 1
        assert "yoe_mismatch" in signals[0]

    def test_empty_career_no_signal(self):
        assert _check_yoe_mismatch(5.0, []) == []

    # --- Medium: impossible chronology ---
    def test_valid_chronology_ok(self):
        career = [{"start_date": "2019-01-01", "end_date": "2022-12-31"}]
        assert _check_impossible_chronology(career) == []

    def test_end_before_start_flagged(self):
        career = [{"start_date": "2022-01-01", "end_date": "2020-12-31"}]
        signals = _check_impossible_chronology(career)
        assert len(signals) == 1

    def test_missing_dates_no_signal(self):
        career = [{"title": "Engineer"}]
        assert _check_impossible_chronology(career) == []

    # --- Medium: overlapping jobs ---
    def test_non_overlapping_ok(self):
        career = [
            {"title": "A", "start_date": "2019-01-01", "end_date": "2021-12-31"},
            {"title": "B", "start_date": "2022-01-01", "end_date": "2024-01-01"},
        ]
        assert _check_overlapping_jobs(career) == []

    def test_major_overlap_flagged(self):
        career = [
            {"title": "A", "start_date": "2019-01-01", "end_date": "2022-12-31"},
            {"title": "B", "start_date": "2019-06-01", "end_date": "2021-06-01"},
        ]
        signals = _check_overlapping_jobs(career)
        assert len(signals) >= 1

    def test_minor_overlap_not_flagged(self):
        # 30-day overlap is below the 60-day threshold
        career = [
            {"title": "A", "start_date": "2019-01-01", "end_date": "2021-01-31"},
            {"title": "B", "start_date": "2021-01-15", "end_date": "2023-01-01"},
        ]
        signals = _check_overlapping_jobs(career)
        assert signals == []

    # --- Medium: multiple current jobs ---
    def test_single_current_ok(self):
        career = [
            {"title": "Current", "end_date": None},
            {"title": "Past", "end_date": "2022-12-31"},
        ]
        assert _check_multiple_current_jobs(career) == []

    def test_two_current_flagged(self):
        career = [
            {"title": "Job A", "end_date": None},
            {"title": "Job B", "end_date": None},
        ]
        signals = _check_multiple_current_jobs(career)
        assert len(signals) == 1

    # --- Weak: salary inversion ---
    def test_normal_salary_ok(self):
        assert _check_salary_inversion(1_000_000, 2_000_000) == []

    def test_inverted_salary_flagged(self):
        signals = _check_salary_inversion(2_000_000, 1_000_000)
        assert len(signals) == 1

    def test_none_salary_ok(self):
        assert _check_salary_inversion(None, None) == []
        assert _check_salary_inversion(100.0, None) == []


class TestHoneypotDetector:
    def setup_method(self):
        self.detector = HoneypotDetector(load_weights())

    def test_clean_candidate_passes(self):
        c = make_candidate()
        result = self.detector.detect(c)
        assert not result.is_excluded
        assert result.gate == pytest.approx(1.0)

    def test_two_strong_signals_excluded(self):
        # 2 expert skills with short duration (2 strong signals)
        c = make_candidate(
            skills=[
                {"name": "PyTorch", "proficiency": "expert", "duration_months": 1},
                {"name": "TensorFlow", "proficiency": "expert", "duration_months": 2},
            ],
            years_of_experience=5.0,
            career_history=[{"duration_months": 60}],
        )
        result = self.detector.detect(c)
        assert result.is_excluded
        assert result.gate == pytest.approx(0.0)

    def test_one_strong_two_medium_excluded(self):
        c = make_candidate(
            skills=[
                {"name": "PyTorch", "proficiency": "expert", "duration_months": 1}
            ],
            career_history=[
                {"title": "A", "start_date": "2020-01-01", "end_date": "2019-01-01"},
                {"title": "B", "start_date": "2019-01-01", "end_date": None},
                {"title": "C", "start_date": "2018-01-01", "end_date": None},
            ],
        )
        result = self.detector.detect(c)
        assert result.is_excluded

    def test_one_strong_one_medium_not_excluded(self):
        c = make_candidate(
            skills=[
                {"name": "PyTorch", "proficiency": "expert", "duration_months": 1}
            ],
            career_history=[
                {"title": "A", "start_date": "2020-01-01", "end_date": "2019-01-01"},
            ],
        )
        result = self.detector.detect(c)
        assert not result.is_excluded

    def test_weak_signal_alone_not_excluded(self):
        c = make_candidate(salary_min=2_000_000.0, salary_max=1_000_000.0)
        result = self.detector.detect(c)
        assert not result.is_excluded
        assert len(result.weak_signals) == 1

    def test_gate_is_zero_or_one(self):
        c = make_candidate()
        result = self.detector.detect(c)
        assert result.gate in (0.0, 1.0)

    def test_batch_returns_correct_shapes(self):
        candidates = [make_candidate(candidate_id=f"C{i}") for i in range(10)]
        gates, results = self.detector.detect_batch(candidates)
        assert gates.shape == (10,)
        assert len(results) == 10

    def test_summary_method(self):
        c = make_candidate()
        result = self.detector.detect(c)
        summary = result.summary()
        assert "TEST_001" in summary
        assert "excluded" in summary