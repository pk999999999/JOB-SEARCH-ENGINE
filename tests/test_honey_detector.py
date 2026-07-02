from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import BehavioralSignals, Candidate, CareerEntry, SkillEntry
from src.honeypot_detector import HoneypotDetector


def make_candidate(**kwargs) -> Candidate:
    """Build a clean baseline candidate, overridable via kwargs."""
    defaults = dict(
        candidate_id="TEST_001",
        headline="ML Engineer",
        summary="Experienced ML engineer.",
        current_title="ML Engineer",
        skills=[SkillEntry(name="python", endorsements=5, duration_months=24)],
        career_history=[
            CareerEntry(
                title="ML Engineer", company="Acme",
                start_date="2019-01-01", end_date="2023-12-31",
                duration_months=60, is_current=True,
            )
        ],
        location="Pune",
        years_of_experience=5.0,
        behavioral_signals=BehavioralSignals(),
        salary_min=1_000_000.0,
        salary_max=2_000_000.0,
    )
    defaults.update(kwargs)
    return Candidate(**defaults)


class TestHoneypotSignalChecks:
    """Tests for the individual signal-check methods."""

    def setup_method(self):
        self.detector = HoneypotDetector()

    # --- Strong: expert skill with short duration ---

    def test_expert_skill_long_duration_not_flagged(self):
        c = make_candidate(
            skills=[SkillEntry(name="pytorch", proficiency="expert", duration_months=24)]
        )
        assert self.detector._check_expert_skill_duration(c) == []

    def test_expert_skill_short_duration_flagged(self):
        c = make_candidate(
            skills=[SkillEntry(name="pytorch", proficiency="expert", duration_months=2)]
        )
        signals = self.detector._check_expert_skill_duration(c)
        assert len(signals) == 1
        assert "pytorch" in signals[0]

    def test_non_expert_short_duration_not_flagged(self):
        c = make_candidate(
            skills=[SkillEntry(name="python", proficiency="beginner", duration_months=1)]
        )
        assert self.detector._check_expert_skill_duration(c) == []

    def test_multiple_expert_short_duration_each_flagged(self):
        c = make_candidate(
            skills=[
                SkillEntry(name="pytorch", proficiency="expert", duration_months=1),
                SkillEntry(name="tensorflow", proficiency="advanced", duration_months=2),
            ]
        )
        signals = self.detector._check_expert_skill_duration(c)
        assert len(signals) == 2

    # --- Strong: YOE mismatch ---

    def test_yoe_matching_career_not_flagged(self):
        c = make_candidate(
            years_of_experience=5.0,
            career_history=[CareerEntry(duration_months=60)],
        )
        assert self.detector._check_yoe_mismatch(c) == []

    def test_yoe_far_exceeding_career_flagged(self):
        c = make_candidate(
            years_of_experience=10.0,
            career_history=[CareerEntry(duration_months=12)],  # 1 year on record
        )
        signals = self.detector._check_yoe_mismatch(c)
        assert len(signals) == 1

    def test_empty_career_history_no_signal(self):
        c = make_candidate(years_of_experience=5.0, career_history=[])
        assert self.detector._check_yoe_mismatch(c) == []

    # --- Medium: impossible chronology ---

    def test_reasonable_career_length_not_flagged(self):
        c = make_candidate(career_history=[CareerEntry(duration_months=120)])  # 10 years
        assert self.detector._check_impossible_chronology(c) == []

    def test_career_exceeding_lifetime_flagged(self):
        c = make_candidate(career_history=[CareerEntry(duration_months=700)])  # >58 years
        signals = self.detector._check_impossible_chronology(c)
        assert len(signals) == 1

    # --- Medium: overlapping / multiple current jobs ---

    def test_single_current_job_not_flagged(self):
        c = make_candidate(career_history=[
            CareerEntry(title="Current", company="A", is_current=True),
            CareerEntry(title="Past", company="B", is_current=False),
        ])
        assert self.detector._check_overlapping_jobs(c) == []
        assert self.detector._check_multiple_current(c) == []

    def test_two_current_jobs_flags_multiple_current_only(self):
        c = make_candidate(career_history=[
            CareerEntry(title="A", company="X", is_current=True),
            CareerEntry(title="B", company="Y", is_current=True),
        ])
        assert self.detector._check_overlapping_jobs(c) == []  # threshold is > 2
        signals = self.detector._check_multiple_current(c)
        assert len(signals) == 1

    def test_three_current_jobs_flags_both_overlap_and_multiple(self):
        c = make_candidate(career_history=[
            CareerEntry(title="A", company="X", is_current=True),
            CareerEntry(title="B", company="Y", is_current=True),
            CareerEntry(title="C", company="Z", is_current=True),
        ])
        assert len(self.detector._check_overlapping_jobs(c)) == 1
        assert len(self.detector._check_multiple_current(c)) == 1

    # --- Weak: salary inversion ---

    def test_normal_salary_not_flagged(self):
        c = make_candidate(salary_min=1_000_000.0, salary_max=2_000_000.0)
        assert self.detector._check_salary_inversion(c) == []

    def test_inverted_salary_flagged(self):
        c = make_candidate(salary_min=2_000_000.0, salary_max=1_000_000.0)
        signals = self.detector._check_salary_inversion(c)
        assert len(signals) == 1

    def test_missing_salary_not_flagged(self):
        c = make_candidate(salary_min=None, salary_max=None)
        assert self.detector._check_salary_inversion(c) == []


class TestHoneypotDetectorExclusionRules:
    def setup_method(self):
        self.detector = HoneypotDetector(strong_threshold=2, strong_plus_medium=(1, 2))

    def test_clean_candidate_passes(self):
        c = make_candidate()
        result = self.detector.detect(c)
        assert not result.is_honeypot
        assert result.gate_value == pytest.approx(1.0)

    def test_two_strong_signals_excluded(self):
        c = make_candidate(
            skills=[
                SkillEntry(name="pytorch", proficiency="expert", duration_months=1),
                SkillEntry(name="tensorflow", proficiency="expert", duration_months=2),
            ],
            years_of_experience=5.0,
            career_history=[CareerEntry(duration_months=60)],
        )
        result = self.detector.detect(c)
        assert result.is_honeypot
        assert result.gate_value == pytest.approx(0.0)

    def test_one_strong_two_medium_excluded(self):
        c = make_candidate(
            skills=[SkillEntry(name="pytorch", proficiency="expert", duration_months=1)],
            career_history=[
                CareerEntry(title="A", company="X", is_current=True),
                CareerEntry(title="B", company="Y", is_current=True),
                CareerEntry(title="C", company="Z", is_current=True),
            ],
        )
        result = self.detector.detect(c)
        assert result.is_honeypot

    def test_one_strong_one_medium_not_excluded(self):
        c = make_candidate(
            skills=[SkillEntry(name="pytorch", proficiency="expert", duration_months=1)],
            career_history=[
                CareerEntry(title="A", company="X", is_current=True),
                CareerEntry(title="B", company="Y", is_current=True),
            ],
        )
        result = self.detector.detect(c)
        assert not result.is_honeypot

    def test_weak_signal_alone_not_excluded(self):
        c = make_candidate(salary_min=2_000_000.0, salary_max=1_000_000.0)
        result = self.detector.detect(c)
        assert not result.is_honeypot
        assert len(result.weak_signals) == 1

    def test_gate_value_is_zero_or_one(self):
        for c in [make_candidate(), make_candidate(
            skills=[SkillEntry(name="pytorch", proficiency="expert", duration_months=1),
                    SkillEntry(name="tensorflow", proficiency="expert", duration_months=1)]
        )]:
            result = self.detector.detect(c)
            assert result.gate_value in (0.0, 1.0)

    def test_detect_batch_returns_correct_length(self):
        candidates = [make_candidate(candidate_id=f"C{i}") for i in range(10)]
        results = self.detector.detect_batch(candidates)
        assert len(results) == 10

    def test_get_gate_values_shape_and_dtype(self):
        candidates = [make_candidate(candidate_id=f"C{i}") for i in range(10)]
        gates = self.detector.get_gate_values(candidates)
        assert gates.shape == (10,)
        assert gates.dtype == np.float64

    def test_excluded_candidates_have_gate_zero_among_batch(self):
        clean = [make_candidate(candidate_id=f"CLEAN_{i}") for i in range(5)]
        dirty = [
            make_candidate(
                candidate_id=f"DIRTY_{i}",
                skills=[
                    SkillEntry(name="pytorch", proficiency="expert", duration_months=1),
                    SkillEntry(name="tensorflow", proficiency="expert", duration_months=1),
                ],
            )
            for i in range(3)
        ]
        gates = self.detector.get_gate_values(clean + dirty)
        assert np.all(gates[:5] == 1.0)
        assert np.all(gates[5:] == 0.0)
