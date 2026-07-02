"""
Honeypot detection module.

Identifies potentially fraudulent or exaggerated candidate profiles
using multi-signal analysis. Gates fraudulent candidates from ranking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
from numpy.typing import NDArray

from src.data_loader import Candidate

logger = logging.getLogger(__name__)


@dataclass
class HoneypotResult:
    """Result of honeypot detection for a single candidate."""

    candidate_id: str = ""
    is_honeypot: bool = False
    strong_signals: List[str] = field(default_factory=list)
    medium_signals: List[str] = field(default_factory=list)
    weak_signals: List[str] = field(default_factory=list)
    gate_value: float = 1.0  # 1.0 = pass, 0.0 = excluded

    @property
    def strong_count(self) -> int:
        """Number of strong signals detected."""
        return len(self.strong_signals)

    @property
    def medium_count(self) -> int:
        """Number of medium signals detected."""
        return len(self.medium_signals)

    @property
    def weak_count(self) -> int:
        """Number of weak signals detected."""
        return len(self.weak_signals)


class HoneypotDetector:
    """Multi-signal honeypot detection for candidate profiles.

    Signal hierarchy:
    - Strong: Expert skill with <=3 months duration, YoE mismatch
    - Medium: Impossible chronology, overlapping jobs, multiple current jobs
    - Weak: Salary min > max

    Exclusion rule:
    - 2+ strong signals OR
    - 1+ strong AND 2+ medium signals
    """

    def __init__(
        self,
        strong_threshold: int = 2,
        strong_plus_medium: tuple[int, int] = (1, 2),
    ) -> None:
        """Initialize the honeypot detector.

        Args:
            strong_threshold: Number of strong signals to trigger exclusion alone.
            strong_plus_medium: Tuple of (min_strong, min_medium) for combined exclusion.
        """
        self.strong_threshold = strong_threshold
        self.strong_plus_medium = strong_plus_medium

    def _check_expert_skill_duration(self, candidate: Candidate) -> List[str]:
        """Check for expert-level skills with very short duration.

        Strong signal: Claiming expert proficiency with <=3 months experience.

        Args:
            candidate: Candidate to check.

        Returns:
            List of signal descriptions.
        """
        signals = []
        for skill in candidate.skills:
            if (
                skill.proficiency.lower() in ("expert", "advanced")
                and 0 < skill.duration_months <= 3
            ):
                signals.append(
                    f"Expert skill '{skill.name}' with only {skill.duration_months} months duration"
                )
        return signals

    def _check_yoe_mismatch(self, candidate: Candidate) -> List[str]:
        """Check for years of experience mismatch with career history.

        Strong signal: Claimed YoE significantly exceeds sum of career durations.

        Args:
            candidate: Candidate to check.

        Returns:
            List of signal descriptions.
        """
        signals = []
        if not candidate.career_history:
            return signals

        total_career_months = sum(
            entry.duration_months for entry in candidate.career_history
        )
        total_career_years = total_career_months / 12.0

        if candidate.years_of_experience > 0 and total_career_years > 0:
            # Allow 20% tolerance
            if candidate.years_of_experience > total_career_years * 1.5 + 2:
                signals.append(
                    f"Claimed {candidate.years_of_experience:.1f} years "
                    f"but career history totals {total_career_years:.1f} years"
                )

        return signals

    def _check_impossible_chronology(self, candidate: Candidate) -> List[str]:
        """Check for impossible timeline in career history.

        Medium signal: Career history dates that don't make logical sense.

        Args:
            candidate: Candidate to check.

        Returns:
            List of signal descriptions.
        """
        signals = []
        if not candidate.career_history:
            return signals

        total_career_months = sum(
            entry.duration_months for entry in candidate.career_history
        )
        total_career_years = total_career_months / 12.0

        # Check if total career duration exceeds a reasonable lifetime
        if total_career_years > 50:
            signals.append(
                f"Total career duration {total_career_years:.1f} years exceeds reasonable limit"
            )

        return signals

    def _check_overlapping_jobs(self, candidate: Candidate) -> List[str]:
        """Check for overlapping job entries.

        Medium signal: Multiple jobs at the same time (excluding legitimate overlaps).

        Args:
            candidate: Candidate to check.

        Returns:
            List of signal descriptions.
        """
        signals = []
        current_jobs = [e for e in candidate.career_history if e.is_current]

        # Check for multiple current jobs (more than 2 is suspicious)
        if len(current_jobs) > 2:
            signals.append(
                f"Has {len(current_jobs)} current jobs simultaneously"
            )

        return signals

    def _check_multiple_current(self, candidate: Candidate) -> List[str]:
        """Check for multiple current job positions.

        Medium signal: Having more than one "current" role.

        Args:
            candidate: Candidate to check.

        Returns:
            List of signal descriptions.
        """
        signals = []
        current_jobs = [e for e in candidate.career_history if e.is_current]

        if len(current_jobs) > 1:
            companies = [e.company for e in current_jobs]
            signals.append(
                f"Multiple current positions at: {', '.join(companies)}"
            )

        return signals

    def _check_salary_inversion(self, candidate: Candidate) -> List[str]:
        """Check for salary min > max.

        Weak signal: Data quality issue.

        Args:
            candidate: Candidate to check.

        Returns:
            List of signal descriptions.
        """
        signals = []
        if (
            candidate.salary_min is not None
            and candidate.salary_max is not None
            and candidate.salary_min > candidate.salary_max
        ):
            signals.append(
                f"Salary min ({candidate.salary_min}) > max ({candidate.salary_max})"
            )
        return signals

    def detect(self, candidate: Candidate) -> HoneypotResult:
        """Run all honeypot checks on a single candidate.

        Args:
            candidate: Candidate to evaluate.

        Returns:
            HoneypotResult with all detected signals and gate value.
        """
        result = HoneypotResult(candidate_id=candidate.candidate_id)

        # Strong signals
        result.strong_signals.extend(self._check_expert_skill_duration(candidate))
        result.strong_signals.extend(self._check_yoe_mismatch(candidate))

        # Medium signals
        result.medium_signals.extend(self._check_impossible_chronology(candidate))
        result.medium_signals.extend(self._check_overlapping_jobs(candidate))
        result.medium_signals.extend(self._check_multiple_current(candidate))

        # Weak signals
        result.weak_signals.extend(self._check_salary_inversion(candidate))

        # Apply exclusion rule
        if result.strong_count >= self.strong_threshold:
            result.is_honeypot = True
            result.gate_value = 0.0
            logger.info(
                "Candidate %s flagged as honeypot: %d strong signals",
                candidate.candidate_id,
                result.strong_count,
            )
        elif (
            result.strong_count >= self.strong_plus_medium[0]
            and result.medium_count >= self.strong_plus_medium[1]
        ):
            result.is_honeypot = True
            result.gate_value = 0.0
            logger.info(
                "Candidate %s flagged as honeypot: %d strong + %d medium signals",
                candidate.candidate_id,
                result.strong_count,
                result.medium_count,
            )

        return result

    def detect_batch(self, candidates: List[Candidate]) -> List[HoneypotResult]:
        """Run honeypot detection on a batch of candidates.

        Args:
            candidates: List of candidates to evaluate.

        Returns:
            List of HoneypotResult objects.
        """
        results = [self.detect(c) for c in candidates]

        flagged_count = sum(1 for r in results if r.is_honeypot)
        logger.info(
            "Honeypot detection: %d/%d candidates flagged",
            flagged_count,
            len(candidates),
        )

        return results

    def get_gate_values(self, candidates: List[Candidate]) -> NDArray[np.float64]:
        """Get gate values (0 or 1) for a batch of candidates.

        Args:
            candidates: List of candidates.

        Returns:
            Array of gate values — 0.0 for honeypots, 1.0 for valid candidates.
        """
        results = self.detect_batch(candidates)
        return np.array(
            [r.gate_value for r in results],
            dtype=np.float64,
        )
