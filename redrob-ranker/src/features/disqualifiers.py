"""
Disqualifier and rule-based adjustment module.

Applies rule-based penalties and bonuses contributing to the
'rule_adjustments' component of the base score.
"""

from __future__ import annotations

import logging
from typing import List, Set

import numpy as np
from numpy.typing import NDArray

from src.data_loader import Candidate

logger = logging.getLogger(__name__)


class DisqualifierChecker:
    """Apply rule-based score adjustments.

    Checks for missing critical skills, career gaps, and other
    disqualifying factors, producing a rule_adjustments score.
    """

    def __init__(
        self,
        required_skills: List[str],
        min_required_skill_fraction: float = 0.3,
        critical_skill_penalty: float = 0.3,
        career_gap_penalty: float = 0.15,
        title_relevance_bonus: float = 0.2,
    ) -> None:
        """Initialize the disqualifier checker.

        Args:
            required_skills: Required skills from the JD.
            min_required_skill_fraction: Minimum fraction of required skills needed.
            critical_skill_penalty: Penalty for missing critical skills.
            career_gap_penalty: Penalty for large career gaps.
            title_relevance_bonus: Bonus for having relevant title.
        """
        self.required_skills: Set[str] = set(s.lower() for s in required_skills)
        self.min_required_skill_fraction = min_required_skill_fraction
        self.critical_skill_penalty = critical_skill_penalty
        self.career_gap_penalty = career_gap_penalty
        self.title_relevance_bonus = title_relevance_bonus

        # Core skills that are absolutely critical
        self.critical_skills = {"python", "machine learning"}

        # Relevant title keywords
        self.relevant_title_keywords = {
            "ai", "machine learning", "ml", "data scientist",
            "deep learning", "nlp", "research", "engineer",
        }

    def _check_critical_skills(self, candidate: Candidate) -> float:
        """Check if candidate has critical required skills.

        Args:
            candidate: The candidate to check.

        Returns:
            Penalty value (0 = no penalty, negative = penalty).
        """
        candidate_skills = set(candidate.get_skill_names())
        missing_critical = self.critical_skills - candidate_skills

        if missing_critical:
            return -self.critical_skill_penalty * (
                len(missing_critical) / len(self.critical_skills)
            )
        return 0.0

    def _check_required_skill_coverage(self, candidate: Candidate) -> float:
        """Check fraction of required skills the candidate has.

        Args:
            candidate: The candidate to check.

        Returns:
            Penalty value if below threshold.
        """
        if not self.required_skills:
            return 0.0

        candidate_skills = set(candidate.get_skill_names())
        coverage = len(candidate_skills & self.required_skills) / len(self.required_skills)

        if coverage < self.min_required_skill_fraction:
            return -0.2 * (1.0 - coverage / self.min_required_skill_fraction)
        return 0.0

    def _check_title_relevance(self, candidate: Candidate) -> float:
        """Check if current title is relevant to the JD.

        Args:
            candidate: The candidate to check.

        Returns:
            Bonus value for relevant title.
        """
        if not candidate.current_title:
            return 0.0

        title_lower = candidate.current_title.lower()
        for keyword in self.relevant_title_keywords:
            if keyword in title_lower:
                return self.title_relevance_bonus
        return 0.0

    def _check_career_continuity(self, candidate: Candidate) -> float:
        """Check for concerning career patterns.

        Args:
            candidate: The candidate to check.

        Returns:
            Penalty for career discontinuity.
        """
        if not candidate.career_history:
            return -self.career_gap_penalty

        # Check if has recent experience (at least one current/recent role)
        has_current = any(entry.is_current for entry in candidate.career_history)
        if not has_current and len(candidate.career_history) > 0:
            return -self.career_gap_penalty * 0.5

        return 0.0

    def score_candidate(self, candidate: Candidate) -> float:
        """Compute rule adjustment score for a single candidate.

        Args:
            candidate: The candidate to evaluate.

        Returns:
            Rule adjustment score (can be negative for penalties).
        """
        adjustment = 0.5  # Start at neutral

        adjustment += self._check_critical_skills(candidate)
        adjustment += self._check_required_skill_coverage(candidate)
        adjustment += self._check_title_relevance(candidate)
        adjustment += self._check_career_continuity(candidate)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, adjustment))

    def score_batch(self, candidates: List[Candidate]) -> NDArray[np.float64]:
        """Score a batch of candidates on rule adjustments.

        Args:
            candidates: List of candidates to evaluate.

        Returns:
            Array of rule adjustment scores.
        """
        scores = np.array(
            [self.score_candidate(c) for c in candidates],
            dtype=np.float64,
        )
        return scores
