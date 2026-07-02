"""
Behavioral availability modifier module.

Computes a multiplier in [0.55, 1.15] from behavioral signals indicating
candidate availability, responsiveness, and engagement.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from src.data_loader import BehavioralSignals

logger = logging.getLogger(__name__)


class BehavioralModifier:
    """Compute behavioral availability multiplier from candidate signals.

    Uses a weighted combination of normalized behavioral signals to produce
    a multiplier in the range [min_multiplier, max_multiplier].

    Default range: [0.55, 1.15]
    """

    DEFAULT_WEIGHTS = {
        "recency": 0.20,
        "recruiter_response_rate": 0.20,
        "response_time": 0.15,
        "interview_completion": 0.15,
        "offer_acceptance": 0.10,
        "verified_email": 0.05,
        "verified_phone": 0.05,
        "linkedin_connected": 0.10,
    }

    def __init__(
        self,
        min_multiplier: float = 0.55,
        max_multiplier: float = 1.15,
        weights: Optional[Dict[str, float]] = None,
        reference_date: Optional[datetime] = None,
        recent_activity_days: int = 90,
    ) -> None:
        """Initialize behavioral modifier.

        Args:
            min_multiplier: Minimum multiplier value.
            max_multiplier: Maximum multiplier value.
            weights: Signal weight overrides.
            reference_date: Reference date for recency computation.
            recent_activity_days: Days threshold for "recent" activity.
        """
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.reference_date = reference_date or datetime.now()
        self.recent_activity_days = recent_activity_days

    def _normalize_recency(self, last_active_date: str) -> float:
        """Normalize recency of last activity to [0, 1].

        Args:
            last_active_date: ISO format date string.

        Returns:
            Recency score — 1.0 for today, decays to 0 for old dates.
        """
        if not last_active_date:
            return 0.3  # Default for missing data

        try:
            # Try multiple date formats
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%d-%m-%Y"):
                try:
                    active_date = datetime.strptime(last_active_date.strip(), fmt)
                    break
                except ValueError:
                    continue
            else:
                return 0.3

            days_ago = (self.reference_date - active_date).days
            if days_ago < 0:
                return 1.0
            if days_ago <= self.recent_activity_days:
                return 1.0 - 0.5 * (days_ago / self.recent_activity_days)
            # Decay beyond threshold
            return max(0.0, 0.5 * np.exp(-days_ago / (self.recent_activity_days * 3)))

        except Exception:
            return 0.3

    def _normalize_response_time(self, avg_hours: float) -> float:
        """Normalize average response time to [0, 1].

        Lower response time is better.

        Args:
            avg_hours: Average response time in hours.

        Returns:
            Normalized score — 1.0 for instant, 0 for very slow.
        """
        if avg_hours <= 0:
            return 0.5  # Missing or invalid

        # < 4 hours = excellent, > 168 hours (1 week) = poor
        if avg_hours <= 4:
            return 1.0
        if avg_hours >= 168:
            return 0.0
        # Linear decay
        return 1.0 - (avg_hours - 4) / (168 - 4)

    def score_single(self, signals: BehavioralSignals) -> float:
        """Compute behavioral multiplier for a single candidate.

        Args:
            signals: Candidate's behavioral signals.

        Returns:
            Multiplier in [min_multiplier, max_multiplier].
        """
        components: Dict[str, float] = {
            "recency": self._normalize_recency(signals.last_active_date),
            "recruiter_response_rate": min(1.0, max(0.0, signals.recruiter_response_rate)),
            "response_time": self._normalize_response_time(signals.avg_response_time_hours),
            "interview_completion": min(1.0, max(0.0, signals.interview_completion_rate)),
            "offer_acceptance": min(1.0, max(0.0, signals.offer_acceptance_rate)),
            "verified_email": 1.0 if signals.verified_email else 0.0,
            "verified_phone": 1.0 if signals.verified_phone else 0.0,
            "linkedin_connected": 1.0 if signals.linkedin_connected else 0.0,
        }

        # Weighted sum → raw score in [0, 1]
        raw_score = sum(
            self.weights.get(key, 0.0) * value
            for key, value in components.items()
        )

        # Scale to [min_multiplier, max_multiplier]
        multiplier = self.min_multiplier + raw_score * (self.max_multiplier - self.min_multiplier)

        return multiplier

    def score_batch(
        self,
        behavioral_signals_list: List[BehavioralSignals],
    ) -> NDArray[np.float64]:
        """Compute behavioral multipliers for a batch of candidates.

        Args:
            behavioral_signals_list: List of behavioral signals per candidate.

        Returns:
            Array of multipliers in [min_multiplier, max_multiplier].
        """
        multipliers = np.array(
            [self.score_single(signals) for signals in behavioral_signals_list],
            dtype=np.float64,
        )
        return multipliers
