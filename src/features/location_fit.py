"""
Location fit scoring module.

Assigns tiered scores based on candidate location matching
against preferred, good, and acceptable locations.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class LocationFitter:
    """Score candidates on location preference tiers.

    Tiered scoring:
    - Preferred (e.g., Pune, Noida) → 1.0
    - Good (e.g., Mumbai, Hyderabad) → 0.8
    - Acceptable country (India) → 0.5
    - Outside country → 0.2
    """

    def __init__(
        self,
        preferred: Optional[List[str]] = None,
        good: Optional[List[str]] = None,
        acceptable_country: str = "India",
        tier_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize location fitter with tier definitions.

        Args:
            preferred: List of preferred city names.
            good: List of good (secondary) city names.
            acceptable_country: Country name for acceptable tier.
            tier_scores: Override scores for each tier.
        """
        self.preferred = [self._normalize(loc) for loc in (preferred or ["Pune", "Noida"])]
        self.good = [self._normalize(loc) for loc in (good or [
            "Mumbai", "Hyderabad", "Delhi NCR", "Delhi", "New Delhi", "Gurgaon", "Gurugram"
        ])]
        self.acceptable_country = self._normalize(acceptable_country)

        defaults = {"preferred": 1.0, "good": 0.8, "acceptable_country": 0.5, "outside_country": 0.2}
        self.tier_scores = tier_scores or defaults

    @staticmethod
    def _normalize(location: str) -> str:
        """Normalize a location string for comparison.

        Args:
            location: Raw location string.

        Returns:
            Lowercased, stripped, de-punctuated location.
        """
        return re.sub(r"[^a-z0-9\s]", "", location.lower().strip())

    def _classify_location(self, location: str) -> str:
        """Classify a location into a tier.

        Args:
            location: Candidate's location string.

        Returns:
            Tier name: 'preferred', 'good', 'acceptable_country', or 'outside_country'.
        """
        normalized = self._normalize(location)

        if not normalized:
            return "outside_country"

        # Check preferred cities
        for city in self.preferred:
            if city in normalized or normalized in city:
                return "preferred"

        # Check good cities
        for city in self.good:
            if city in normalized or normalized in city:
                return "good"

        # Check if in acceptable country
        if self.acceptable_country in normalized:
            return "acceptable_country"

        # Common India indicators
        india_indicators = [
            "india", "bengaluru", "bangalore", "chennai", "kolkata",
            "ahmedabad", "jaipur", "lucknow", "kanpur", "nagpur",
            "indore", "thane", "bhopal", "visakhapatnam", "patna",
            "vadodara", "coimbatore", "surat", "kochi", "trivandrum",
            "chandigarh", "mysore", "mysuru",
        ]
        for indicator in india_indicators:
            if indicator in normalized:
                return "acceptable_country"

        return "outside_country"

    def score_batch(self, locations: List[str]) -> NDArray[np.float64]:
        """Score a batch of candidate locations.

        Args:
            locations: List of location strings, one per candidate.

        Returns:
            Array of location fit scores.
        """
        scores = np.array(
            [self.tier_scores[self._classify_location(loc)] for loc in locations],
            dtype=np.float64,
        )
        return scores

    def score_single(self, location: str) -> float:
        """Score a single candidate location.

        Args:
            location: Candidate's location string.

        Returns:
            Location fit score.
        """
        tier = self._classify_location(location)
        return self.tier_scores[tier]
