"""
Education fit scoring module.

Low-weight scorer used primarily as a tie-breaker.
Scores based on degree level and field relevance to the JD.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from src.data_loader import EducationEntry

logger = logging.getLogger(__name__)


# Degree level hierarchy
DEGREE_SCORES: Dict[str, float] = {
    "phd": 1.0,
    "ph.d": 1.0,
    "doctorate": 1.0,
    "m.tech": 0.85,
    "mtech": 0.85,
    "m.s.": 0.80,
    "ms": 0.80,
    "masters": 0.80,
    "master": 0.80,
    "mba": 0.70,
    "m.e.": 0.80,
    "b.tech": 0.60,
    "btech": 0.60,
    "b.e.": 0.60,
    "be": 0.60,
    "bachelors": 0.60,
    "bachelor": 0.60,
    "b.s.": 0.60,
    "bs": 0.60,
    "b.sc": 0.55,
    "bsc": 0.55,
    "diploma": 0.30,
    "certificate": 0.20,
}

# Field relevance mapping
RELEVANT_FIELDS = {
    "exact": [
        "computer science", "artificial intelligence", "machine learning",
        "data science", "deep learning", "nlp", "natural language processing",
        "computational linguistics", "cs"
    ],
    "related": [
        "statistics", "mathematics", "electrical engineering",
        "electronics", "information technology", "software engineering",
        "operations research", "applied mathematics", "physics",
        "computational science", "bioinformatics", "it"
    ],
}


class EducationFitter:
    """Score candidates on education fit.

    Combines degree level score with field relevance score.
    Designed as a low-weight tie-breaker in the final ranking.
    """

    def __init__(
        self,
        preferred_fields: Optional[List[str]] = None,
        degree_scores: Optional[Dict[str, float]] = None,
        field_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize education fitter."""
        self.preferred_fields = [f.lower() for f in (preferred_fields or [])]
        self.degree_scores = degree_scores or DEGREE_SCORES
        self.field_scores = field_scores or {
            "exact_match": 1.0,
            "related": 0.7,
            "unrelated": 0.3,
        }

    def _score_degree(self, degree: str) -> float:
        """Score a degree level with word boundary isolation."""
        if not degree:
            return 0.3

        # Normalize text completely
        normalized = re.sub(r"[^a-z0-9\s.]", "", degree.lower().strip())

        # Exact match check
        if normalized in self.degree_scores:
            return self.degree_scores[normalized]

        # 🛠️ FIXED SUBSTRING MATCH: Checks tokens or explicit word boundaries 
        # so words like "management" or "business" don't falsely match "me" or "be"
        tokens = set(normalized.split())
        for key in sorted(self.degree_scores.keys(), key=len, reverse=True):
            if key in tokens:
                return self.degree_scores[key]
            # Handle boundary patterns like 'b.tech' or 'm.s.'
            if len(key) > 2 and key in normalized:
                return self.degree_scores[key]

        return 0.3

    def _score_field(self, field_of_study: str) -> float:
        """Score a field of study for relevance using explicit boundary checks."""
        normalized = field_of_study.lower().strip()

        if not normalized:
            return self.field_scores["unrelated"]

        # 🛠️ FIXED DIRECTIONAL LOOKUP: Ensures short candidate strings (like 'cs')
        # match criteria properly without evaluating backwards.
        for exact_field in RELEVANT_FIELDS["exact"]:
            if exact_field in normalized or (len(normalized) >= 2 and normalized in exact_field):
                return self.field_scores["exact_match"]

        for related_field in RELEVANT_FIELDS["related"]:
            if related_field in normalized or (len(normalized) >= 2 and normalized in related_field):
                return self.field_scores["related"]

        return self.field_scores["unrelated"]

    def score_candidate(self, education: List[EducationEntry]) -> float:
        """Score a single candidate's education."""
        if not education:
            return 0.0  # Reset to 0.0 if entirely missing from resume text

        best_score = 0.0
        for entry in education:
            # Prevent crashes if fields are missing or empty
            deg_text = getattr(entry, 'degree', '') or ''
            field_text = getattr(entry, 'field_of_study', '') or ''
            
            degree_score = self._score_degree(deg_text)
            field_score = self._score_field(field_text)
            
            combined = 0.6 * degree_score + 0.4 * field_score
            best_score = max(best_score, combined)

        return best_score

    def score_batch(
        self,
        education_lists: List[List[EducationEntry]],
    ) -> NDArray[np.float64]:
        """Score a batch of candidates on education fit."""
        scores = np.array(
            [self.score_candidate(edu) for edu in education_lists],
            dtype=np.float64,
        )
        return scores