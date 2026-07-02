"""
Experience fit scoring module.

Uses Gaussian decay to score candidates based on how well their years
of experience match the ideal range for the role.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ExperienceFitter:
    """Score candidates on experience fit using Gaussian decay.

    Perfect score (1.0) for candidates within the ideal range.
    Gaussian decay applied outside the range, parameterized by sigma.
    """

    def __init__(
        self,
        ideal_min: float = 5.0,
        ideal_max: float = 9.0,
        sigma: float = 3.0,
    ) -> None:
        """Initialize with experience range parameters.

        Args:
            ideal_min: Lower bound of ideal experience range (years).
            ideal_max: Upper bound of ideal experience range (years).
            sigma: Standard deviation for Gaussian decay outside range.
        """
        self.ideal_min = ideal_min
        self.ideal_max = ideal_max
        self.midpoint = (ideal_min + ideal_max) / 2.0
        self.sigma = sigma

    def score_batch(self, years_of_experience: NDArray[np.float64]) -> NDArray[np.float64]:
        """Score a batch of candidates on experience fit.

        Candidates within [ideal_min, ideal_max] receive 1.0.
        Those outside receive exp(-0.5 * ((yoe - nearest_boundary) / sigma)^2).

        Args:
            years_of_experience: Array of years of experience values.

        Returns:
            Array of experience fit scores in [0, 1].
        """
        scores = np.ones_like(years_of_experience, dtype=np.float64)

        # Below ideal range
        below_mask = years_of_experience < self.ideal_min
        if np.any(below_mask):
            deviation = (years_of_experience[below_mask] - self.ideal_min) / self.sigma
            scores[below_mask] = np.exp(-0.5 * deviation ** 2)

        # Above ideal range
        above_mask = years_of_experience > self.ideal_max
        if np.any(above_mask):
            deviation = (years_of_experience[above_mask] - self.ideal_max) / self.sigma
            scores[above_mask] = np.exp(-0.5 * deviation ** 2)

        return scores

    def score_single(self, years_of_experience: float) -> float:
        """Score a single candidate on experience fit.

        Args:
            years_of_experience: Candidate's years of experience.

        Returns:
            Experience fit score in [0, 1].
        """
        arr = np.array([years_of_experience], dtype=np.float64)
        return float(self.score_batch(arr)[0])
