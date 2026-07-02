"""
Scoring module — computes base and final scores for candidates.

Combines all feature scores with configurable weights and applies
multiplicative modifiers (coherence, behavioral, honeypot gate).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import yaml
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _load_weights(config_path: str | Path) -> Dict[str, Any]:
    """Load weights from a YAML configuration file.

    Args:
        config_path: Path to weights.yaml.

    Returns:
        Dictionary of weight configurations.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Scorer:
    """Compute base and final scores for candidate ranking.

    Base score formula:
        base = Σ(weight_i × feature_i) for all features

    Final score formula:
        final = base × coherence_factor × behavioral_modifier × honeypot_gate

    All operations are vectorized for batch processing.
    """

    DEFAULT_WEIGHTS = {
        "career_match": 0.30,
        "skill_match": 0.25,
        "experience_fit": 0.15,
        "location_fit": 0.10,
        "education_fit": 0.05,
        "rule_adjustments": 0.15,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        config_path: Optional[str | Path] = None,
    ) -> None:
        """Initialize the scorer.

        Args:
            weights: Base score component weights. Must sum to 1.0.
            config_path: Path to weights.yaml (alternative to weights dict).
        """
        if config_path:
            config = _load_weights(config_path)
            self.weights = config.get("base_score", self.DEFAULT_WEIGHTS)
        else:
            self.weights = weights or self.DEFAULT_WEIGHTS

        # Validate weights sum to ~1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "Base score weights sum to %.3f (expected 1.0). Normalizing.",
                total,
            )
            self.weights = {k: v / total for k, v in self.weights.items()}

    def compute_base_scores(
        self,
        career_match: NDArray[np.float64],
        skill_match: NDArray[np.float64],
        experience_fit: NDArray[np.float64],
        location_fit: NDArray[np.float64],
        education_fit: NDArray[np.float64],
        rule_adjustments: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute weighted base scores for all candidates.

        Args:
            career_match: Career match scores.
            skill_match: Skill match scores.
            experience_fit: Experience fit scores.
            location_fit: Location fit scores.
            education_fit: Education fit scores.
            rule_adjustments: Rule adjustment scores.

        Returns:
            Array of base scores.
        """
        base = (
            self.weights["career_match"] * career_match
            + self.weights["skill_match"] * skill_match
            + self.weights["experience_fit"] * experience_fit
            + self.weights["location_fit"] * location_fit
            + self.weights["education_fit"] * education_fit
            + self.weights["rule_adjustments"] * rule_adjustments
        )

        return base

    def compute_final_scores(
        self,
        base_scores: NDArray[np.float64],
        coherence_factors: NDArray[np.float64],
        behavioral_modifiers: NDArray[np.float64],
        honeypot_gates: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute final scores with multiplicative modifiers.

        final = base × coherence × behavioral × honeypot_gate

        Args:
            base_scores: Base scores from compute_base_scores().
            coherence_factors: Coherence multiplicative factors.
            behavioral_modifiers: Behavioral availability multipliers.
            honeypot_gates: Honeypot gate values (0 or 1).

        Returns:
            Array of final scores.
        """
        final = (
            base_scores
            * coherence_factors
            * behavioral_modifiers
            * honeypot_gates
        )

        return final

    def get_score_breakdown(
        self,
        index: int,
        career_match: NDArray[np.float64],
        skill_match: NDArray[np.float64],
        experience_fit: NDArray[np.float64],
        location_fit: NDArray[np.float64],
        education_fit: NDArray[np.float64],
        rule_adjustments: NDArray[np.float64],
        coherence_factors: NDArray[np.float64],
        behavioral_modifiers: NDArray[np.float64],
        honeypot_gates: NDArray[np.float64],
    ) -> Dict[str, float]:
        """Get detailed score breakdown for a single candidate.

        Args:
            index: Candidate index in the arrays.
            career_match: Career match scores array.
            skill_match: Skill match scores array.
            experience_fit: Experience fit scores array.
            location_fit: Location fit scores array.
            education_fit: Education fit scores array.
            rule_adjustments: Rule adjustments array.
            coherence_factors: Coherence factors array.
            behavioral_modifiers: Behavioral modifiers array.
            honeypot_gates: Honeypot gates array.

        Returns:
            Dictionary with all score components.
        """
        base = (
            self.weights["career_match"] * career_match[index]
            + self.weights["skill_match"] * skill_match[index]
            + self.weights["experience_fit"] * experience_fit[index]
            + self.weights["location_fit"] * location_fit[index]
            + self.weights["education_fit"] * education_fit[index]
            + self.weights["rule_adjustments"] * rule_adjustments[index]
        )

        final = (
            base
            * coherence_factors[index]
            * behavioral_modifiers[index]
            * honeypot_gates[index]
        )

        return {
            "career_match": float(career_match[index]),
            "skill_match": float(skill_match[index]),
            "experience_fit": float(experience_fit[index]),
            "location_fit": float(location_fit[index]),
            "education_fit": float(education_fit[index]),
            "rule_adjustments": float(rule_adjustments[index]),
            "base_score": float(base),
            "coherence_factor": float(coherence_factors[index]),
            "behavioral_modifier": float(behavioral_modifiers[index]),
            "honeypot_gate": float(honeypot_gates[index]),
            "final_score": float(final),
        }
