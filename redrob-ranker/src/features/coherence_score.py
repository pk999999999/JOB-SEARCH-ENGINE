"""
Narrative coherence scoring module.

Computes cross-section coherence by measuring cosine similarity between
different parts of a candidate's profile (headline+summary, career history, skills).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class CoherenceScorer:
    """Score narrative coherence across candidate profile sections.

    A coherent candidate has consistent messaging across their headline,
    summary, career history, and skills. Low coherence may indicate
    copy-paste profiles or inconsistent information.

    Coherence is computed as average pairwise cosine similarity between
    section embeddings.
    """

    def __init__(
        self,
        min_factor: float = 0.70,
        max_factor: float = 1.10,
    ) -> None:
        """Initialize coherence scorer.

        Args:
            min_factor: Minimum coherence factor (applied to very incoherent profiles).
            max_factor: Maximum coherence factor (applied to very coherent profiles).
        """
        self.min_factor = min_factor
        self.max_factor = max_factor

    @staticmethod
    def compute_pairwise_similarity(
        embeddings_a: NDArray[np.float32],
        embeddings_b: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Compute pairwise cosine similarity between two sets of embeddings.

        Args:
            embeddings_a: (N, D) first set of embeddings.
            embeddings_b: (N, D) second set of embeddings.

        Returns:
            (N,) array of cosine similarities.
        """
        norms_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-8
        norms_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-8

        normalized_a = embeddings_a / norms_a
        normalized_b = embeddings_b / norms_b

        # Row-wise dot product
        similarities = np.sum(normalized_a * normalized_b, axis=1)

        return np.clip(similarities.astype(np.float64), 0.0, 1.0)

    def score_batch(
        self,
        narrative_embeddings: NDArray[np.float32],
        career_embeddings: NDArray[np.float32],
        skill_embeddings: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Compute coherence scores for a batch of candidates.

        Computes average pairwise cosine similarity between:
        - narrative (headline+summary) vs career history
        - narrative vs skills
        - career history vs skills

        Args:
            narrative_embeddings: (N, D) headline+summary embeddings.
            career_embeddings: (N, D) career history embeddings.
            skill_embeddings: (N, D) skill text embeddings.

        Returns:
            Array of coherence scores in [0, 1].
        """
        sim_nc = self.compute_pairwise_similarity(narrative_embeddings, career_embeddings)
        sim_ns = self.compute_pairwise_similarity(narrative_embeddings, skill_embeddings)
        sim_cs = self.compute_pairwise_similarity(career_embeddings, skill_embeddings)

        # Average pairwise similarity
        coherence = (sim_nc + sim_ns + sim_cs) / 3.0

        return coherence

    def coherence_to_factor(
        self,
        coherence_scores: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert raw coherence scores to multiplicative factors.

        Maps [0, 1] coherence to [min_factor, max_factor] range.

        Args:
            coherence_scores: Raw coherence scores in [0, 1].

        Returns:
            Coherence factors for score multiplication.
        """
        return self.min_factor + coherence_scores * (self.max_factor - self.min_factor)

    def score_from_precomputed(
        self,
        precomputed_coherence: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert precomputed coherence scores to factors.

        Used during the ranking phase when coherence scores are
        loaded from artifacts rather than computed on the fly.

        Args:
            precomputed_coherence: Pre-computed coherence scores from artifacts.

        Returns:
            Coherence factors for score multiplication.
        """
        return self.coherence_to_factor(precomputed_coherence)
