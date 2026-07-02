"""
Title and career match module.

Computes semantic similarity between candidate narrative embeddings
(headline + summary + recent career history) and the JD embedding.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class CareerMatcher:
    """Compute career match scores using precomputed narrative embeddings.

    Uses cosine similarity between candidate narrative embeddings and
    the JD embedding to produce career match scores.
    """

    def __init__(self, jd_embedding: Optional[NDArray[np.float32]] = None) -> None:
        """Initialize with the JD embedding.

        Args:
            jd_embedding: Pre-computed embedding vector for the job description.
        """
        self.jd_embedding = jd_embedding
        if jd_embedding is not None:
            norm = np.linalg.norm(jd_embedding)
            self.jd_embedding_normalized = jd_embedding / (norm + 1e-8)
        else:
            self.jd_embedding_normalized = None

    def score_batch(
        self,
        candidate_embeddings: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Compute career match scores for a batch of candidates.

        Uses matrix-vector cosine similarity for efficiency.

        Args:
            candidate_embeddings: (N, D) array of candidate narrative embeddings.

        Returns:
            Array of career match scores in [0, 1].
        """
        if self.jd_embedding_normalized is None:
            logger.warning("No JD embedding provided; returning zeros.")
            return np.zeros(len(candidate_embeddings), dtype=np.float64)

        # Normalize candidate embeddings
        norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        normalized = candidate_embeddings / (norms + 1e-8)

        # Matrix-vector dot product for cosine similarity
        similarities = normalized @ self.jd_embedding_normalized

        # Clamp to [0, 1] — negative similarities treated as 0
        return np.clip(similarities.astype(np.float64), 0.0, 1.0)

    def score_single(self, candidate_embedding: NDArray[np.float32]) -> float:
        """Compute career match score for a single candidate.

        Args:
            candidate_embedding: (D,) embedding vector for the candidate.

        Returns:
            Career match score in [0, 1].
        """
        result = self.score_batch(candidate_embedding.reshape(1, -1))
        return float(result[0])
