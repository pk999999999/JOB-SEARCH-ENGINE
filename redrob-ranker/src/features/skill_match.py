"""
Skill matching module with hybrid scoring.

Combines lexical overlap, BM25 retrieval scoring, and embedding-based
semantic similarity for robust skill matching against job requirements.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SkillMatcher:
    """Hybrid skill matching using lexical, BM25, and embedding similarity.

    Attributes:
        required_skills: Normalized list of required skills from JD.
        preferred_skills: Normalized list of preferred skills from JD.
        weights: Sub-weights for lexical, BM25, and embedding components.
    """

    def __init__(
        self,
        required_skills: List[str],
        preferred_skills: List[str],
        jd_skill_embedding: Optional[NDArray[np.float32]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the skill matcher.

        Args:
            required_skills: Required skill names from the JD (lowercased).
            preferred_skills: Preferred skill names from the JD (lowercased).
            jd_skill_embedding: Pre-computed embedding for JD skills text.
            weights: Sub-weights dict with keys 'lexical_weight', 'bm25_weight', 'embedding_weight'.
        """
        self.required_skills = set(s.lower().strip() for s in required_skills)
        self.preferred_skills = set(s.lower().strip() for s in preferred_skills)
        self.all_jd_skills = self.required_skills | self.preferred_skills
        self.jd_skill_embedding = jd_skill_embedding
        self.weights = weights or {
            "lexical_weight": 0.25,
            "bm25_weight": 0.35,
            "embedding_weight": 0.40,
        }

    def lexical_overlap(self, candidate_skills: List[str]) -> float:
        """Compute Jaccard-style lexical overlap between candidate and JD skills.

        Args:
            candidate_skills: List of candidate skill names (lowercased).

        Returns:
            Score in [0, 1] — weighted overlap of required + preferred skills.
        """
        candidate_set = set(s.lower().strip() for s in candidate_skills)

        if not self.all_jd_skills:
            return 0.0

        # Required skills are weighted more heavily
        required_match = len(candidate_set & self.required_skills)
        preferred_match = len(candidate_set & self.preferred_skills)

        required_score = required_match / max(len(self.required_skills), 1)
        preferred_score = preferred_match / max(len(self.preferred_skills), 1)

        # 70% weight on required, 30% on preferred
        return 0.7 * required_score + 0.3 * preferred_score

    def compute_trust_multiplier(
        self,
        endorsements: NDArray[np.float64],
        duration_months: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute trust multiplier for skill claims.

        trust = min(1, endorsements/10) * min(1, duration_months/12)

        Args:
            endorsements: Array of endorsement counts per candidate.
            duration_months: Array of average skill duration in months per candidate.

        Returns:
            Trust multiplier array in [0, 1].
        """
        endorsement_factor = np.minimum(1.0, endorsements / 10.0)
        duration_factor = np.minimum(1.0, duration_months / 12.0)
        return endorsement_factor * duration_factor

    def score_batch_lexical(
        self,
        candidate_skill_lists: List[List[str]],
    ) -> NDArray[np.float64]:
        """Score multiple candidates using lexical overlap.

        Args:
            candidate_skill_lists: List of skill name lists, one per candidate.

        Returns:
            Array of lexical overlap scores.
        """
        scores = np.array(
            [self.lexical_overlap(skills) for skills in candidate_skill_lists],
            dtype=np.float64,
        )
        return scores

    def score_batch_embedding(
        self,
        candidate_skill_embeddings: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Score multiple candidates using embedding cosine similarity.

        Args:
            candidate_skill_embeddings: (N, D) array of candidate skill embeddings.

        Returns:
            Array of cosine similarity scores in [0, 1].
        """
        if self.jd_skill_embedding is None:
            logger.warning("No JD skill embedding provided; returning zeros.")
            return np.zeros(len(candidate_skill_embeddings), dtype=np.float64)

        # Normalize embeddings
        jd_norm = self.jd_skill_embedding / (
            np.linalg.norm(self.jd_skill_embedding) + 1e-8
        )
        cand_norms = np.linalg.norm(candidate_skill_embeddings, axis=1, keepdims=True)
        cand_normalized = candidate_skill_embeddings / (cand_norms + 1e-8)

        # Matrix-vector cosine similarity
        similarities = cand_normalized @ jd_norm
        # Clamp to [0, 1]
        return np.clip(similarities.astype(np.float64), 0.0, 1.0)

    def score_batch_bm25(
        self,
        bm25_scores: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Normalize pre-computed BM25 scores to [0, 1].

        Args:
            bm25_scores: Raw BM25 scores per candidate.

        Returns:
            Normalized BM25 scores in [0, 1].
        """
        if bm25_scores.max() == 0:
            return np.zeros_like(bm25_scores)
        # Min-max normalization
        min_score = bm25_scores.min()
        max_score = bm25_scores.max()
        if max_score == min_score:
            return np.ones_like(bm25_scores) * 0.5
        return (bm25_scores - min_score) / (max_score - min_score)

    def score_batch(
        self,
        candidate_skill_lists: List[List[str]],
        candidate_skill_embeddings: NDArray[np.float32],
        bm25_scores: NDArray[np.float64],
        endorsements: NDArray[np.float64],
        duration_months: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute hybrid skill match scores for a batch of candidates.

        Combines lexical overlap, BM25, and embedding similarity, then
        applies trust multiplier.

        Args:
            candidate_skill_lists: Skill name lists per candidate.
            candidate_skill_embeddings: (N, D) skill embeddings.
            bm25_scores: Raw BM25 scores per candidate.
            endorsements: Average endorsements per candidate.
            duration_months: Average skill duration per candidate.

        Returns:
            Final hybrid skill match scores.
        """
        lexical = self.score_batch_lexical(candidate_skill_lists)
        embedding = self.score_batch_embedding(candidate_skill_embeddings)
        bm25 = self.score_batch_bm25(bm25_scores)
        trust = self.compute_trust_multiplier(endorsements, duration_months)

        hybrid = (
            self.weights["lexical_weight"] * lexical
            + self.weights["bm25_weight"] * bm25
            + self.weights["embedding_weight"] * embedding
        )

        # Apply trust multiplier — blend between raw and trusted
        # trust=1.0 means full trust, trust=0 means halved score
        result = hybrid * (0.5 + 0.5 * trust)

        return result
