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
        """Initialize the skill matcher."""
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
        """Compute Jaccard-style lexical overlap between candidate and JD skills."""
        candidate_set = set(s.lower().strip() for s in candidate_skills)

        if not self.all_jd_skills:
            return 0.0

        required_match = len(candidate_set & self.required_skills)
        preferred_match = len(candidate_set & self.preferred_skills)

        required_score = required_match / max(len(self.required_skills), 1)
        preferred_score = preferred_match / max(len(self.preferred_skills), 1)

        return 0.7 * required_score + 0.3 * preferred_score

    def compute_trust_multiplier(
        self,
        endorsements: NDArray[np.float64],
        duration_months: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute trust multiplier for skill claims."""
        endorsement_factor = np.minimum(1.0, endorsements / 10.0)
        duration_factor = np.minimum(1.0, duration_months / 12.0)
        return endorsement_factor * duration_factor

    def score_batch_lexical(
        self,
        candidate_skill_lists: List[List[str]],
    ) -> NDArray[np.float64]:
        """Score multiple candidates using lexical overlap."""
        scores = np.array(
            [self.lexical_overlap(skills) for skills in candidate_skill_lists],
            dtype=np.float64,
        )
        return scores

    def score_batch_embedding(
        self,
        candidate_ids: list[str],
    ) -> NDArray[np.float64]:
        """Score multiple candidates using embedding cosine similarity via ChromaDB."""
        if self.jd_skill_embedding is None or not candidate_ids:
            logger.warning("No JD skill embedding provided or empty candidates; returning zeros.")
            return np.zeros(len(candidate_ids), dtype=np.float64)

        jd_norm = self.jd_skill_embedding / (
            np.linalg.norm(self.jd_skill_embedding) + 1e-8
        )

        from src.vector_store import get_skill_collection
        
        try:
            col = get_skill_collection()
            count = col.count()
            if count == 0:
                return np.zeros(len(candidate_ids), dtype=np.float64)
                
            results = col.query(
                query_embeddings=[jd_norm.tolist()],
                n_results=count
            )
            
            dist_map = {}
            if results["ids"] and results["distances"]:
                for cid, dist in zip(results["ids"][0], results["distances"][0]):
                    dist_map[cid] = dist
                    
            scores = []
            for cid in candidate_ids:
                if cid in dist_map:
                    sim = 1.0 - dist_map[cid]
                    scores.append(max(0.0, min(1.0, sim)))
                else:
                    scores.append(0.0)
                    
            return np.array(scores, dtype=np.float64)
            
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            return np.zeros(len(candidate_ids), dtype=np.float64)

    def score_batch_bm25(
        self,
        bm25_scores: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Normalize pre-computed BM25 scores to [0, 1]."""
        if bm25_scores.max() == 0:
            return np.zeros_like(bm25_scores)
        min_score = bm25_scores.min()
        max_score = bm25_scores.max()
        if max_score == min_score:
            return np.ones_like(bm25_scores) * 0.5
        return (bm25_scores - min_score) / (max_score - min_score)

    def score_batch(
        self,
        candidate_ids: list[str],
        candidate_skill_lists: List[List[str]],
        bm25_scores: NDArray[np.float64],
        endorsements: NDArray[np.float64],
        duration_months: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute hybrid skill match scores for a batch of candidates."""
        lexical = self.score_batch_lexical(candidate_skill_lists)
        embedding = self.score_batch_embedding(candidate_ids)
        bm25 = self.score_batch_bm25(bm25_scores)
        trust = self.compute_trust_multiplier(endorsements, duration_months)

        target_len = len(candidate_ids)

        # 🛠️ FIXED ALIGNMENT: Drops the query score at index 0 to map 
        # candidate scores accurately back to their profiles without shifting.
        if len(bm25) == target_len + 1:
            bm25 = bm25[1:]  
        elif len(bm25) > target_len:
            bm25 = bm25[:target_len]

        # Guard rails for lengths of other arrays
        def final_ensure(vec: NDArray[np.float64]) -> NDArray[np.float64]:
            if len(vec) == target_len:
                return vec
            if len(vec) > target_len:
                return vec[:target_len]
            padded = np.zeros(target_len, dtype=vec.dtype)
            padded[:len(vec)] = vec
            return padded

        lexical = final_ensure(lexical)
        embedding = final_ensure(final_ensure(embedding))
        bm25 = final_ensure(bm25)
        trust = final_ensure(trust)

        hybrid = (
            self.weights["lexical_weight"] * lexical
            + self.weights["bm25_weight"] * bm25
            + self.weights["embedding_weight"] * embedding
        )

        result = hybrid * (0.5 + 0.5 * trust)
        return result