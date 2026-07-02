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
        candidate_ids: list[str],
    ) -> NDArray[np.float64]:
        """Compute career match scores for a batch of candidates.

        Queries ChromaDB for cosine distances and maps back to scores.

        Args:
            candidate_ids: List of candidate IDs to score.

        Returns:
            Array of career match scores in [0, 1].
        """
        if self.jd_embedding_normalized is None or not candidate_ids:
            logger.warning("No JD embedding provided or empty candidates; returning zeros.")
            return np.zeros(len(candidate_ids), dtype=np.float64)

        from src.vector_store import get_narrative_collection
        
        try:
            col = get_narrative_collection()
            count = col.count()
            if count == 0:
                return np.zeros(len(candidate_ids), dtype=np.float64)
                
            results = col.query(
                query_embeddings=[self.jd_embedding_normalized.tolist()],
                n_results=count
            )
            
            # Map ids to distances
            dist_map = {}
            if results["ids"] and results["distances"]:
                for cid, dist in zip(results["ids"][0], results["distances"][0]):
                    dist_map[cid] = dist
                    
            # Reconstruct array in original candidate_ids order
            scores = []
            for cid in candidate_ids:
                if cid in dist_map:
                    # distance is cosine distance (1 - cos_sim)
                    sim = 1.0 - dist_map[cid]
                    scores.append(max(0.0, min(1.0, sim)))
                else:
                    scores.append(0.0)
                    
            return np.array(scores, dtype=np.float64)
            
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            return np.zeros(len(candidate_ids), dtype=np.float64)
