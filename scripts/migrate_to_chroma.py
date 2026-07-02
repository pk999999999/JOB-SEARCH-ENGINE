"""
Migrate embeddings from .npy files to ChromaDB.
"""
import sys
import os
from pathlib import Path
import logging
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import get_narrative_collection, get_skill_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

def migrate():
    artifacts_dir = Path("artifacts")
    
    emb_path = artifacts_dir / "embeddings.npy"
    skill_emb_path = artifacts_dir / "skill_embeddings.npy"
    ids_path = artifacts_dir / "candidate_ids.npy"
    
    if not (emb_path.exists() and skill_emb_path.exists() and ids_path.exists()):
        logger.error("Artifact files (.npy) not found. Nothing to migrate.")
        return
        
    logger.info("Loading .npy files...")
    embeddings = np.load(emb_path)
    skill_embeddings = np.load(skill_emb_path)
    candidate_ids = np.load(ids_path)
    
    num_candidates = len(candidate_ids)
    logger.info(f"Found {num_candidates} candidates to migrate.")
    
    narrative_col = get_narrative_collection()
    skill_col = get_skill_collection()
    
    # Process in batches to avoid memory/network overload (though local Chroma is fast)
    batch_size = 500
    for i in range(0, num_candidates, batch_size):
        end = min(i + batch_size, num_candidates)
        
        batch_ids = candidate_ids[i:end].tolist()
        batch_emb = embeddings[i:end].tolist()
        batch_skill_emb = skill_embeddings[i:end].tolist()
        
        logger.info(f"Upserting batch {i} to {end}...")
        
        narrative_col.upsert(
            ids=batch_ids,
            embeddings=batch_emb,
            metadatas=[{"source": "migration"} for _ in batch_ids]
        )
        
        skill_col.upsert(
            ids=batch_ids,
            embeddings=batch_skill_emb,
            metadatas=[{"source": "migration"} for _ in batch_ids]
        )
        
    logger.info("Migration complete!")

if __name__ == "__main__":
    migrate()
