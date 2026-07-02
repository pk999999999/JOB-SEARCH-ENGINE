"""
Vector Database connection manager using ChromaDB.
"""
import os

import chromadb
from chromadb.config import Settings
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Singleton pattern for the ChromaDB client
_client = None

def get_chroma_client() -> chromadb.ClientAPI:
    """Get or create the ChromaDB client instance."""
    global _client
    if _client is None:
        # Resolve relative to project root (parent of src/)
        project_root = Path(__file__).resolve().parent.parent
        db_path = project_root / "data" / "chroma"
        db_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at {db_path.absolute()}")
        _client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    return _client

def get_narrative_collection():
    """Get the collection for candidate narrative (career) embeddings."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="candidates_narrative",
        metadata={"hnsw:space": "cosine"}
    )

def get_skill_collection():
    """Get the collection for candidate skill embeddings."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="candidates_skills",
        metadata={"hnsw:space": "cosine"}
    )
