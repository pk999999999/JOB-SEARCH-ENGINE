#!/usr/bin/env python3
"""
Precompute embeddings and artifacts for offline ranking.

This script runs BEFORE the ranking stage (requires internet for model download).
It generates all artifacts needed by rank.py.

Usage:
    python scripts/precompute_embeddings.py --input data/candidates.csv --output artifacts/

Outputs:
    artifacts/
        embeddings.npy          — (N, 384) narrative embeddings
        candidate_ids.npy       — (N,) candidate ID array
        skill_embeddings.npy    — (N, 384) skill text embeddings
        jd_embedding.npy        — (384,) JD embedding
        jd_skill_embedding.npy  — (384,) JD skills embedding
        coherence_scores.parquet
        honeypot_flags.parquet
        bm25_index.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import CandidateLoader
from src.jd_parser import JDParser
from src.honeypot_detector import HoneypotDetector
from src.features.coherence_score import CoherenceScorer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute embeddings and artifacts for offline ranking."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/candidates.csv",
        help="Path to candidate data file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/",
        help="Path to output artifacts directory.",
    )
    parser.add_argument(
        "--jd",
        type=str,
        default="config/jd_requirements.yaml",
        help="Path to JD requirements YAML.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding generation.",
    )
    return parser.parse_args()


def generate_embeddings(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> np.ndarray:
    """Generate embeddings for a list of texts using sentence-transformers.

    Args:
        texts: List of text strings to embed.
        model_name: Name of the sentence-transformer model.
        batch_size: Batch size for encoding.

    Returns:
        (N, D) numpy array of embeddings (float32).
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading model: %s", model_name)
    model = SentenceTransformer(model_name)

    logger.info("Generating embeddings for %d texts (batch_size=%d)", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype(np.float32)


def main() -> int:
    """Main entry point for precomputation.

    Returns:
        Exit code.
    """
    args = parse_args()
    start_time = time.time()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PRECOMPUTE EMBEDDINGS — Starting")
    logger.info("=" * 60)

    # 1. Load candidates
    logger.info("Loading candidates from: %s", args.input)
    loader = CandidateLoader(args.input)
    candidates = loader.load()
    logger.info("Loaded %d candidates", len(candidates))

    # 2. Parse JD
    logger.info("Parsing JD from: %s", args.jd)
    jd_parser = JDParser(args.jd)
    jd = jd_parser.parse()

    # 3. Build narrative texts
    logger.info("Building narrative texts...")
    narrative_texts = [c.get_narrative_text() for c in candidates]
    skill_texts = [c.get_skills_text() for c in candidates]
    career_texts = [c.get_career_text() for c in candidates]

    # Replace empty texts with placeholder
    narrative_texts = [t if t.strip() else "no information available" for t in narrative_texts]
    skill_texts = [t if t.strip() else "no skills listed" for t in skill_texts]
    career_texts = [t if t.strip() else "no career history" for t in career_texts]

    # 4. Generate embeddings
    logger.info("Generating narrative embeddings...")
    narrative_embeddings = generate_embeddings(narrative_texts, args.model, args.batch_size)

    logger.info("Generating skill embeddings...")
    skill_embeddings = generate_embeddings(skill_texts, args.model, args.batch_size)

    logger.info("Generating career embeddings...")
    career_embeddings = generate_embeddings(career_texts, args.model, args.batch_size)

    # 5. Generate JD embeddings
    jd_text = jd.get_jd_text()
    jd_skills_text = " ".join(jd.get_all_skills())

    logger.info("Generating JD embeddings...")
    jd_embedding = generate_embeddings([jd_text], args.model, args.batch_size)[0]
    jd_skill_embedding = generate_embeddings([jd_skills_text], args.model, args.batch_size)[0]

    # 6. Compute coherence scores
    logger.info("Computing coherence scores...")
    coherence_scorer = CoherenceScorer()
    coherence_scores = coherence_scorer.score_batch(
        narrative_embeddings=narrative_embeddings,
        career_embeddings=career_embeddings,
        skill_embeddings=skill_embeddings,
    )

    # 7. Detect honeypots
    logger.info("Running honeypot detection...")
    honeypot_detector = HoneypotDetector()
    honeypot_results = honeypot_detector.detect_batch(candidates)

    # 8. Build BM25 index
    logger.info("Building BM25 index...")
    tokenized_skills = [text.lower().split() for text in skill_texts]
    bm25_index = BM25Okapi(tokenized_skills)

    # 9. Save artifacts
    logger.info("Saving artifacts to: %s", output_dir)

    candidate_ids = np.array([c.candidate_id for c in candidates])

    np.save(str(output_dir / "embeddings.npy"), narrative_embeddings)
    np.save(str(output_dir / "candidate_ids.npy"), candidate_ids)
    np.save(str(output_dir / "skill_embeddings.npy"), skill_embeddings)
    np.save(str(output_dir / "jd_embedding.npy"), jd_embedding)
    np.save(str(output_dir / "jd_skill_embedding.npy"), jd_skill_embedding)

    # Coherence scores
    coherence_df = pd.DataFrame({
        "candidate_id": candidate_ids,
        "coherence_score": coherence_scores,
    })
    coherence_df.to_parquet(str(output_dir / "coherence_scores.parquet"), index=False)

    # Honeypot flags
    honeypot_df = pd.DataFrame({
        "candidate_id": [r.candidate_id for r in honeypot_results],
        "is_honeypot": [r.is_honeypot for r in honeypot_results],
        "gate_value": [r.gate_value for r in honeypot_results],
        "strong_count": [r.strong_count for r in honeypot_results],
        "medium_count": [r.medium_count for r in honeypot_results],
        "weak_count": [r.weak_count for r in honeypot_results],
    })
    honeypot_df.to_parquet(str(output_dir / "honeypot_flags.parquet"), index=False)

    # BM25 index
    with open(str(output_dir / "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25_index, f)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PRECOMPUTE COMPLETE")
    logger.info("  Candidates: %d", len(candidates))
    logger.info("  Embedding dim: %d", narrative_embeddings.shape[1])
    logger.info("  Honeypots flagged: %d", sum(1 for r in honeypot_results if r.is_honeypot))
    logger.info("  Time: %.2f seconds", elapsed)
    logger.info("  Artifacts saved to: %s", output_dir)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
