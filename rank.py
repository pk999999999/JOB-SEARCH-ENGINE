#!/usr/bin/env python3
"""
rank.py — Main entry point for the Redrob candidate ranking system.

Usage:
    python rank.py --candidates data/candidates.csv --artifacts artifacts/ --output output/top_100.csv

Flow:
    Load artifacts → Compute scores → Apply penalties → Apply modifiers
    → Sort descending → Tie break by candidate_id → Top 100
    → Generate reasoning → Write CSV
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src.data_loader import CandidateLoader
from src.jd_parser import JDParser
from src.ranker import RankingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rank")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rank candidates against a job description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python rank.py \\
        --candidates data/candidates.csv \\
        --artifacts artifacts/ \\
        --output output/top_100.csv \\
        --jd config/jd_requirements.yaml \\
        --weights config/weights.yaml \\
        --top-k 100
        """,
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="data/candidates.csv",
        help="Path to candidate data file (CSV or JSON).",
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default="artifacts/",
        help="Path to precomputed artifacts directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/top_100.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--jd",
        type=str,
        default="config/jd_requirements.yaml",
        help="Path to JD requirements YAML.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="config/weights.yaml",
        help="Path to weights YAML configuration.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top candidates to output.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for ranking.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("REDROB AI RANKER — Starting ranking pipeline")
    logger.info("=" * 60)

    try:
        # 1. Load candidates
        logger.info("Loading candidates from: %s", args.candidates)
        loader = CandidateLoader(args.candidates)
        candidates = loader.load()
        logger.info("Loaded %d candidates", len(candidates))

        # 2. Parse JD
        logger.info("Parsing job description from: %s", args.jd)
        jd_parser = JDParser(args.jd)
        jd = jd_parser.parse()
        logger.info("JD: %s", jd.job_title)

        # 3. Run ranking pipeline
        weights_path = Path(args.weights)
        weights_config = str(weights_path) if weights_path.exists() else None
        if weights_config is None:
            logger.warning("Weights config not found at %s — using defaults", args.weights)

        pipeline = RankingPipeline(
            candidates=candidates,
            jd=jd,
            artifacts_dir=args.artifacts,
            weights_config=weights_config,
            top_k=args.top_k,
        )

        results = pipeline.rank()

        # 4. Export
        pipeline.export_csv(results, args.output)

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("RANKING COMPLETE")
        logger.info("  Candidates processed: %d", len(candidates))
        logger.info("  Top %d exported to: %s", args.top_k, args.output)
        logger.info("  Total time: %.2f seconds", elapsed)
        logger.info("  Top score: %.4f", results["score"].iloc[0] if len(results) > 0 else 0)
        logger.info("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except Exception as e:
        logger.exception("Ranking failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
