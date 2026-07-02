"""
Core ranker module — orchestrates the full ranking pipeline.

Loads precomputed artifacts, computes all feature scores,
applies modifiers, sorts, and generates reasoning for top-K.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from src.data_loader import Candidate, CandidateLoader
from src.jd_parser import JDParser, JobDescription
from src.features.skill_match import SkillMatcher
from src.features.title_career_match import CareerMatcher
from src.features.experience_fit import ExperienceFitter
from src.features.location_fit import LocationFitter
from src.features.education_fit import EducationFitter
from src.features.disqualifiers import DisqualifierChecker
from src.features.behavioral_modifier import BehavioralModifier
from src.features.coherence_score import CoherenceScorer
from src.honeypot_detector import HoneypotDetector
from src.scoring import Scorer
from src.reasoning import ReasoningGenerator

logger = logging.getLogger(__name__)


class RankingPipeline:
    """End-to-end candidate ranking pipeline.

    Orchestrates:
    1. Loading precomputed artifacts
    2. Computing feature scores
    3. Applying modifiers (coherence, behavioral, honeypot)
    4. Sorting and tie-breaking
    5. Generating reasoning for top-K
    6. CSV export
    """

    def __init__(
        self,
        candidates: List[Candidate],
        jd: JobDescription,
        artifacts_dir: str | Path,
        weights_config: Optional[str | Path] = None,
        top_k: int = 100,
    ) -> None:
        """Initialize the ranking pipeline.

        Args:
            candidates: List of all candidate objects.
            jd: Parsed job description.
            artifacts_dir: Path to precomputed artifacts directory.
            weights_config: Path to weights.yaml (optional).
            top_k: Number of top candidates to output.
        """
        self.candidates = candidates
        self.jd = jd
        self.artifacts_dir = Path(artifacts_dir)
        self.top_k = top_k

        # Load weights config
        self.weights_config: Dict[str, Any] = {}
        if weights_config:
            with open(weights_config, "r", encoding="utf-8") as f:
                self.weights_config = yaml.safe_load(f)

        # Initialize components
        self._init_components()

    def _init_components(self) -> None:
        """Initialize all scoring components."""
        # Load artifacts
        raw_embeddings = self._load_artifact("embeddings.npy")
        artifact_ids = self._load_artifact("candidate_ids.npy")

        # Align embeddings to the order of self.candidates using the stored
        # candidate_ids index.  Without this, any reordering or filtering of
        # the input CSV between precompute and rank stages would silently
        # assign every candidate the wrong embedding.
        if raw_embeddings is not None and artifact_ids is not None:
            artifact_id_to_pos = {str(cid): i for i, cid in enumerate(artifact_ids)}
            n = len(self.candidates)
            dim = raw_embeddings.shape[1]
            self.embeddings = np.zeros((n, dim), dtype=raw_embeddings.dtype)
            missing = 0
            for row_idx, cand in enumerate(self.candidates):
                pos = artifact_id_to_pos.get(str(cand.candidate_id))
                if pos is not None:
                    self.embeddings[row_idx] = raw_embeddings[pos]
                else:
                    missing += 1
            if missing:
                logger.warning(
                    "%d/%d candidates have no precomputed embedding; "
                    "their rows will be zero-vectors (career_match → 0).",
                    missing, n,
                )

        
        # Load JD embeddings directly
        jd_emb_path = self.artifacts_dir / "jd_embedding.npy"
        if jd_emb_path.exists():
            self.jd_embedding = np.load(str(jd_emb_path))
        else:
            self.jd_embedding = None

        jd_skill_path = self.artifacts_dir / "jd_skill_embedding.npy"
        if jd_skill_path.exists():
            self.jd_skill_embedding = np.load(str(jd_skill_path))
        else:
            self.jd_skill_embedding = None

        # Load coherence scores
        coherence_path = self.artifacts_dir / "coherence_scores.parquet"
        if coherence_path.exists():
            self.coherence_df = pd.read_parquet(str(coherence_path))
        else:
            self.coherence_df = None

        # Load honeypot flags
        honeypot_path = self.artifacts_dir / "honeypot_flags.parquet"
        if honeypot_path.exists():
            self.honeypot_df = pd.read_parquet(str(honeypot_path))
        else:
            self.honeypot_df = None

        # Load BM25 index
        bm25_path = self.artifacts_dir / "bm25_index.pkl"
        if bm25_path.exists():
            with open(str(bm25_path), "rb") as f:
                self.bm25_index = pickle.load(f)
        else:
            self.bm25_index = None

        # Scoring weights
        base_weights = self.weights_config.get("base_score", None)
        self.scorer = Scorer(weights=base_weights)

        # Feature modules
        skill_weights = self.weights_config.get("skill_match", None)
        self.skill_matcher = SkillMatcher(
            required_skills=self.jd.get_required_skills_lower(),
            preferred_skills=[s.lower() for s in self.jd.preferred_skills],
            jd_skill_embedding=self.jd_skill_embedding,
            weights=skill_weights,
        )

        self.career_matcher = CareerMatcher(jd_embedding=self.jd_embedding)

        exp_config = self.weights_config.get("experience", {})
        self.experience_fitter = ExperienceFitter(
            ideal_min=exp_config.get("ideal_min", 5),
            ideal_max=exp_config.get("ideal_max", 9),
            sigma=exp_config.get("sigma", 3.0),
        )

        loc_config = self.weights_config.get("location", {})
        # Only pass tier_scores if it contains the required tier keys
        valid_tier_keys = {"preferred", "good", "acceptable_country", "outside_country"}
        if loc_config and valid_tier_keys.issubset(loc_config.keys()):
            tier_scores = {k: loc_config[k] for k in valid_tier_keys}
        else:
            tier_scores = None
        self.location_fitter = LocationFitter(
            preferred=self.jd.location.preferred,
            good=self.jd.location.good,
            acceptable_country=self.jd.location.acceptable_country,
            tier_scores=tier_scores,
        )

        self.education_fitter = EducationFitter(
            preferred_fields=[f.lower() for f in self.jd.education.preferred_fields],
        )

        self.disqualifier = DisqualifierChecker(
            required_skills=self.jd.get_required_skills_lower(),
        )

        beh_config = self.weights_config.get("behavioral", {})
        self.behavioral_modifier = BehavioralModifier(
            min_multiplier=beh_config.get("min_multiplier", 0.55),
            max_multiplier=beh_config.get("max_multiplier", 1.15),
            weights=beh_config.get("weights", None),
        )

        coh_config = self.weights_config.get("coherence", {})
        self.coherence_scorer = CoherenceScorer(
            min_factor=coh_config.get("min_factor", 0.70),
            max_factor=coh_config.get("max_factor", 1.10),
        )

        self.honeypot_detector = HoneypotDetector()
        self.reasoning_generator = ReasoningGenerator()

    def _load_artifact(self, filename: str) -> Optional[np.ndarray]:
        """Load a numpy artifact file.

        Args:
            filename: Name of the artifact file.

        Returns:
            Numpy array or None if not found.
        """
        path = self.artifacts_dir / filename
        if path.exists():
            return np.load(str(path))
        logger.warning("Artifact not found: %s", path)
        return None

    def _compute_bm25_scores(self) -> NDArray[np.float64]:
        """Compute BM25 scores for all candidates.

        Returns:
            Array of BM25 scores.
        """
        if self.bm25_index is None:
            return np.zeros(len(self.candidates), dtype=np.float64)

        query = " ".join(self.jd.get_all_skills())
        query_tokens = query.lower().split()
        scores = self.bm25_index.get_scores(query_tokens)
        return np.array(scores, dtype=np.float64)

    def _compute_endorsements_and_durations(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Extract average endorsements and durations per candidate.

        Returns:
            Tuple of (endorsements, durations) arrays.
        """
        endorsements = []
        durations = []
        for c in self.candidates:
            if c.skills:
                avg_end = np.mean([s.endorsements for s in c.skills])
                avg_dur = np.mean([s.duration_months for s in c.skills])
            else:
                avg_end = 0.0
                avg_dur = 0.0
            endorsements.append(avg_end)
            durations.append(avg_dur)

        return (
            np.array(endorsements, dtype=np.float64),
            np.array(durations, dtype=np.float64),
        )

    def rank(self) -> pd.DataFrame:
        """Execute the full ranking pipeline.

        Returns:
            DataFrame with columns: candidate_id, rank, score, reasoning
        """
        start_time = time.time()
        n = len(self.candidates)
        logger.info("Starting ranking pipeline for %d candidates", n)

        candidate_ids_list = [c.candidate_id for c in self.candidates]
        
        # 1. Career match (embedding similarity)
        if self.jd_embedding is not None:
            career_scores = self.career_matcher.score_batch(candidate_ids_list)
        else:
            career_scores = np.full(n, 0.5, dtype=np.float64)

        # 2. Skill match (hybrid)
        skill_lists = [c.get_skill_names() for c in self.candidates]
        bm25_scores = self._compute_bm25_scores()
        endorsements, durations = self._compute_endorsements_and_durations()

        if self.jd_skill_embedding is not None:
            skill_scores = self.skill_matcher.score_batch(
                candidate_ids=candidate_ids_list,
                candidate_skill_lists=skill_lists,
                bm25_scores=bm25_scores,
                endorsements=endorsements,
                duration_months=durations,
            )
        else:
            lexical = self.skill_matcher.score_batch_lexical(skill_lists)
            trust = self.skill_matcher.compute_trust_multiplier(endorsements, durations)
            skill_scores = lexical * (0.5 + 0.5 * trust)

        # 3. Experience fit
        yoe_array = np.array(
            [c.years_of_experience for c in self.candidates], dtype=np.float64
        )
        experience_scores = self.experience_fitter.score_batch(yoe_array)

        # 4. Location fit
        locations = [c.location for c in self.candidates]
        location_scores = self.location_fitter.score_batch(locations)

        # 5. Education fit
        education_lists = [c.education for c in self.candidates]
        education_scores = self.education_fitter.score_batch(education_lists)

        # 6. Rule adjustments (disqualifiers)
        rule_scores = self.disqualifier.score_batch(self.candidates)

        # 7. Behavioral modifier
        behavioral_signals = [c.behavioral_signals for c in self.candidates]
        behavioral_modifiers = self.behavioral_modifier.score_batch(behavioral_signals)

        # 8. Coherence factor
        if self.coherence_df is not None:
            # Map coherence scores by candidate_id
            coherence_map = dict(
                zip(
                    self.coherence_df["candidate_id"].astype(str),
                    self.coherence_df["coherence_score"],
                )
            )
            raw_coherence = np.array(
                [coherence_map.get(c.candidate_id, 0.5) for c in self.candidates],
                dtype=np.float64,
            )
        else:
            raw_coherence = np.full(n, 0.5, dtype=np.float64)

        coherence_factors = self.coherence_scorer.coherence_to_factor(raw_coherence)

        # 9. Honeypot gate
        if self.honeypot_df is not None:
            honeypot_map = dict(
                zip(
                    self.honeypot_df["candidate_id"].astype(str),
                    self.honeypot_df["gate_value"],
                )
            )
            honeypot_gates = np.array(
                [honeypot_map.get(c.candidate_id, 1.0) for c in self.candidates],
                dtype=np.float64,
            )
        else:
            honeypot_gates = self.honeypot_detector.get_gate_values(self.candidates)

        # 10. Compute base and final scores
        base_scores = self.scorer.compute_base_scores(
            career_match=career_scores,
            skill_match=skill_scores,
            experience_fit=experience_scores,
            location_fit=location_scores,
            education_fit=education_scores,
            rule_adjustments=rule_scores,
        )

        final_scores = self.scorer.compute_final_scores(
            base_scores=base_scores,
            coherence_factors=coherence_factors,
            behavioral_modifiers=behavioral_modifiers,
            honeypot_gates=honeypot_gates,
        )

        # 11. Sort: descending by score, tie-break by candidate_id ascending
        candidate_id_array = np.array(
            [c.candidate_id for c in self.candidates]
        )
        # Create sort order: primary = -score, secondary = candidate_id
        sort_indices = np.lexsort((candidate_id_array, -final_scores))

        # Take top K
        top_indices = sort_indices[: self.top_k]

        # 12. Generate reasoning for top K
        top_candidates = [self.candidates[i] for i in top_indices]
        top_breakdowns = []
        for i in top_indices:
            breakdown = self.scorer.get_score_breakdown(
                index=i,
                career_match=career_scores,
                skill_match=skill_scores,
                experience_fit=experience_scores,
                location_fit=location_scores,
                education_fit=education_scores,
                rule_adjustments=rule_scores,
                coherence_factors=coherence_factors,
                behavioral_modifiers=behavioral_modifiers,
                honeypot_gates=honeypot_gates,
            )
            top_breakdowns.append(breakdown)

        top_ranks = list(range(1, len(top_indices) + 1))
        reasonings = self.reasoning_generator.generate_batch(
            candidates=top_candidates,
            score_breakdowns=top_breakdowns,
            ranks=top_ranks,
        )

        # 13. Build output DataFrame
        output = pd.DataFrame({
            "candidate_id": [self.candidates[i].candidate_id for i in top_indices],
            "rank": top_ranks,
            "score": [float(final_scores[i]) for i in top_indices],
            "reasoning": reasonings,
        })

        elapsed = time.time() - start_time
        logger.info(
            "Ranking complete: %d candidates ranked in %.2f seconds. Top score: %.4f",
            n,
            elapsed,
            output["score"].iloc[0] if len(output) > 0 else 0,
        )

        return output

    def export_csv(self, output: pd.DataFrame, filepath: str | Path) -> None:
        """Export ranking results to CSV.

        Args:
            output: Ranking results DataFrame.
            filepath: Output CSV file path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(filepath, index=False)
        logger.info("Results exported to %s", filepath)
