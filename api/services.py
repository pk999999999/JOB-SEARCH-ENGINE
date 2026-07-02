"""
Business logic services for the API layer.

Wraps the existing ranking engine and provides CRUD for candidates and jobs.
Caches loaded artifacts in memory so repeated requests don't reload from disk.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from api.config import settings
from src.data_loader import (
    BehavioralSignals,
    Candidate,
    CandidateLoader,
    CareerEntry,
    EducationEntry,
    SkillEntry,
)
from src.jd_parser import (
    BehavioralThresholds,
    EducationRequirements,
    ExperienceRequirements,
    JobDescription,
    LocationRequirements,
)
from src.ranker import RankingPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate Service
# ---------------------------------------------------------------------------


class CandidateService:
    """Manages candidate data loading, caching, and queries."""

    def __init__(self) -> None:
        self._candidates: List[Candidate] = []
        self._candidates_map: Dict[str, Candidate] = {}
        self._loaded = False

    def load(self, filepath: Optional[str] = None) -> None:
        """Load candidates from disk.

        Args:
            filepath: Path to candidate CSV/JSON. Defaults to settings.
        """
        path = filepath or str(settings.data_path / "candidates.csv")
        if not Path(path).exists():
            logger.warning("Candidate file not found: %s", path)
            self._candidates = []
            self._candidates_map = {}
            self._loaded = True
            return

        loader = CandidateLoader(path)
        self._candidates = loader.load()
        self._candidates_map = {c.candidate_id: c for c in self._candidates}
        self._loaded = True
        logger.info("Loaded %d candidates into memory", len(self._candidates))

    @property
    def candidates(self) -> List[Candidate]:
        if not self._loaded:
            self.load()
        return self._candidates

    def get_by_id(self, candidate_id: str) -> Optional[Candidate]:
        """Get a single candidate by ID."""
        if not self._loaded:
            self.load()
        return self._candidates_map.get(candidate_id)

    def get_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        location: Optional[str] = None,
        min_experience: Optional[float] = None,
        max_experience: Optional[float] = None,
        skill: Optional[str] = None,
    ) -> Tuple[List[Candidate], int]:
        """Get filtered, paginated candidates.

        Returns:
            Tuple of (candidates_page, total_matching).
        """
        if not self._loaded:
            self.load()

        filtered = self._candidates

        if location:
            loc_lower = location.lower()
            filtered = [
                c for c in filtered if loc_lower in c.location.lower()
            ]

        if min_experience is not None:
            filtered = [
                c for c in filtered if c.years_of_experience >= min_experience
            ]

        if max_experience is not None:
            filtered = [
                c for c in filtered if c.years_of_experience <= max_experience
            ]

        if skill:
            skill_lower = skill.lower()
            filtered = [
                c for c in filtered
                if any(skill_lower in s.name.lower() for s in c.skills)
            ]

        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        return filtered[start:end], total

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics about loaded candidates."""
        if not self._loaded:
            self.load()

        if not self._candidates:
            return {
                "total_candidates": 0,
                "avg_experience_years": 0.0,
                "top_locations": [],
                "skill_distribution": [],
            }

        # Average experience
        yoe_values = [c.years_of_experience for c in self._candidates]
        avg_yoe = sum(yoe_values) / len(yoe_values)

        # Top locations
        location_counts = Counter(c.location for c in self._candidates if c.location)
        top_locations = [
            {"location": loc, "count": count}
            for loc, count in location_counts.most_common(10)
        ]

        # Skill distribution
        all_skills: List[str] = []
        for c in self._candidates:
            all_skills.extend(s.name for s in c.skills if s.name)
        skill_counts = Counter(all_skills)
        skill_distribution = [
            {"skill": skill, "count": count}
            for skill, count in skill_counts.most_common(15)
        ]

        return {
            "total_candidates": len(self._candidates),
            "avg_experience_years": round(avg_yoe, 1),
            "top_locations": top_locations,
            "skill_distribution": skill_distribution,
        }

    def add_candidate(self, cand_data: Dict[str, Any]) -> Candidate:
        """Ingest a new candidate, generate embeddings, and save to disk."""
        if not self._loaded:
            self.load()
            
        import pandas as pd
        from src.data_loader import _parse_candidate_row
        
        # 1. Ensure ID
        if not cand_data.get("candidate_id"):
            cand_data["candidate_id"] = str(uuid.uuid4())
            
        cand_id = cand_data["candidate_id"]
        
        # 2. Parse candidate
        candidate = _parse_candidate_row(cand_data)
        
        # 3. Generate embeddings on the fly
        # Lazy load model to save memory/startup time if unused
        if not hasattr(self, "_embed_model"):
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model for ingestion...")
            self._embed_model = SentenceTransformer(settings.embedding_model)
            
        narrative_text = candidate.get_narrative_text()
        skill_text = candidate.get_skills_text()
        
        narrative_text = narrative_text if narrative_text.strip() else "no information available"
        skill_text = skill_text if skill_text.strip() else "no skills listed"
        
        narrative_emb = self._embed_model.encode(
            [narrative_text], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        
        skill_emb = self._embed_model.encode(
            [skill_text], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        
        # 4. Append to ChromaDB collections
        from src.vector_store import get_narrative_collection, get_skill_collection
        
        narrative_col = get_narrative_collection()
        narrative_col.upsert(
            ids=[cand_id],
            embeddings=narrative_emb.tolist(),
            metadatas=[{"source": "api"}]
        )
        
        skill_col = get_skill_collection()
        skill_col.upsert(
            ids=[cand_id],
            embeddings=skill_emb.tolist(),
            metadatas=[{"source": "api"}]
        )
            
        # 5. Append to CSV
        csv_path = settings.data_path / "candidates.csv"
        # Sanitize years_of_experience to avoid NaN propagation (Bug #6)
        yoe = candidate.years_of_experience
        if yoe != yoe:  # NaN check
            yoe = 0.0
        # Flatten dict for CSV writing
        csv_row = {
            "candidate_id": cand_id,
            "headline": candidate.headline,
            "summary": candidate.summary,
            "current_title": candidate.current_title,
            "location": candidate.location,
            "years_of_experience": yoe,
            "skills": json.dumps([{"name": s.name, "endorsements": s.endorsements, "duration_months": s.duration_months, "proficiency": s.proficiency} for s in candidate.skills]),
            "career_history": json.dumps([{"title": c.title, "company": c.company, "start_date": c.start_date, "end_date": c.end_date, "description": c.description, "duration_months": c.duration_months, "is_current": c.is_current} for c in candidate.career_history]),
            "education": json.dumps([{"degree": e.degree, "field_of_study": e.field_of_study, "institution": e.institution, "year": e.year} for e in candidate.education]),
            "behavioral_signals": json.dumps({
                "last_active_date": candidate.behavioral_signals.last_active_date,
                "recruiter_response_rate": candidate.behavioral_signals.recruiter_response_rate,
                "avg_response_time_hours": candidate.behavioral_signals.avg_response_time_hours,
                "interview_completion_rate": candidate.behavioral_signals.interview_completion_rate,
                "offer_acceptance_rate": candidate.behavioral_signals.offer_acceptance_rate,
                "verified_email": candidate.behavioral_signals.verified_email,
                "verified_phone": candidate.behavioral_signals.verified_phone,
                "linkedin_connected": candidate.behavioral_signals.linkedin_connected,
            }),
            "salary_min": candidate.salary_min,
            "salary_max": candidate.salary_max,
        }
        df_new = pd.DataFrame([csv_row])
        df_new.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        
        # 6. Update in-memory state
        self._candidates.append(candidate)
        self._candidates_map[cand_id] = candidate
        
        logger.info("Successfully ingested candidate %s", cand_id)
        return candidate


# ---------------------------------------------------------------------------
# Job Service
# ---------------------------------------------------------------------------


class JobService:
    """Manages job description CRUD with YAML file storage."""

    def __init__(self) -> None:
        self._jobs_dir = settings.jobs_path
        self._jobs_dir.mkdir(parents=True, exist_ok=True)

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all saved job descriptions."""
        jobs = []

        # Include the default JD
        default_jd_path = settings.config_path / "jd_requirements.yaml"
        if default_jd_path.exists():
            with open(default_jd_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            data["id"] = "default"
            jobs.append(data)

        # Include custom JDs
        for path in sorted(self._jobs_dir.glob("*.yaml")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                data["id"] = path.stem
                jobs.append(data)
            except Exception as e:
                logger.warning("Failed to load job %s: %s", path, e)

        return jobs

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a single job description by ID."""
        if job_id == "default":
            path = settings.config_path / "jd_requirements.yaml"
        else:
            path = self._jobs_dir / f"{job_id}.yaml"

        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data["id"] = job_id
        return data

    def create_job(self, job_data: Dict[str, Any]) -> str:
        """Create a new job description.

        Returns:
            The generated job ID.
        """
        job_id = str(uuid.uuid4())[:8]
        path = self._jobs_dir / f"{job_id}.yaml"

        # Remove fields that are internal
        save_data = {k: v for k, v in job_data.items() if k != "id"}

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(save_data, f, default_flow_style=False, allow_unicode=True)

        logger.info("Created job %s at %s", job_id, path)
        return job_id

    def delete_job(self, job_id: str) -> bool:
        """Delete a job description. Cannot delete the default."""
        if job_id == "default":
            return False
        path = self._jobs_dir / f"{job_id}.yaml"
        if path.exists():
            path.unlink()
            return True
        return False


# ---------------------------------------------------------------------------
# Ranking Service
# ---------------------------------------------------------------------------


class RankingService:
    """Wraps the RankingPipeline for API use.

    Caches loaded artifacts so they don't need to be reloaded on every request.
    """

    def __init__(self, candidate_service: CandidateService) -> None:
        self._candidate_service = candidate_service

    def _build_jd_from_request(self, request_data: Dict[str, Any]) -> JobDescription:
        """Convert an API rank request into a JobDescription dataclass."""
        exp_data = request_data.get("experience", {})
        loc_data = request_data.get("location", {})
        edu_data = request_data.get("education", {})

        return JobDescription(
            job_title=request_data.get("job_title", "Senior AI Engineer"),
            description=request_data.get("description", ""),
            required_skills=request_data.get("required_skills", []),
            preferred_skills=request_data.get("preferred_skills", []),
            experience=ExperienceRequirements(
                ideal_min_years=exp_data.get("ideal_min_years", 5),
                ideal_max_years=exp_data.get("ideal_max_years", 9),
                absolute_min_years=exp_data.get("absolute_min_years", 3),
                absolute_max_years=exp_data.get("absolute_max_years", 15),
            ),
            location=LocationRequirements(
                preferred=loc_data.get("preferred", ["Pune", "Noida"]),
                good=loc_data.get("good", ["Mumbai", "Hyderabad", "Delhi NCR"]),
                acceptable_country=loc_data.get("acceptable_country", "India"),
            ),
            education=EducationRequirements(
                preferred_degrees=edu_data.get("preferred_degrees", []),
                preferred_fields=edu_data.get("preferred_fields", []),
                acceptable_degrees=edu_data.get("acceptable_degrees", []),
            ),
        )

    def rank(
        self,
        request_data: Dict[str, Any],
        top_k: int = 100,
    ) -> Dict[str, Any]:
        """Execute the ranking pipeline.

        Args:
            request_data: JD fields from the API request.
            top_k: Number of top candidates to return.

        Returns:
            Dictionary with ranking results including candidates, timing, etc.
        """
        start_time = time.time()

        candidates = self._candidate_service.candidates
        if not candidates:
            return {
                "job_title": request_data.get("job_title", ""),
                "total_candidates": 0,
                "ranked_count": 0,
                "elapsed_seconds": 0.0,
                "candidates": [],
            }

        jd = self._build_jd_from_request(request_data)

        # Load weights config
        weights_path = settings.config_path / "weights.yaml"
        weights_config = str(weights_path) if weights_path.exists() else None

        # Run pipeline
        pipeline = RankingPipeline(
            candidates=candidates,
            jd=jd,
            artifacts_dir=str(settings.artifacts_path),
            weights_config=weights_config,
            top_k=top_k,
        )

        results_df = pipeline.rank()
        elapsed = time.time() - start_time

        # Build response
        ranked_candidates = []
        candidates_map = {c.candidate_id: c for c in candidates}

        for _, row in results_df.iterrows():
            cand = candidates_map.get(row["candidate_id"])
            
            # Format education summary
            edu_str = ""
            if cand and cand.education:
                first_edu = cand.education[0]
                if first_edu.degree and first_edu.field_of_study:
                    edu_str = f"{first_edu.degree} in {first_edu.field_of_study}"
                elif first_edu.degree:
                    edu_str = first_edu.degree
                elif first_edu.field_of_study:
                    edu_str = first_edu.field_of_study
            
            entry: Dict[str, Any] = {
                "candidate_id": row["candidate_id"],
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "reasoning": row.get("reasoning", ""),
                "headline": cand.headline if cand else "",
                "current_title": cand.current_title if cand else "",
                "location": cand.location if cand else "",
                "years_of_experience": cand.years_of_experience if cand else 0.0,
                "education": edu_str,
                "skills": (
                    [
                        {
                            "name": s.name,
                            "endorsements": s.endorsements,
                            "duration_months": s.duration_months,
                            "proficiency": s.proficiency,
                        }
                        for s in cand.skills
                    ]
                    if cand
                    else []
                ),
            }
            ranked_candidates.append(entry)

        return {
            "job_title": jd.job_title,
            "total_candidates": len(candidates),
            "ranked_count": len(ranked_candidates),
            "elapsed_seconds": round(elapsed, 2),
            "candidates": ranked_candidates,
        }
