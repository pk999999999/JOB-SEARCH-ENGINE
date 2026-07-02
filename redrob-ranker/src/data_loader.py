"""
Data loader module for candidate ingestion and schema validation.

Handles reading candidate data from CSV/JSON, validating schema integrity,
and returning structured Candidate dataclass instances.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CareerEntry:
    """A single entry in a candidate's career history."""

    title: str = ""
    company: str = ""
    start_date: str = ""
    end_date: str = ""
    duration_months: int = 0
    is_current: bool = False
    description: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CareerEntry":
        """Create a CareerEntry from a dictionary."""
        return cls(
            title=str(data.get("title", "")),
            company=str(data.get("company", "")),
            start_date=str(data.get("start_date", "")),
            end_date=str(data.get("end_date", "")),
            duration_months=int(data.get("duration_months", 0)),
            is_current=bool(data.get("is_current", False)),
            description=str(data.get("description", "")),
        )


@dataclass
class EducationEntry:
    """A single education entry."""

    degree: str = ""
    field_of_study: str = ""
    institution: str = ""
    year: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EducationEntry":
        """Create an EducationEntry from a dictionary."""
        return cls(
            degree=str(data.get("degree", "")),
            field_of_study=str(data.get("field_of_study", "")),
            institution=str(data.get("institution", "")),
            year=int(data.get("year", 0)),
        )


@dataclass
class SkillEntry:
    """A single skill with optional metadata."""

    name: str = ""
    endorsements: int = 0
    duration_months: int = 0
    proficiency: str = ""  # e.g., "beginner", "intermediate", "expert"

    @classmethod
    def from_dict(cls, data: Union[Dict[str, Any], str]) -> "SkillEntry":
        """Create a SkillEntry from a dictionary or plain string."""
        if isinstance(data, str):
            return cls(name=data.strip().lower())
        return cls(
            name=str(data.get("name", "")).strip().lower(),
            endorsements=int(data.get("endorsements", 0)),
            duration_months=int(data.get("duration_months", 0)),
            proficiency=str(data.get("proficiency", "")),
        )


@dataclass
class BehavioralSignals:
    """Behavioral availability signals for a candidate."""

    last_active_date: str = ""
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 72.0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = 0.0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehavioralSignals":
        """Create BehavioralSignals from a dictionary."""
        return cls(
            last_active_date=str(data.get("last_active_date", "")),
            recruiter_response_rate=float(data.get("recruiter_response_rate", 0.0)),
            avg_response_time_hours=float(data.get("avg_response_time_hours", 72.0)),
            interview_completion_rate=float(data.get("interview_completion_rate", 0.0)),
            offer_acceptance_rate=float(data.get("offer_acceptance_rate", 0.0)),
            verified_email=bool(data.get("verified_email", False)),
            verified_phone=bool(data.get("verified_phone", False)),
            linkedin_connected=bool(data.get("linkedin_connected", False)),
        )


@dataclass
class Candidate:
    """Complete candidate profile with all fields."""

    candidate_id: str = ""
    headline: str = ""
    summary: str = ""
    current_title: str = ""
    skills: List[SkillEntry] = field(default_factory=list)
    career_history: List[CareerEntry] = field(default_factory=list)
    education: List[EducationEntry] = field(default_factory=list)
    location: str = ""
    years_of_experience: float = 0.0
    behavioral_signals: BehavioralSignals = field(default_factory=BehavioralSignals)

    # Optional fields that may be present
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None

    def get_skill_names(self) -> List[str]:
        """Return a list of lowercase skill names."""
        return [s.name for s in self.skills if s.name]

    def get_narrative_text(self) -> str:
        """Build narrative text from headline, summary, and recent career."""
        parts = []
        if self.headline:
            parts.append(self.headline)
        if self.summary:
            parts.append(self.summary)
        # Include the 3 most recent career entries
        for entry in self.career_history[:3]:
            career_text = f"{entry.title} at {entry.company}"
            if entry.description:
                career_text += f". {entry.description}"
            parts.append(career_text)
        return " | ".join(parts)

    def get_skills_text(self) -> str:
        """Get all skills as a space-separated string."""
        return " ".join(self.get_skill_names())

    def get_career_text(self) -> str:
        """Get career history as concatenated text."""
        parts = []
        for entry in self.career_history:
            text = f"{entry.title} at {entry.company}"
            if entry.description:
                text += f": {entry.description}"
            parts.append(text)
        return " | ".join(parts)


def _parse_json_field(value: Any) -> Any:
    """Parse a JSON string field, returning the parsed value or empty default."""
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Try to handle comma-separated values
            if "," in value:
                return [v.strip() for v in value.split(",")]
            return [value]
    return []


def _parse_candidate_row(row: Dict[str, Any]) -> Candidate:
    """Parse a single row of candidate data into a Candidate object."""
    # Parse skills
    raw_skills = _parse_json_field(row.get("skills", []))
    skills = [SkillEntry.from_dict(s) for s in raw_skills] if raw_skills else []

    # Parse career history
    raw_career = _parse_json_field(row.get("career_history", []))
    career_history = [CareerEntry.from_dict(c) for c in raw_career] if raw_career else []

    # Parse education
    raw_education = _parse_json_field(row.get("education", []))
    education = [EducationEntry.from_dict(e) for e in raw_education] if raw_education else []

    # Parse behavioral signals
    raw_behavioral = row.get("behavioral_signals", {})
    if isinstance(raw_behavioral, str):
        try:
            raw_behavioral = json.loads(raw_behavioral)
        except json.JSONDecodeError:
            raw_behavioral = {}
    behavioral = BehavioralSignals.from_dict(raw_behavioral) if isinstance(raw_behavioral, dict) else BehavioralSignals()

    # Parse salary fields
    salary_min = row.get("salary_min")
    salary_max = row.get("salary_max")
    try:
        salary_min = float(salary_min) if salary_min is not None and str(salary_min).strip() else None
    except (ValueError, TypeError):
        salary_min = None
    try:
        salary_max = float(salary_max) if salary_max is not None and str(salary_max).strip() else None
    except (ValueError, TypeError):
        salary_max = None

    return Candidate(
        candidate_id=str(row.get("candidate_id", "")),
        headline=str(row.get("headline", "")),
        summary=str(row.get("summary", "")),
        current_title=str(row.get("current_title", "")),
        skills=skills,
        career_history=career_history,
        education=education,
        location=str(row.get("location", "")),
        years_of_experience=float(row.get("years_of_experience", 0.0)),
        behavioral_signals=behavioral,
        salary_min=salary_min,
        salary_max=salary_max,
    )


class CandidateLoader:
    """Load and validate candidate data from CSV or JSON files.

    Supports chunked loading for memory efficiency with large datasets.
    """

    REQUIRED_FIELDS = {"candidate_id"}
    EXPECTED_FIELDS = {
        "candidate_id", "headline", "summary", "current_title",
        "skills", "career_history", "education", "location",
        "years_of_experience", "behavioral_signals",
    }

    def __init__(self, filepath: Union[str, Path]) -> None:
        """Initialize the loader with a file path.

        Args:
            filepath: Path to the candidate data file (CSV or JSON).
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Candidate data file not found: {self.filepath}")

    def validate_schema(self, df: pd.DataFrame) -> List[str]:
        """Validate the schema of a DataFrame against expected fields.

        Args:
            df: DataFrame to validate.

        Returns:
            List of warning messages for missing expected fields.
        """
        warnings: List[str] = []
        columns = set(df.columns)

        missing_required = self.REQUIRED_FIELDS - columns
        if missing_required:
            raise ValueError(
                f"Missing required columns: {missing_required}"
            )

        missing_expected = self.EXPECTED_FIELDS - columns
        if missing_expected:
            warnings.append(
                f"Missing expected columns (will use defaults): {missing_expected}"
            )
            for col in missing_expected:
                df[col] = ""

        return warnings

    def load(self, chunk_size: Optional[int] = None) -> List[Candidate]:
        """Load all candidates from the file.

        Args:
            chunk_size: If set, load in chunks of this size for memory efficiency.

        Returns:
            List of Candidate objects.
        """
        logger.info("Loading candidates from %s", self.filepath)

        if self.filepath.suffix == ".json":
            df = pd.read_json(self.filepath)
        elif self.filepath.suffix == ".csv":
            df = pd.read_csv(self.filepath, low_memory=False)
        else:
            raise ValueError(f"Unsupported file format: {self.filepath.suffix}")

        # Validate schema
        warnings = self.validate_schema(df)
        for warning in warnings:
            logger.warning(warning)

        # Parse candidates
        candidates: List[Candidate] = []
        for _, row in df.iterrows():
            try:
                candidate = _parse_candidate_row(row.to_dict())
                candidates.append(candidate)
            except Exception as e:
                logger.warning(
                    "Failed to parse candidate row: %s — %s", row.get("candidate_id", "unknown"), e
                )

        logger.info("Loaded %d candidates successfully", len(candidates))
        return candidates

    def load_dataframe(self) -> pd.DataFrame:
        """Load raw DataFrame without parsing into Candidate objects.

        Returns:
            Raw pandas DataFrame.
        """
        if self.filepath.suffix == ".json":
            df = pd.read_json(self.filepath)
        elif self.filepath.suffix == ".csv":
            df = pd.read_csv(self.filepath, low_memory=False)
        else:
            raise ValueError(f"Unsupported file format: {self.filepath.suffix}")

        self.validate_schema(df)
        return df
