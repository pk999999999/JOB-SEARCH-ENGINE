"""
Job Description parser module.

Parses the JD requirements YAML configuration into structured dataclasses
for use throughout the ranking pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExperienceRequirements:
    """Experience range requirements."""

    ideal_min_years: int = 5
    ideal_max_years: int = 9
    absolute_min_years: int = 3
    absolute_max_years: int = 15


@dataclass
class LocationRequirements:
    """Location tier definitions."""

    preferred: List[str] = field(default_factory=lambda: ["Pune", "Noida"])
    good: List[str] = field(
        default_factory=lambda: ["Mumbai", "Hyderabad", "Delhi NCR", "Delhi", "New Delhi", "Gurgaon", "Gurugram"]
    )
    acceptable_country: str = "India"


@dataclass
class EducationRequirements:
    """Education preferences."""

    preferred_degrees: List[str] = field(default_factory=list)
    preferred_fields: List[str] = field(default_factory=list)
    acceptable_degrees: List[str] = field(default_factory=list)


@dataclass
class BehavioralThresholds:
    """Behavioral availability thresholds."""

    min_recruiter_response_rate: float = 0.3
    max_avg_response_time_hours: float = 72.0
    min_interview_completion_rate: float = 0.5
    recent_activity_days: int = 90


@dataclass
class JobDescription:
    """Complete structured job description."""

    job_title: str = "Senior AI Engineer"
    description: str = ""
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    experience: ExperienceRequirements = field(default_factory=ExperienceRequirements)
    location: LocationRequirements = field(default_factory=LocationRequirements)
    education: EducationRequirements = field(default_factory=EducationRequirements)
    behavioral_thresholds: BehavioralThresholds = field(default_factory=BehavioralThresholds)

    def get_all_skills(self) -> List[str]:
        """Return all skills (required + preferred) as lowercase."""
        return [s.lower() for s in self.required_skills + self.preferred_skills]

    def get_required_skills_lower(self) -> List[str]:
        """Return required skills as lowercase list."""
        return [s.lower() for s in self.required_skills]

    def get_jd_text(self) -> str:
        """Build a text representation of the JD for embedding."""
        parts = [
            self.job_title,
            self.description,
            "Required skills: " + ", ".join(self.required_skills),
            "Preferred skills: " + ", ".join(self.preferred_skills),
        ]
        return " ".join(parts)


def _parse_experience(data: Dict[str, Any]) -> ExperienceRequirements:
    """Parse experience requirements from config dict."""
    return ExperienceRequirements(
        ideal_min_years=data.get("ideal_min_years", 5),
        ideal_max_years=data.get("ideal_max_years", 9),
        absolute_min_years=data.get("absolute_min_years", 3),
        absolute_max_years=data.get("absolute_max_years", 15),
    )


def _parse_location(data: Dict[str, Any]) -> LocationRequirements:
    """Parse location requirements from config dict."""
    return LocationRequirements(
        preferred=data.get("preferred", ["Pune", "Noida"]),
        good=data.get("good", ["Mumbai", "Hyderabad", "Delhi NCR"]),
        acceptable_country=data.get("acceptable_country", "India"),
    )


def _parse_education(data: Dict[str, Any]) -> EducationRequirements:
    """Parse education requirements from config dict."""
    return EducationRequirements(
        preferred_degrees=data.get("preferred_degrees", []),
        preferred_fields=data.get("preferred_fields", []),
        acceptable_degrees=data.get("acceptable_degrees", []),
    )


def _parse_behavioral(data: Dict[str, Any]) -> BehavioralThresholds:
    """Parse behavioral thresholds from config dict."""
    return BehavioralThresholds(
        min_recruiter_response_rate=data.get("min_recruiter_response_rate", 0.3),
        max_avg_response_time_hours=data.get("max_avg_response_time_hours", 72.0),
        min_interview_completion_rate=data.get("min_interview_completion_rate", 0.5),
        recent_activity_days=data.get("recent_activity_days", 90),
    )


class JDParser:
    """Parser for job description YAML configuration files."""

    def __init__(self, config_path: str | Path) -> None:
        """Initialize with path to JD requirements YAML.

        Args:
            config_path: Path to the jd_requirements.yaml file.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"JD config file not found: {self.config_path}")

    def parse(self) -> JobDescription:
        """Parse the YAML config into a JobDescription object.

        Returns:
            Structured JobDescription instance.
        """
        logger.info("Parsing JD requirements from %s", self.config_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        jd = JobDescription(
            job_title=config.get("job_title", "Senior AI Engineer"),
            description=config.get("description", ""),
            required_skills=config.get("required_skills", []),
            preferred_skills=config.get("preferred_skills", []),
            experience=_parse_experience(config.get("experience", {})),
            location=_parse_location(config.get("location", {})),
            education=_parse_education(config.get("education", {})),
            behavioral_thresholds=_parse_behavioral(config.get("behavioral_thresholds", {})),
        )

        logger.info(
            "Parsed JD: %s — %d required skills, %d preferred skills",
            jd.job_title,
            len(jd.required_skills),
            len(jd.preferred_skills),
        )
        return jd
