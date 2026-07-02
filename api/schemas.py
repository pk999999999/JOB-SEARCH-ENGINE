"""
Pydantic schemas for API request/response validation.

Defines the contract between frontend and backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class SkillSchema(BaseModel):
    """Skill entry in a candidate profile."""
    name: str = ""
    endorsements: int = 0
    duration_months: int = 0
    proficiency: str = ""


class CareerEntrySchema(BaseModel):
    """Career history entry."""
    title: str = ""
    company: str = ""
    start_date: str = ""
    end_date: str = ""
    duration_months: int = 0
    is_current: bool = False
    description: str = ""


class EducationEntrySchema(BaseModel):
    """Education entry."""
    degree: str = ""
    field_of_study: str = ""
    institution: str = ""
    year: int = 0


class BehavioralSignalsSchema(BaseModel):
    """Behavioral signals for a candidate."""
    last_active_date: str = ""
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 72.0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = 0.0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False


class ScoreBreakdownSchema(BaseModel):
    """Detailed score breakdown for a ranked candidate."""
    career_match: float = 0.0
    skill_match: float = 0.0
    experience_fit: float = 0.0
    location_fit: float = 0.0
    education_fit: float = 0.0
    rule_adjustments: float = 0.0
    base_score: float = 0.0
    coherence_factor: float = 0.0
    behavioral_modifier: float = 0.0
    honeypot_gate: float = 0.0
    final_score: float = 0.0


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class UserCreate(BaseModel):
    """Schema for user registration."""
    username: str
    password: str

class ExperienceRange(BaseModel):
    """Experience range for a job description."""
    ideal_min_years: int = 5
    ideal_max_years: int = 9
    absolute_min_years: int = 3
    absolute_max_years: int = 15


class LocationPreferences(BaseModel):
    """Location preferences for a job description."""
    preferred: List[str] = Field(default_factory=lambda: ["Pune", "Noida"])
    good: List[str] = Field(
        default_factory=lambda: ["Mumbai", "Hyderabad", "Delhi NCR"]
    )
    acceptable_country: str = "India"


class EducationPreferences(BaseModel):
    """Education preferences for a job description."""
    preferred_degrees: List[str] = Field(default_factory=list)
    preferred_fields: List[str] = Field(default_factory=list)
    acceptable_degrees: List[str] = Field(default_factory=list)


class RankRequest(BaseModel):
    """Request body for the ranking endpoint."""
    job_title: str = "Senior AI Engineer"
    description: str = ""
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    experience: ExperienceRange = Field(default_factory=ExperienceRange)
    location: LocationPreferences = Field(default_factory=LocationPreferences)
    education: EducationPreferences = Field(default_factory=EducationPreferences)
    top_k: int = Field(default=100, ge=1, le=500)


class JobDescriptionCreate(BaseModel):
    """Request body for creating a new job description."""
    job_title: str
    description: str = ""
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    experience: ExperienceRange = Field(default_factory=ExperienceRange)
    location: LocationPreferences = Field(default_factory=LocationPreferences)
    education: EducationPreferences = Field(default_factory=EducationPreferences)


class CandidateFilterParams(BaseModel):
    """Query parameters for filtering candidates."""
    location: Optional[str] = None
    min_experience: Optional[float] = None
    max_experience: Optional[float] = None
    skill: Optional[str] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class CandidateIngest(BaseModel):
    """Schema for ingesting a new candidate via Webhook."""
    candidate_id: Optional[str] = None
    headline: str = ""
    summary: str = ""
    current_title: str = ""
    location: str = ""
    years_of_experience: float = 0.0
    skills: List[SkillSchema] = Field(default_factory=list)
    career_history: List[CareerEntrySchema] = Field(default_factory=list)
    education: List[EducationEntrySchema] = Field(default_factory=list)
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class RankedCandidateResponse(BaseModel):
    """A single ranked candidate in the response."""
    candidate_id: str
    rank: int
    score: float
    reasoning: str = ""
    headline: str = ""
    current_title: str = ""
    location: str = ""
    years_of_experience: float = 0.0
    skills: List[SkillSchema] = Field(default_factory=list)
    education: str = ""
    score_breakdown: Optional[ScoreBreakdownSchema] = None


class RankResponse(BaseModel):
    """Response for the ranking endpoint."""
    job_title: str
    total_candidates: int
    ranked_count: int
    elapsed_seconds: float
    candidates: List[RankedCandidateResponse]


class CandidateDetailResponse(BaseModel):
    """Full candidate detail response."""
    candidate_id: str
    headline: str = ""
    summary: str = ""
    current_title: str = ""
    location: str = ""
    years_of_experience: float = 0.0
    skills: List[SkillSchema] = Field(default_factory=list)
    career_history: List[CareerEntrySchema] = Field(default_factory=list)
    education: List[EducationEntrySchema] = Field(default_factory=list)
    behavioral_signals: BehavioralSignalsSchema = Field(
        default_factory=BehavioralSignalsSchema
    )
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None


class CandidateListResponse(BaseModel):
    """Paginated candidate list response."""
    total: int
    page: int
    page_size: int
    total_pages: int
    candidates: List[CandidateDetailResponse]


class JobDescriptionResponse(BaseModel):
    """Job description response."""
    id: str
    job_title: str
    description: str = ""
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    experience: ExperienceRange = Field(default_factory=ExperienceRange)
    location: LocationPreferences = Field(default_factory=LocationPreferences)
    education: EducationPreferences = Field(default_factory=EducationPreferences)


class JobListResponse(BaseModel):
    """Job description list response."""
    total: int
    jobs: List[JobDescriptionResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    candidates_loaded: int = 0
    artifacts_available: bool = False


class StatsResponse(BaseModel):
    """Dashboard statistics response."""
    total_candidates: int = 0
    avg_experience_years: float = 0.0
    top_locations: List[Dict[str, Any]] = Field(default_factory=list)
    total_jobs: int = 0
    skill_distribution: List[Dict[str, Any]] = Field(default_factory=list)
