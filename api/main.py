"""
FastAPI application — main entry point for the Redrob AI Ranker API.

Provides REST endpoints for:
- Ranking candidates against job descriptions
- Browsing and filtering candidates
- Managing job descriptions
- Health checks and statistics
"""

from __future__ import annotations
import os
import platform

# Fix PyTorch c10.dll initialization bug on Windows
if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec
    try:
        spec = find_spec("torch")
        if spec and spec.origin:
            dll_path = os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
            if os.path.exists(dll_path):
                ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass
import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Depends, status, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.config import settings
from api.auth import (
    create_access_token,
    verify_password,
    get_current_user,
    Token,
    user_service,
)
from api.dependencies import (
    get_candidate_service,
    get_job_service,
    get_ranking_service,
)
from api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from api.schemas import (
    CandidateDetailResponse,
    CandidateListResponse,
    EducationEntrySchema,
    CareerEntrySchema,
    BehavioralSignalsSchema,
    HealthResponse,
    JobDescriptionCreate,
    JobDescriptionResponse,
    JobListResponse,
    RankRequest,
    RankResponse,
    RankedCandidateResponse,
    SkillSchema,
    StatsResponse,
    ExperienceRange,
    LocationPreferences,
    EducationPreferences,
    UserCreate,
    CandidateIngest,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Redrob AI Ranker",
    description="Production-grade AI candidate ranking engine",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Middleware (order matters — outermost first)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: Candidate dataclass → response schema
# ---------------------------------------------------------------------------


def _candidate_to_response(candidate) -> CandidateDetailResponse:
    """Convert a Candidate dataclass to a CandidateDetailResponse schema."""
    return CandidateDetailResponse(
        candidate_id=candidate.candidate_id,
        headline=candidate.headline,
        summary=candidate.summary,
        current_title=candidate.current_title,
        location=candidate.location,
        years_of_experience=candidate.years_of_experience,
        skills=[
            SkillSchema(
                name=s.name,
                endorsements=s.endorsements,
                duration_months=s.duration_months,
                proficiency=s.proficiency,
            )
            for s in candidate.skills
        ],
        career_history=[
            CareerEntrySchema(
                title=e.title,
                company=e.company,
                start_date=e.start_date,
                end_date=e.end_date,
                duration_months=e.duration_months,
                is_current=e.is_current,
                description=e.description,
            )
            for e in candidate.career_history
        ],
        education=[
            EducationEntrySchema(
                degree=e.degree,
                field_of_study=e.field_of_study,
                institution=e.institution,
                year=e.year,
            )
            for e in candidate.education
        ],
        behavioral_signals=BehavioralSignalsSchema(
            last_active_date=candidate.behavioral_signals.last_active_date,
            recruiter_response_rate=candidate.behavioral_signals.recruiter_response_rate,
            avg_response_time_hours=candidate.behavioral_signals.avg_response_time_hours,
            interview_completion_rate=candidate.behavioral_signals.interview_completion_rate,
            offer_acceptance_rate=candidate.behavioral_signals.offer_acceptance_rate,
            verified_email=candidate.behavioral_signals.verified_email,
            verified_phone=candidate.behavioral_signals.verified_phone,
            linkedin_connected=candidate.behavioral_signals.linkedin_connected,
        ),
        salary_min=candidate.salary_min,
        salary_max=candidate.salary_max,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/auth/signup", response_model=Token, tags=["Authentication"])
async def signup(user_data: UserCreate):
    """Register a new user and return an access token."""
    if not user_service.create_user(user_data.username, user_data.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    access_token = create_access_token(data={"sub": user_data.username})
    return Token(access_token=access_token, token_type="bearer")


@app.post("/api/auth/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 compatible token login, get an access token for future requests."""
    user = user_service.get_user(form_data.username)
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": form_data.username})
    return Token(access_token=access_token, token_type="bearer")


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    svc = get_candidate_service()
    artifacts_path = settings.artifacts_path
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        candidates_loaded=len(svc.candidates),
        artifacts_available=(artifacts_path / "embeddings.npy").exists(),
    )


@app.get("/api/stats", response_model=StatsResponse, tags=["System"])
async def get_stats(current_user: str = Depends(get_current_user)):
    """Get dashboard statistics."""
    candidate_svc = get_candidate_service()
    job_svc = get_job_service()

    stats = candidate_svc.get_stats()
    jobs = job_svc.list_jobs()

    return StatsResponse(
        total_candidates=stats["total_candidates"],
        avg_experience_years=stats["avg_experience_years"],
        top_locations=stats["top_locations"],
        total_jobs=len(jobs),
        skill_distribution=stats["skill_distribution"],
    )


@app.post("/api/rank", response_model=RankResponse, tags=["Ranking"])
async def rank_candidates(request: RankRequest, current_user: str = Depends(get_current_user)):
    """Rank candidates against a job description.

    Accepts a JD specification and returns the top-K candidates
    with scores, reasoning, and optional score breakdowns.
    """
    ranking_svc = get_ranking_service()

    result = ranking_svc.rank(
        request_data=request.model_dump(),
        top_k=request.top_k,
    )

    ranked = [
        RankedCandidateResponse(
            candidate_id=c["candidate_id"],
            rank=c["rank"],
            score=c["score"],
            reasoning=c.get("reasoning", ""),
            headline=c.get("headline", ""),
            current_title=c.get("current_title", ""),
            location=c.get("location", ""),
            years_of_experience=c.get("years_of_experience", 0.0),
            skills=[SkillSchema(**s) for s in c.get("skills", [])],
        )
        for c in result["candidates"]
    ]

    return RankResponse(
        job_title=result["job_title"],
        total_candidates=result["total_candidates"],
        ranked_count=result["ranked_count"],
        elapsed_seconds=result["elapsed_seconds"],
        candidates=ranked,
    )


@app.get("/api/candidates", response_model=CandidateListResponse, tags=["Candidates"])
async def list_candidates(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    location: Optional[str] = Query(default=None),
    min_experience: Optional[float] = Query(default=None),
    max_experience: Optional[float] = Query(default=None),
    skill: Optional[str] = Query(default=None),
    current_user: str = Depends(get_current_user),
):
    """Get paginated, filterable list of candidates."""
    svc = get_candidate_service()

    candidates, total = svc.get_paginated(
        page=page,
        page_size=page_size,
        location=location,
        min_experience=min_experience,
        max_experience=max_experience,
        skill=skill,
    )

    total_pages = math.ceil(total / page_size) if total > 0 else 0

    return CandidateListResponse(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        candidates=[_candidate_to_response(c) for c in candidates],
    )


@app.get(
    "/api/candidates/{candidate_id}",
    response_model=CandidateDetailResponse,
    tags=["Candidates"],
)
async def get_candidate(candidate_id: str, current_user: str = Depends(get_current_user)):
    """Get full candidate profile by ID."""
    svc = get_candidate_service()
    candidate = svc.get_by_id(candidate_id)

    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")

    return _candidate_to_response(candidate)


@app.get("/api/jobs", response_model=JobListResponse, tags=["Jobs"])
async def list_jobs(current_user: str = Depends(get_current_user)):
    """List all saved job descriptions."""
    svc = get_job_service()
    jobs = svc.list_jobs()

    job_responses = []
    for job in jobs:
        exp = job.get("experience", {})
        loc = job.get("location", {})
        edu = job.get("education", {})
        job_responses.append(
            JobDescriptionResponse(
                id=job.get("id", ""),
                job_title=job.get("job_title", ""),
                description=job.get("description", ""),
                required_skills=job.get("required_skills", []),
                preferred_skills=job.get("preferred_skills", []),
                experience=ExperienceRange(**exp) if exp else ExperienceRange(),
                location=LocationPreferences(**loc) if loc else LocationPreferences(),
                education=EducationPreferences(**edu) if edu else EducationPreferences(),
            )
        )

    return JobListResponse(total=len(job_responses), jobs=job_responses)


@app.post("/api/jobs", response_model=JobDescriptionResponse, tags=["Jobs"])
async def create_job(request: JobDescriptionCreate, current_user: str = Depends(get_current_user)):
    """Create a new job description."""
    svc = get_job_service()
    job_data = request.model_dump()
    job_id = svc.create_job(job_data)

    return JobDescriptionResponse(id=job_id, **job_data)


@app.delete("/api/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str, current_user: str = Depends(get_current_user)):
    """Delete a job description."""
    svc = get_job_service()
    if not svc.delete_job(job_id):
        raise HTTPException(
            status_code=404, detail="Job not found or cannot delete default"
        )
    return {"detail": "Deleted"}


@app.post("/api/webhooks/ingest", response_model=CandidateDetailResponse, tags=["Webhooks"])
async def ingest_candidate_webhook(request: CandidateIngest, current_user: str = Depends(get_current_user)):
    """Webhook for ATS/LinkedIn to ingest a new candidate directly."""
    svc = get_candidate_service()
    try:
        candidate = svc.add_candidate(request.model_dump())
        return _candidate_to_response(candidate)
    except Exception as e:
        logger.error("Failed to ingest candidate: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/candidates/upload", response_model=CandidateDetailResponse, tags=["Candidates"])
async def upload_candidate_resume(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    """Upload a PDF resume, extract text, parse heuristically, and ingest."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        import PyPDF2
        import io
        import re
        
        content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
                
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
            
        # Heuristic Parsing
        text_lower = text.lower()
        skills = []
        common_skills = [
            "python", "java", "c++", "c#", "javascript", "typescript", "react", "angular", "vue",
            "node.js", "machine learning", "deep learning", "nlp", "computer vision",
            "docker", "kubernetes", "aws", "gcp", "azure", "sql", "nosql", "mongodb",
            "postgresql", "mysql", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
            "golang", "rust", "ruby", "php", "html", "css", "agile", "scrum", "git"
        ]
        
        for s in common_skills:
            if s in text_lower:
                skills.append(SkillSchema(name=s, endorsements=1, duration_months=24, proficiency="intermediate"))
                
        # Heuristic for years of experience
        yoe = 0.0
        # Look for things like "5 years", "10+ years"
        match = re.search(r'(\d+)\+?\s*years?', text_lower)
        if match:
            yoe = float(match.group(1))
            
        # Truncate text to avoid massive payloads, but keep enough for embeddings
        clean_text = text.replace('\n', ' ').strip()
        summary_text = clean_text[:1000]
        exp_text = clean_text[1000:3000] if len(clean_text) > 1000 else ""
        
        ingest_data = CandidateIngest(
            candidate_id=None,
            headline=f"Resume: {file.filename}",
            summary=summary_text,
            current_title="Candidate",
            location="Unknown",
            years_of_experience=yoe,
            skills=skills,
            career_history=[
                CareerEntrySchema(
                    title="Experience",
                    company="Various",
                    start_date="Unknown",
                    end_date="Unknown",
                    duration_months=int(yoe * 12),
                    is_current=False,
                    description=exp_text
                )
            ]
        )
        
        svc = get_candidate_service()
        candidate = svc.add_candidate(ingest_data.model_dump())
        
        # Save the original PDF to disk
        uploads_dir = Path("data/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = uploads_dir / f"{candidate.candidate_id}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(content)
            
        return _candidate_to_response(candidate)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to parse and ingest PDF resume: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to process resume: {str(e)}")

@app.get("/api/candidates/{candidate_id}/resume", tags=["Candidates"])
async def download_candidate_resume(candidate_id: str, current_user: str = Depends(get_current_user)):
    """Download the original uploaded PDF resume."""
    pdf_path = Path("data/uploads") / f"{candidate_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Original resume not found")
    return FileResponse(
        str(pdf_path), 
        media_type="application/pdf", 
        filename=f"resume_{candidate_id}.pdf"
    )

# ---------------------------------------------------------------------------
# Serve frontend static files (for production Docker deployment)
# ---------------------------------------------------------------------------

FRONTEND_BUILD_DIR = Path(__file__).parent.parent / "frontend" / "out"

if FRONTEND_BUILD_DIR.exists():
    # Serve Next.js static export
    app.mount(
        "/_next",
        StaticFiles(directory=str(FRONTEND_BUILD_DIR / "_next")),
        name="next_static",
    )

    @app.get("/{full_path:path}", tags=["Frontend"])
    async def serve_frontend(full_path: str):
        """Serve the Next.js static export for all non-API routes."""
        # Try exact file match first
        file_path = FRONTEND_BUILD_DIR / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))

        # Try with .html extension
        html_path = FRONTEND_BUILD_DIR / f"{full_path}.html"
        if html_path.is_file():
            return FileResponse(str(html_path))

        # Try index.html in directory
        index_path = FRONTEND_BUILD_DIR / full_path / "index.html"
        if index_path.is_file():
            return FileResponse(str(index_path))

        # Fallback to root index.html (SPA routing)
        root_index = FRONTEND_BUILD_DIR / "index.html"
        if root_index.is_file():
            return FileResponse(str(root_index))

        raise HTTPException(status_code=404, detail="Not found")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Pre-load data on startup for fast first-request response."""
    logger.info("=" * 60)
    logger.info("REDROB AI RANKER — API Starting")
    logger.info("=" * 60)

    # Pre-load candidates
    svc = get_candidate_service()
    logger.info("Candidates loaded: %d", len(svc.candidates))

    # Check artifacts
    artifacts_path = settings.artifacts_path
    if (artifacts_path / "embeddings.npy").exists():
        logger.info("Artifacts directory: %s ✓", artifacts_path)
    else:
        logger.warning("Artifacts not found at %s — ranking will use fallback scores", artifacts_path)

    logger.info("API ready at http://%s:%d", settings.api_host, settings.api_port)
    logger.info("Docs at http://%s:%d/api/docs", settings.api_host, settings.api_port)
    logger.info("=" * 60)
    yield  # Application runs here


app.router.lifespan_context = lifespan
