"""
FastAPI dependency injection for shared services.

Provides singleton instances of services that are shared across requests.
NOTE: With multi-worker uvicorn (--workers N), each worker gets its own
singleton. Candidate ingestion in one worker won't be visible in another
until the worker reloads. For true multi-worker consistency, use an
external database instead of in-memory lists.
"""

from __future__ import annotations

import threading
from functools import lru_cache

from api.services import CandidateService, JobService, RankingService

_lock = threading.Lock()
_candidate_service: CandidateService | None = None


def get_candidate_service() -> CandidateService:
    """Get or create the singleton CandidateService (thread-safe)."""
    global _candidate_service
    if _candidate_service is None:
        with _lock:
            if _candidate_service is None:
                service = CandidateService()
                service.load()
                _candidate_service = service
    return _candidate_service


@lru_cache(maxsize=1)
def get_job_service() -> JobService:
    """Get or create the singleton JobService."""
    return JobService()


@lru_cache(maxsize=1)
def get_ranking_service() -> RankingService:
    """Get or create the singleton RankingService."""
    candidate_service = get_candidate_service()
    return RankingService(candidate_service=candidate_service)
