"""
Application configuration via environment variables.

Uses pydantic-settings for type-safe configuration with .env file support.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Paths
    artifacts_dir: str = "./artifacts"
    data_dir: str = "./data"
    config_dir: str = "./config"
    jobs_dir: str = "./config/jobs"

    # Ranking
    top_k_default: int = 100
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # JWT Authentication
    # SECURITY: Override secret_key via env var or .env file in production!
    secret_key: str = os.environ.get("SECRET_KEY", secrets.token_urlsafe(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days
    admin_username: str = "admin"
    # Default hash for "admin123" — override ADMIN_PASSWORD_HASH in production
    admin_password_hash: str = "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"

    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    # Logging
    log_level: str = "INFO"

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir).resolve()

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).resolve()

    @property
    def config_path(self) -> Path:
        return Path(self.config_dir).resolve()

    @property
    def jobs_path(self) -> Path:
        return Path(self.jobs_dir).resolve()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
