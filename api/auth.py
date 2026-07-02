"""
JWT authentication module for the API.

Handles token generation, password hashing (using raw bcrypt to avoid passlib bugs), 
token verification, and basic SQLite user management.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from api.config import settings

logger = logging.getLogger(__name__)

# OAuth2 scheme for token extraction (points to the token endpoint)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


class Token(BaseModel):
    """JWT response model."""
    access_token: str
    token_type: str


# ---------------------------------------------------------------------------
# Password Hashing
# ---------------------------------------------------------------------------

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'), 
            hashed_password.encode('utf-8')
        )
    except ValueError:
        return False


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


# ---------------------------------------------------------------------------
# JWT Tokens
# ---------------------------------------------------------------------------

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a new JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
        
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.secret_key, 
        algorithm=settings.algorithm
    )
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Dependency that extracts and validates the JWT token.
    
    Returns the username if valid, raises HTTPException otherwise.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token, 
            settings.secret_key, 
            algorithms=[settings.algorithm]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
            
    except jwt.PyJWTError:
        raise credentials_exception
        
    return username


# ---------------------------------------------------------------------------
# User Service (SQLite)
# ---------------------------------------------------------------------------

class UserService:
    """Simple SQLite-backed user service."""
    
    def __init__(self):
        self.db_path = settings.data_path / "users.db"
        self._init_db()
        self._ensure_admin()
        
    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
    def _ensure_admin(self):
        """Ensure the default admin user exists from settings."""
        if not self.get_user(settings.admin_username):
            # The setting is already a bcrypt hash
            self._insert_user(settings.admin_username, settings.admin_password_hash)
            
    def _insert_user(self, username: str, password_hash: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash)
            )
            
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cur.fetchone()
            return dict(row) if row else None
            
    def create_user(self, username: str, password: str) -> bool:
        """Create a new user. Returns False if user already exists."""
        if self.get_user(username):
            return False
            
        password_hash = get_password_hash(password)
        try:
            self._insert_user(username, password_hash)
            return True
        except sqlite3.IntegrityError:
            return False

# Global singleton
user_service = UserService()
