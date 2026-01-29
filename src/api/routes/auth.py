"""
Authentication API routes for user signup, login, and session management.
"""
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Response, Cookie, Depends
from pydantic import BaseModel, EmailStr, Field

from src.database.sqlite_manager import SQLiteManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Simple in-memory session store (for production, use Redis or database)
sessions: dict[str, dict] = {}

# Session configuration
SESSION_DURATION_HOURS = 24


class SignupRequest(BaseModel):
    """Request body for user signup."""
    email: EmailStr
    password: str = Field(..., min_length=6)
    name: Optional[str] = None


class LoginRequest(BaseModel):
    """Request body for user login."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Response containing user data."""
    id: str
    email: str
    name: Optional[str]
    created_at: Optional[str]


class AuthResponse(BaseModel):
    """Response after successful authentication."""
    user: UserResponse
    message: str


def create_session(user_id: str, email: str, name: Optional[str]) -> str:
    """Create a new session and return the session token."""
    token = secrets.token_urlsafe(32)
    sessions[token] = {
        "user_id": user_id,
        "email": email,
        "name": name,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(hours=SESSION_DURATION_HOURS),
    }
    return token


def get_session(token: str) -> Optional[dict]:
    """Get session data if valid and not expired."""
    if not token or token not in sessions:
        return None

    session = sessions[token]
    if datetime.utcnow() > session["expires_at"]:
        # Session expired, remove it
        del sessions[token]
        return None

    return session


def invalidate_session(token: str) -> bool:
    """Invalidate a session. Returns True if session existed."""
    if token in sessions:
        del sessions[token]
        return True
    return False


async def get_current_user(session_token: Optional[str] = Cookie(default=None, alias="session")) -> Optional[dict]:
    """Dependency to get current user from session cookie."""
    if not session_token:
        return None
    return get_session(session_token)


@router.post("/auth/signup", response_model=AuthResponse)
async def signup(request: SignupRequest, response: Response):
    """
    Create a new user account.

    Returns user data and sets session cookie.
    """
    logger.info(f"Signup attempt for email: {request.email}")

    try:
        db = SQLiteManager()
        db.initialize()

        # Check if user already exists
        existing = await db.get_user_by_email(request.email)
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create user
        user = await db.create_user(
            email=request.email,
            password=request.password,
            name=request.name,
        )

        # Create session
        token = create_session(user.id, user.email, user.name)

        # Set cookie
        response.set_cookie(
            key="session",
            value=token,
            httponly=True,
            max_age=SESSION_DURATION_HOURS * 3600,
            samesite="lax",
        )

        logger.info(f"User created successfully: {user.email}")

        return AuthResponse(
            user=UserResponse(
                id=user.id,
                email=user.email,
                name=user.name,
                created_at=user.created_at.isoformat() if user.created_at else None,
            ),
            message="Account created successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create account")


@router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest, response: Response):
    """
    Login with email and password.

    Returns user data and sets session cookie.
    """
    logger.info(f"Login attempt for email: {request.email}")

    try:
        db = SQLiteManager()
        db.initialize()

        # Verify credentials
        user = await db.verify_user(request.email, request.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Create session
        token = create_session(user.id, user.email, user.name)

        # Set cookie
        response.set_cookie(
            key="session",
            value=token,
            httponly=True,
            max_age=SESSION_DURATION_HOURS * 3600,
            samesite="lax",
        )

        logger.info(f"User logged in: {user.email}")

        return AuthResponse(
            user=UserResponse(
                id=user.id,
                email=user.email,
                name=user.name,
                created_at=user.created_at.isoformat() if user.created_at else None,
            ),
            message="Login successful",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Login failed")


@router.post("/auth/logout")
async def logout(
    response: Response,
    session_token: Optional[str] = Cookie(default=None, alias="session")
):
    """
    Logout and clear session.
    """
    if session_token:
        invalidate_session(session_token)

    response.delete_cookie(key="session")

    return {"message": "Logged out successfully"}


@router.get("/auth/me")
async def get_current_user_info(
    session_token: Optional[str] = Cookie(default=None, alias="session")
):
    """
    Get current logged-in user info.

    Returns user data if logged in, null otherwise.
    """
    if not session_token:
        return {"user": None}

    session = get_session(session_token)
    if not session:
        return {"user": None}

    return {
        "user": {
            "id": session["user_id"],
            "email": session["email"],
            "name": session["name"],
        }
    }


@router.get("/auth/history")
async def get_search_history(
    session_token: Optional[str] = Cookie(default=None, alias="session")
):
    """
    Get current user's recent search history.

    Returns last 5 searches for logged-in users.
    """
    if not session_token:
        return {"history": [], "authenticated": False}

    session = get_session(session_token)
    if not session:
        return {"history": [], "authenticated": False}

    try:
        db = SQLiteManager()
        db.initialize()

        history = await db.get_recent_searches(session["user_id"], limit=5)

        return {
            "history": [h.to_dict() for h in history],
            "authenticated": True,
        }

    except Exception as e:
        logger.error(f"Failed to get search history: {e}", exc_info=True)
        return {"history": [], "authenticated": True, "error": str(e)}


@router.delete("/auth/history")
async def clear_search_history(
    session_token: Optional[str] = Cookie(default=None, alias="session")
):
    """
    Clear current user's search history.
    """
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    session = get_session(session_token)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired")

    try:
        db = SQLiteManager()
        db.initialize()

        count = await db.clear_search_history(session["user_id"])

        return {"message": f"Cleared {count} search history entries"}

    except Exception as e:
        logger.error(f"Failed to clear search history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear history")
