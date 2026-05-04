"""
Smart Door Lock API Routes
==========================
Provides three public endpoints consumed by the ESP32:
  POST /door/unlock  → sets command = UNLOCK
  POST /door/lock    → sets command = LOCK
  GET  /door/status  → returns current command (polled by ESP32)

The door_status table is auto-created on first use and holds
exactly one row (id = 1).  No authentication is required for
/door/status so the ESP32 can poll without credentials.

For /door/unlock and /door/lock the caller must supply the shared
DOOR_LOCK_API_KEY header (configured via CAPERCLUB_DOOR_LOCK_API_KEY
environment variable) to prevent random internet traffic from
controlling the lock.
"""

from __future__ import annotations

import os
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

if __package__:
    from .db import engine
else:
    from db import engine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Set this in Railway environment variables (and in .env locally).
# The ESP32 and front-end must send this value in the X-Door-Key header.
_DOOR_API_KEY: str = os.getenv("CAPERCLUB_DOOR_LOCK_API_KEY", "caperclub-door-2026")

router = APIRouter(prefix="/door", tags=["door-lock"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_table() -> None:
    """Create door_status table if it does not exist yet."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS door_status (
                    id          INT          NOT NULL DEFAULT 1,
                    command     VARCHAR(10)  NOT NULL DEFAULT 'LOCK',
                    updated_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                                ON UPDATE CURRENT_TIMESTAMP,
                    PRIMARY KEY (id),
                    CONSTRAINT chk_door_id CHECK (id = 1)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        )
        # Seed the single row if missing
        conn.execute(
            text(
                """
                INSERT IGNORE INTO door_status (id, command, updated_at)
                VALUES (1, 'LOCK', NOW())
                """
            )
        )


def _set_command(command: str) -> dict:
    """Persist command to DB and return a status dict."""
    _ensure_table()
    now = datetime.utcnow()
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE door_status SET command = :cmd, updated_at = :ts WHERE id = 1"
            ),
            {"cmd": command, "ts": now},
        )
    return {
        "command": command,
        "updatedAt": now.isoformat() + "Z",
        "ok": True,
    }


def _get_command() -> dict:
    """Read current command from DB."""
    _ensure_table()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT command, updated_at FROM door_status WHERE id = 1")
        ).first()
    if row is None:
        return {"command": "LOCK", "updatedAt": None}
    return {
        "command": row.command,
        "updatedAt": row.updated_at.isoformat() + "Z" if row.updated_at else None,
    }


def _verify_key(x_door_key: str | None) -> None:
    if x_door_key != _DOOR_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Door-Key header.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/unlock", summary="Unlock the door")
def door_unlock(x_door_key: str | None = Header(default=None)) -> JSONResponse:
    """
    Sets door command to UNLOCK.
    Requires X-Door-Key header matching CAPERCLUB_DOOR_LOCK_API_KEY.
    """
    _verify_key(x_door_key)
    result = _set_command("UNLOCK")
    return JSONResponse(status_code=200, content=result)


@router.post("/lock", summary="Lock the door")
def door_lock(x_door_key: str | None = Header(default=None)) -> JSONResponse:
    """
    Sets door command to LOCK.
    Requires X-Door-Key header matching CAPERCLUB_DOOR_LOCK_API_KEY.
    """
    _verify_key(x_door_key)
    result = _set_command("LOCK")
    return JSONResponse(status_code=200, content=result)


@router.get("/status", summary="Get current door command (polled by ESP32)")
def door_status() -> JSONResponse:
    """
    Returns current door command.  No authentication required so ESP32
    can poll without storing credentials.

    Response:
        { "command": "LOCK" | "UNLOCK", "updatedAt": "..." }
    """
    result = _get_command()
    return JSONResponse(status_code=200, content=result)
