"""
Smart Door Lock API routes used by the ESP32 relay controller.
"""

from __future__ import annotations

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse

if __package__:
  from .door_lock_service import (
    DOOR_COMMAND_LOCK,
    DOOR_COMMAND_UNLOCK,
    get_door_state,
    set_door_state,
    verify_door_api_key,
  )
else:
  from door_lock_service import (
    DOOR_COMMAND_LOCK,
    DOOR_COMMAND_UNLOCK,
    get_door_state,
    set_door_state,
    verify_door_api_key,
  )

router = APIRouter(prefix='/door', tags=['door-lock'])


@router.get('/status', summary='Get current door command')
def door_status(x_door_key: str | None = Header(default=None)) -> JSONResponse:
  verify_door_api_key(x_door_key)
  return JSONResponse(status_code=200, content=get_door_state())


@router.post('/unlock', summary='Unlock the door')
def door_unlock(x_door_key: str | None = Header(default=None)) -> JSONResponse:
  verify_door_api_key(x_door_key)
  return JSONResponse(status_code=200, content=set_door_state(DOOR_COMMAND_UNLOCK))


@router.post('/lock', summary='Lock the door')
def door_lock(x_door_key: str | None = Header(default=None)) -> JSONResponse:
  verify_door_api_key(x_door_key)
  return JSONResponse(status_code=200, content=set_door_state(DOOR_COMMAND_LOCK))
