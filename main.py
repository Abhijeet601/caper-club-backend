"""
CaperClub Backend API
=====================

ALWAYS USE RAILWAY DATABASE
- Backend is configured for Railway MySQL only
- Local MySQL development is not supported
- All data operations use Railway remote database
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jwt

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

if __package__:
  from .db import SessionLocal, get_db, get_settings
  from .door_control import sync_door_for_detection
  from .door_lock_service import get_door_state
  from .door_lock_routes import router as door_lock_router
  from .models import User, UserRole

  from .schemas import (
    AttendanceInput,
    CreateAnnouncementInput,
    CreateMembershipInput,
    CreateSlotInput,
    CreateUserInput,
    DescriptorEnrollmentInput,
    LoginInput,
    RegisterInput,
    SessionEndInput,
    SessionStartInput,
    TTSRequest,
    UpdateSlotInput,
    UpdateUserInput,
  )

  from .security import decode_access_token

  from .service import (
    ApiError,
    authenticate_user,
    clear_live_feed,
    create_announcement,
    create_membership,
    create_slot,
    create_user,
    delete_slot,
    delete_user,
    delete_user_embeddings,
    end_session,
    generate_tts,
    get_admin_announcements,
    get_admin_dashboard,
    get_admin_reports,
    get_admin_sessions,
    get_admin_slots,
    get_admin_users,
    get_face_enrollment_status,
    get_current_user_payload,
    get_session_timer,
    get_user_embeddings,
    get_user_dashboard,
    get_user_history,
    get_user_notifications,
    get_user_payments,
    get_user_profile,
    get_user_report,
    mark_attendance,
    register_user,
    save_user_embeddings,
    start_session,
    update_slot,
    update_user,
  )

else:
  from db import SessionLocal, get_db, get_settings
  from door_control import sync_door_for_detection
  from door_lock_service import get_door_state
  from door_lock_routes import router as door_lock_router
  from models import User, UserRole

  from schemas import (
    AttendanceInput,
    CreateAnnouncementInput,
    CreateMembershipInput,
    CreateSlotInput,
    CreateUserInput,
    DescriptorEnrollmentInput,
    LoginInput,
    RegisterInput,
    SessionEndInput,
    SessionStartInput,
    TTSRequest,
    UpdateSlotInput,
    UpdateUserInput,
  )

  from security import decode_access_token

  from service import (
    ApiError,
    authenticate_user,
    clear_live_feed,
    create_announcement,
    create_membership,
    create_slot,
    create_user,
    delete_slot,
    delete_user,
    delete_user_embeddings,
    end_session,
    generate_tts,
    get_admin_announcements,
    get_admin_dashboard,
    get_admin_reports,
    get_admin_sessions,
    get_admin_slots,
    get_admin_users,
    get_face_enrollment_status,
    get_current_user_payload,
    get_session_timer,
    get_user_embeddings,
    get_user_dashboard,
    get_user_history,
    get_user_notifications,
    get_user_payments,
    get_user_profile,
    get_user_report,
    mark_attendance,
    register_user,
    save_user_embeddings,
    start_session,
    update_slot,
    update_user,
  )

settings = get_settings()

bearer_scheme = HTTPBearer(auto_error=False)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / 'Frontend'

# =========================
# FASTAPI APP
# =========================

app = FastAPI(
  title='CaperClub API',
  docs_url='/docs',
  redoc_url='/redoc',
  openapi_url='/openapi.json',
)

# =========================
# CORS
# =========================

app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)

# =========================
# ROUTERS
# =========================

app.include_router(door_lock_router)

# =========================
# AUTH HELPERS
# =========================

def _resolve_user(
  credentials: HTTPAuthorizationCredentials | None,
  db: Session,
) -> User:

  if credentials is None:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail='Authentication required.',
    )

  try:
    payload = decode_access_token(credentials.credentials)

  except jwt.PyJWTError as error:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail='Invalid or expired token.',
    ) from error

  user = db.get(User, payload.get('sub'))

  if user is None:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail='Authenticated user no longer exists.',
    )

  return user


def get_current_user(
  credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
  db: Session = Depends(get_db),
) -> User:
  return _resolve_user(credentials, db)


def get_current_admin(
  user: User = Depends(get_current_user),
) -> User:

  if user.role != UserRole.ADMIN:
    raise HTTPException(
      status_code=status.HTTP_403_FORBIDDEN,
      detail='Admin access required.',
    )

  return user

# =========================
# EXCEPTION HANDLERS
# =========================

@app.exception_handler(ApiError)
async def handle_api_error(_: Request, error: ApiError):
  return JSONResponse(
    status_code=error.status_code,
    content={'message': error.message},
  )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, error: RequestValidationError):
  return JSONResponse(
    status_code=400,
    content={
      'message': 'Invalid request payload.',
      'issues': error.errors(),
    },
  )


@app.exception_handler(Exception)
async def handle_unexpected_error(_: Request, error: Exception):
  return JSONResponse(
    status_code=500,
    content={
      'message': str(error) or 'Unexpected server error.',
    },
  )

# =========================
# PUBLIC ROUTES
# =========================

@app.get('/')
def root():
  return {'message': 'CaperClub Backend Running'}


@app.get('/health')
def health_check():
  return {'ok': True}


@app.get('/swagger')
def swagger_redirect():
  return RedirectResponse(url='/docs')

# =========================
# AUTH ROUTES
# =========================

@app.post('/login')
def login(
  input_data: LoginInput,
  db: Session = Depends(get_db),
):
  return authenticate_user(db, input_data)


@app.post('/auth/login')
def login_legacy(
  input_data: LoginInput,
  db: Session = Depends(get_db),
):
  return authenticate_user(db, input_data)


@app.post('/register', status_code=201)
def register(
  input_data: RegisterInput,
  db: Session = Depends(get_db),
):
  return register_user(db, input_data)


@app.get('/auth/me')
def auth_me(
  user: User = Depends(get_current_user),
):
  return get_current_user_payload(user)

# =========================
# ADMIN ROUTES
# =========================

@app.get('/admin/dashboard')
def admin_dashboard(
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
):
  return get_admin_dashboard(db)

# =========================
# ATTENDANCE
# =========================

@app.post('/attendance')
def attendance_mark(
  input_data: AttendanceInput,
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
):

  result = mark_attendance(db, input_data)

  status_text = str(result.get('status') or '').lower()

  sync_door_for_detection(
    known_face=status_text not in {'unknown', 'retry', 'denied'},
    name=result.get('name'),
  )

  return result

# =========================
# DOOR
# =========================

@app.get('/door/state')
def door_state(
  _: User = Depends(get_current_admin),
):
  return get_door_state()

# =========================
# TTS
# =========================

@app.post('/tts')
def tts_generate(
  input_data: TTSRequest,
  _: User = Depends(get_current_admin),
):

  audio_bytes, media_type = generate_tts(input_data.text)

  return Response(
    content=audio_bytes,
    media_type=media_type,
  )

# =========================
# FRONTEND
# =========================

if FRONTEND_DIR.exists():
  app.mount(
    '/app',
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name='frontend',
  )

# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":

  import os
  import uvicorn

  uvicorn.run(
    app,
    host="0.0.0.0",
    port=int(os.getenv('PORT', '8001')),
  )