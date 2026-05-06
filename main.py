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
  # =========================
# USERS
# =========================

@app.get('/admin/users')
def admin_users(
  scope: str = Query(default='full', pattern='^(full|live)$'),
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> list[dict]:
  return get_admin_users(db, scope=scope)


@app.post('/admin/create-user', status_code=201)
def admin_create_user(
  input_data: CreateUserInput,
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return create_user(db, input_data)


@app.post('/users', status_code=201)
def users_create(
  input_data: CreateUserInput,
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return create_user(db, input_data)


@app.put('/admin/update-user/{user_id}')
def admin_update_user(
  user_id: str,
  input_data: UpdateUserInput,
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return update_user(db, user_id=user_id, input_data=input_data)


@app.delete('/admin/delete-user/{user_id}')
def admin_delete_user(
  user_id: str,
  current_user: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return delete_user(db, user_id=user_id, current_user=current_user)

# =========================
# FACE EMBEDDINGS
# =========================

@app.get('/users/embeddings')
def users_embeddings(
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> list[dict]:
  return get_user_embeddings(db)


@app.post('/users/embeddings')
def users_embeddings_save(
  input_data: DescriptorEnrollmentInput,
  current_user: User = Depends(get_current_user),
  db: Session = Depends(get_db),
) -> dict:
  return save_user_embeddings(db, current_user, input_data)


@app.delete('/admin/users/{user_id}/embeddings')
def admin_delete_user_embeddings(
  user_id: str,
  current_user: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return delete_user_embeddings(
    db,
    user_id=user_id,
    current_user=current_user,
  )

# =========================
# MEMBERSHIP
# =========================

@app.post('/admin/create-membership')
def admin_create_membership(
  input_data: CreateMembershipInput,
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return create_membership(db, input_data)

# =========================
# SLOTS
# =========================

@app.get('/admin/slots')
def admin_slots(
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> list[dict]:
  return get_admin_slots(db)


@app.post('/admin/slots', status_code=201)
def admin_create_slot(
  input_data: CreateSlotInput,
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return create_slot(db, input_data)


@app.put('/admin/slots/{slot_id}')
def admin_update_slot(
  slot_id: str,
  input_data: UpdateSlotInput,
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return update_slot(db, slot_id, input_data)


@app.delete('/admin/slots/{slot_id}')
def admin_delete_slot(
  slot_id: str,
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return delete_slot(db, slot_id)

# =========================
# ANNOUNCEMENTS
# =========================

@app.get('/admin/announcements')
def admin_announcements(
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> list[dict]:
  return get_admin_announcements(db)


@app.post('/admin/announcements')
def admin_create_announcement(
  input_data: CreateAnnouncementInput,
  current_admin: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return create_announcement(
    db,
    creator=current_admin,
    input_data=input_data,
  )

# =========================
# REPORTS
# =========================

@app.get('/admin/reports')
def admin_reports(
  scope: str = Query(default='full', pattern='^(full|live)$'),
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return get_admin_reports(db, scope=scope)


@app.get('/admin/user/{user_id}/report')
def user_report(
  user_id: str,
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return get_user_report(db, user_id)

# =========================
# SESSIONS
# =========================

@app.get('/admin/sessions')
def admin_sessions(
  scope: str = Query(default='full', pattern='^(full|live)$'),
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> list[dict]:
  return get_admin_sessions(db, scope=scope)


@app.post('/session/start')
def session_start(
  input_data: SessionStartInput,
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return start_session(db, input_data)


@app.post('/session/end')
def session_end(
  input_data: SessionEndInput,
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return end_session(db, input_data)


@app.get('/admin/session/timer/{session_id}')
def session_timer(
  session_id: str,
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return get_session_timer(db, session_id)

# =========================
# USER DASHBOARD
# =========================

@app.get('/user/dashboard')
def user_dashboard(
  user: User = Depends(get_current_user),
  db: Session = Depends(get_db),
) -> dict:
  return get_user_dashboard(db, user)


@app.get('/user/profile')
def user_profile(
  user: User = Depends(get_current_user),
  db: Session = Depends(get_db),
) -> dict:
  return get_user_profile(db, user)


@app.get('/user/history')
def user_history(
  user: User = Depends(get_current_user),
  db: Session = Depends(get_db),
) -> list[dict]:
  return get_user_history(db, user)


@app.get('/user/payments')
def user_payments(
  user: User = Depends(get_current_user),
  db: Session = Depends(get_db),
) -> list[dict]:
  return get_user_payments(db, user)


@app.get('/user/notifications')
def user_notifications(
  user: User = Depends(get_current_user),
  db: Session = Depends(get_db),
) -> list[dict]:
  return get_user_notifications(db, user)

# =========================
# LIVE FEED
# =========================

@app.delete('/admin/live-feed')
def admin_clear_live_feed(
  _: User = Depends(get_current_admin),
) -> dict:
  return clear_live_feed()

# =========================
# FACE STATUS
# =========================

@app.get('/admin/face-enrollment-status')
def face_enrollment_status(
  _: User = Depends(get_current_admin),
  db: Session = Depends(get_db),
) -> dict:
  return get_face_enrollment_status(db)
# =========================
# LEGACY FACE ROUTES
# =========================

@app.post('/admin/upload-face')
def admin_upload_face(
  _: User = Depends(get_current_admin),
) -> dict:
  raise ApiError(
    'Legacy image-based enrollment is disabled. '
    'Use /users/embeddings from the browser.',
    410,
  )


@app.post('/user/upload-face')
def user_upload_face(
  _: User = Depends(get_current_user),
) -> dict:
  raise ApiError(
    'Legacy image-based enrollment is disabled. '
    'Use /users/embeddings from the browser.',
    410,
  )


@app.post('/access/scan')
def access_scan(
  _: User = Depends(get_current_admin),
) -> dict:
  raise ApiError(
    'Legacy image-based scanning is disabled. '
    'Use browser recognition and POST /attendance.',
    410,
  )

# =========================
# DOOR DETECTION
# =========================

@app.post('/door/detection')
def door_detection(
  payload: dict[str, Any],
  _: User = Depends(get_current_admin),
) -> dict[str, Any]:

  status_text = str(payload.get('status') or '').lower()

  known_face = (
    bool(payload.get('knownFace'))
    and status_text not in {'unknown', 'retry', 'denied'}
  )

  force_lock = (
    bool(payload.get('forceLock'))
    or not known_face
  )

  name = (
    payload.get('name')
    if isinstance(payload.get('name'), str)
    else None
  )

  return sync_door_for_detection(
    known_face=known_face,
    name=name,
    force_lock=force_lock,
  )

# =========================
# TTS LEGACY
# =========================

@app.post('/api/tts')
def tts_generate_legacy(
  input_data: TTSRequest,
  _: User = Depends(get_current_admin),
) -> Response:

  audio_bytes, media_type = generate_tts(input_data.text)

  return Response(
    content=audio_bytes,
    media_type=media_type,
  )

# =========================
# FINAL FRONTEND MOUNT
# =========================

if FRONTEND_DIR.exists():

  app.mount(
    '/app',
    StaticFiles(
      directory=FRONTEND_DIR,
      html=True,
    ),
    name='frontend',
  )

# =========================
# SERVER START
# =========================

if __name__ == "__main__":

  import os
  import uvicorn

  uvicorn.run(
    app,
    host="0.0.0.0",
    port=int(os.getenv('PORT', '8001')),
  )