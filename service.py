from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import tempfile
import time
from collections import Counter, deque
from datetime import date, datetime, time as dt_time, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any
from urllib import error as urllib_error
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, load_only, selectinload

if __package__:
  from .db import get_settings
  from .media_storage import MediaStorage
  from .models import (
    Announcement,
    FaceEmbedding,
    Notification,
    PaymentHistory,
    SessionRecord,
    SessionStatus,
    TimeSlot,
    TimelineEventType,
    Tone,
    User,
    UserRole,
    UserTimeline,
    generate_uuid,
    utcnow,
  )
  from .schemas import (
    AccessScanInput,
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
    UpdateSlotInput,
    UpdateUserInput,
    UploadFaceInput,
  )
  from .security import create_access_token, hash_password, verify_password
else:
  from db import get_settings
  from media_storage import MediaStorage
  from models import (
    Announcement,
    FaceEmbedding,
    Notification,
    PaymentHistory,
    SessionRecord,
    SessionStatus,
    TimeSlot,
    TimelineEventType,
    Tone,
    User,
    UserRole,
    UserTimeline,
    generate_uuid,
    utcnow,
  )
  from schemas import (
    AccessScanInput,
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
    UpdateSlotInput,
    UpdateUserInput,
    UploadFaceInput,
  )
  from security import create_access_token, hash_password, verify_password



STORAGE_ROOT = Path(__file__).resolve().parent / 'storage'
USER_STORAGE = STORAGE_ROOT / 'users'
SESSION_STORAGE = STORAGE_ROOT / 'sessions'
UNKNOWN_STORAGE = STORAGE_ROOT / 'unknown'

for directory in (USER_STORAGE, SESSION_STORAGE, UNKNOWN_STORAGE):
  directory.mkdir(parents=True, exist_ok=True)

MEDIA_STORAGE = MediaStorage(get_settings(), STORAGE_ROOT)

FACE_MATCH_THRESHOLD = 0.47
FACE_RETRY_THRESHOLD = 0.58
COOLDOWN_SECONDS = 300
ENTRY_DUPLICATE_SECONDS = 30
MIN_EXIT_SECONDS = 300
SESSION_LIMIT_MINUTES = 70
SESSION_WARNING_MINUTES = 5
MEMBERSHIP_VISIT_LIMITS = {
  'Monthly': 30,
  '2 Month': 60,
  'Quarterly': 90,
  'Half-Yearly': 180,
  'Yearly': 365,
}
CLUB_TIMEZONE = ZoneInfo('Asia/Kolkata')
RECENT_SCAN_EVENTS: deque[dict[str, Any]] = deque(maxlen=12)
ELEVENLABS_MODEL_ID = 'eleven_multilingual_v2'
DEFAULT_VOICE_ID = 'pNInz6obpgDQGcFmaJgB'
READ_SIDE_EXPIRY_TTL_SECONDS = 5.0
USER_EMBEDDINGS_CACHE_TTL_SECONDS = 60.0
LIVE_DASHBOARD_SESSION_LIMIT = 120
LIVE_DASHBOARD_PAYMENT_LIMIT = 80

_expire_overdue_sessions_lock = Lock()
_expire_overdue_sessions_last_all_at = 0.0
_expire_overdue_sessions_last_by_user: dict[str, float] = {}
_user_embeddings_cache_lock = Lock()
_user_embeddings_cache_payload: list[dict[str, Any]] | None = None
_user_embeddings_cache_at = 0.0


class ApiError(Exception):
  def __init__(self, message: str, status_code: int = 400) -> None:
    super().__init__(message)
    self.message = message
    self.status_code = status_code


def _safe_json_loads(value: str) -> dict[str, Any] | None:
  try:
    parsed = json.loads(value)
  except json.JSONDecodeError:
    return None

  return parsed if isinstance(parsed, dict) else None


def _default_member_id(user: User) -> str:
  return f"CC-{user.id.replace('-', '')[:8].upper()}"


def _normalize_member_id(value: str | None, fallback: str) -> str:
  if not value:
    return fallback

  normalized = ''.join(
    character if character.isalnum() else '-'
    for character in value.strip().upper()
  ).strip('-')
  normalized = '-'.join(part for part in normalized.split('-') if part)

  return normalized or fallback


def _member_id_prefix(role: UserRole | str) -> str:
  normalized_role = role.value if isinstance(role, UserRole) else str(role).strip().lower()
  return 'ADMIN' if normalized_role == 'admin' else 'CSC'


def _generate_member_id(db: Session, role: UserRole | str) -> str:
  prefix = _member_id_prefix(role)
  width = 4 if prefix == 'ADMIN' else 3
  pattern = re.compile(rf'^{re.escape(prefix)}-(\d+)$')
  existing_ids = db.scalars(
    select(User.member_id).where(User.member_id.like(f'{prefix}-%'))
  ).all()

  highest = 0
  for member_id in existing_ids:
    match = pattern.match(str(member_id or '').strip().upper())
    if match:
      highest = max(highest, int(match.group(1)))

  return f'{prefix}-{highest + 1:0{width}d}'


def _legacy_member_metadata(user: User) -> dict[str, Any]:
  parsed = _safe_json_loads(user.note or '')

  if parsed is None or parsed.get('tag') != 'member-profile-v1':
    return {}

  return parsed


def _resolve_member_id(user: User) -> str:
  fallback = _default_member_id(user)
  direct_value = getattr(user, 'member_id', None)
  legacy_value = _legacy_member_metadata(user).get('memberId')

  if isinstance(direct_value, str) and direct_value.strip():
    normalized_direct = _normalize_member_id(direct_value, fallback)
    if (
      normalized_direct != fallback
      or not isinstance(legacy_value, str)
      or not legacy_value.strip()
    ):
      return normalized_direct

  return _normalize_member_id(legacy_value if isinstance(legacy_value, str) else None, fallback)


def _resolve_payment_amount(user: User) -> float:
  direct_value = float(user.payment_amount or 0)

  if direct_value > 0:
    return direct_value

  legacy_value = _legacy_member_metadata(user).get('paymentAmount')
  if isinstance(legacy_value, (int, float)):
    return max(0.0, float(legacy_value))

  if isinstance(legacy_value, str):
    try:
      return max(0.0, float(legacy_value))
    except ValueError:
      return direct_value

  return direct_value


def _resolve_payment_mode(user: User) -> str:
  direct_value = (user.payment_mode or '').strip()
  legacy_value = _legacy_member_metadata(user).get('paymentMode')

  if direct_value in {'Cash', 'UPI', 'Card'} and not (
    direct_value == 'UPI' and legacy_value in {'Cash', 'UPI', 'Card'}
  ):
    return direct_value

  return legacy_value if legacy_value in {'Cash', 'UPI', 'Card'} else 'UPI'


def _resolve_payment_status(user: User) -> str:
  direct_value = (user.payment_status or '').strip()
  legacy_value = _legacy_member_metadata(user).get('paymentStatus')

  if direct_value in {'Paid', 'Pending', 'Expired'} and not (
    direct_value == 'Pending' and legacy_value in {'Paid', 'Pending', 'Expired'}
  ):
    return direct_value

  return legacy_value if legacy_value in {'Paid', 'Pending', 'Expired'} else 'Pending'


def _resolve_sport(user: User) -> str:
  return (user.sport or '').strip() or 'General'


def _resolve_membership_level(user: User) -> str:
  return (user.membership_level or '').strip()


def _resolve_due_amount(user: User) -> float:
  return max(0.0, float(user.due_amount or 0))


def _resolve_admin_note(user: User) -> str:
  parsed = _legacy_member_metadata(user)

  if parsed:
    admin_note = parsed.get('adminNote')
    return admin_note.strip() if isinstance(admin_note, str) else ''

  return (user.note or '').strip()


def _serialize_datetime(value: datetime | None) -> str | None:
  if value is None:
    return None

  return value.isoformat() + 'Z'


def _serialize_date(value: date | None) -> str | None:
  if value is None:
    return None

  return value.isoformat()


def _serialize_time(value: dt_time | None) -> str | None:
  if value is None:
    return None

  return value.strftime('%H:%M')


def _serialize_attendance_action(value: str | None) -> str | None:
  if value is None:
    return None

  normalized = str(value).strip().upper()
  return normalized if normalized in {'IN', 'OUT'} else None


def _action_cooldown_remaining_seconds(
  user: User,
  *,
  now: datetime | None = None,
) -> int:
  if _serialize_attendance_action(getattr(user, 'last_action', None)) != 'OUT':
    return 0

  last_action_at = getattr(user, 'last_action_at', None)
  if last_action_at is None:
    return 0

  reference = now or utcnow()
  elapsed = int((reference - last_action_at).total_seconds())
  remaining = COOLDOWN_SECONDS - elapsed
  return max(0, remaining)


def _club_now() -> datetime:
  return datetime.now(CLUB_TIMEZONE)


def _club_today() -> date:
  return _club_now().date()


def _utc_to_local(value: datetime) -> datetime:
  return value.replace(tzinfo=timezone.utc).astimezone(CLUB_TIMEZONE)


def _local_to_utc_naive(value: datetime) -> datetime:
  return value.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_slot_time(value: str) -> dt_time:
  return datetime.strptime(value, '%H:%M').time()


def _deterministic_hue(seed: str) -> int:
  return int(sha256(seed.encode('utf-8')).hexdigest()[:6], 16) % 360


def _membership_status(user: User) -> str:
  if not user.membership_expiry:
    return 'expired'

  remaining_days = (user.membership_expiry - _club_today()).days

  if remaining_days < 0:
    return 'expired'

  if remaining_days <= 7:
    return 'warning'

  return 'active'


def _days_left(user: User) -> int:
  if not user.membership_expiry:
    return 0

  return max(0, (user.membership_expiry - _club_today()).days)


def _membership_visit_limit(user: User) -> int:
  override_value = getattr(user, 'visit_limit', None)
  if override_value is not None:
    return max(0, int(override_value))

  configured_limit = MEMBERSHIP_VISIT_LIMITS.get(str(user.membership_plan or '').strip())
  if configured_limit is not None:
    return configured_limit

  if user.membership_start and user.membership_expiry:
    return max(0, (user.membership_expiry - user.membership_start).days)

  return 0


def _membership_visit_stats(user: User) -> dict[str, int]:
  entry_timelines = [
    timeline for timeline in user.timelines if timeline.event_type == TimelineEventType.ENTRY
  ]
  membership_start = user.membership_start
  membership_expiry = user.membership_expiry

  membership_visits_used = 0
  for timeline in entry_timelines:
    occurred_at = getattr(timeline, 'occurred_at', None)
    if occurred_at is None:
      continue

    local_date = _utc_to_local(occurred_at).date()
    if membership_start and local_date < membership_start:
      continue
    if membership_expiry and local_date > membership_expiry:
      continue
    membership_visits_used += 1

  membership_visits_allowed = _membership_visit_limit(user)
  membership_visits_remaining = max(0, membership_visits_allowed - membership_visits_used)

  return {
    'totalVisits': len(entry_timelines),
    'membershipVisitsAllowed': membership_visits_allowed,
    'membershipVisitsUsed': membership_visits_used,
    'membershipVisitsRemaining': membership_visits_remaining,
  }


def _slot_window_for_local_date(slot: TimeSlot, target_date: date) -> tuple[datetime, datetime]:
  start_local = datetime.combine(target_date, slot.start_time, CLUB_TIMEZONE)
  end_local = datetime.combine(target_date, slot.end_time, CLUB_TIMEZONE)

  if end_local <= start_local:
    end_local += timedelta(days=1)

  return _local_to_utc_naive(start_local), _local_to_utc_naive(end_local)


def _resolve_slot_window(slot: TimeSlot, reference_utc: datetime | None = None) -> tuple[datetime, datetime]:
  now_utc = reference_utc or utcnow()
  now_local = _utc_to_local(now_utc)
  candidate_date = now_local.date()

  if slot.end_time <= slot.start_time and now_local.time() < slot.end_time:
    candidate_date = candidate_date - timedelta(days=1)

  return _slot_window_for_local_date(slot, candidate_date)


def _resolve_session_slot_window(
  slot: TimeSlot | None,
  *,
  reference_utc: datetime | None = None,
) -> tuple[datetime | None, datetime | None]:
  if slot is None:
    return None, None

  return _resolve_slot_window(slot, reference_utc)


def _slot_status_payload(
  slot: TimeSlot | None,
  *,
  reference_utc: datetime | None = None,
) -> dict[str, Any]:
  if slot is None:
    return {
      'status': 'unassigned',
      'startsAt': None,
      'endsAt': None,
      'secondsRemaining': 0,
      'secondsUntilStart': 0,
      'attendanceLocked': True,
    }

  now_utc = reference_utc or utcnow()
  start_utc, end_utc = _resolve_slot_window(slot, now_utc)

  if now_utc < start_utc:
    return {
      'status': 'upcoming',
      'startsAt': _serialize_datetime(start_utc),
      'endsAt': _serialize_datetime(end_utc),
      'secondsRemaining': 0,
      'secondsUntilStart': max(0, int((start_utc - now_utc).total_seconds())),
      'attendanceLocked': True,
    }

  if now_utc <= end_utc:
    return {
      'status': 'active',
      'startsAt': _serialize_datetime(start_utc),
      'endsAt': _serialize_datetime(end_utc),
      'secondsRemaining': max(0, int((end_utc - now_utc).total_seconds())),
      'secondsUntilStart': 0,
      'attendanceLocked': False,
    }

  next_start_utc, next_end_utc = _slot_window_for_local_date(
    slot,
    _utc_to_local(now_utc).date() + timedelta(days=1),
  )
  return {
    'status': 'ended',
    'startsAt': _serialize_datetime(next_start_utc),
    'endsAt': _serialize_datetime(next_end_utc),
    'secondsRemaining': 0,
    'secondsUntilStart': max(0, int((next_start_utc - now_utc).total_seconds())),
    'attendanceLocked': True,
  }


def _serialize_slot(slot: TimeSlot | None) -> dict[str, Any] | None:
  if slot is None:
    return None

  return {
    'id': slot.id,
    'name': slot.name,
    'startTime': _serialize_time(slot.start_time),
    'endTime': _serialize_time(slot.end_time),
    'createdAt': _serialize_datetime(slot.created_at),
    'updatedAt': _serialize_datetime(slot.updated_at),
  }


def _slot_gate_message(slot: TimeSlot | None) -> tuple[bool, str, str]:
  if slot is None:
    return False, 'No attendance slot is assigned to this user yet.', 'No slot assigned.'

  slot_state = _slot_status_payload(slot)
  window_label = f'{slot.name} ({_serialize_time(slot.start_time)}-{_serialize_time(slot.end_time)})'

  if slot_state['status'] == 'upcoming':
    return (
      False,
      f'Attendance opens at {window_label}.',
      f'Your slot starts at {window_label}.',
    )

  if slot_state['status'] == 'ended':
    return (
      False,
      f'Attendance window closed for {window_label}.',
      'Your session has ended.',
    )

  return True, '', ''


def _session_duration_minutes(session: SessionRecord) -> int:
  end_time = session.ended_at or utcnow()
  return max(0, int((end_time - session.started_at).total_seconds() // 60))


def _session_elapsed_seconds(session: SessionRecord, now: datetime | None = None) -> int:
  reference_time = now or utcnow()
  return max(0, int((reference_time - session.started_at).total_seconds()))


def _exit_lock_remaining_seconds(session: SessionRecord, now: datetime | None = None) -> int:
  elapsed = _session_elapsed_seconds(session, now)
  return max(0, MIN_EXIT_SECONDS - elapsed)


def _session_limit_deadline(session: SessionRecord) -> datetime:
  return session.started_at + timedelta(minutes=SESSION_LIMIT_MINUTES)


def _session_deadline(session: SessionRecord) -> datetime:
  limit_deadline = _session_limit_deadline(session)

  if session.slot_end_at is None:
    return limit_deadline

  return min(limit_deadline, session.slot_end_at)


def _session_deadline_note(session: SessionRecord) -> str:
  if session.slot_end_at is not None and session.slot_end_at <= _session_limit_deadline(session):
    return 'Session ended when the assigned slot closed.'

  return f'Session ended automatically after {SESSION_LIMIT_MINUTES} minutes.'


def _session_remaining_seconds(session: SessionRecord, now: datetime | None = None) -> int:
  if session.status != SessionStatus.ACTIVE:
    return 0

  reference_time = now or utcnow()
  return max(0, int((_session_deadline(session) - reference_time).total_seconds()))


def _session_tts(session: SessionRecord) -> str:
  if session.status == SessionStatus.ACTIVE:
    return 'Attendance active'

  if session.status == SessionStatus.DENIED:
    return 'Access denied'

  if session.status == SessionStatus.EXPIRED:
    return 'Your session has ended'

  return 'Attendance marked'


def _serialize_session(session: SessionRecord) -> dict[str, Any]:
  started_at = _serialize_datetime(session.started_at)
  ended_at = _serialize_datetime(session.ended_at)
  duration_minutes = _session_duration_minutes(session)
  remaining_seconds = _session_remaining_seconds(session)

  return {
    'id': session.id,
    'userId': session.user_id,
    'memberId': _resolve_member_id(session.user),
    'name': session.user.name,
    'email': session.user.email,
    'area': session.area,
    'status': session.status.value,
    'confidence': round(float(session.confidence or 0), 2),
    'startedAt': started_at,
    'endedAt': ended_at,
    'checkIn': started_at,
    'checkOut': ended_at,
    'slotId': session.slot_id,
    'slotName': session.slot.name if session.slot else None,
    'slotStartTime': _serialize_time(session.slot.start_time) if session.slot else None,
    'slotEndTime': _serialize_time(session.slot.end_time) if session.slot else None,
    'slotStartAt': _serialize_datetime(session.slot_start_at),
    'slotEndAt': _serialize_datetime(session.slot_end_at),
    'durationMinutes': duration_minutes,
    'membershipPlan': session.user.membership_plan,
    'plan': session.user.membership_plan,
    'entryTime': started_at,
    'remainingMinutes': (remaining_seconds + 59) // 60 if remaining_seconds else 0,
    'remainingSeconds': remaining_seconds,
    'tts': _session_tts(session),
  }


def _serialize_timeline(item: UserTimeline) -> dict[str, Any]:
  return {
    'id': item.id,
    'eventType': item.event_type.value,
    'area': item.area,
    'occurredAt': _serialize_datetime(item.occurred_at),
    'totalMinutes': item.total_minutes,
    'note': item.note,
  }


def _serialize_payment(item: PaymentHistory) -> dict[str, Any]:
  user = getattr(item, 'user', None)
  return {
    'id': item.id,
    'userId': item.user_id,
    'userName': user.name if user else None,
    'memberId': _resolve_member_id(user) if user else None,
    'plan': item.plan,
    'amount': float(item.amount),
    'paymentMode': item.payment_mode,
    'paymentStatus': item.payment_status,
    'membershipStart': _serialize_date(item.membership_start),
    'membershipExpiry': _serialize_date(item.membership_expiry),
    'source': item.source,
    'createdAt': _serialize_datetime(item.created_at),
  }


def _serialize_notification(item: Notification) -> dict[str, Any]:
  return {
    'id': item.id,
    'title': item.title,
    'message': item.message,
    'tone': item.tone.value,
    'createdAt': _serialize_datetime(item.created_at),
    'isRead': item.is_read,
  }


def _serialize_announcement(item: Announcement) -> dict[str, Any]:
  return {
    'id': item.id,
    'title': item.title,
    'message': item.message,
    'detail': item.message,
    'time': _serialize_datetime(item.created_at),
    'tone': item.tone.value,
    'targetName': item.target_user.name if item.target_user else None,
    'createdAt': _serialize_datetime(item.created_at),
  }


def _normalize_admin_scope(scope: str | None) -> str:
  normalized = str(scope or 'full').strip().lower()
  return 'live' if normalized == 'live' else 'full'


def _admin_user_summary_query():
  return (
    select(User)
    .options(
      load_only(
        User.id,
        User.name,
        User.email,
        User.role,
        User.mobile_number,
        User.member_id,
        User.slot_id,
        User.sport,
        User.membership_plan,
        User.membership_level,
        User.membership_start,
        User.membership_expiry,
        User.visit_limit,
        User.payment_amount,
        User.due_amount,
        User.payment_mode,
        User.payment_status,
        User.last_action,
        User.last_action_at,
        User.note,
        User.face_images_count,
        User.created_at,
        User.updated_at,
      ),
      selectinload(User.slot).load_only(
        TimeSlot.id,
        TimeSlot.name,
        TimeSlot.start_time,
        TimeSlot.end_time,
      ),
    )
    .where(User.role == UserRole.USER)
    .order_by(User.created_at.desc())
  )


def _admin_session_query():
  return (
    select(SessionRecord)
    .options(
      load_only(
        SessionRecord.id,
        SessionRecord.user_id,
        SessionRecord.slot_id,
        SessionRecord.area,
        SessionRecord.status,
        SessionRecord.confidence,
        SessionRecord.started_at,
        SessionRecord.ended_at,
        SessionRecord.slot_start_at,
        SessionRecord.slot_end_at,
      ),
      selectinload(SessionRecord.user).load_only(
        User.id,
        User.name,
        User.email,
        User.member_id,
        User.membership_plan,
        User.note,
      ),
      selectinload(SessionRecord.slot).load_only(
        TimeSlot.id,
        TimeSlot.name,
        TimeSlot.start_time,
        TimeSlot.end_time,
      ),
    )
  )


def _first_face_asset_url(user: User) -> str | None:
  for embedding in user.face_embeddings:
    image_data = str(embedding.image_data or '').strip()
    if image_data.startswith('http://') or image_data.startswith('https://'):
      return image_data
  return None


def _serialize_auth_user(user: User) -> dict[str, Any]:
  last_action = _serialize_attendance_action(user.last_action)
  last_action_at = _serialize_datetime(user.last_action_at)
  return {
    'id': user.id,
    'memberId': _resolve_member_id(user),
    'name': user.name,
    'email': user.email,
    'mobileNumber': user.mobile_number,
    'mobile_number': user.mobile_number,
    'role': user.role.value,
    'membershipPlan': user.membership_plan,
    'sport': _resolve_sport(user),
    'membershipLevel': _resolve_membership_level(user),
    'membershipStatus': _membership_status(user),
    'membershipExpiry': _serialize_date(user.membership_expiry),
    'visitLimit': user.visit_limit,
    'paymentAmount': _resolve_payment_amount(user),
    'dueAmount': _resolve_due_amount(user),
    'paymentMode': _resolve_payment_mode(user),
    'paymentStatus': _resolve_payment_status(user),
    'faceImageUrl': _first_face_asset_url(user),
    'lastAction': last_action,
    'lastActionAt': last_action_at,
    'lastTimestamp': last_action_at,
    'cooldownRemainingSeconds': _action_cooldown_remaining_seconds(user),
    'slotId': user.slot_id,
    'slotName': user.slot.name if user.slot else None,
    'slotStartTime': _serialize_time(user.slot.start_time) if user.slot else None,
    'slotEndTime': _serialize_time(user.slot.end_time) if user.slot else None,
    'slot': _serialize_slot(user.slot),
  }


def _serialize_user_summary(user: User) -> dict[str, Any]:
  last_action = _serialize_attendance_action(user.last_action)
  last_action_at = _serialize_datetime(user.last_action_at)
  membership_visits_allowed = _membership_visit_limit(user)

  return {
    'id': user.id,
    'memberId': _resolve_member_id(user),
    'name': user.name,
    'email': user.email,
    'mobileNumber': user.mobile_number,
    'mobile_number': user.mobile_number,
    'role': user.role.value,
    'slotId': user.slot_id,
    'slotName': user.slot.name if user.slot else None,
    'slotStartTime': _serialize_time(user.slot.start_time) if user.slot else None,
    'slotEndTime': _serialize_time(user.slot.end_time) if user.slot else None,
    'slot': _serialize_slot(user.slot),
    'sport': _resolve_sport(user),
    'membershipLevel': _resolve_membership_level(user),
    'membershipPlan': user.membership_plan,
    'membershipStatus': _membership_status(user),
    'membershipStart': _serialize_date(user.membership_start),
    'membershipExpiry': _serialize_date(user.membership_expiry),
    'visitLimit': user.visit_limit,
    'paymentAmount': _resolve_payment_amount(user),
    'dueAmount': _resolve_due_amount(user),
    'paymentMode': _resolve_payment_mode(user),
    'paymentStatus': _resolve_payment_status(user),
    'faceImageUrl': None,
    'lastAction': last_action,
    'lastActionAt': last_action_at,
    'lastTimestamp': last_action_at,
    'cooldownRemainingSeconds': _action_cooldown_remaining_seconds(user),
    'daysLeft': _days_left(user),
    'faceImageCount': user.face_images_count,
    'createdAt': _serialize_datetime(user.created_at),
    'updatedAt': _serialize_datetime(user.updated_at),
    'plan': user.membership_plan,
    'startDate': _serialize_date(user.membership_start),
    'expiry': _serialize_date(user.membership_expiry),
    'status': _membership_status(user),
    'imageCount': user.face_images_count,
    'hue': _deterministic_hue(user.id),
    'visits': 0,
    'membershipVisitsAllowed': membership_visits_allowed,
    'membershipVisitsUsed': 0,
    'membershipVisitsRemaining': membership_visits_allowed,
    'confidence': 0,
  }


def _serialize_user(user: User) -> dict[str, Any]:
  visit_stats = _membership_visit_stats(user)
  latest_session = max(user.sessions, key=lambda item: item.started_at, default=None)
  last_action = _serialize_attendance_action(user.last_action)
  last_action_at = _serialize_datetime(user.last_action_at)

  return {
    'id': user.id,
    'memberId': _resolve_member_id(user),
    'name': user.name,
    'email': user.email,
    'mobileNumber': user.mobile_number,
    'mobile_number': user.mobile_number,
    'role': user.role.value,
    'slotId': user.slot_id,
    'slotName': user.slot.name if user.slot else None,
    'slotStartTime': _serialize_time(user.slot.start_time) if user.slot else None,
    'slotEndTime': _serialize_time(user.slot.end_time) if user.slot else None,
    'slot': _serialize_slot(user.slot),
    'sport': _resolve_sport(user),
    'membershipLevel': _resolve_membership_level(user),
    'membershipPlan': user.membership_plan,
    'membershipStatus': _membership_status(user),
    'membershipStart': _serialize_date(user.membership_start),
    'membershipExpiry': _serialize_date(user.membership_expiry),
    'visitLimit': user.visit_limit,
    'paymentAmount': _resolve_payment_amount(user),
    'dueAmount': _resolve_due_amount(user),
    'paymentMode': _resolve_payment_mode(user),
    'paymentStatus': _resolve_payment_status(user),
    'faceImageUrl': _first_face_asset_url(user),
    'lastAction': last_action,
    'lastActionAt': last_action_at,
    'lastTimestamp': last_action_at,
    'cooldownRemainingSeconds': _action_cooldown_remaining_seconds(user),
    'daysLeft': _days_left(user),
    'faceImageCount': user.face_images_count,
    'note': _resolve_admin_note(user),
    'adminNote': _resolve_admin_note(user),
    'createdAt': _serialize_datetime(user.created_at),
    'updatedAt': _serialize_datetime(user.updated_at),
    'plan': user.membership_plan,
    'startDate': _serialize_date(user.membership_start),
    'expiry': _serialize_date(user.membership_expiry),
    'status': _membership_status(user),
    'imageCount': user.face_images_count,
    'hue': _deterministic_hue(user.id),
    'visits': visit_stats['totalVisits'],
    'membershipVisitsAllowed': visit_stats['membershipVisitsAllowed'],
    'membershipVisitsUsed': visit_stats['membershipVisitsUsed'],
    'membershipVisitsRemaining': visit_stats['membershipVisitsRemaining'],
    'confidence': round(float(latest_session.confidence), 2) if latest_session else 0,
  }


def _create_auth_payload(user: User) -> dict[str, Any]:
  return {
    'accessToken': create_access_token(subject=user.id, role=user.role.value),
    'tokenType': 'bearer',
    'role': user.role.value,
    'user': _serialize_auth_user(user),
  }


def _create_notification(
  db: Session,
  *,
  user: User,
  title: str,
  message: str,
  tone: Tone = Tone.BLUE,
) -> Notification:
  notification = Notification(
    user=user,
    title=title,
    message=message,
    tone=tone,
  )
  db.add(notification)
  return notification


def _create_announcement(
  db: Session,
  *,
  created_by: User | None,
  title: str,
  message: str,
  tone: Tone,
  target_user: User | None = None,
) -> Announcement:
  announcement = Announcement(
    title=title,
    message=message,
    tone=tone,
    target_user=target_user,
    created_by=created_by,
  )
  db.add(announcement)
  return announcement


def _create_timeline_event(
  db: Session,
  *,
  user: User,
  event_type: TimelineEventType,
  area: str,
  total_minutes: int | None = None,
  note: str = '',
) -> UserTimeline:
  item = UserTimeline(
    user=user,
    event_type=event_type,
    area=area,
    total_minutes=total_minutes,
    note=note,
  )
  db.add(item)
  return item


def _append_payment_entry(
  db: Session,
  *,
  user: User,
  plan: str,
  amount: float,
  payment_mode: str,
  payment_status: str,
  membership_start: date | None,
  membership_expiry: date | None,
  source: str,
) -> PaymentHistory:
  payment = PaymentHistory(
    user=user,
    plan=plan,
    amount=amount,
    payment_mode=payment_mode,
    payment_status=payment_status,
    membership_start=membership_start,
    membership_expiry=membership_expiry,
    source=source,
  )
  db.add(payment)
  return payment


def _get_user_by_id(db: Session, user_id: str) -> User:
  user = db.scalar(
    select(User)
    .options(
      selectinload(User.face_embeddings),
      selectinload(User.sessions),
      selectinload(User.timelines),
      selectinload(User.notifications),
      selectinload(User.payments),
      selectinload(User.slot),
    )
    .where(User.id == user_id)
  )

  if user is None:
    raise ApiError('User not found.', 404)

  return user


def _get_user_with_slot_by_id(db: Session, user_id: str) -> User:
  user = db.scalar(
    select(User)
    .options(selectinload(User.slot))
    .where(User.id == user_id)
  )

  if user is None:
    raise ApiError('User not found.', 404)

  return user


def _get_user_by_email(db: Session, email: str) -> User | None:
  return db.scalar(
    select(User)
    .options(
      selectinload(User.slot),
    )
    .where(User.email == email)
  )


def _get_user_by_member_id(db: Session, member_id: str) -> User | None:
  return db.scalar(
    select(User)
    .where(User.member_id == member_id)
  )


def _get_user_by_mobile_number(db: Session, mobile_number: str | None) -> User | None:
  if not mobile_number:
    return None

  return db.scalar(
    select(User)
    .where(User.mobile_number == mobile_number)
  )


def _get_slot_by_id(db: Session, slot_id: str) -> TimeSlot:
  slot = db.scalar(select(TimeSlot).where(TimeSlot.id == slot_id))

  if slot is None:
    raise ApiError('Slot not found.', 404)

  return slot


def _get_session_by_id(db: Session, session_id: str) -> SessionRecord:
  session = db.scalar(
    select(SessionRecord)
    .options(selectinload(SessionRecord.user), selectinload(SessionRecord.slot))
    .where(SessionRecord.id == session_id)
  )

  if session is None:
    raise ApiError('Session not found.', 404)

  return session


def _active_session_for_user(db: Session, user_id: str) -> SessionRecord | None:
  return db.scalar(
    select(SessionRecord)
    .options(selectinload(SessionRecord.user), selectinload(SessionRecord.slot))
    .where(
      SessionRecord.user_id == user_id,
      SessionRecord.status == SessionStatus.ACTIVE,
    )
    .order_by(SessionRecord.started_at.desc())
    .limit(1)
  )


def _latest_session_for_user(db: Session, user_id: str) -> SessionRecord | None:
  return db.scalar(
    select(SessionRecord)
    .options(selectinload(SessionRecord.user), selectinload(SessionRecord.slot))
    .where(SessionRecord.user_id == user_id)
    .order_by(SessionRecord.updated_at.desc(), SessionRecord.started_at.desc())
    .limit(1)
  )


def _day_window_for_local_date(target_date: date) -> tuple[datetime, datetime]:
  day_start_local = datetime.combine(target_date, dt_time.min, CLUB_TIMEZONE)
  day_end_local = day_start_local + timedelta(days=1)
  return _local_to_utc_naive(day_start_local), _local_to_utc_naive(day_end_local)


def _latest_session_for_user_on_date(
  db: Session,
  user_id: str,
  target_date: date,
) -> SessionRecord | None:
  day_start_utc, day_end_utc = _day_window_for_local_date(target_date)
  return db.scalar(
    select(SessionRecord)
    .options(selectinload(SessionRecord.user), selectinload(SessionRecord.slot))
    .where(
      SessionRecord.user_id == user_id,
      SessionRecord.started_at >= day_start_utc,
      SessionRecord.started_at < day_end_utc,
    )
    .order_by(SessionRecord.updated_at.desc(), SessionRecord.started_at.desc())
    .limit(1)
  )


def _expire_session(
  db: Session,
  *,
  session: SessionRecord,
  note: str = 'Session ended when the assigned slot closed.',
) -> None:
  if session.status != SessionStatus.ACTIVE:
    return

  ended_at = _session_deadline(session)
  session.status = SessionStatus.EXPIRED
  session.ended_at = ended_at
  _create_timeline_event(
    db,
    user=session.user,
    event_type=TimelineEventType.EXIT,
    area=session.area,
    total_minutes=max(0, int((ended_at - session.started_at).total_seconds() // 60)),
    note=note,
  )


def _expire_overdue_sessions(db: Session, user_id: str | None = None) -> None:
  now = utcnow()
  fixed_limit_cutoff = now - timedelta(minutes=SESSION_LIMIT_MINUTES)
  query = (
    select(SessionRecord)
    .options(selectinload(SessionRecord.user), selectinload(SessionRecord.slot))
    .where(
      SessionRecord.status == SessionStatus.ACTIVE,
      or_(
        (SessionRecord.started_at <= fixed_limit_cutoff),
        (SessionRecord.slot_end_at.is_not(None) & (SessionRecord.slot_end_at <= now)),
      ),
    )
  )

  if user_id:
    query = query.where(SessionRecord.user_id == user_id)

  sessions = db.scalars(query).all()

  if not sessions:
    return

  for session in sessions:
    _expire_session(
      db,
      session=session,
      note=_session_deadline_note(session),
    )

  db.commit()


def _expire_overdue_sessions_cached(
  db: Session,
  user_id: str | None = None,
  *,
  ttl_seconds: float = READ_SIDE_EXPIRY_TTL_SECONDS,
) -> None:
  global _expire_overdue_sessions_last_all_at

  now = time.monotonic()
  normalized_user_id = str(user_id or '').strip()

  with _expire_overdue_sessions_lock:
    if (now - _expire_overdue_sessions_last_all_at) < ttl_seconds:
      return

    if normalized_user_id:
      last_at = _expire_overdue_sessions_last_by_user.get(normalized_user_id, 0.0)
      if (now - last_at) < ttl_seconds:
        return

    _expire_overdue_sessions(db, normalized_user_id or None)

    completed_at = time.monotonic()
    if normalized_user_id:
      _expire_overdue_sessions_last_by_user[normalized_user_id] = completed_at
      stale_cutoff = completed_at - max(ttl_seconds * 4, 30.0)
      stale_user_ids = [
        cached_user_id
        for cached_user_id, cached_at in _expire_overdue_sessions_last_by_user.items()
        if cached_at < stale_cutoff
      ]
      for stale_user_id in stale_user_ids:
        _expire_overdue_sessions_last_by_user.pop(stale_user_id, None)
    else:
      _expire_overdue_sessions_last_all_at = completed_at
      _expire_overdue_sessions_last_by_user.clear()


def _tone_from_scan_status(status: str) -> Tone:
  if status == 'granted':
    return Tone.GREEN

  if status in {'retry', 'cooldown', 'duplicate'}:
    return Tone.AMBER

  return Tone.RED


def _feed_status_from_scan_status(status: str) -> str:
  if status == 'granted':
    return 'valid'

  if status in {'retry', 'cooldown', 'duplicate'}:
    return 'retry'

  return 'unknown'


def _distance_to_confidence(distance: float) -> float:
  scaled = max(0.0, min(1.0, 1.0 - (distance / 0.65)))
  return round(scaled, 2)


def _decode_image_payload(image_data: str) -> bytes:
  payload = image_data.strip()

  if ',' in payload and payload.startswith('data:image'):
    payload = payload.split(',', 1)[1]

  try:
    return base64.b64decode(payload)
  except Exception as error:  # noqa: BLE001
    raise ApiError('Invalid image payload.', 400) from error


def _load_rgb_image(image_bytes: bytes) -> np.ndarray:
  from PIL import Image

  image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
  return np.ascontiguousarray(np.array(image))


def _largest_face_location(
  locations: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
  return max(
    locations,
    key=lambda item: max(item[2] - item[0], 0) * max(item[1] - item[3], 0),
  )


def _normalize_face_box(
  location: tuple[int, int, int, int],
  *,
  width: int,
  height: int,
) -> dict[str, float]:
  top, right, bottom, left = location
  return {
    'top': round(top / height, 4),
    'left': round(left / width, 4),
    'width': round((right - left) / width, 4),
    'height': round((bottom - top) / height, 4),
  }


def _extract_face_encoding(
  image_bytes: bytes,
) -> tuple[np.ndarray | None, dict[str, float] | None, str | None]:
  import face_recognition

  try:
    image = _load_rgb_image(image_bytes)
  except Exception as error:  # noqa: BLE001
    raise ApiError('Unable to read image payload.', 400) from error

  locations = face_recognition.face_locations(image, model='hog')

  if not locations:
    return None, None, 'No face detected. Align your face and try again.'

  location = _largest_face_location(locations)
  encodings = face_recognition.face_encodings(image, [location])
  box = _normalize_face_box(location, width=image.shape[1], height=image.shape[0])

  if not encodings:
    return None, box, 'Face detected, but encoding failed. Try again.'

  return encodings[0], box, None


def _encoding_to_bytes(encoding: np.ndarray) -> bytes:
  return np.asarray(encoding, dtype=np.float64).tobytes()


def _encoding_from_bytes(raw: Any) -> np.ndarray | None:
  if raw is None:
    return None

  if isinstance(raw, memoryview):
    raw = raw.tobytes()

  if isinstance(raw, (bytes, bytearray)):
    buffer = bytes(raw)
    if len(buffer) % 8 == 0:
      return np.frombuffer(buffer, dtype=np.float64)

    try:
      parsed = json.loads(buffer.decode('utf-8'))
    except Exception:  # noqa: BLE001
      return None

    if isinstance(parsed, list):
      return np.asarray(parsed, dtype=np.float64)

    return None

  if isinstance(raw, str):
    try:
      parsed = json.loads(raw)
    except Exception:  # noqa: BLE001
      return None

    if isinstance(parsed, list):
      return np.asarray(parsed, dtype=np.float64)

  return None


def _descriptor_from_iterable(values: list[float]) -> np.ndarray:
  try:
    descriptor = np.asarray(values, dtype=np.float64)
  except Exception as error:  # noqa: BLE001
    raise ApiError('Descriptor payload is invalid.', 400) from error

  if descriptor.ndim != 1 or descriptor.size != 128:
    raise ApiError('Each face descriptor must contain 128 values.', 400)

  return descriptor


def _descriptor_to_payload(encoding: np.ndarray) -> list[float]:
  values = np.asarray(encoding, dtype=np.float64).tolist()
  return [round(float(value), 6) for value in values]


def _descriptor_centroid(encodings: list[np.ndarray]) -> list[float] | None:
  if not encodings:
    return None

  matrix = np.stack(encodings)
  centroid = np.mean(matrix, axis=0)
  magnitude = float(np.linalg.norm(centroid))

  if magnitude > 0:
    centroid = centroid / magnitude

  return _descriptor_to_payload(centroid)


def _invalidate_user_embeddings_cache() -> None:
  global _user_embeddings_cache_payload, _user_embeddings_cache_at

  with _user_embeddings_cache_lock:
    _user_embeddings_cache_payload = None
    _user_embeddings_cache_at = 0.0


def _save_user_face_image(user_id: str, image_bytes: bytes, index: int) -> str:
  timestamp = utcnow().strftime('%Y%m%d_%H%M%S_%f')
  key = f'users/{user_id}/faces/face_{index}_{timestamp}.jpg'
  return MEDIA_STORAGE.save_bytes(key, image_bytes, content_type='image/jpeg')


def _save_session_image(session_id: str, image_bytes: bytes, prefix: str) -> None:
  timestamp = utcnow().strftime('%Y%m%d_%H%M%S_%f')
  key = f'sessions/{session_id}/{prefix}_{timestamp}.jpg'
  MEDIA_STORAGE.save_bytes(key, image_bytes, content_type='image/jpeg')


def _save_unknown_image(image_bytes: bytes) -> None:
  timestamp = utcnow().strftime('%Y%m%d_%H%M%S_%f')
  key = f'unknown/unknown_{timestamp}.jpg'
  MEDIA_STORAGE.save_bytes(key, image_bytes, content_type='image/jpeg')


def _cooldown_remaining_seconds(db: Session, user_id: str) -> int:
  latest_session = _latest_session_for_user(db, user_id)

  if latest_session is None or latest_session.ended_at is None:
    return 0

  elapsed = int((utcnow() - latest_session.ended_at).total_seconds())
  remaining = COOLDOWN_SECONDS - elapsed
  return max(0, remaining)


def _entry_duplicate_remaining_seconds(session: SessionRecord) -> int:
  elapsed = int((utcnow() - session.started_at).total_seconds())
  remaining = ENTRY_DUPLICATE_SECONDS - elapsed
  return max(0, remaining)

def _format_wait_time(seconds: int) -> str:
  minutes, remaining_seconds = divmod(max(seconds, 0), 60)

  if minutes and remaining_seconds:
    return f'{minutes}m {remaining_seconds}s'

  if minutes:
    return f'{minutes}m'

  return f'{remaining_seconds}s'


def _record_scan_event(
  *,
  status: str,
  message: str,
  name: str | None,
  user_id: str | None,
  confidence: float,
  area: str,
  frames_captured: int,
  tts_message: str,
  duplicate_warning: bool,
  face_box: dict[str, float] | None,
  attendance_action: str | None,
) -> dict[str, Any]:
  event = {
    'id': generate_uuid(),
    'status': status,
    'message': message,
    'name': name,
    'userId': user_id,
    'confidence': round(confidence, 2),
    'area': area,
    'framesCaptured': frames_captured,
    'ttsMessage': tts_message,
    'duplicateWarning': duplicate_warning,
    'faceBox': face_box,
    'attendanceAction': attendance_action,
    'scannedAt': _serialize_datetime(utcnow()),
    'tone': _tone_from_scan_status(status).value,
  }
  RECENT_SCAN_EVENTS.appendleft(event)
  return event


def _build_scan_response(
  *,
  status: str,
  message: str,
  confidence: float,
  name: str | None,
  duplicate_warning: bool,
  tts_message: str,
  attendance_action: str | None,
  session: dict[str, Any] | None,
  cooldown_remaining_seconds: int,
  face_box: dict[str, float] | None,
  area: str,
  frames_captured: int,
) -> dict[str, Any]:
  resolved_user_id = (session or {}).get('userId')
  event = _record_scan_event(
    status=status,
    message=message,
    name=name,
    user_id=resolved_user_id,
    confidence=confidence,
    area=area,
    frames_captured=frames_captured,
    tts_message=tts_message,
    duplicate_warning=duplicate_warning,
    face_box=face_box,
    attendance_action=attendance_action,
  )

  return {
    'status': status,
    'message': message,
    'tone': event['tone'],
    'confidence': confidence,
    'name': name,
    'userId': resolved_user_id,
    'duplicateWarning': duplicate_warning,
    'ttsMessage': tts_message,
    'attendanceAction': attendance_action,
    'session': session,
    'cooldownRemainingSeconds': cooldown_remaining_seconds,
    'faceBox': face_box,
    'scannedAt': event['scannedAt'],
  }


def _serialize_live_feed_event(event: dict[str, Any]) -> dict[str, Any]:
  return {
    'id': event['id'],
    'userId': event.get('userId'),
    'name': event['name'] or 'Unknown Face',
    'confidence': event['confidence'],
    'status': _feed_status_from_scan_status(event['status']),
    'area': event['area'],
    'framesCaptured': event['framesCaptured'],
    'ttsActive': bool(event['ttsMessage']),
    'duplicateWarning': event['duplicateWarning'],
    'faceBox': event['faceBox'],
    'message': event['message'],
    'attendanceAction': event['attendanceAction'],
    'scannedAt': event['scannedAt'],
  }


def _serialize_recent_activity(event: dict[str, Any]) -> dict[str, Any]:
  name = event['name'] or 'Unknown face'
  action = 'Attendance marked'
  if event['status'] == 'duplicate':
    action = 'Duplicate attendance'
  elif event['status'] == 'retry':
    action = 'Try again'
  elif event['status'] == 'denied':
    action = 'Access denied'
  elif event['status'] == 'unknown':
    action = 'Unknown face'

  return {
    'id': event['id'],
    'headline': action,
    'detail': f"{name} at {event['area']}",
    'message': event['message'],
    'tone': event['tone'],
    'confidence': event['confidence'],
    'time': event['scannedAt'],
  }


def _live_feed_items(db: Session) -> list[dict[str, Any]]:
  if RECENT_SCAN_EVENTS:
    return [_serialize_live_feed_event(event) for event in list(RECENT_SCAN_EVENTS)]

  sessions = db.scalars(
    select(SessionRecord)
    .options(
      load_only(
        SessionRecord.id,
        SessionRecord.area,
        SessionRecord.confidence,
        SessionRecord.started_at,
      ),
      selectinload(SessionRecord.user).load_only(
        User.id,
        User.name,
        User.face_images_count,
      ),
    )
    .order_by(SessionRecord.started_at.desc())
    .limit(4)
  ).all()

  fallback: list[dict[str, Any]] = []
  for session in sessions:
    fallback.append(
      {
        'id': session.id,
        'name': session.user.name,
        'confidence': round(float(session.confidence or 0), 2),
        'status': 'valid',
        'area': session.area,
        'framesCaptured': max(5, session.user.face_images_count),
        'ttsActive': False,
        'duplicateWarning': False,
        'faceBox': None,
        'message': 'Recent attendance activity',
        'attendanceAction': 'in',
        'scannedAt': _serialize_datetime(session.started_at),
      }
    )

  return fallback


def _reports_payload(
  db: Session,
  *,
  payments_limit: int | None = None,
) -> dict[str, Any]:
  timeline_items = db.scalars(
    select(UserTimeline)
    .options(
      load_only(
        UserTimeline.id,
        UserTimeline.user_id,
        UserTimeline.event_type,
        UserTimeline.area,
        UserTimeline.occurred_at,
        UserTimeline.note,
      ),
      selectinload(UserTimeline.user).load_only(User.id, User.name),
    )
    .order_by(UserTimeline.occurred_at.desc())
    .limit(50)
  ).all()

  attendance_counter: Counter[str] = Counter()
  peak_counter: Counter[str] = Counter()

  for item in timeline_items:
    if item.event_type == TimelineEventType.ENTRY:
      attendance_counter[item.occurred_at.strftime('%a')] += 1
      peak_counter[item.occurred_at.strftime('%I %p')] += 1

  weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
  attendance_bars = [
    {'label': day, 'count': attendance_counter.get(day, 0)}
    for day in weekday_order
  ]

  peak_hours = [
    {
      'label': label,
      'value': value,
      'note': 'Highest recognition traffic',
    }
    for label, value in peak_counter.most_common(4)
  ]

  entry_logs = [
    {
      'id': item.id,
      'name': item.user.name,
      'action': item.event_type.value.upper(),
      'zone': item.area,
      'time': _serialize_datetime(item.occurred_at),
    }
    for item in timeline_items[:8]
  ]

  users = db.scalars(
    select(User)
    .options(
      load_only(
        User.id,
        User.name,
        User.role,
        User.membership_plan,
        User.membership_expiry,
        User.due_amount,
      ),
    )
    .where(User.role == UserRole.USER)
    .order_by(User.updated_at.desc())
  ).all()
  payments_query = (
    select(PaymentHistory)
    .options(
      load_only(
        PaymentHistory.id,
        PaymentHistory.user_id,
        PaymentHistory.plan,
        PaymentHistory.amount,
        PaymentHistory.payment_mode,
        PaymentHistory.payment_status,
        PaymentHistory.membership_start,
        PaymentHistory.membership_expiry,
        PaymentHistory.source,
        PaymentHistory.created_at,
      ),
      selectinload(PaymentHistory.user).load_only(
        User.id,
        User.name,
        User.member_id,
        User.note,
      ),
    )
    .order_by(PaymentHistory.created_at.desc())
  )
  if payments_limit is not None:
    payments_query = payments_query.limit(max(1, int(payments_limit)))
  payments = db.scalars(payments_query).all()
  defaulters: list[dict[str, Any]] = []
  for user in users:
    if user.role != UserRole.USER:
      continue

    due_amount = _resolve_due_amount(user)
    membership_status = _membership_status(user)

    if due_amount > 0:
      defaulters.append(
        {
          'id': user.id,
          'name': user.name,
          'issue': 'Due amount',
          'detail': f'Outstanding due: Rs. {due_amount:,.2f}',
        }
      )
      continue

    if membership_status != 'active':
      defaulters.append(
        {
          'id': user.id,
          'name': user.name,
          'issue': 'Membership expired' if membership_status == 'expired' else 'Expiring soon',
          'detail': f"{user.membership_plan} plan ends on {_serialize_date(user.membership_expiry)}",
        }
      )

  defaulters = defaulters[:6]

  if RECENT_SCAN_EVENTS:
    recent_activity = [
      _serialize_recent_activity(event)
      for event in list(RECENT_SCAN_EVENTS)
    ]
  else:
    recent_activity = [
      {
        'id': item.id,
        'headline': item.event_type.value.title(),
        'detail': f"{item.user.name} at {item.area}",
        'message': item.note or f"{item.user.name} {item.event_type.value} recorded",
        'tone': 'green' if item.event_type == TimelineEventType.ENTRY else 'amber',
        'confidence': None,
        'time': _serialize_datetime(item.occurred_at),
      }
      for item in timeline_items[:6]
    ]

  return {
    'attendanceBars': attendance_bars,
    'peakHours': peak_hours,
    'entryLogs': entry_logs,
    'payments': [_serialize_payment(item) for item in payments],
    'defaulters': defaulters,
    'recentActivity': recent_activity,
  }


def authenticate_user(db: Session, input_data: LoginInput) -> dict[str, Any]:
  user = _get_user_by_email(db, input_data.email)

  if user is None or not verify_password(input_data.password, user.password_hash):
    raise ApiError('Invalid email or password.', 401)

  return _create_auth_payload(user)


def register_user(db: Session, input_data: RegisterInput) -> dict[str, Any]:
  _ = db
  _ = input_data
  raise ApiError('Only admins can create member IDs. Ask an admin to create your account.', 403)


def seed_database(db: Session) -> None:
  changed = False
  slot_seeds = [
    ('Morning Prime', '06:00', '07:00'),
    ('Morning Flex', '07:00', '08:00'),
    ('Evening Prime', '18:00', '19:00'),
  ]

  for name, start_time, end_time in slot_seeds:
    existing_slot = db.scalar(select(TimeSlot).where(TimeSlot.name == name))

    if existing_slot is not None:
      continue

    db.add(
      TimeSlot(
        name=name,
        start_time=_parse_slot_time(start_time),
        end_time=_parse_slot_time(end_time),
      )
    )
    changed = True

  seeds = [
    {
      'name': 'Club Admin',
      'email': 'admin@caperclub.ai',
      'password': 'caperclub',
      'role': UserRole.ADMIN,
      'plan': 'Yearly',
      'days': 730,
      'note': 'Default admin for local development.',
    },
  ]

  for seed in seeds:
    if _get_user_by_email(db, seed['email']) is not None:
      continue

    db.add(
      User(
        name=seed['name'],
        email=seed['email'],
        password_hash=hash_password(seed['password']),
        role=seed['role'],
        member_id='ADMIN-0001',
        membership_plan=seed['plan'],
        membership_start=_club_today(),
        membership_expiry=_club_today() + timedelta(days=seed['days']),
        payment_amount=0,
        payment_mode='UPI',
        payment_status='Paid',
        face_images_count=0,
        note=seed['note'],
      )
    )
    changed = True

  if changed:
    db.commit()


def get_current_user_payload(user: User) -> dict[str, Any]:
  return _serialize_auth_user(user)


def get_admin_slots(db: Session) -> list[dict[str, Any]]:
  slots = db.scalars(select(TimeSlot).order_by(TimeSlot.start_time.asc(), TimeSlot.name.asc())).all()
  return [_serialize_slot(slot) for slot in slots if slot is not None]


def create_slot(db: Session, input_data: CreateSlotInput) -> dict[str, Any]:
  existing = db.scalar(select(TimeSlot).where(func.lower(TimeSlot.name) == input_data.name.lower()))

  if existing is not None:
    raise ApiError('A slot with this name already exists.', 409)

  slot = TimeSlot(
    name=input_data.name,
    start_time=_parse_slot_time(input_data.startTime),
    end_time=_parse_slot_time(input_data.endTime),
  )
  db.add(slot)
  db.commit()
  db.refresh(slot)
  return _serialize_slot(slot) or {}


def update_slot(db: Session, slot_id: str, input_data: UpdateSlotInput) -> dict[str, Any]:
  slot = _get_slot_by_id(db, slot_id)
  existing = db.scalar(
    select(TimeSlot).where(
      func.lower(TimeSlot.name) == input_data.name.lower(),
      TimeSlot.id != slot.id,
    )
  )

  if existing is not None:
    raise ApiError('A slot with this name already exists.', 409)

  slot.name = input_data.name
  slot.start_time = _parse_slot_time(input_data.startTime)
  slot.end_time = _parse_slot_time(input_data.endTime)
  slot.updated_at = utcnow()
  db.commit()
  db.refresh(slot)
  return _serialize_slot(slot) or {}


def delete_slot(db: Session, slot_id: str) -> dict[str, Any]:
  slot = _get_slot_by_id(db, slot_id)
  assigned_users = db.scalar(select(func.count(User.id)).where(User.slot_id == slot.id)) or 0

  if assigned_users > 0:
    raise ApiError('Reassign users before deleting this slot.', 400)

  db.delete(slot)
  db.commit()
  return {'message': f'{slot.name} deleted successfully.'}


def get_admin_dashboard(db: Session) -> dict[str, Any]:
  _expire_overdue_sessions_cached(db)
  total_users = db.scalar(select(func.count(User.id)).where(User.role == UserRole.USER)) or 0
  day_start_utc, day_end_utc = _day_window_for_local_date(_club_today())
  attendance_today = db.scalar(
    select(func.count(UserTimeline.id)).where(
      UserTimeline.event_type == TimelineEventType.ENTRY,
      UserTimeline.occurred_at >= day_start_utc,
      UserTimeline.occurred_at < day_end_utc,
    )
  ) or 0
  expired_memberships = db.scalar(
    select(func.count(User.id)).where(
      User.role == UserRole.USER,
      User.membership_expiry.is_not(None),
      User.membership_expiry < _club_today(),
    )
  ) or 0
  revenue_summary = db.scalar(select(func.coalesce(func.sum(PaymentHistory.amount), 0))) or 0

  stats = [
    {
      'id': 'users',
      'label': 'Total Members',
      'value': total_users,
      'subLabel': 'Profiles managed by admin',
      'tone': 'blue',
      'iconKey': 'users',
      'delta': 'Roster',
      'progress': min(100, total_users * 5),
      'detail': 'Members available for face enrollment and attendance',
    },
    {
      'id': 'attendance-today',
      'label': 'Active Today',
      'value': attendance_today,
      'subLabel': 'Face-confirmed check-ins today',
      'tone': 'green',
      'iconKey': 'clock',
      'delta': 'Live',
      'progress': min(100, attendance_today * 12),
      'detail': 'Live club traffic across the attendance lane',
    },
    {
      'id': 'expired-memberships',
      'label': 'Expired Memberships',
      'value': expired_memberships,
      'subLabel': 'Members blocked until renewal',
      'tone': 'red',
      'iconKey': 'alerts',
      'delta': 'Renew',
      'progress': min(100, expired_memberships * 20),
      'detail': 'Membership expiry blocks fresh attendance automatically',
    },
    {
      'id': 'revenue-summary',
      'label': 'Revenue Summary',
      'value': float(revenue_summary),
      'subLabel': 'Recorded membership and renewal payments',
      'tone': 'amber',
      'iconKey': 'sessions',
      'delta': 'Collected',
      'progress': min(100, int(float(revenue_summary) // 1000) + 8),
      'detail': 'Payment history is tracked for report exports',
    },
  ]

  return {
    'stats': stats,
    'liveFeed': _live_feed_items(db),
  }


def get_user_dashboard(db: Session, user: User) -> dict[str, Any]:
  _expire_overdue_sessions_cached(db, user.id)
  hydrated_user = _get_user_by_id(db, user.id)
  active_session = _active_session_for_user(db, hydrated_user.id)
  timeline = db.scalars(
    select(UserTimeline)
    .where(UserTimeline.user_id == hydrated_user.id)
    .order_by(UserTimeline.occurred_at.desc())
  ).all()
  payments = db.scalars(
    select(PaymentHistory)
    .where(PaymentHistory.user_id == hydrated_user.id)
    .order_by(PaymentHistory.created_at.desc())
  ).all()
  notifications = db.scalars(
    select(Notification)
    .where(Notification.user_id == hydrated_user.id)
    .order_by(Notification.created_at.desc())
  ).all()
  announcements = db.scalars(
    select(Announcement)
    .options(selectinload(Announcement.target_user))
    .where(or_(Announcement.user_id == None, Announcement.user_id == hydrated_user.id))
    .order_by(Announcement.created_at.desc())
  ).all()

  return {
    'profile': _serialize_user(hydrated_user),
    'activeSession': _serialize_session(active_session) if active_session else None,
    'assignedSlot': _serialize_slot(hydrated_user.slot),
    'currentSlot': _slot_status_payload(hydrated_user.slot),
    'attendanceHistory': [_serialize_timeline(item) for item in timeline],
    'payments': [_serialize_payment(item) for item in payments],
    'notifications': [_serialize_notification(item) for item in notifications],
    'announcements': [_serialize_announcement(item) for item in announcements],
  }


def get_admin_users(
  db: Session,
  scope: str = 'full',
) -> list[dict[str, Any]]:
  _ = _normalize_admin_scope(scope)
  _expire_overdue_sessions_cached(db)
  users = db.scalars(_admin_user_summary_query()).all()
  return [_serialize_user_summary(user) for user in users]

def get_user_report(db: Session, user_id: str) -> dict[str, Any]:
  """
  Full report for admin - all user data
  """
  user = _get_user_by_id(db, user_id)
  _expire_overdue_sessions_cached(db, user_id)
  
  sessions = db.scalars(
    select(SessionRecord).where(SessionRecord.user_id == user_id)
    .options(selectinload(SessionRecord.slot))
    .order_by(SessionRecord.started_at.desc())
  ).all()
  
  timelines = db.scalars(
    select(UserTimeline).where(UserTimeline.user_id == user_id)
    .order_by(UserTimeline.occurred_at.desc())
  ).all()
  
  payments = db.scalars(
    select(PaymentHistory).where(PaymentHistory.user_id == user_id)
    .order_by(PaymentHistory.created_at.desc())
  ).all()
  
  return {
    'profile': _serialize_user(user),
    'sessions': [_serialize_session(s) for s in sessions],
    'timelines': [_serialize_timeline(t) for t in timelines],
    'payments': [_serialize_payment(p) for p in payments],
    'recentActivity': _live_feed_items(db),  # Shared live context
  }


def create_user(db: Session, input_data: CreateUserInput) -> dict[str, Any]:
  if _get_user_by_email(db, input_data.email):
    raise ApiError('User with this email already exists.', 409)

  role = UserRole(input_data.role)
  if input_data.memberId:
    member_id = _normalize_member_id(input_data.memberId, _generate_member_id(db, role))
    if _get_user_by_member_id(db, member_id):
      raise ApiError('User with this member ID already exists.', 409)
  else:
    member_id = _generate_member_id(db, role)
    while _get_user_by_member_id(db, member_id):
      member_id = _generate_member_id(db, role)

  slot = _get_slot_by_id(db, input_data.slotId) if input_data.slotId else None
  user = User(
    name=input_data.name,
    email=input_data.email,
    mobile_number=input_data.mobileNumber,
    password_hash=hash_password(input_data.password),
    role=role,
    member_id=member_id,
    slot=slot,
    sport=input_data.sport or 'General',
    membership_plan=input_data.membershipPlan,
    membership_level=input_data.membershipLevel or '',
    membership_start=input_data.membershipStart,
    membership_expiry=input_data.membershipExpiry,
    visit_limit=input_data.visitLimit,
    payment_amount=input_data.paymentAmount,
    due_amount=input_data.dueAmount,
    payment_mode=input_data.paymentMode,
    payment_status=input_data.paymentStatus,
    face_images_count=0,
    note=input_data.note,
  )
  db.add(user)

  if user.role == UserRole.USER:
    _append_payment_entry(
      db,
      user=user,
      plan=input_data.membershipPlan,
      amount=input_data.paymentAmount,
      payment_mode=input_data.paymentMode,
      payment_status=input_data.paymentStatus,
      membership_start=input_data.membershipStart,
      membership_expiry=input_data.membershipExpiry,
      source='Admin onboarding',
    )

  db.commit()
  db.refresh(user)
  return _serialize_user(_get_user_by_id(db, user.id))


def create_announcement(
  db: Session,
  creator: User,
  input_data: CreateAnnouncementInput,
) -> dict[str, Any]:
  target_user = _get_user_by_id(db, input_data.userId) if input_data.userId else None
  tone = Tone(input_data.tone)
  announcement = _create_announcement(
    db,
    created_by=creator,
    title=input_data.title,
    message=input_data.message,
    tone=tone,
    target_user=target_user,
  )
  db.commit()
  db.refresh(announcement)
  return _serialize_announcement(announcement)


def create_membership(db: Session, input_data: CreateMembershipInput) -> dict[str, Any]:
  user = _get_user_by_id(db, input_data.userId)
  user.membership_plan = input_data.plan
  user.membership_start = input_data.startDate
  user.membership_expiry = input_data.expiryDate
  user.visit_limit = input_data.visitLimit
  user.payment_amount = input_data.paymentAmount
  user.due_amount = 0 if input_data.paymentStatus == 'Paid' else input_data.paymentAmount
  user.payment_mode = input_data.paymentMode
  user.payment_status = input_data.paymentStatus
  user.updated_at = utcnow()

  payment = _append_payment_entry(
    db,
    user=user,
    plan=input_data.plan,
    amount=input_data.paymentAmount,
    payment_mode=input_data.paymentMode,
    payment_status=input_data.paymentStatus,
    membership_start=input_data.startDate,
    membership_expiry=input_data.expiryDate,
    source=input_data.source,
  )
  db.commit()
  db.refresh(payment)
  _invalidate_user_embeddings_cache()
  return {
    'user': _serialize_user(_get_user_by_id(db, user.id)),
    'payment': _serialize_payment(payment),
  }


def delete_user(db: Session, user_id: str, current_user: User) -> dict[str, Any]:
  if current_user.id == user_id:
    raise ApiError('Current admin account cannot delete itself.', 400)

  user = _get_user_by_id(db, user_id)
  db.delete(user)
  db.commit()
  _invalidate_user_embeddings_cache()
  return {'message': f'{user.name} deleted successfully.'}


def delete_user_embeddings(db: Session, user_id: str, current_user: User) -> dict[str, Any]:
  if current_user.role != UserRole.ADMIN:
    raise ApiError('Admin access required.', 403)

  user = _get_user_by_id(db, user_id)
  db.query(FaceEmbedding).filter(FaceEmbedding.user_id == user.id).delete()
  user.face_images_count = 0
  user.updated_at = utcnow()
  db.commit()
  _invalidate_user_embeddings_cache()

  return {
    'message': f'Face enrollment removed for {user.name}.',
    'user': _serialize_user(_get_user_by_id(db, user.id)),
  }


def update_user(db: Session, user_id: str, input_data: UpdateUserInput) -> dict[str, Any]:
  user = _get_user_by_id(db, user_id)
  existing = _get_user_by_email(db, input_data.email)
  member_id = _normalize_member_id(input_data.memberId, _default_member_id(user))
  existing_member = _get_user_by_member_id(db, member_id)
  slot = _get_slot_by_id(db, input_data.slotId) if input_data.slotId else None

  if existing is not None and existing.id != user.id:
    raise ApiError('Another user already uses this email address.', 409)

  if existing_member is not None and existing_member.id != user.id:
    raise ApiError('Another user already uses this member ID.', 409)

  membership_changed = any(
    (
      user.membership_plan != input_data.membershipPlan,
      user.membership_start != input_data.membershipStart,
      user.membership_expiry != input_data.membershipExpiry,
      float(user.payment_amount or 0) != float(input_data.paymentAmount),
      (user.payment_mode or '').strip() != input_data.paymentMode,
      (user.payment_status or '').strip() != input_data.paymentStatus,
    )
  )

  user.name = input_data.name
  user.email = input_data.email
  user.mobile_number = input_data.mobileNumber
  user.role = UserRole(input_data.role)
  user.member_id = member_id
  user.slot = slot
  user.sport = input_data.sport or 'General'
  user.membership_plan = input_data.membershipPlan
  user.membership_level = input_data.membershipLevel or ''
  user.membership_start = input_data.membershipStart
  user.membership_expiry = input_data.membershipExpiry
  user.visit_limit = input_data.visitLimit
  user.payment_amount = input_data.paymentAmount
  user.due_amount = input_data.dueAmount
  user.payment_mode = input_data.paymentMode
  user.payment_status = input_data.paymentStatus
  user.note = input_data.note
  user.updated_at = utcnow()

  if input_data.password:
    user.password_hash = hash_password(input_data.password)

  if user.role == UserRole.USER and membership_changed:
    _append_payment_entry(
      db,
      user=user,
      plan=input_data.membershipPlan,
      amount=input_data.paymentAmount,
      payment_mode=input_data.paymentMode,
      payment_status=input_data.paymentStatus,
      membership_start=input_data.membershipStart,
      membership_expiry=input_data.membershipExpiry,
      source='Admin membership update',
    )

  db.commit()
  _invalidate_user_embeddings_cache()
  return _serialize_user(_get_user_by_id(db, user.id))


def upload_faces(db: Session, actor: User, input_data: UploadFaceInput) -> dict[str, Any]:
  target_user_id = input_data.userId or actor.id
  user = _get_user_by_id(db, target_user_id)

  if actor.role != UserRole.ADMIN and actor.id != user.id:
    raise ApiError('You can only upload face data for your own account.', 403)

  valid_encodings: list[np.ndarray] = []
  db.query(FaceEmbedding).filter(FaceEmbedding.user_id == user.id).delete()

  for index, image_data in enumerate(input_data.images, start=1):
    image_bytes = _decode_image_payload(image_data)
    encoding, _, _ = _extract_face_encoding(image_bytes)

    if encoding is None:
      continue

    valid_encodings.append(encoding)
    saved_frame_path = _save_user_face_image(user.id, image_bytes, index)
    db.add(
      FaceEmbedding(
        user=user,
        image_data=saved_frame_path,
        embedding_vector=_encoding_to_bytes(encoding),
      )
    )

  if len(valid_encodings) < 5:
    db.rollback()
    raise ApiError('Capture at least 5 clear face images before saving.', 400)

  user.face_images_count = len(valid_encodings)
  user.updated_at = utcnow()
  db.commit()
  _invalidate_user_embeddings_cache()
  return {
    'user': _serialize_user(_get_user_by_id(db, user.id)),
    'embeddingCount': len(valid_encodings),
    'message': 'Face enrollment saved successfully.',
  }


def save_user_embeddings(
  db: Session,
  actor: User,
  input_data: DescriptorEnrollmentInput,
) -> dict[str, Any]:
  user = _get_user_by_id(db, input_data.userId)

  if actor.role != UserRole.ADMIN and actor.id != user.id:
    raise ApiError('You can only update face data for your own account.', 403)

  descriptors = [_descriptor_from_iterable(values) for values in input_data.descriptors]

  if len(descriptors) < 3:
    raise ApiError('Capture at least 3 clear face descriptors before saving.', 400)

  db.query(FaceEmbedding).filter(FaceEmbedding.user_id == user.id).delete()

  for index, descriptor in enumerate(descriptors, start=1):
    db.add(
      FaceEmbedding(
        user=user,
        image_data=f'frontend-descriptor:{index}',
        embedding_vector=_encoding_to_bytes(descriptor),
      )
    )

  user.face_images_count = len(descriptors)
  user.updated_at = utcnow()
  db.commit()
  _invalidate_user_embeddings_cache()

  return {
    'user': _serialize_user(_get_user_by_id(db, user.id)),
    'embeddingCount': len(descriptors),
    'descriptor': _descriptor_centroid(descriptors),
    'message': 'Face descriptors saved successfully.',
  }


def get_user_embeddings(db: Session) -> list[dict[str, Any]]:
  global _user_embeddings_cache_payload, _user_embeddings_cache_at

  with _user_embeddings_cache_lock:
    if (
      _user_embeddings_cache_payload is not None
      and (time.monotonic() - _user_embeddings_cache_at) < USER_EMBEDDINGS_CACHE_TTL_SECONDS
    ):
      return _user_embeddings_cache_payload

  users = db.scalars(
    select(User)
    .options(selectinload(User.face_embeddings), selectinload(User.slot))
    .where(User.role == UserRole.USER)
    .order_by(User.updated_at.desc(), User.created_at.desc())
  ).all()

  payload: list[dict[str, Any]] = []

  for user in users:
    encodings = [
      value
      for value in (
        _encoding_from_bytes(embedding.embedding_vector)
        for embedding in user.face_embeddings
      )
      if value is not None and value.shape == (128,)
    ]

    centroid = _descriptor_centroid(encodings)
    if centroid is None:
      continue
    descriptors = [_descriptor_to_payload(encoding) for encoding in encodings]

    payload.append(
      {
        'id': user.id,
        'memberId': _resolve_member_id(user),
        'name': user.name,
        'descriptor': centroid,
        'descriptors': descriptors,
        'sampleCount': len(encodings),
        'faceImageUrl': _first_face_asset_url(user),
        'lastAction': _serialize_attendance_action(user.last_action),
        'lastActionAt': _serialize_datetime(user.last_action_at),
        'membershipStatus': _membership_status(user),
        'membershipExpiry': _serialize_date(user.membership_expiry),
        'slotId': user.slot_id,
        'slotName': user.slot.name if user.slot else None,
        'updatedAt': _serialize_datetime(user.updated_at),
      }
    )

  with _user_embeddings_cache_lock:
    _user_embeddings_cache_payload = payload
    _user_embeddings_cache_at = time.monotonic()

  return payload


def _find_best_user_match(
  db: Session,
  probe_encoding: np.ndarray,
) -> tuple[User | None, float]:
  import face_recognition

  users = db.scalars(
    select(User)
    .options(selectinload(User.face_embeddings), selectinload(User.slot))
    .order_by(User.created_at.asc())
  ).all()

  best_user: User | None = None
  best_distance = 1.0

  for user in users:
    if not user.face_embeddings:
      continue

    known_encodings = [
      value
      for value in (
        _encoding_from_bytes(embedding.embedding_vector)
        for embedding in user.face_embeddings
      )
      if value is not None and value.shape == probe_encoding.shape
    ]
    if not known_encodings:
      continue
    distances = face_recognition.face_distance(known_encodings, probe_encoding)
    if len(distances) == 0:
      continue

    user_distance = float(np.min(distances))
    if user_distance < best_distance:
      best_distance = user_distance
      best_user = user

  return best_user, best_distance


def mark_attendance(db: Session, input_data: AttendanceInput) -> dict[str, Any]:
  user = _get_user_with_slot_by_id(db, input_data.userId)
  _expire_overdue_sessions(db, user.id)

  attendance_time = utcnow()
  action = _serialize_attendance_action(input_data.action) or 'IN'
  confidence = round(float(input_data.confidence or 0), 2)
  area = input_data.area
  active_session = _active_session_for_user(db, user.id)
  today_session = _latest_session_for_user_on_date(db, user.id, _club_today())

  if _membership_status(user) == 'expired':
    return _build_scan_response(
      status='denied',
      message='Your membership has expired.',
      confidence=confidence,
      name=user.name,
      duplicate_warning=False,
      tts_message='Your membership has expired.',
      attendance_action=None,
      session=None,
      cooldown_remaining_seconds=0,
      face_box=None,
      area=area,
      frames_captured=1,
    )

  if active_session is not None:
    if action == 'IN':
      return _build_scan_response(
        status='duplicate',
        message='Entry already marked. Exit is pending.',
        confidence=confidence,
        name=user.name,
        duplicate_warning=True,
        tts_message='Attendance already recorded.',
        attendance_action='in',
        session=_serialize_session(active_session),
        cooldown_remaining_seconds=0,
        face_box=None,
        area=area,
        frames_captured=1,
      )

    exit_lock_remaining = _exit_lock_remaining_seconds(active_session, attendance_time)
    if exit_lock_remaining > 0:
      return _build_scan_response(
        status='cooldown',
        message=f'Please wait {_format_wait_time(exit_lock_remaining)} before exit.',
        confidence=confidence,
        name=user.name,
        duplicate_warning=True,
        tts_message='Please wait before exit.',
        attendance_action='out',
        session=_serialize_session(active_session),
        cooldown_remaining_seconds=exit_lock_remaining,
        face_box=None,
        area=area,
        frames_captured=1,
      )

    active_session.ended_at = attendance_time
    active_session.status = SessionStatus.ENDED
    duration_minutes = _session_duration_minutes(active_session)
    user.last_action = 'OUT'
    user.last_action_at = attendance_time
    user.updated_at = attendance_time
    _create_timeline_event(
      db,
      user=user,
      event_type=TimelineEventType.EXIT,
      area=area,
      total_minutes=duration_minutes,
      note='Exit marked by browser recognition',
    )
    db.commit()

    return _build_scan_response(
      status='granted',
      message='Exit marked successfully.',
      confidence=confidence,
      name=user.name,
      duplicate_warning=False,
      tts_message=f'Exit marked successfully for {user.name}.',
      attendance_action='out',
      session=_serialize_session(_get_session_by_id(db, active_session.id)),
      cooldown_remaining_seconds=0,
      face_box=None,
      area=area,
      frames_captured=1,
    )

  if today_session is not None and today_session.ended_at is not None:
    return _build_scan_response(
      status='duplicate',
      message='Exit already marked.' if action == 'OUT' else 'Attendance already marked.',
      confidence=confidence,
      name=user.name,
      duplicate_warning=True,
      tts_message='Exit already recorded.' if action == 'OUT' else 'Attendance already recorded.',
      attendance_action=action.lower(),
      session=_serialize_session(today_session),
      cooldown_remaining_seconds=0,
      face_box=None,
      area=area,
      frames_captured=1,
    )

  if action == 'OUT':
    return _build_scan_response(
      status='duplicate',
      message='No active session is open for this member.',
      confidence=confidence,
      name=user.name,
      duplicate_warning=True,
      tts_message='No active session is open for this member.',
      attendance_action='out',
      session=None,
      cooldown_remaining_seconds=0,
      face_box=None,
      area=area,
      frames_captured=1,
    )

  cooldown_remaining = _action_cooldown_remaining_seconds(user, now=attendance_time)
  if cooldown_remaining > 0:
    return _build_scan_response(
      status='cooldown',
      message=f'Please wait {_format_wait_time(cooldown_remaining)} before next action.',
      confidence=confidence,
      name=user.name,
      duplicate_warning=True,
      tts_message='Please wait before marking again.',
      attendance_action=None,
      session=None,
      cooldown_remaining_seconds=cooldown_remaining,
      face_box=None,
      area=area,
      frames_captured=1,
    )

  slot_start_at, slot_end_at = _resolve_session_slot_window(user.slot, reference_utc=attendance_time)
  session = SessionRecord(
    user=user,
    slot=user.slot,
    area=area,
    status=SessionStatus.ACTIVE,
    confidence=confidence,
    started_at=attendance_time,
    slot_start_at=slot_start_at,
    slot_end_at=slot_end_at,
  )
  db.add(session)
  db.flush()
  user.last_action = 'IN'
  user.last_action_at = attendance_time
  user.updated_at = attendance_time
  _create_timeline_event(
    db,
    user=user,
    event_type=TimelineEventType.ENTRY,
    area=area,
    note='Attendance marked by browser recognition',
  )
  db.commit()

  return _build_scan_response(
    status='granted',
    message='Attendance marked successfully.',
    confidence=confidence,
    name=user.name,
    duplicate_warning=False,
    tts_message=f'Entry marked successfully for {user.name}.',
    attendance_action='in',
    session=_serialize_session(_get_session_by_id(db, session.id)),
    cooldown_remaining_seconds=0,
    face_box=None,
    area=area,
    frames_captured=1,
  )


def perform_access_scan(db: Session, input_data: AccessScanInput) -> dict[str, Any]:
  image_bytes = _decode_image_payload(input_data.image)
  encoding, face_box, error_message = _extract_face_encoding(image_bytes)

  if encoding is None:
    return _build_scan_response(
      status='retry',
      message=error_message or 'Face alignment failed.',
      confidence=0,
      name=None,
      duplicate_warning=False,
      tts_message='Please look at the camera and try again.',
      attendance_action=None,
      session=None,
      cooldown_remaining_seconds=0,
      face_box=face_box,
      area=input_data.area,
      frames_captured=input_data.capturedFrames,
    )

  user, best_distance = _find_best_user_match(db, encoding)
  confidence = _distance_to_confidence(best_distance)

  if user is None or best_distance > FACE_RETRY_THRESHOLD:
  # _save_unknown_image(image_bytes)  # Disabled to prevent unknown storage
    return _build_scan_response(
      status='unknown',
      message='Unknown face. No enrolled member matched this scan.',
      confidence=confidence,
      name=None,
      duplicate_warning=False,
      tts_message='Unknown face detected.',
      attendance_action=None,
      session=None,
      cooldown_remaining_seconds=0,
      face_box=face_box,
      area=input_data.area,
      frames_captured=input_data.capturedFrames,
    )

  if best_distance > FACE_MATCH_THRESHOLD:
    return _build_scan_response(
      status='retry',
      message='Face detected, but confidence is low. Try again.',
      confidence=confidence,
      name=user.name,
      duplicate_warning=False,
      tts_message='Confidence is low. Please try again.',
      attendance_action=None,
      session=None,
      cooldown_remaining_seconds=0,
      face_box=face_box,
      area=input_data.area,
      frames_captured=input_data.capturedFrames,
    )

  if _membership_status(user) == 'expired':
    return _build_scan_response(
      status='denied',
      message='Your membership has expired.',
      confidence=confidence,
      name=user.name,
      duplicate_warning=False,
      tts_message='Your membership has expired.',
      attendance_action=None,
      session=None,
      cooldown_remaining_seconds=0,
      face_box=face_box,
      area=input_data.area,
      frames_captured=input_data.capturedFrames,
    )

  _expire_overdue_sessions(db, user.id)
  attendance_time = utcnow()
  active_session = _active_session_for_user(db, user.id)

  if active_session is not None:
    duplicate_remaining = _entry_duplicate_remaining_seconds(active_session)

    if duplicate_remaining > 0:
      return _build_scan_response(
        status='duplicate',
        message='Attendance already marked',
        confidence=confidence,
        name=user.name,
        duplicate_warning=True,
        tts_message='Attendance already marked.',
        attendance_action=None,
        session=_serialize_session(active_session),
        cooldown_remaining_seconds=duplicate_remaining,
        face_box=face_box,
        area=input_data.area,
        frames_captured=input_data.capturedFrames,
      )

    active_session.ended_at = attendance_time
    active_session.status = SessionStatus.ENDED
    duration_minutes = _session_duration_minutes(active_session)
    over_limit = duration_minutes >= SESSION_LIMIT_MINUTES
    user.last_action = 'OUT'
    user.last_action_at = attendance_time
    user.updated_at = attendance_time
    _save_session_image(active_session.id, image_bytes, 'exit')
    _create_timeline_event(
      db,
      user=user,
      event_type=TimelineEventType.EXIT,
      area=input_data.area,
      total_minutes=duration_minutes,
      note='Auto-marked exit after session limit' if over_limit else 'Exit marked by face scan',
    )
    db.commit()

    refreshed_session = _get_session_by_id(db, active_session.id)
    return _build_scan_response(
      status='granted',
      message='Exit marked successfully',
      confidence=confidence,
      name=user.name,
      duplicate_warning=False,
      tts_message=(
        'Exit marked successfully.'
        if over_limit
        else f'Exit marked successfully. See you soon, {user.name}.'
      ),
      attendance_action='out',
      session=_serialize_session(refreshed_session),
      cooldown_remaining_seconds=0,
      face_box=face_box,
      area=input_data.area,
      frames_captured=input_data.capturedFrames,
    )

  cooldown_remaining = _cooldown_remaining_seconds(db, user.id)

  if cooldown_remaining > 0:
    return _build_scan_response(
      status='cooldown',
      message='Please wait 10 minutes before marking exit again',
      confidence=confidence,
      name=user.name,
      duplicate_warning=True,
      tts_message='Please wait 10 minutes before marking exit again.',
      attendance_action=None,
      session=None,
      cooldown_remaining_seconds=cooldown_remaining,
      face_box=face_box,
      area=input_data.area,
      frames_captured=input_data.capturedFrames,
    )

  slot_start_at, slot_end_at = _resolve_session_slot_window(user.slot, reference_utc=attendance_time)
  session = SessionRecord(
    user=user,
    slot=user.slot,
    area=input_data.area,
    status=SessionStatus.ACTIVE,
    confidence=confidence,
    started_at=attendance_time,
    slot_start_at=slot_start_at,
    slot_end_at=slot_end_at,
  )
  db.add(session)
  db.flush()
  user.last_action = 'IN'
  user.last_action_at = attendance_time
  user.updated_at = attendance_time
  _save_session_image(session.id, image_bytes, 'entry')
  _create_timeline_event(
    db,
    user=user,
    event_type=TimelineEventType.ENTRY,
    area=input_data.area,
    note='Attendance marked by face scan',
  )
  db.commit()
  refreshed_session = _get_session_by_id(db, session.id)
  return _build_scan_response(
    status='granted',
    message='Attendance marked successfully',
    confidence=confidence,
    name=user.name,
    duplicate_warning=False,
    tts_message=f'Welcome, {user.name}. Attendance marked successfully.',
    attendance_action='in',
    session=_serialize_session(refreshed_session),
    cooldown_remaining_seconds=0,
    face_box=face_box,
    area=input_data.area,
    frames_captured=input_data.capturedFrames,
  )


def start_session(db: Session, input_data: SessionStartInput) -> dict[str, Any]:
  user = _get_user_with_slot_by_id(db, input_data.userId)

  if _membership_status(user) == 'expired':
    raise ApiError('Membership expired. Cannot start a session.', 400)

  _expire_overdue_sessions(db, user.id)
  if _active_session_for_user(db, user.id):
    raise ApiError('This user already has an active session.', 400)

  started_at = utcnow()
  slot_start_at, slot_end_at = _resolve_session_slot_window(user.slot, reference_utc=started_at)

  session = SessionRecord(
    user=user,
    slot=user.slot,
    area=input_data.area,
    status=SessionStatus.ACTIVE,
    confidence=input_data.confidence,
    started_at=started_at,
    slot_start_at=slot_start_at,
    slot_end_at=slot_end_at,
  )
  db.add(session)
  db.flush()
  user.last_action = 'IN'
  user.last_action_at = started_at
  user.updated_at = started_at
  _create_timeline_event(
    db,
    user=user,
    event_type=TimelineEventType.ENTRY,
    area=input_data.area,
    note='Manual session start',
  )
  db.commit()
  return _serialize_session(_get_session_by_id(db, session.id))


def end_session(db: Session, input_data: SessionEndInput) -> dict[str, Any]:
  _expire_overdue_sessions(db)
  session = _get_session_by_id(db, input_data.sessionId)

  if session.status != SessionStatus.ACTIVE:
    raise ApiError('Session is not active.', 400)

  ended_at = utcnow()
  exit_lock_remaining = _exit_lock_remaining_seconds(session, ended_at)
  if exit_lock_remaining > 0:
    raise ApiError(f'Please wait {_format_wait_time(exit_lock_remaining)} before exit.', 400)

  session.status = SessionStatus.ENDED
  session.ended_at = ended_at
  session.user.last_action = 'OUT'
  session.user.last_action_at = ended_at
  session.user.updated_at = ended_at
  duration_minutes = _session_duration_minutes(session)
  _create_timeline_event(
    db,
    user=session.user,
    event_type=TimelineEventType.EXIT,
    area=session.area,
    total_minutes=duration_minutes,
    note='Manual session end',
  )
  db.commit()
  return _serialize_session(_get_session_by_id(db, session.id))


def get_admin_announcements(db: Session) -> list[dict[str, Any]]:
  announcements = db.scalars(
    select(Announcement)
    .options(selectinload(Announcement.target_user))
    .order_by(Announcement.created_at.desc())
  ).all()
  return [_serialize_announcement(item) for item in announcements]


def get_admin_reports(
  db: Session,
  scope: str = 'full',
) -> dict[str, Any]:
  normalized_scope = _normalize_admin_scope(scope)
  _expire_overdue_sessions_cached(db)
  payments_limit = LIVE_DASHBOARD_PAYMENT_LIMIT if normalized_scope == 'live' else None
  return _reports_payload(db, payments_limit=payments_limit)


def get_admin_sessions(
  db: Session,
  scope: str = 'full',
) -> list[dict[str, Any]]:
  normalized_scope = _normalize_admin_scope(scope)
  _expire_overdue_sessions_cached(db)
  session_query = _admin_session_query()

  if normalized_scope == 'live':
    # Return all sessions for live dashboard to ensure all active are shown
    sessions = db.scalars(
      session_query.order_by(SessionRecord.started_at.desc())
    ).all()
  else:
    sessions = db.scalars(
      session_query.order_by(SessionRecord.started_at.desc())
    ).all()
  return [_serialize_session(item) for item in sessions]


def get_user_history(db: Session, user: User) -> list[dict[str, Any]]:
  items = db.scalars(
    select(UserTimeline)
    .where(UserTimeline.user_id == user.id)
    .order_by(UserTimeline.occurred_at.desc())
  ).all()
  return [_serialize_timeline(item) for item in items]


def get_user_notifications(db: Session, user: User) -> list[dict[str, Any]]:
  notifications = db.scalars(
    select(Notification)
    .where(Notification.user_id == user.id)
    .order_by(Notification.created_at.desc())
  ).all()
  return [_serialize_notification(item) for item in notifications]


def get_user_payments(db: Session, user: User) -> list[dict[str, Any]]:
  payments = db.scalars(
    select(PaymentHistory)
    .where(PaymentHistory.user_id == user.id)
    .order_by(PaymentHistory.created_at.desc())
  ).all()
  return [_serialize_payment(item) for item in payments]



def get_session_timer(db: Session, session_id: str) -> dict[str, Any]:
  _expire_overdue_sessions_cached(db)
  session = _get_session_by_id(db, session_id)
  duration_min = _session_duration_minutes(session)
  remaining_sec = _session_remaining_seconds(session)
  remaining_min = remaining_sec // 60
  warning_5min = remaining_sec > 0 and remaining_sec <= (SESSION_WARNING_MINUTES * 60)
  overtime_count = max(0, (duration_min - SESSION_LIMIT_MINUTES) // 10)
  overtime_count = min(overtime_count, 5)  # Cap at 5
  
  # Success announcement every hour
  hourly_annc = (duration_min % 60 == 0 and duration_min > 0)
  
  return {
    'sessionId': session.id,
    'userId': session.user_id,
    'name': session.user.name,
    'status': session.status.value,
    'limitMinutes': SESSION_LIMIT_MINUTES,
    'durationMinutes': duration_min,
    'remainingSeconds': remaining_sec,
    'remainingMinutes': remaining_min,
    'warning5Min': warning_5min,
    'overtimeCount': overtime_count,
    'hourlyAnnouncement': hourly_annc,
    'ttsWarning': f"Warning {session.user.name}: {remaining_min} min left!" if warning_5min else None,
    'ttsOvertime': f"Time over {session.user.name}! ({overtime_count}/5)" if overtime_count > 0 else None,
    'ttsHourly': (
      f"Hourly check: {session.user.name}, you have been here for {duration_min} minutes."
      if hourly_annc else None
    ),
  }


def get_face_enrollment_status(db: Session) -> dict[str, Any]:
  from sqlalchemy import select
  from sqlalchemy.orm import selectinload
  
  # Count totals
  total_users = db.scalar(select(func.count(User.id)).where(User.role == UserRole.USER)) or 0
  enrolled_count = db.scalar(
    select(func.count(User.id))
    .where(User.role == UserRole.USER, User.face_images_count > 0)
  ) or 0
  pending_count = total_users - enrolled_count
  
  # Fetch enrolled users (top 20 recently updated)
  enrolled = db.scalars(
    select(User)
    .options(selectinload(User.slot))
    .where(User.role == UserRole.USER, User.face_images_count > 0)
    .order_by(User.updated_at.desc())
    .limit(20)
  ).all()
  
  # Fetch pending users (all, ordered by creation)
  pending = db.scalars(
    select(User)
    .options(selectinload(User.slot))
    .where(User.role == UserRole.USER, User.face_images_count == 0)
    .order_by(User.created_at.asc())
  ).all()
  
  return {
    'total_users': total_users,
    'enrolled_count': enrolled_count,
    'pending_count': pending_count,
    'enrolled_percentage': round((enrolled_count / max(total_users, 1)) * 100, 1),
    'enrolled': [_serialize_user_summary(user) for user in enrolled],
    'pending': [_serialize_user_summary(user) for user in pending],
  }


def get_user_profile(db: Session, user: User) -> dict[str, Any]:
  return _serialize_user(_get_user_by_id(db, user.id))


def _generate_windows_tts_bytes(text: str) -> bytes:
  if os.name != 'nt':
    raise ApiError('Local Windows TTS fallback is unavailable on this system.', 503)

  with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
    temp_path = Path(temp_file.name)
  script = """
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Volume = 100
$synth.Rate = 0
$synth.SetOutputToWaveFile($env:CAPPERCLUB_TTS_PATH)
$synth.Speak($env:CAPPERCLUB_TTS_TEXT)
$synth.Dispose()
"""

  try:
    environment = os.environ.copy()
    environment['CAPPERCLUB_TTS_PATH'] = str(temp_path)
    environment['CAPPERCLUB_TTS_TEXT'] = text

    completed = subprocess.run(
      ['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
      capture_output=True,
      text=True,
      timeout=30,
      env=environment,
      check=False,
    )

    if completed.returncode != 0 or not temp_path.exists():
      error_output = (completed.stderr or completed.stdout or '').strip() or 'unknown error'
      raise ApiError(f'Local TTS generation failed: {error_output}', 500)

    return temp_path.read_bytes()
  except subprocess.TimeoutExpired as error:
    raise ApiError('Local TTS generation timed out.', 500) from error
  finally:
    temp_path.unlink(missing_ok=True)



def _audio_mime_type(output_format: str) -> str:
  normalized = str(output_format or '').strip().lower()

  if normalized.startswith('wav_') or normalized.startswith('pcm_'):
    return 'audio/wav'

  if normalized.startswith('ulaw_'):
    return 'audio/basic'

  return 'audio/mpeg'


def _generate_elevenlabs_tts_bytes(text: str) -> tuple[bytes, str]:
  settings = get_settings()
  api_key = settings.elevenlabs_api_key.strip()

  if not api_key:
    raise ApiError(
      'ElevenLabs API key is not configured. Set CAPERCLUB_ELEVENLABS_API_KEY.',
      503,
    )

  voice_id = settings.elevenlabs_voice_id.strip() or DEFAULT_VOICE_ID
  model_id = settings.elevenlabs_model_id.strip() or ELEVENLABS_MODEL_ID
  output_format = settings.elevenlabs_output_format.strip() or 'mp3_44100_128'
  query = urlencode({'output_format': output_format})
  url = f'https://api.elevenlabs.io/v1/text-to-speech/{quote(voice_id, safe="")}?{query}'
  payload = json.dumps({
    'text': text,
    'model_id': model_id,
  }).encode('utf-8')
  request = Request(
    url,
    data=payload,
    headers={
      'Accept': _audio_mime_type(output_format),
      'Content-Type': 'application/json',
      'xi-api-key': api_key,
    },
    method='POST',
  )

  try:
    with urlopen(request, timeout=30) as response:
      audio_bytes = response.read()

    if not audio_bytes:
      raise ApiError('ElevenLabs returned an empty audio response.', 502)

    return audio_bytes, _audio_mime_type(output_format)
  except urllib_error.HTTPError as error:
    detail = error.read().decode('utf-8', errors='ignore').strip()
    message = detail or f'ElevenLabs request failed with HTTP {error.code}.'
    raise ApiError(message, 502) from error
  except urllib_error.URLError as error:
    raise ApiError('Unable to reach ElevenLabs from the backend server.', 502) from error


def generate_tts(text: str) -> tuple[bytes, str]:
  normalized = ' '.join(str(text or '').split())

  if not normalized:
    raise ApiError('Text is required.', 400)

  return _generate_elevenlabs_tts_bytes(normalized)
