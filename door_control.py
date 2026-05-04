from __future__ import annotations

import os
import time
from threading import Lock
from typing import Any

import requests


DOOR_API_BASE = os.getenv('DOOR_API_BASE', 'http://127.0.0.1:5000').rstrip('/')
DOOR_REQUEST_TIMEOUT_SECONDS = float(os.getenv('DOOR_REQUEST_TIMEOUT_SECONDS', '1.5'))
DOOR_LOCK_DELAY_SECONDS = float(os.getenv('DOOR_LOCK_DELAY_SECONDS', '2.0'))

_state_lock = Lock()
door_open = False
_last_unlock_at = 0.0
_last_action = 'locked'
_lock_confirmed = False


def _send_door_request(action: str) -> bool:
  endpoint = 'unlock' if action == 'unlock' else 'lock'
  try:
    response = requests.get(
      f'{DOOR_API_BASE}/{endpoint}',
      timeout=DOOR_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
  except requests.RequestException as error:
    print(f'Door {action} request failed: {error}')
    return False

  return True


def unlock_door() -> dict[str, Any]:
  global door_open, _last_unlock_at, _last_action

  with _state_lock:
    if door_open:
      return {'doorOpen': True, 'action': 'unchanged', 'reason': 'already_unlocked'}

    if not _send_door_request('unlock'):
      return {'doorOpen': door_open, 'action': 'failed', 'reason': 'unlock_request_failed'}

    door_open = True
    _last_unlock_at = time.monotonic()
    _last_action = 'unlocked'
    return {'doorOpen': True, 'action': 'unlocked', 'reason': 'known_face'}


def lock_door(*, force: bool = False) -> dict[str, Any]:
  global door_open, _last_action, _lock_confirmed

  with _state_lock:
    if not door_open and _lock_confirmed and not force:
      return {'doorOpen': False, 'action': 'unchanged', 'reason': 'already_locked'}

    if door_open and not force:
      elapsed = time.monotonic() - _last_unlock_at
      if elapsed < DOOR_LOCK_DELAY_SECONDS:
        return {
          'doorOpen': True,
          'action': 'delayed',
          'reason': 'lock_delay_active',
          'remainingSeconds': round(DOOR_LOCK_DELAY_SECONDS - elapsed, 2),
        }

    if not _send_door_request('lock'):
      return {'doorOpen': door_open, 'action': 'failed', 'reason': 'lock_request_failed'}

    door_open = False
    _lock_confirmed = True
    _last_action = 'locked'
    return {'doorOpen': False, 'action': 'locked', 'reason': 'unknown_or_no_face'}


def sync_door_for_detection(
  *,
  known_face: bool,
  name: str | None = None,
  force_lock: bool = False,
) -> dict[str, Any]:
  result = unlock_door() if known_face else lock_door(force=force_lock)
  result['name'] = name
  result['lastAction'] = _last_action
  return result
