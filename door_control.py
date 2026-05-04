from __future__ import annotations

import os
import time
from threading import Lock
from typing import Any

if __package__:
  from .door_lock_service import (
    DOOR_COMMAND_LOCK,
    DOOR_COMMAND_UNLOCK,
    get_door_state,
    set_door_state,
  )
else:
  from door_lock_service import (
    DOOR_COMMAND_LOCK,
    DOOR_COMMAND_UNLOCK,
    get_door_state,
    set_door_state,
  )

DOOR_LOCK_DELAY_SECONDS = float(os.getenv('DOOR_LOCK_DELAY_SECONDS', '5.0'))

_state_lock = Lock()
_door_open = False
_last_unlock_at = 0.0
_last_action = DOOR_COMMAND_LOCK.lower()


def _sync_cached_state(command: str) -> None:
  global _door_open, _last_action
  normalized = str(command or DOOR_COMMAND_LOCK).upper()
  _door_open = normalized == DOOR_COMMAND_UNLOCK
  _last_action = normalized.lower()


def unlock_door() -> dict[str, Any]:
  global _last_unlock_at

  with _state_lock:
    current_state = get_door_state()
    current_command = str(current_state.get('command') or DOOR_COMMAND_LOCK).upper()
    _sync_cached_state(current_command)

    if current_command == DOOR_COMMAND_UNLOCK:
      return {
        'doorOpen': True,
        'command': current_command,
        'action': 'unchanged',
        'reason': 'already_unlocked',
        'updatedAt': current_state.get('updatedAt'),
      }

    state = set_door_state(DOOR_COMMAND_UNLOCK)
    _last_unlock_at = time.monotonic()
    _sync_cached_state(DOOR_COMMAND_UNLOCK)
    return {
      'doorOpen': True,
      'command': state['command'],
      'updatedAt': state['updatedAt'],
      'action': 'unlocked',
      'reason': 'known_face',
    }


def lock_door(*, force: bool = False) -> dict[str, Any]:
  with _state_lock:
    current_state = get_door_state()
    current_command = str(current_state.get('command') or DOOR_COMMAND_LOCK).upper()
    _sync_cached_state(current_command)

    if current_command == DOOR_COMMAND_LOCK and not force:
      return {
        'doorOpen': False,
        'command': current_command,
        'action': 'unchanged',
        'reason': 'already_locked',
        'updatedAt': current_state.get('updatedAt'),
      }

    if current_command == DOOR_COMMAND_UNLOCK and not force:
      elapsed = time.monotonic() - _last_unlock_at
      if elapsed < DOOR_LOCK_DELAY_SECONDS:
        return {
          'doorOpen': True,
          'command': current_command,
          'action': 'delayed',
          'reason': 'lock_delay_active',
          'updatedAt': current_state.get('updatedAt'),
          'remainingSeconds': round(DOOR_LOCK_DELAY_SECONDS - elapsed, 2),
        }

    state = set_door_state(DOOR_COMMAND_LOCK)
    _sync_cached_state(DOOR_COMMAND_LOCK)
    return {
      'doorOpen': False,
      'command': state['command'],
      'updatedAt': state['updatedAt'],
      'action': 'locked',
      'reason': 'unknown_or_no_face',
    }


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
