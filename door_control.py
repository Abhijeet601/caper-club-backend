from __future__ import annotations

import os
import time
from threading import Lock, Timer
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

DOOR_AUTO_RELOCK_SECONDS = float(os.getenv('DOOR_AUTO_RELOCK_SECONDS', '3.0'))

_state_lock = Lock()
_door_open = False
_last_unlock_at = 0.0
_last_action = DOOR_COMMAND_LOCK.lower()
_relock_timer: Timer | None = None
_auto_relock_at = 0.0


def _sync_cached_state(command: str) -> None:
  global _door_open, _last_action
  normalized = str(command or DOOR_COMMAND_LOCK).upper()
  _door_open = normalized == DOOR_COMMAND_UNLOCK
  _last_action = normalized.lower()


def _cancel_relock_timer() -> None:
  global _relock_timer, _auto_relock_at
  if _relock_timer is not None:
    _relock_timer.cancel()
    _relock_timer = None
  _auto_relock_at = 0.0


def _schedule_auto_relock() -> None:
  global _relock_timer, _auto_relock_at

  _cancel_relock_timer()
  _auto_relock_at = time.monotonic() + DOOR_AUTO_RELOCK_SECONDS
  _relock_timer = Timer(DOOR_AUTO_RELOCK_SECONDS, _auto_relock_callback)
  _relock_timer.daemon = True
  _relock_timer.start()


def _auto_relock_callback() -> None:
  lock_door(force=True, reason='auto_relock')


def _relock_metadata() -> dict[str, Any]:
  if not _door_open:
    return {
      'autoRelockSeconds': int(DOOR_AUTO_RELOCK_SECONDS),
      'relocking': False,
      'remainingSeconds': 0,
    }

  remaining = max(0.0, _auto_relock_at - time.monotonic()) if _auto_relock_at else 0.0
  timer_active = remaining > 0
  return {
    'autoRelockSeconds': int(DOOR_AUTO_RELOCK_SECONDS),
    'relocking': timer_active,
    'remainingSeconds': round(remaining, 2) if timer_active else 0,
  }


def unlock_door() -> dict[str, Any]:
  global _last_unlock_at

  with _state_lock:
    current_state = get_door_state()
    current_command = str(current_state.get('command') or DOOR_COMMAND_LOCK).upper()
    _sync_cached_state(current_command)
    state = (
      current_state
      if current_command == DOOR_COMMAND_UNLOCK
      else set_door_state(DOOR_COMMAND_UNLOCK)
    )
    _last_unlock_at = time.monotonic()
    _sync_cached_state(DOOR_COMMAND_UNLOCK)
    _schedule_auto_relock()
    result = {
      'doorOpen': True,
      'command': state['command'],
      'updatedAt': state['updatedAt'],
      'action': 'unlocked' if current_command != DOOR_COMMAND_UNLOCK else 'timer_reset',
      'reason': 'known_face',
    }
    result.update(_relock_metadata())
    return result


def lock_door(*, force: bool = False, reason: str = 'unknown_or_no_face') -> dict[str, Any]:
  with _state_lock:
    current_state = get_door_state()
    current_command = str(current_state.get('command') or DOOR_COMMAND_LOCK).upper()
    _sync_cached_state(current_command)

    if current_command == DOOR_COMMAND_LOCK and not force:
      _cancel_relock_timer()
      result = {
        'doorOpen': False,
        'command': current_command,
        'action': 'unchanged',
        'reason': 'already_locked',
        'updatedAt': current_state.get('updatedAt'),
      }
      result.update(_relock_metadata())
      return result

    state = set_door_state(DOOR_COMMAND_LOCK)
    _cancel_relock_timer()
    _sync_cached_state(DOOR_COMMAND_LOCK)
    return {
      'doorOpen': False,
      'command': state['command'],
      'updatedAt': state['updatedAt'],
      'action': 'locked',
      'reason': reason,
      'autoRelockSeconds': int(DOOR_AUTO_RELOCK_SECONDS),
      'relocking': False,
      'remainingSeconds': 0,
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


def get_door_control_state() -> dict[str, Any]:
  with _state_lock:
    state = get_door_state()
    _sync_cached_state(str(state.get('command') or DOOR_COMMAND_LOCK).upper())
    if not _door_open:
      _cancel_relock_timer()
    result = {
      'doorOpen': _door_open,
      'lastAction': _last_action,
      **state,
    }
    result.update(_relock_metadata())
    return result
