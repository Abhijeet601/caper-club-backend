from __future__ import annotations

import os
from datetime import datetime

from fastapi import HTTPException, status
from sqlalchemy import text

if __package__:
  from .db import engine
else:
  from db import engine

DOOR_COMMAND_LOCK = 'LOCK'
DOOR_COMMAND_UNLOCK = 'UNLOCK'
DOOR_COMMANDS = {DOOR_COMMAND_LOCK, DOOR_COMMAND_UNLOCK}
DOOR_API_KEY_ENV = 'CAPERCLUB_DOOR_LOCK_API_KEY'


def get_door_api_key() -> str:
  return os.getenv(DOOR_API_KEY_ENV, '').strip()


def verify_door_api_key(x_door_key: str | None) -> None:
  expected = get_door_api_key()
  if not expected or x_door_key != expected:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail='Invalid or missing X-Door-Key header.',
    )


def ensure_door_state_table() -> None:
  with engine.begin() as conn:
    conn.execute(
      text(
        '''
        CREATE TABLE IF NOT EXISTS door_status (
          id INT NOT NULL DEFAULT 1,
          command VARCHAR(10) NOT NULL DEFAULT 'LOCK',
          updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            ON UPDATE CURRENT_TIMESTAMP,
          PRIMARY KEY (id),
          CONSTRAINT chk_door_status_singleton CHECK (id = 1)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        '''
      )
    )
    conn.execute(
      text(
        '''
        INSERT IGNORE INTO door_status (id, command, updated_at)
        VALUES (1, 'LOCK', NOW())
        '''
      )
    )


def get_door_state() -> dict[str, str | None]:
  ensure_door_state_table()
  with engine.connect() as conn:
    row = conn.execute(
      text('SELECT command, updated_at FROM door_status WHERE id = 1')
    ).first()

  if row is None:
    return {'command': DOOR_COMMAND_LOCK, 'updatedAt': None}

  return {
    'command': str(row.command or DOOR_COMMAND_LOCK).upper(),
    'updatedAt': row.updated_at.isoformat() + 'Z' if row.updated_at else None,
  }


def set_door_state(command: str) -> dict[str, str | bool | None]:
  normalized_command = str(command or '').strip().upper()
  if normalized_command not in DOOR_COMMANDS:
    raise ValueError(f'Unsupported door command: {command}')

  ensure_door_state_table()
  now = datetime.utcnow()
  with engine.begin() as conn:
    conn.execute(
      text(
        '''
        UPDATE door_status
        SET command = :command, updated_at = :updated_at
        WHERE id = 1
        '''
      ),
      {'command': normalized_command, 'updated_at': now},
    )

  return {
    'command': normalized_command,
    'updatedAt': now.isoformat() + 'Z',
    'ok': True,
  }
