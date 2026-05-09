from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

if __package__:
  from .db import get_settings
else:
  from db import get_settings

HASH_ITERATIONS = 120_000


def hash_password(password: str) -> str:
  salt = secrets.token_hex(16)
  digest = hashlib.pbkdf2_hmac(
    'sha256',
    password.encode('utf-8'),
    bytes.fromhex(salt),
    HASH_ITERATIONS,
  )
  return f'pbkdf2_sha256${HASH_ITERATIONS}${salt}${digest.hex()}'


def verify_password(password: str, stored_hash: str) -> bool:
  try:
    algorithm, iteration_text, salt, digest = stored_hash.split('$', 3)
  except ValueError:
    return False

  if algorithm != 'pbkdf2_sha256':
    return False

  candidate = hashlib.pbkdf2_hmac(
    'sha256',
    password.encode('utf-8'),
    bytes.fromhex(salt),
    int(iteration_text),
  ).hex()
  return hmac.compare_digest(candidate, digest)


def create_access_token(*, subject: str, role: str) -> str:
  settings = get_settings()
  expires_at = datetime.now(timezone.utc) + timedelta(
    minutes=settings.access_token_expiry_minutes
  )
  payload: dict[str, Any] = {
    'sub': subject,
    'role': role,
    'exp': expires_at,
  }
  return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
  settings = get_settings()
  return jwt.decode(
    token,
    settings.jwt_secret,
    algorithms=[settings.jwt_algorithm],
  )
