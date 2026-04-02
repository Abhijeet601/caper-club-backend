from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

if TYPE_CHECKING:
  from .db import Settings


logger = logging.getLogger(__name__)


class MediaStorage:
  def __init__(self, settings: 'Settings', storage_root: Path) -> None:
    self.settings = settings
    self.storage_root = storage_root
    self.storage_root.mkdir(parents=True, exist_ok=True)

  def save_bytes(
    self,
    key: str,
    payload: bytes,
    *,
    content_type: str = 'application/octet-stream',
  ) -> str:
    normalized_key = self._normalize_key(key)
    if self._use_r2():
      try:
        return self._save_to_r2(normalized_key, payload, content_type=content_type)
      except Exception as error:
        logger.warning('R2 upload failed for %s, falling back to local storage: %s', normalized_key, error)

    return self._save_to_local(normalized_key, payload)

  def _use_r2(self) -> bool:
    if str(self.settings.media_backend).strip().lower() != 'r2':
      return False

    return bool(
      self.settings.r2_endpoint_url
      and self.settings.r2_public_base_url
      and self._resolved_bucket_name()
      and self.settings.r2_access_key_id
      and self.settings.r2_secret_access_key
    )

  def _resolved_bucket_name(self) -> str:
    configured = str(self.settings.r2_bucket or '').strip()
    if configured:
      return configured

    parsed = urlparse(str(self.settings.r2_endpoint_url or '').strip())
    return parsed.path.strip('/').split('/')[0] if parsed.path.strip('/') else ''

  def _resolved_endpoint_url(self) -> str:
    raw_value = str(self.settings.r2_endpoint_url or '').strip()
    if not raw_value:
      return ''

    parsed = urlparse(raw_value)
    if not parsed.scheme or not parsed.netloc:
      return raw_value.rstrip('/')

    return urlunparse((parsed.scheme, parsed.netloc, '', '', '', '')).rstrip('/')

  def _save_to_r2(
    self,
    key: str,
    payload: bytes,
    *,
    content_type: str,
  ) -> str:
    import boto3

    bucket_name = self._resolved_bucket_name()
    endpoint_url = self._resolved_endpoint_url()
    if not bucket_name or not endpoint_url:
      raise RuntimeError('R2 bucket or endpoint is not configured.')

    client = boto3.client(
      's3',
      endpoint_url=endpoint_url,
      region_name=self.settings.r2_region or 'auto',
      aws_access_key_id=self.settings.r2_access_key_id,
      aws_secret_access_key=self.settings.r2_secret_access_key,
    )
    client.put_object(
      Bucket=bucket_name,
      Key=key,
      Body=payload,
      ContentType=content_type,
    )

    public_base = str(self.settings.r2_public_base_url or '').rstrip('/')
    return f'{public_base}/{key}'

  def _save_to_local(self, key: str, payload: bytes) -> str:
    destination = self.storage_root / Path(key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return key.replace('\\', '/')

  @staticmethod
  def _normalize_key(value: str) -> str:
    cleaned = str(value or '').replace('\\', '/').strip().strip('/')
    if not cleaned:
      raise ValueError('Storage key is required.')

    return cleaned

