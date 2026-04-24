from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from urllib.parse import quote_plus

import pymysql
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

BACKEND_DIR = Path(__file__).resolve().parent
ENV_FILES = (
  str(BACKEND_DIR / '.env'),
  str(BACKEND_DIR.parent / '.env'),
)


class Settings(BaseSettings):
  db_host: str = '127.0.0.1'
  db_port: int = 3306
  db_name: str = 'caperclub'
  db_user: str = 'root'
  db_password: str = 'Abhijeet@7654'
  database_url_value: str = Field(
    default='',
    validation_alias=AliasChoices(
      'CAPERCLUB_DATABASE_URL',
      'MYSQL_URL',
      'DATABASE_URL',
      'MYSQL_PUBLIC_URL',
      'MYSQL_PRIVATE_URL',
    ),
  )
  mysql_host_value: str = Field(
    default='',
    validation_alias=AliasChoices('MYSQLHOST', 'DB_HOST'),
  )
  mysql_port_value: int | None = Field(
    default=None,
    validation_alias=AliasChoices('MYSQLPORT', 'DB_PORT'),
  )
  mysql_database_value: str = Field(
    default='',
    validation_alias=AliasChoices('MYSQLDATABASE', 'DB_NAME'),
  )
  mysql_user_value: str = Field(
    default='',
    validation_alias=AliasChoices('MYSQLUSER', 'DB_USER'),
  )
  mysql_password_value: str = Field(
    default='',
    validation_alias=AliasChoices('MYSQLPASSWORD', 'DB_PASSWORD'),
  )
  media_backend: str = 'r2'
  r2_endpoint_url: str = 'https://94b941e2dd9341b958247c2cb68276e7.r2.cloudflarestorage.com/caperclubdata'
  r2_bucket: str = 'caperclubdata'
  r2_public_base_url: str = 'https://pub-9ed9b2e4413a49cb847df3a991647b68.r2.dev'
  r2_region: str = 'auto'
  r2_access_key_id: str = ''
  r2_secret_access_key: str = ''
  jwt_secret: str = 'caperclub-dev-secret-key-2026-rotate'
  jwt_algorithm: str = 'HS256'
  access_token_expiry_minutes: int = 12 * 60
  cors_origin: str = 'http://localhost:5173'
  cors_origins: str = ''
  cors_origin_regex: str = (
    r'^https?://((localhost|127\.0\.0\.1)(:\d+)?|[a-z0-9-]+\.trycloudflare\.com)$'
  )
  elevenlabs_api_key: str = Field(
    default='',
    validation_alias=AliasChoices('CAPERCLUB_ELEVENLABS_API_KEY', 'ELEVENLABS_API_KEY'),
  )
  elevenlabs_voice_id: str = Field(
    default='pNInz6obpgDQGcFmaJgB',
    validation_alias=AliasChoices('CAPERCLUB_ELEVENLABS_VOICE_ID', 'ELEVENLABS_VOICE_ID'),
  )
  elevenlabs_model_id: str = Field(
    default='eleven_multilingual_v2',
    validation_alias=AliasChoices('CAPERCLUB_ELEVENLABS_MODEL_ID', 'ELEVENLABS_MODEL_ID'),
  )
  elevenlabs_output_format: str = Field(
    default='mp3_44100_128',
    validation_alias=AliasChoices(
      'CAPERCLUB_ELEVENLABS_OUTPUT_FORMAT',
      'ELEVENLABS_OUTPUT_FORMAT',
    ),
  )

  model_config = SettingsConfigDict(
    env_prefix='CAPERCLUB_',
    extra='ignore',
    env_file=ENV_FILES,
    env_file_encoding='utf-8',
  )

  @property
  def database_url(self) -> str:
    override = self.database_url_override
    if override:
      if override.startswith('mysql://'):
        return override.replace('mysql://', 'mysql+pymysql://', 1)
      return override

    config = self.database_config
    password = quote_plus(config['password'])
    return (
      f"mysql+pymysql://{config['user']}:{password}"
      f"@{config['host']}:{config['port']}/{config['name']}"
    )

  @property
  def cors_origin_list(self) -> list[str]:
    configured_origins = f'{self.cors_origin},{self.cors_origins}'
    origins = [origin.strip() for origin in configured_origins.split(',') if origin.strip()]
    return list(dict.fromkeys(origins))

  @property
  def database_url_override(self) -> str:
    return self.database_url_value.strip()

  @property
  def database_config(self) -> dict[str, str | int]:
    return {
      'host': (self.mysql_host_value or self.db_host).strip(),
      'port': self.mysql_port_value if self.mysql_port_value is not None else self.db_port,
      'name': (self.mysql_database_value or self.db_name).strip(),
      'user': (self.mysql_user_value or self.db_user).strip(),
      'password': self.mysql_password_value or self.db_password,
    }

  @property
  def is_managed_database(self) -> bool:
    return bool(
      self.database_url_override
      or self.mysql_host_value
      or self.mysql_database_value
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
  return Settings()


class Base(DeclarativeBase):
  pass


settings = get_settings()
engine = create_engine(
  settings.database_url,
  pool_pre_ping=True,
  future=True,
)
SessionLocal = sessionmaker(
  bind=engine,
  autoflush=False,
  autocommit=False,
  expire_on_commit=False,
  class_=Session,
)


def get_db():
  database = SessionLocal()

  try:
    yield database
  finally:
    database.close()


EXPECTED_TABLE_COLUMNS = {
  'users': {
    'id',
    'name',
    'email',
    'password_hash',
    'role',
    'mobile_number',
    'member_id',
    'sport',
    'membership_level',
    'membership_plan',
    'membership_start',
    'membership_expiry',
    'visit_limit',
    'payment_amount',
    'due_amount',
    'payment_mode',
    'payment_status',
    'last_action',
    'last_action_at',
    'note',
    'face_images_count',
    'created_at',
    'updated_at',
  },
  'face_embeddings': {
    'id',
    'user_id',
    'image_data',
    'embedding_vector',
    'created_at',
  },
  'sessions': {
    'id',
    'user_id',
    'area',
    'status',
    'confidence',
    'started_at',
    'ended_at',
    'created_at',
    'updated_at',
  },
  'user_timelines': {
    'id',
    'user_id',
    'event_type',
    'area',
    'occurred_at',
    'total_minutes',
    'note',
  },
  'payment_history': {
    'id',
    'user_id',
    'plan',
    'amount',
    'payment_mode',
    'payment_status',
    'membership_start',
    'membership_expiry',
    'source',
    'created_at',
  },
  'announcements': {
    'id',
    'title',
    'message',
    'tone',
    'user_id',
    'created_by_id',
    'created_at',
  },
  'notifications': {
    'id',
    'user_id',
    'title',
    'message',
    'tone',
    'is_read',
    'created_at',
  },
  'time_slots': {
    'id',
    'name',
    'start_time',
    'end_time',
    'created_at',
    'updated_at',
  },
}

MIGRATABLE_MISSING_COLUMNS = {
  'users': {
    'member_id',
    'mobile_number',
    'sport',
    'membership_level',
    'visit_limit',
    'payment_amount',
    'due_amount',
    'payment_mode',
    'payment_status',
    'last_action',
    'last_action_at',
  },
  'payment_history': {
    'payment_mode',
    'payment_status',
    'membership_start',
    'membership_expiry',
  },
}

KNOWN_APP_TABLES = {
  'activity_feed',
  'announcement_history',
  'announcement_templates',
  'announcements',
  'app_meta',
  'camera_feeds',
  'entry_logs',
  'face_embeddings',
  'membership_plans',
  'notifications',
  'payment_history',
  'sessions',
  'settings',
  'time_slots',
  'user_notifications',
  'user_timelines',
  'users',
}


def _is_railway_environment() -> bool:
  return bool(
    os.getenv('RAILWAY_PROJECT_ID')
    or os.getenv('RAILWAY_SERVICE_ID')
    or os.getenv('RAILWAY_ENVIRONMENT')
    or os.getenv('RAILWAY_ENVIRONMENT_NAME')
  )


def _ensure_database_exists() -> None:
  if settings.is_managed_database:
    return

  config = settings.database_config

  if str(config['host']).strip().lower() in {'127.0.0.1', 'localhost'} and _is_railway_environment():
    raise RuntimeError(
      'Railway deployment started without MySQL connection settings. '
      'Set CAPERCLUB_DATABASE_URL, MYSQL_URL, DATABASE_URL, MYSQL_PUBLIC_URL, '
      'MYSQL_PRIVATE_URL, or MYSQLHOST/MYSQLPORT/MYSQLDATABASE/MYSQLUSER/MYSQLPASSWORD '
      'on the backend service.'
    )

  connection = pymysql.connect(
    host=str(config['host']),
    port=int(config['port']),
    user=str(config['user']),
    password=str(config['password']),
    charset='utf8mb4',
    autocommit=True,
  )

  try:
    with connection.cursor() as cursor:
      cursor.execute(
        'CREATE DATABASE IF NOT EXISTS '
        f"`{config['name']}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
      )
  finally:
    connection.close()


def _is_schema_legacy(table_columns: dict[str, set[str]]) -> bool:
  for table_name, expected_columns in EXPECTED_TABLE_COLUMNS.items():
    existing_columns = table_columns.get(table_name)

    if existing_columns is None:
      continue

    missing_columns = expected_columns - existing_columns

    if not missing_columns:
      continue

    migratable_columns = MIGRATABLE_MISSING_COLUMNS.get(table_name, set())

    if not missing_columns.issubset(migratable_columns):
      return True

  legacy_only_tables = {
    'activity_feed',
    'announcement_history',
    'announcement_templates',
    'app_meta',
    'camera_feeds',
    'entry_logs',
    'membership_plans',
    'settings',
    'user_notifications',
  }
  return any(table_name in table_columns for table_name in legacy_only_tables)


def _ensure_no_legacy_schema() -> None:
  inspector = inspect(engine)
  table_names = set(inspector.get_table_names())

  if not table_names:
    return

  relevant_tables = table_names & KNOWN_APP_TABLES
  table_columns = {
    table_name: {column['name'] for column in inspector.get_columns(table_name)}
    for table_name in relevant_tables
  }

  if not _is_schema_legacy(table_columns):
    return

  non_empty_tables: list[str] = []

  with engine.begin() as connection:
    for table_name in sorted(relevant_tables):
      result = connection.execute(
        text(f'SELECT 1 FROM `{table_name}` LIMIT 1')
      ).first()

      if result is not None:
        non_empty_tables.append(table_name)

    if non_empty_tables:
      formatted_tables = ', '.join(non_empty_tables)
      raise RuntimeError(
        'Legacy CaperClub schema detected with existing data. '
        f'Migrate or clear these tables first: {formatted_tables}.'
      )

    connection.execute(text('SET FOREIGN_KEY_CHECKS = 0'))

    for table_name in sorted(relevant_tables):
      connection.execute(text(f'DROP TABLE IF EXISTS `{table_name}`'))

    connection.execute(text('SET FOREIGN_KEY_CHECKS = 1'))


def _ensure_index_exists(
  inspector,
  *,
  table_name: str,
  index_name: str,
  create_sql: str,
) -> None:
  indexes = {index['name'] for index in inspector.get_indexes(table_name)}

  if index_name in indexes:
    return

  with engine.begin() as connection:
    connection.execute(text(create_sql))


def _drop_index_if_exists(
  inspector,
  *,
  table_name: str,
  index_name: str,
) -> None:
  indexes = {index['name'] for index in inspector.get_indexes(table_name)}

  if index_name not in indexes:
    return

  with engine.begin() as connection:
    connection.execute(text(f'ALTER TABLE `{table_name}` DROP INDEX `{index_name}`'))


def _ensure_foreign_key_exists(
  inspector,
  *,
  table_name: str,
  foreign_key_name: str,
  create_sql: str,
) -> None:
  foreign_keys = {foreign_key['name'] for foreign_key in inspector.get_foreign_keys(table_name)}

  if foreign_key_name in foreign_keys:
    return

  with engine.begin() as connection:
    connection.execute(text(create_sql))


def _ensure_slot_schema() -> None:
  inspector = inspect(engine)
  table_names = set(inspector.get_table_names())

  if 'users' not in table_names or 'sessions' not in table_names or 'time_slots' not in table_names:
    return

  user_columns = {column['name'] for column in inspector.get_columns('users')}
  session_columns = {column['name'] for column in inspector.get_columns('sessions')}

  with engine.begin() as connection:
    if 'slot_id' not in user_columns:
      connection.execute(
        text(
          'ALTER TABLE `users` '
          'ADD COLUMN `slot_id` VARCHAR(36) NULL AFTER `role`'
        )
      )

    if 'slot_id' not in session_columns:
      connection.execute(
        text(
          'ALTER TABLE `sessions` '
          'ADD COLUMN `slot_id` VARCHAR(36) NULL AFTER `user_id`'
        )
      )

    if 'slot_start_at' not in session_columns:
      connection.execute(
        text(
          'ALTER TABLE `sessions` '
          'ADD COLUMN `slot_start_at` DATETIME NULL AFTER `ended_at`'
        )
      )

    if 'slot_end_at' not in session_columns:
      connection.execute(
        text(
          'ALTER TABLE `sessions` '
          'ADD COLUMN `slot_end_at` DATETIME NULL AFTER `slot_start_at`'
        )
      )

  inspector = inspect(engine)
  _ensure_index_exists(
    inspector,
    table_name='users',
    index_name='idx_users_slot_id',
    create_sql='CREATE INDEX `idx_users_slot_id` ON `users` (`slot_id`)',
  )
  _ensure_index_exists(
    inspector,
    table_name='sessions',
    index_name='idx_sessions_slot_id',
    create_sql='CREATE INDEX `idx_sessions_slot_id` ON `sessions` (`slot_id`)',
  )
  _ensure_foreign_key_exists(
    inspector,
    table_name='users',
    foreign_key_name='fk_users_slot_id',
    create_sql=(
      'ALTER TABLE `users` '
      'ADD CONSTRAINT `fk_users_slot_id` '
      'FOREIGN KEY (`slot_id`) REFERENCES `time_slots` (`id`) ON DELETE SET NULL'
    ),
  )
  _ensure_foreign_key_exists(
    inspector,
    table_name='sessions',
    foreign_key_name='fk_sessions_slot_id',
    create_sql=(
      'ALTER TABLE `sessions` '
      'ADD CONSTRAINT `fk_sessions_slot_id` '
      'FOREIGN KEY (`slot_id`) REFERENCES `time_slots` (`id`) ON DELETE SET NULL'
    ),
  )


def _ensure_member_profile_schema() -> None:
  inspector = inspect(engine)
  table_names = set(inspector.get_table_names())

  if 'users' not in table_names:
    return

  user_columns = {column['name'] for column in inspector.get_columns('users')}
  payment_columns = (
    {column['name'] for column in inspector.get_columns('payment_history')}
    if 'payment_history' in table_names
    else set()
  )

  with engine.begin() as connection:
    if 'mobile_number' not in user_columns:
      connection.execute(
        text(
          'ALTER TABLE `users` '
          'ADD COLUMN `mobile_number` VARCHAR(15) NULL AFTER `role`'
        )
      )

    if 'member_id' not in user_columns:
      connection.execute(
        text(
          'ALTER TABLE `users` '
          'ADD COLUMN `member_id` VARCHAR(64) NULL AFTER `role`'
        )
      )

    if 'sport' not in user_columns:
      connection.execute(
        text(
          'ALTER TABLE `users` '
          "ADD COLUMN `sport` VARCHAR(64) NOT NULL DEFAULT 'General' AFTER `slot_id`"
        )
      )

    if 'membership_level' not in user_columns:
      connection.execute(
        text(
          'ALTER TABLE `users` '
          "ADD COLUMN `membership_level` VARCHAR(120) NOT NULL DEFAULT '' AFTER `membership_plan`"
        )
      )

    if 'payment_amount' not in user_columns:
      connection.execute(
        text(
          'ALTER TABLE `users` '
          'ADD COLUMN `payment_amount` DECIMAL(10, 2) NOT NULL DEFAULT 0 AFTER `membership_expiry`'
        )
      )

    if 'visit_limit' not in user_columns:
      connection.execute(
        text(
          'ALTER TABLE `users` '
          'ADD COLUMN `visit_limit` INT NULL AFTER `membership_expiry`'
        )
      )

    if 'due_amount' not in user_columns:
      connection.execute(
        text(
          'ALTER TABLE `users` '
          'ADD COLUMN `due_amount` DECIMAL(10, 2) NOT NULL DEFAULT 0 AFTER `payment_amount`'
        )
      )

    if 'payment_mode' not in user_columns:
      connection.execute(
        text(
          'ALTER TABLE `users` '
          "ADD COLUMN `payment_mode` VARCHAR(16) NOT NULL DEFAULT 'UPI' AFTER `due_amount`"
        )
      )

    if 'payment_status' not in user_columns:
      connection.execute(
        text(
          'ALTER TABLE `users` '
          "ADD COLUMN `payment_status` VARCHAR(16) NOT NULL DEFAULT 'Pending' AFTER `payment_mode`"
        )
      )

    if 'last_action' not in user_columns:
      connection.execute(
        text(
          'ALTER TABLE `users` '
          'ADD COLUMN `last_action` VARCHAR(8) NULL AFTER `payment_status`'
        )
      )

    if 'last_action_at' not in user_columns:
      connection.execute(
        text(
          'ALTER TABLE `users` '
          'ADD COLUMN `last_action_at` DATETIME NULL AFTER `last_action`'
        )
      )

    connection.execute(
      text(
        "UPDATE `users` "
        "SET `member_id` = CONCAT('CC-', UPPER(LEFT(REPLACE(`id`, '-', ''), 8))) "
        "WHERE `member_id` IS NULL OR TRIM(`member_id`) = ''"
      )
    )

    if 'payment_history' in table_names:
      if 'payment_mode' not in payment_columns:
        connection.execute(
          text(
            'ALTER TABLE `payment_history` '
            "ADD COLUMN `payment_mode` VARCHAR(16) NOT NULL DEFAULT 'UPI' AFTER `amount`"
          )
        )

      if 'payment_status' not in payment_columns:
        connection.execute(
          text(
            'ALTER TABLE `payment_history` '
            "ADD COLUMN `payment_status` VARCHAR(16) NOT NULL DEFAULT 'Pending' AFTER `payment_mode`"
          )
        )

      if 'membership_start' not in payment_columns:
        connection.execute(
          text(
            'ALTER TABLE `payment_history` '
            'ADD COLUMN `membership_start` DATE NULL AFTER `payment_status`'
          )
        )

      if 'membership_expiry' not in payment_columns:
        connection.execute(
          text(
            'ALTER TABLE `payment_history` '
            'ADD COLUMN `membership_expiry` DATE NULL AFTER `membership_start`'
          )
        )

  inspector = inspect(engine)
  indexes = inspector.get_indexes('users')
  unique_mobile_indexes = [
    index['name']
    for index in indexes
    if index.get('unique') and index.get('column_names') == ['mobile_number']
  ]

  for index_name in unique_mobile_indexes:
    _drop_index_if_exists(
      inspector,
      table_name='users',
      index_name=index_name,
    )

  inspector = inspect(engine)
  _ensure_index_exists(
    inspector,
    table_name='users',
    index_name='ix_users_mobile_number',
    create_sql='CREATE INDEX `ix_users_mobile_number` ON `users` (`mobile_number`)',
  )
  _ensure_index_exists(
    inspector,
    table_name='users',
    index_name='ix_users_member_id',
    create_sql='CREATE UNIQUE INDEX `ix_users_member_id` ON `users` (`member_id`)',
  )


def _ensure_query_performance_indexes() -> None:
  inspector = inspect(engine)
  _ensure_index_exists(
    inspector,
    table_name='users',
    index_name='idx_users_role_created_at',
    create_sql='CREATE INDEX `idx_users_role_created_at` ON `users` (`role`, `created_at`)',
  )
  _ensure_index_exists(
    inspector,
    table_name='users',
    index_name='idx_users_role_updated_at',
    create_sql='CREATE INDEX `idx_users_role_updated_at` ON `users` (`role`, `updated_at`)',
  )
  _ensure_index_exists(
    inspector,
    table_name='sessions',
    index_name='idx_sessions_user_status_started_at',
    create_sql='CREATE INDEX `idx_sessions_user_status_started_at` ON `sessions` (`user_id`, `status`, `started_at`)',
  )
  _ensure_index_exists(
    inspector,
    table_name='sessions',
    index_name='idx_sessions_status_started_at',
    create_sql='CREATE INDEX `idx_sessions_status_started_at` ON `sessions` (`status`, `started_at`)',
  )
  _ensure_index_exists(
    inspector,
    table_name='sessions',
    index_name='idx_sessions_status_slot_end_at',
    create_sql='CREATE INDEX `idx_sessions_status_slot_end_at` ON `sessions` (`status`, `slot_end_at`)',
  )
  _ensure_index_exists(
    inspector,
    table_name='user_timelines',
    index_name='idx_user_timelines_event_occurred_at',
    create_sql='CREATE INDEX `idx_user_timelines_event_occurred_at` ON `user_timelines` (`event_type`, `occurred_at`)',
  )
  _ensure_index_exists(
    inspector,
    table_name='user_timelines',
    index_name='idx_user_timelines_user_occurred_at',
    create_sql='CREATE INDEX `idx_user_timelines_user_occurred_at` ON `user_timelines` (`user_id`, `occurred_at`)',
  )
  _ensure_index_exists(
    inspector,
    table_name='payment_history',
    index_name='idx_payment_history_user_created_at',
    create_sql='CREATE INDEX `idx_payment_history_user_created_at` ON `payment_history` (`user_id`, `created_at`)',
  )
  _ensure_index_exists(
    inspector,
    table_name='payment_history',
    index_name='idx_payment_history_created_at',
    create_sql='CREATE INDEX `idx_payment_history_created_at` ON `payment_history` (`created_at`)',
  )
  _ensure_index_exists(
    inspector,
    table_name='announcements',
    index_name='idx_announcements_user_created_at',
    create_sql='CREATE INDEX `idx_announcements_user_created_at` ON `announcements` (`user_id`, `created_at`)',
  )


def _ensure_runtime_compatible_schema() -> None:
  inspector = inspect(engine)

  if 'face_embeddings' not in inspector.get_table_names():
    return

  columns = {
    column['name']: type(column['type']).__name__.lower()
    for column in inspector.get_columns('face_embeddings')
  }

  if columns.get('embedding_vector') == 'json':
    with engine.begin() as connection:
      connection.execute(
        text(
          'ALTER TABLE `face_embeddings` '
          'MODIFY COLUMN `embedding_vector` LONGBLOB NOT NULL'
        )
      )


def initialize_database() -> None:
  _ensure_database_exists()
  _ensure_no_legacy_schema()

  if __package__:
    from . import models  # noqa: F401
  else:
    import models  # noqa: F401

  Base.metadata.create_all(bind=engine)
  _ensure_slot_schema()
  _ensure_member_profile_schema()
  _ensure_query_performance_indexes()
  _ensure_runtime_compatible_schema()
