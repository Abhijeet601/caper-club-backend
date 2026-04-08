from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from sqlalchemy import select

if __package__:
  from .db import SessionLocal, initialize_database
  from .models import PaymentHistory, User, UserRole, utcnow
  from .security import hash_password
else:
  from db import SessionLocal, initialize_database
  from models import PaymentHistory, User, UserRole, utcnow
  from security import hash_password


IMPORT_EMAIL_DOMAIN = 'import.caperclub.local'
DEFAULT_PASSWORD = 'Caper@123'
DEFAULT_PAYMENT_MODE = 'Cash'
MAX_REASONABLE_YEAR = 2100
PLAN_DAYS = {
  'Monthly': 30,
  '2 Month': 60,
  'Quarterly': 90,
  'Half-Yearly': 180,
  'Yearly': 365,
}


@dataclass(slots=True)
class WorkbookRow:
  excel_row: int
  member_id: str
  name: str
  mobile_number: str | None
  membership_level: str
  membership_plan: str
  payment_amount: float
  due_amount: float
  membership_start: date | None
  membership_expiry: date | None
  reg_date: date | None
  issues: list[str] = field(default_factory=list)


def _clean_text(value: Any) -> str:
  if value is None:
    return ''
  return str(value).strip()


def _to_float(value: Any) -> float:
  if value in (None, ''):
    return 0.0
  if isinstance(value, (int, float)):
    return round(float(value), 2)

  text = _clean_text(value).replace(',', '')
  if not text:
    return 0.0

  try:
    return round(float(text), 2)
  except ValueError:
    return 0.0


def _normalize_mobile(value: Any) -> str | None:
  if isinstance(value, int):
    digits = str(value)
    return digits or None

  if isinstance(value, float):
    if value.is_integer():
      digits = str(int(value))
      return digits or None

  digits = ''.join(character for character in _clean_text(value) if character.isdigit())
  return digits or None


def _normalize_member_id(value: str, *, row_number: int) -> str:
  source = _clean_text(value) or f'SW-IMPORT-{row_number:04d}'
  normalized = ''.join(character if character.isalnum() else '-' for character in source.upper())
  normalized = '-'.join(part for part in normalized.split('-') if part)
  return normalized or f'SW-IMPORT-{row_number:04d}'


def _maybe_date(value: Any) -> date | None:
  if isinstance(value, datetime):
    if 2000 <= value.year <= MAX_REASONABLE_YEAR:
      return value.date()
    return None

  if isinstance(value, date):
    if 2000 <= value.year <= MAX_REASONABLE_YEAR:
      return value
    return None

  return None


def _display_name(first_name: Any, last_name: Any) -> str:
  first = _clean_text(first_name)
  last = _clean_text(last_name)
  parts = [part for part in (first, last) if part]
  return ' '.join(parts)


def _infer_plan(level: str, start_date: date | None, expiry_date: date | None) -> str:
  normalized = level.upper()

  if 'HALF' in normalized or normalized.endswith(' HY') or 'SWIMMING HY' in normalized:
    return 'Half-Yearly'
  if 'QTY' in normalized or 'QUARTER' in normalized:
    return 'Quarterly'
  if '2 MONTH' in normalized or '2MONTH' in normalized:
    return '2 Month'
  if normalized.endswith(' Y') or 'YEAR' in normalized:
    return 'Yearly'
  if 'MOTH' in normalized or 'MONTH' in normalized:
    return 'Monthly'

  if start_date and expiry_date:
    duration = (expiry_date - start_date).days
    if duration >= 330:
      return 'Yearly'
    if duration >= 170:
      return 'Half-Yearly'
    if duration >= 75:
      return 'Quarterly'
    if duration >= 45:
      return '2 Month'
    if duration >= 25:
      return 'Monthly'

  return 'Custom'


def _infer_expiry(start_date: date | None, plan: str) -> date | None:
  if start_date is None:
    return None

  days = PLAN_DAYS.get(plan)
  if days is None:
    return None

  return start_date + timedelta(days=days)


def _derive_payment_status(*, due_amount: float, membership_expiry: date | None, today: date) -> str:
  if due_amount > 0:
    return 'Pending'
  if membership_expiry and membership_expiry < today:
    return 'Expired'
  return 'Paid'


def _is_import_email(email: str | None) -> bool:
  if not email:
    return False
  return email.strip().lower().endswith(f'@{IMPORT_EMAIL_DOMAIN}')


def _import_email(member_id: str) -> str:
  local_part = member_id.lower().replace('/', '-')
  return f'{local_part}@{IMPORT_EMAIL_DOMAIN}'


def _row_source(sheet_name: str, row_number: int) -> str:
  return f'Workbook import: {sheet_name} row {row_number}'


def _should_record_payment(item: WorkbookRow) -> bool:
  return bool(
    item.membership_level
    or item.payment_amount > 0
    or item.due_amount > 0
    or item.membership_expiry is not None
  )


def _parse_sheet(*, workbook_path: Path, sheet_name: str) -> tuple[list[WorkbookRow], dict[str, int]]:
  workbook = load_workbook(workbook_path, data_only=True)
  worksheet = workbook[sheet_name]
  rows: list[WorkbookRow] = []
  stats = defaultdict(int)

  for excel_row, row in enumerate(worksheet.iter_rows(min_row=9, values_only=True), start=9):
    reg_no, reg_date, last_name, first_name, mobile, level, fees, paid, due, start, expiry, _days = row[:12]
    name = _display_name(first_name, last_name)
    mobile_number = _normalize_mobile(mobile)

    if not name and not mobile_number:
      stats['skipped_blank_identity'] += 1
      continue

    normalized_member_id = _normalize_member_id(_clean_text(reg_no), row_number=excel_row)
    reg_date_value = _maybe_date(reg_date)
    start_date = _maybe_date(start)
    expiry_date = _maybe_date(expiry)
    membership_level = _clean_text(level)
    issues: list[str] = []

    if start_date is None and reg_date_value is not None:
      start_date = reg_date_value
      issues.append('membership_start_from_reg_date')

    membership_plan = _infer_plan(membership_level, start_date, expiry_date)

    if expiry_date is None:
      inferred_expiry = _infer_expiry(start_date, membership_plan)
      if inferred_expiry is not None:
        expiry_date = inferred_expiry
        issues.append('membership_expiry_inferred')

    if start_date is None and expiry_date is None:
      stats['skipped_missing_membership_window'] += 1
      continue

    rows.append(
      WorkbookRow(
        excel_row=excel_row,
        member_id=normalized_member_id,
        name=name,
        mobile_number=mobile_number,
        membership_level=membership_level,
        membership_plan=membership_plan,
        payment_amount=_to_float(paid),
        due_amount=_to_float(due),
        membership_start=start_date,
        membership_expiry=expiry_date,
        reg_date=reg_date_value,
        issues=issues,
      )
    )

    if issues:
      for issue in issues:
        stats[issue] += 1

  return rows, dict(stats)


def _state_sort_key(item: WorkbookRow) -> tuple[date, date, date, int]:
  return (
    item.membership_start or date.min,
    item.membership_expiry or date.min,
    item.reg_date or date.min,
    item.excel_row,
  )


def _group_rows(rows: list[WorkbookRow]) -> dict[str, list[WorkbookRow]]:
  grouped: dict[str, list[WorkbookRow]] = defaultdict(list)
  for item in rows:
    grouped[item.member_id].append(item)
  return grouped


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Bulk import members from a workbook sheet.')
  parser.add_argument('--workbook', required=True, help='Path to the workbook (.xlsx).')
  parser.add_argument('--sheet', required=True, help='Worksheet name to import.')
  parser.add_argument('--sport', default='Swimming', help='Sport label to assign to imported users.')
  parser.add_argument(
    '--default-password',
    default=DEFAULT_PASSWORD,
    help='Default password for newly created imported users.',
  )
  parser.add_argument(
    '--summary-only',
    action='store_true',
    help='Parse and summarize the workbook without connecting to the database.',
  )
  parser.add_argument(
    '--execute',
    action='store_true',
    help='Apply the import to the configured database.',
  )
  parser.add_argument(
    '--skip-init',
    action='store_true',
    help='Skip initialize_database() and use the current schema as-is.',
  )
  return parser.parse_args()


def main() -> int:
  args = _parse_args()
  workbook_path = Path(args.workbook).expanduser().resolve()

  if not workbook_path.exists():
    raise FileNotFoundError(f'Workbook not found: {workbook_path}')

  rows, parse_stats = _parse_sheet(workbook_path=workbook_path, sheet_name=args.sheet)
  grouped = _group_rows(rows)

  print(f'Workbook: {workbook_path}')
  print(f'Sheet: {args.sheet}')
  print(f'Parsed membership rows: {len(rows)}')
  print(f'Distinct members: {len(grouped)}')
  if parse_stats:
    print('Parse notes:')
    for key in sorted(parse_stats):
      print(f'  {key}: {parse_stats[key]}')

  if args.summary_only and not args.execute:
    return 0

  if not args.skip_init:
    initialize_database()
  db = SessionLocal()
  today = date.today()
  created = 0
  updated = 0
  payments_added = 0

  try:
    member_ids = list(grouped.keys())
    existing_users = db.scalars(select(User).where(User.member_id.in_(member_ids))).all()
    user_by_member_id = {user.member_id: user for user in existing_users}

    existing_payment_sources: dict[str, set[str]] = defaultdict(set)
    existing_user_ids = [user.id for user in existing_users]
    if existing_user_ids:
      payment_rows = db.scalars(
        select(PaymentHistory).where(PaymentHistory.user_id.in_(existing_user_ids))
      ).all()
      for payment in payment_rows:
        existing_payment_sources[payment.user_id].add(payment.source)

    for member_id, member_rows in grouped.items():
      latest = max(member_rows, key=_state_sort_key)
      user = user_by_member_id.get(member_id)
      payment_status = _derive_payment_status(
        due_amount=latest.due_amount,
        membership_expiry=latest.membership_expiry,
        today=today,
      )
      note = (
        f'Imported from workbook "{workbook_path.name}" '
        f'({args.sheet}, latest row {latest.excel_row}).'
      )

      if user is None:
        user = User(
          name=latest.name,
          email=_import_email(member_id),
          password_hash=hash_password(args.default_password),
          role=UserRole.USER,
          mobile_number=latest.mobile_number,
          member_id=member_id,
          sport=args.sport,
          membership_plan=latest.membership_plan,
          membership_level=latest.membership_level,
          membership_start=latest.membership_start,
          membership_expiry=latest.membership_expiry,
          payment_amount=latest.payment_amount,
          due_amount=latest.due_amount,
          payment_mode=DEFAULT_PAYMENT_MODE,
          payment_status=payment_status,
          face_images_count=0,
          note=note,
          updated_at=utcnow(),
        )
        db.add(user)
        db.flush()
        user_by_member_id[member_id] = user
        created += 1
      else:
        if _is_import_email(user.email):
          user.email = _import_email(member_id)
        user.name = latest.name
        user.mobile_number = latest.mobile_number
        user.sport = args.sport
        user.membership_plan = latest.membership_plan
        user.membership_level = latest.membership_level
        user.membership_start = latest.membership_start
        user.membership_expiry = latest.membership_expiry
        user.payment_amount = latest.payment_amount
        user.due_amount = latest.due_amount
        user.payment_mode = DEFAULT_PAYMENT_MODE
        user.payment_status = payment_status
        if not user.note or user.note.startswith('Imported from workbook "'):
          user.note = note
        user.updated_at = utcnow()
        updated += 1

      for item in sorted(member_rows, key=_state_sort_key):
        if not _should_record_payment(item):
          continue

        source = _row_source(args.sheet, item.excel_row)
        if source in existing_payment_sources[user.id]:
          continue

        db.add(
          PaymentHistory(
            user=user,
            plan=item.membership_plan,
            amount=item.payment_amount,
            payment_mode=DEFAULT_PAYMENT_MODE,
            payment_status=_derive_payment_status(
              due_amount=item.due_amount,
              membership_expiry=item.membership_expiry,
              today=today,
            ),
            membership_start=item.membership_start,
            membership_expiry=item.membership_expiry,
            source=source,
            created_at=datetime.combine(
              item.reg_date or item.membership_start or today,
              datetime.min.time(),
            ),
          )
        )
        existing_payment_sources[user.id].add(source)
        payments_added += 1

    if not args.execute:
      db.rollback()
      print('Dry run only. No database changes were committed.')
    else:
      db.commit()
      print('Import committed.')

    print(f'Users created: {created}')
    print(f'Users updated: {updated}')
    print(f'Payment history rows added: {payments_added}')
    if created:
      print(f'Default password for newly created users: {args.default_password}')
    return 0
  finally:
    db.close()


if __name__ == '__main__':
  raise SystemExit(main())
