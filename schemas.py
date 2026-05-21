from __future__ import annotations

from datetime import date, datetime, time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RoleLiteral = Literal['admin', 'user']
PlanLiteral = Literal['Monthly', '2 Month', 'Quarterly', 'Half-Yearly', 'Yearly', 'Custom']
ToneLiteral = Literal['blue', 'purple', 'green', 'red', 'amber']
PaymentModeLiteral = Literal['Cash', 'UPI', 'Card']
PaymentStatusLiteral = Literal['Paid', 'Pending', 'Expired']
AttendanceActionLiteral = Literal['IN', 'OUT']


class StrictModel(BaseModel):
  model_config = ConfigDict(extra='forbid')


def _normalize_text(value: object) -> object:
  if isinstance(value, str):
    return value.strip()

  return value


def _validate_email(value: str) -> str:
  normalized = value.strip().lower()

  if '@' not in normalized or '.' not in normalized.split('@')[-1]:
    raise ValueError('Enter a valid email address.')

  return normalized


def _validate_mobile(value: str | None) -> str | None:
  if value is None:
    return None

  normalized = ''.join(character for character in value.strip() if character.isdigit())
  if not normalized:
    return None

  if len(normalized) < 7 or len(normalized) > 15:
    raise ValueError('Enter a valid mobile number.')

  return normalized


def _validate_time_value(value: str) -> time:
  normalized = value.strip()

  try:
    parsed = datetime.strptime(normalized, '%H:%M').time()
  except ValueError as error:
    raise ValueError('Enter time in HH:MM format.') from error

  return parsed


class LoginInput(StrictModel):
  email: str
  password: str = Field(min_length=6, max_length=128)

  @field_validator('email')
  @classmethod
  def validate_email(cls, value: str) -> str:
    return _validate_email(value)


class RegisterInput(StrictModel):
  name: str = Field(min_length=2, max_length=120)
  email: str
  password: str = Field(min_length=8, max_length=128)

  @field_validator('name', mode='before')
  @classmethod
  def strip_name(cls, value: object) -> object:
    return _normalize_text(value)

  @field_validator('email')
  @classmethod
  def validate_email(cls, value: str) -> str:
    return _validate_email(value)


class CreateUserInput(StrictModel):
  name: str = Field(min_length=2, max_length=120)
  memberId: str | None = Field(default=None, max_length=64)
  sport: str = Field(default='General', max_length=64)
  membershipLevel: str = Field(default='', max_length=120)
  email: str
  password: str = Field(min_length=8, max_length=128)
  mobileNumber: str | None = Field(default=None, max_length=15)
  role: RoleLiteral
  slotId: str | None = None
  membershipPlan: PlanLiteral = 'Monthly'
  membershipStart: date
  membershipExpiry: date
  visitLimit: int | None = Field(default=None, ge=0)
  paymentAmount: float = Field(default=0, ge=0)
  dueAmount: float = Field(default=0, ge=0)
  paymentMode: PaymentModeLiteral = 'UPI'
  paymentStatus: PaymentStatusLiteral = 'Pending'
  note: str = Field(default='', max_length=500)

  @field_validator(
    'name',
    'memberId',
    'sport',
    'membershipLevel',
    'note',
    'slotId',
    'mobileNumber',
    mode='before',
  )
  @classmethod
  def strip_text(cls, value: object) -> object:
    return _normalize_text(value)

  @field_validator('email')
  @classmethod
  def validate_email(cls, value: str) -> str:
    return _validate_email(value)

  @field_validator('mobileNumber')
  @classmethod
  def validate_mobile(cls, value: str | None) -> str | None:
    return _validate_mobile(value)

  @model_validator(mode='after')
  def validate_date_range(self) -> 'CreateUserInput':
    if self.membershipExpiry < self.membershipStart:
      raise ValueError('Membership expiry must be on or after the start date.')

    if self.role == 'user' and not self.slotId:
      raise ValueError('Assigned slot is required for users.')

    return self


class UpdateUserInput(StrictModel):
  name: str = Field(min_length=2, max_length=120)
  memberId: str = Field(min_length=2, max_length=64)
  sport: str = Field(default='General', max_length=64)
  membershipLevel: str = Field(default='', max_length=120)
  email: str
  mobileNumber: str | None = Field(default=None, max_length=15)
  role: RoleLiteral
  slotId: str | None = None
  membershipPlan: PlanLiteral = 'Monthly'
  membershipStart: date
  membershipExpiry: date
  visitLimit: int | None = Field(default=None, ge=0)
  paymentAmount: float = Field(default=0, ge=0)
  dueAmount: float = Field(default=0, ge=0)
  paymentMode: PaymentModeLiteral = 'UPI'
  paymentStatus: PaymentStatusLiteral = 'Pending'
  note: str = Field(default='', max_length=500)
  password: str | None = Field(default=None, min_length=8, max_length=128)

  @field_validator(
    'name',
    'memberId',
    'sport',
    'membershipLevel',
    'note',
    'password',
    'slotId',
    'mobileNumber',
    mode='before',
  )
  @classmethod
  def strip_text(cls, value: object) -> object:
    return _normalize_text(value)

  @field_validator('email')
  @classmethod
  def validate_email(cls, value: str) -> str:
    return _validate_email(value)

  @field_validator('mobileNumber')
  @classmethod
  def validate_mobile(cls, value: str | None) -> str | None:
    return _validate_mobile(value)

  @model_validator(mode='after')
  def validate_date_range(self) -> 'UpdateUserInput':
    if self.membershipExpiry < self.membershipStart:
      raise ValueError('Membership expiry must be on or after the start date.')

    if self.role == 'user' and not self.slotId:
      raise ValueError('Assigned slot is required for users.')

    return self


class CreateSlotInput(StrictModel):
  name: str = Field(min_length=2, max_length=120)
  startTime: str
  endTime: str

  @field_validator('name', mode='before')
  @classmethod
  def strip_name(cls, value: object) -> object:
    return _normalize_text(value)

  @field_validator('startTime', 'endTime')
  @classmethod
  def validate_times(cls, value: str) -> str:
    parsed = _validate_time_value(value)
    return parsed.strftime('%H:%M')

  @model_validator(mode='after')
  def validate_range(self) -> 'CreateSlotInput':
    if self.startTime == self.endTime:
      raise ValueError('Slot end time must be different from start time.')

    return self


class UpdateSlotInput(CreateSlotInput):
  pass


class UploadFaceInput(StrictModel):
  userId: str | None = None
  images: list[str] = Field(default_factory=list, min_length=1, max_length=10)

  @field_validator('userId', mode='before')
  @classmethod
  def strip_user_id(cls, value: object) -> object:
    return _normalize_text(value)


class AccessScanInput(StrictModel):
  userId: str | None = None
  area: str = Field(min_length=2, max_length=120)
  image: str = Field(min_length=20)
  capturedFrames: int = Field(default=3, ge=1, le=10)

  @field_validator('userId', 'area', mode='before')
  @classmethod
  def strip_values(cls, value: object) -> object:
    return _normalize_text(value)


class CreateMembershipInput(StrictModel):
  userId: str
  plan: PlanLiteral
  startDate: date
  expiryDate: date
  visitLimit: int | None = Field(default=None, ge=0)
  paymentAmount: float = Field(default=0, ge=0)
  paymentMode: PaymentModeLiteral = 'UPI'
  paymentStatus: PaymentStatusLiteral = 'Pending'
  source: str = Field(default='Manual Entry', min_length=2, max_length=120)

  @field_validator('userId', 'source', mode='before')
  @classmethod
  def strip_values(cls, value: object) -> object:
    return _normalize_text(value)

  @model_validator(mode='after')
  def validate_date_range(self) -> 'CreateMembershipInput':
    if self.expiryDate < self.startDate:
      raise ValueError('Membership expiry must be on or after the start date.')

    return self


class CreateAnnouncementInput(StrictModel):
  title: str = Field(min_length=3, max_length=120)
  message: str = Field(min_length=3, max_length=600)
  tone: ToneLiteral = 'blue'
  userId: str | None = None

  @field_validator('title', 'message', 'userId', mode='before')
  @classmethod
  def strip_values(cls, value: object) -> object:
    return _normalize_text(value)


class SessionStartInput(StrictModel):
  userId: str
  area: str = Field(min_length=2, max_length=120)
  confidence: float = Field(default=0.9, ge=0, le=1)

  @field_validator('userId', 'area', mode='before')
  @classmethod
  def strip_values(cls, value: object) -> object:
    return _normalize_text(value)




class SessionEndInput(StrictModel):
  sessionId: str

  @field_validator('sessionId', mode='before')
  @classmethod
  def strip_values(cls, value: object) -> object:
    return _normalize_text(value)


class TTSRequest(StrictModel):
  text: str = Field(min_length=1, max_length=500)


class DescriptorEnrollmentInput(StrictModel):
  userId: str
  descriptors: list[list[float]] = Field(default_factory=list, min_length=1, max_length=10)

  @field_validator('userId', mode='before')
  @classmethod
  def strip_user_id(cls, value: object) -> object:
    return _normalize_text(value)

  @field_validator('descriptors')
  @classmethod
  def validate_descriptors(cls, value: list[list[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []

    for descriptor in value:
      if len(descriptor) != 128:
        raise ValueError('Each face descriptor must contain 128 values.')

      normalized.append([float(item) for item in descriptor])

    return normalized


class AttendanceInput(StrictModel):
  userId: str
  action: AttendanceActionLiteral
  area: str = Field(default='Capper Sports Club Entry', min_length=2, max_length=120)
  confidence: float = Field(default=0, ge=0, le=1)

  @field_validator('userId', 'area', mode='before')
  @classmethod
  def strip_values(cls, value: object) -> object:
    return _normalize_text(value)
