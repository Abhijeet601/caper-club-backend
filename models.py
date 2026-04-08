from __future__ import annotations

from datetime import date, datetime, time
from enum import Enum
from uuid import uuid4

from sqlalchemy import Boolean, Date, DateTime, Enum as SqlEnum, Float, ForeignKey, Integer, LargeBinary, Numeric, String, Text, Time
from sqlalchemy.orm import Mapped, mapped_column, relationship

if __package__:
  from .db import Base
else:
  from db import Base


def generate_uuid() -> str:
  return str(uuid4())


def utcnow() -> datetime:
  return datetime.utcnow()


class UserRole(str, Enum):
  ADMIN = 'admin'
  USER = 'user'


class SessionStatus(str, Enum):
  ACTIVE = 'active'
  ENDED = 'ended'
  EXPIRED = 'expired'
  DENIED = 'denied'


class TimelineEventType(str, Enum):
  ENTRY = 'entry'
  EXIT = 'exit'
  DENIED = 'denied'


class Tone(str, Enum):
  BLUE = 'blue'
  PURPLE = 'purple'
  GREEN = 'green'
  RED = 'red'
  AMBER = 'amber'


class TimeSlot(Base):
  __tablename__ = 'time_slots'

  id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
  name: Mapped[str] = mapped_column(String(120), unique=True)
  start_time: Mapped[time] = mapped_column(Time)
  end_time: Mapped[time] = mapped_column(Time)
  created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
  updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

  users: Mapped[list['User']] = relationship(back_populates='slot')
  sessions: Mapped[list['SessionRecord']] = relationship(back_populates='slot')


class User(Base):
  __tablename__ = 'users'

  id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
  name: Mapped[str] = mapped_column(String(120))
  email: Mapped[str] = mapped_column(String(190), unique=True, index=True)
  password_hash: Mapped[str] = mapped_column(String(255))
  role: Mapped[UserRole] = mapped_column(SqlEnum(UserRole))
  mobile_number: Mapped[str | None] = mapped_column(
    String(15),
    index=True,
    nullable=True,
  )
  member_id: Mapped[str] = mapped_column(String(64), index=True)
  slot_id: Mapped[str | None] = mapped_column(
    ForeignKey('time_slots.id', ondelete='SET NULL'),
    nullable=True,
  )
  sport: Mapped[str] = mapped_column(String(64), default='General')
  membership_plan: Mapped[str] = mapped_column(String(32), default='Monthly')
  membership_level: Mapped[str] = mapped_column(String(120), default='')
  membership_start: Mapped[date | None] = mapped_column(Date, nullable=True)
  membership_expiry: Mapped[date | None] = mapped_column(Date, nullable=True)
  payment_amount: Mapped[float] = mapped_column(Numeric(10, 2), default=0)
  due_amount: Mapped[float] = mapped_column(Numeric(10, 2), default=0)
  payment_mode: Mapped[str] = mapped_column(String(16), default='UPI')
  payment_status: Mapped[str] = mapped_column(String(16), default='Pending')
  last_action: Mapped[str | None] = mapped_column(String(8), nullable=True)
  last_action_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
  note: Mapped[str] = mapped_column(Text, default='')
  face_images_count: Mapped[int] = mapped_column(Integer, default=0)
  created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
  updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

  face_embeddings: Mapped[list['FaceEmbedding']] = relationship(
    back_populates='user',
    cascade='all, delete-orphan',
  )
  sessions: Mapped[list['SessionRecord']] = relationship(
    back_populates='user',
    cascade='all, delete-orphan',
  )
  timelines: Mapped[list['UserTimeline']] = relationship(
    back_populates='user',
    cascade='all, delete-orphan',
  )
  payments: Mapped[list['PaymentHistory']] = relationship(
    back_populates='user',
    cascade='all, delete-orphan',
  )
  notifications: Mapped[list['Notification']] = relationship(
    back_populates='user',
    cascade='all, delete-orphan',
  )
  targeted_announcements: Mapped[list['Announcement']] = relationship(
    back_populates='target_user',
    foreign_keys='Announcement.user_id',
  )
  created_announcements: Mapped[list['Announcement']] = relationship(
    back_populates='created_by',
    foreign_keys='Announcement.created_by_id',
  )
  slot: Mapped[TimeSlot | None] = relationship(back_populates='users')


class FaceEmbedding(Base):
  __tablename__ = 'face_embeddings'

  id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
  user_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'))
  image_data: Mapped[str] = mapped_column(Text)
  embedding_vector: Mapped[bytes] = mapped_column(LargeBinary)
  created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

  user: Mapped[User] = relationship(back_populates='face_embeddings')


class SessionRecord(Base):
  __tablename__ = 'sessions'

  id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
  user_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'))
  slot_id: Mapped[str | None] = mapped_column(
    ForeignKey('time_slots.id', ondelete='SET NULL'),
    nullable=True,
  )
  area: Mapped[str] = mapped_column(String(120))
  status: Mapped[SessionStatus] = mapped_column(SqlEnum(SessionStatus), default=SessionStatus.ACTIVE)
  confidence: Mapped[float] = mapped_column(Float, default=0.0)
  started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
  ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
  slot_start_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
  slot_end_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
  created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
  updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

  user: Mapped[User] = relationship(back_populates='sessions')
  slot: Mapped[TimeSlot | None] = relationship(back_populates='sessions')


class UserTimeline(Base):
  __tablename__ = 'user_timelines'

  id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
  user_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'))
  event_type: Mapped[TimelineEventType] = mapped_column(SqlEnum(TimelineEventType))
  area: Mapped[str] = mapped_column(String(120))
  occurred_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
  total_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
  note: Mapped[str] = mapped_column(Text, default='')

  user: Mapped[User] = relationship(back_populates='timelines')


class PaymentHistory(Base):
  __tablename__ = 'payment_history'

  id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
  user_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'))
  plan: Mapped[str] = mapped_column(String(32))
  amount: Mapped[float] = mapped_column(Numeric(10, 2), default=0)
  payment_mode: Mapped[str] = mapped_column(String(16), default='UPI')
  payment_status: Mapped[str] = mapped_column(String(16), default='Pending')
  membership_start: Mapped[date | None] = mapped_column(Date, nullable=True)
  membership_expiry: Mapped[date | None] = mapped_column(Date, nullable=True)
  source: Mapped[str] = mapped_column(String(120))
  created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

  user: Mapped[User] = relationship(back_populates='payments')


class Announcement(Base):
  __tablename__ = 'announcements'

  id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
  title: Mapped[str] = mapped_column(String(120))
  message: Mapped[str] = mapped_column(Text)
  tone: Mapped[Tone] = mapped_column(SqlEnum(Tone), default=Tone.BLUE)
  user_id: Mapped[str | None] = mapped_column(ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
  created_by_id: Mapped[str | None] = mapped_column(ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
  created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

  target_user: Mapped[User | None] = relationship(
    back_populates='targeted_announcements',
    foreign_keys=[user_id],
  )
  created_by: Mapped[User | None] = relationship(
    back_populates='created_announcements',
    foreign_keys=[created_by_id],
  )


class Notification(Base):
  __tablename__ = 'notifications'

  id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
  user_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'))
  title: Mapped[str] = mapped_column(String(120))
  message: Mapped[str] = mapped_column(Text)
  tone: Mapped[Tone] = mapped_column(SqlEnum(Tone), default=Tone.BLUE)
  is_read: Mapped[bool] = mapped_column(Boolean, default=False)
  created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

  user: Mapped[User] = relationship(back_populates='notifications')
