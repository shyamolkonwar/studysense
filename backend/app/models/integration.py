from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, JSON, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class IntegrationProvider(str, enum.Enum):
    GOOGLE_CALENDAR = "google_calendar"
    SLACK = "slack"
    MICROSOFT_TEAMS = "microsoft_teams"
    CANVAS = "canvas"
    MOODLE = "moodle"
    GOOGLE_FIT = "google_fit"
    APPLE_HEALTH = "apple_health"


class IntegrationStatus(str, enum.Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    EXPIRED = "expired"


class IntegrationAccount(Base):
    __tablename__ = "integration_accounts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    provider = Column(Enum(IntegrationProvider), nullable=False)
    external_user_id = Column(String(255), nullable=True)  # ID in external system
    external_account_name = Column(String(255), nullable=True)

    # OAuth and authentication
    access_token = Column(Text, nullable=True)  # Encrypted
    refresh_token = Column(Text, nullable=True)  # Encrypted
    token_expires_at = Column(DateTime(timezone=True), nullable=True)
    scope = Column(Text, nullable=True)  # OAuth scope string

    # Integration configuration
    config = Column(JSON, nullable=True)  # Provider-specific settings
    sync_frequency_minutes = Column(Integer, default=60, nullable=False)
    last_sync_at = Column(DateTime(timezone=True), nullable=True)

    # Status
    status = Column(Enum(IntegrationStatus), default=IntegrationStatus.DISCONNECTED, nullable=False)
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="integrations")
    messages = relationship("Message", back_populates="integration")
    activities = relationship("Activity", back_populates="integration")