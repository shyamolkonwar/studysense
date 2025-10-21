from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class ConsentType(str, enum.Enum):
    MESSAGES_METADATA = "messages_metadata"
    MESSAGES_CONTENT = "messages_content"
    CALENDAR_EVENTS = "calendar_events"
    LMS_DATA = "lms_data"
    ACTIVITY_PATTERNS = "activity_patterns"
    LOCATION_DATA = "location_data"


class ConsentStatus(str, enum.Enum):
    PENDING = "pending"
    GRANTED = "granted"
    REVOKED = "revoked"
    EXPIRED = "expired"


class Consent(Base):
    __tablename__ = "consents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    consent_type = Column(Enum(ConsentType), nullable=False)
    status = Column(Enum(ConsentStatus), default=ConsentStatus.PENDING, nullable=False)

    # Consent details
    purpose = Column(Text, nullable=False)  # Plain language description
    data_retention_days = Column(Integer, nullable=True)
    scope_details = Column(Text, nullable=True)  # JSON string with specific data fields

    # Timestamps
    granted_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=False)

    # Audit
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    legal_version = Column(String(20), nullable=False)  # Version of privacy policy/T&Cs

    # Relationships
    user = relationship("User", back_populates="consents")