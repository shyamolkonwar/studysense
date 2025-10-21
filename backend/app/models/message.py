from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Float, JSON, Boolean, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class MessageType(str, enum.Enum):
    SENT = "sent"
    RECEIVED = "received"


class MessageChannel(str, enum.Enum):
    SLACK = "slack"
    TEAMS = "teams"
    EMAIL = "email"
    SMS = "sms"
    CHAT = "chat"


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    integration_id = Column(Integer, ForeignKey("integration_accounts.id"), nullable=True)

    # Source information
    external_id = Column(String(255), nullable=False)  # ID in source system
    source = Column(String(100), nullable=False)  # Integration provider
    channel = Column(Enum(MessageChannel), nullable=False)
    thread_id = Column(String(255), nullable=True)  # For grouping conversations

    # Timestamp and direction
    timestamp = Column(DateTime(timezone=True), nullable=False)
    message_type = Column(Enum(MessageType), nullable=False)

    # Content (privacy-focused design)
    content = Column(Text, nullable=True)  # May be null for metadata-only analysis
    content_hash = Column(String(64), nullable=True)  # SHA-256 hash for deduplication
    content_redacted = Column(Text, nullable=True)  # PII-redacted version
    word_count = Column(Integer, nullable=True)

    # NLP and sentiment analysis
    sentiment_score = Column(Float, nullable=True)  # VADER compound score (-1 to 1)
    sentiment_label = Column(String(20), nullable=True)  # positive/neutral/negative
    emotions = Column(JSON, nullable=True)  # Emotion scores (joy, anger, fear, etc.)
    stress_indicators = Column(Float, nullable=True)  # Stress-related language score

    # Linguistic features
    urgency_indicators = Column(Integer, default=0, nullable=False)
    question_count = Column(Integer, default=0, nullable=False)
    exclamation_count = Column(Integer, default=0, nullable=False)
    caps_ratio = Column(Float, default=0.0, nullable=False)

    # Contextual features
    deadline_proximity_hours = Column(Integer, nullable=True)
    time_of_day = Column(Integer, nullable=False)  # Hour of day (0-23)
    is_weekend = Column(Boolean, default=False, nullable=False)
    response_time_seconds = Column(Integer, nullable=True)  # Time since previous message

    # Processing metadata
    processed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    embedding_ref = Column(String(255), nullable=True)  # Reference to vector store
    risk_features = Column(JSON, nullable=True)  # Feature vector for risk scoring

    # Privacy and compliance
    is_pii_detected = Column(Boolean, default=False, nullable=False)
    retention_expires_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="messages")
    integration = relationship("IntegrationAccount", back_populates="messages")

    # Indexes
    __table_args__ = (
        {"schema": "public"},
    )