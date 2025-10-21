from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class ConversationType(str, enum.Enum):
    CHAT_BOT = "chat_bot"
    COUNSELOR = "counselor"
    PEER_SUPPORT = "peer_support"
    SELF_REFLECTION = "self_reflection"


class ConversationStatus(str, enum.Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ESCALATED = "escalated"


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    conversation_type = Column(Enum(ConversationType), nullable=False)
    status = Column(Enum(ConversationStatus), default=ConversationStatus.ACTIVE, nullable=False)

    # Conversation metadata
    title = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)  # AI-generated summary
    topic_tags = Column(Text, nullable=True)  # JSON array of topics

    # Emotional context
    dominant_emotion = Column(String(50), nullable=True)
    stress_level_start = Column(Integer, nullable=True)  # 1-10 scale
    stress_level_end = Column(Integer, nullable=True)  # 1-10 scale
    sentiment_trend = Column(String(20), nullable=True)  # improving/stable/declining

    # Outcomes and actions
    recommendations_provided = Column(Text, nullable=True)  # JSON array
    resources_shared = Column(Text, nullable=True)  # JSON array of resource IDs
    actions_taken = Column(Text, nullable=True)  # JSON array of user actions
    resolution_status = Column(String(50), nullable=True)

    # Counselor/peer info (if applicable)
    counselor_id = Column(Integer, nullable=True)  # FK to counselors table
    escalation_reason = Column(Text, nullable=True)
    escalation_timestamp = Column(DateTime(timezone=True), nullable=True)

    # Analytics
    message_count = Column(Integer, default=0, nullable=False)
    duration_minutes = Column(Integer, nullable=True)
    satisfaction_score = Column(Integer, nullable=True)  # 1-5, user feedback
    was_helpful = Column(Boolean, nullable=True)

    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_message_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=False)

    # Privacy and retention
    is_exportable = Column(Boolean, default=True, nullable=False)
    retention_expires_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="conversations")