from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean, JSON, Enum, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class RecommendationType(str, enum.Enum):
    COPING_STRATEGY = "coping_strategy"
    STUDY_TECHNIQUE = "study_technique"
    WELLNESS_ACTIVITY = "wellness_activity"
    RESOURCE_REFERRAL = "resource_referral"
    SOCIAL_CONNECTION = "social_connection"
    PROFESSIONAL_HELP = "professional_help"
    TIME_MANAGEMENT = "time_management"
    SLEEP_HYGIENE = "sleep_hygiene"


class RecommendationStatus(str, enum.Enum):
    PENDING = "pending"
    VIEWED = "viewed"
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)

    # Recommendation classification
    recommendation_type = Column(Enum(RecommendationType), nullable=False)
    status = Column(Enum(RecommendationStatus), default=RecommendationStatus.PENDING, nullable=False)

    # Content
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    detailed_instructions = Column(Text, nullable=True)
    estimated_time_minutes = Column(Integer, nullable=True)
    difficulty_level = Column(String(20), nullable=True)  # easy/medium/hard

    # Personalization
    personalization_factors = Column(JSON, nullable=True)  # Why this was recommended
    user_preferences_considered = Column(JSON, nullable=True)
    previous_effectiveness = Column(Float, nullable=True)  # Based on user history

    # RAG and evidence
    evidence_sources = Column(JSON, nullable=True)  # Sources from knowledge base
    confidence_score = Column(Float, nullable=False)  # 0-1 confidence in relevance
    rag_citations = Column(JSON, nullable=True)  # Citations from retrieved content

    # Categorization and filtering
    tags = Column(JSON, nullable=True)  # JSON array of tags
    categories = Column(JSON, nullable=True)  # JSON array of categories
    suitable_for = Column(JSON, nullable=True)  # Risk levels this applies to

    # Action and engagement
    action_steps = Column(JSON, nullable=True)  # JSON array of specific steps
    progress_tracking = Column(JSON, nullable=True)  # Progress milestones
    reminder_schedule = Column(JSON, nullable=True)  # When to remind user

    # Delivery and presentation
    priority = Column(Integer, default=5, nullable=False)  # 1-10 priority
    display_format = Column(String(50), nullable=True)  # card/list/modal/etc
    is_interactive = Column(Boolean, default=False, nullable=False)
    external_resource_url = Column(String(500), nullable=True)

    # Effectiveness tracking
    user_rating = Column(Integer, nullable=True)  # 1-5 after completion
    user_feedback = Column(Text, nullable=True)
    actual_completion_time = Column(Integer, nullable=True)
    perceived_helpfulness = Column(Integer, nullable=True)  # 1-5

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    viewed_at = Column(DateTime(timezone=True), nullable=True)
    accepted_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="recommendations")
    alert = relationship("Alert", foreign_keys=[alert_id])
    conversation = relationship("Conversation", foreign_keys=[conversation_id])
