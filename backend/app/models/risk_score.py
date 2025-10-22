from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, JSON, Enum, Boolean, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRISIS = "crisis"


class RiskScore(Base):
    __tablename__ = "risk_scores"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Risk classification
    risk_level = Column(Enum(RiskLevel), nullable=False)
    overall_score = Column(Float, nullable=False)  # 0-1 normalized score
    confidence_score = Column(Float, nullable=False)  # 0-1 confidence in assessment

    # Component scores (for explainability)
    sentiment_score = Column(Float, nullable=False)  # Weighted sentiment analysis
    emotion_score = Column(Float, nullable=False)  # Emotion-based risk
    behavioral_score = Column(Float, nullable=False)  # Activity pattern analysis
    academic_score = Column(Float, nullable=False)  # Academic pressure indicators
    social_score = Column(Float, nullable=False)  # Social interaction patterns
    sleep_score = Column(Float, nullable=True)  # Sleep pattern analysis

    # Feature vector (for model analysis)
    feature_vector = Column(JSON, nullable=False)  # All normalized features
    feature_weights = Column(JSON, nullable=False)  # Weight configuration used

    # Trend analysis
    previous_score = Column(Float, nullable=True)
    score_change_24h = Column(Float, nullable=True)
    score_change_7d = Column(Float, nullable=True)
    trend_direction = Column(String(20), nullable=True)  # improving/stable/declining

    # Contextual factors
    academic_calendar_phase = Column(String(50), nullable=True)  # exam/break/normal
    deadline_density = Column(Integer, default=0, nullable=False)  # Deadlines in next 7 days
    recent_stress_events = Column(JSON, nullable=True)  # Recent triggers

    # Alert triggering
    alert_triggered = Column(Boolean, default=False, nullable=False)
    alert_reason = Column(Text, nullable=True)
    alert_threshold_breached = Column(String(100), nullable=True)

    # Temporal information
    calculated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    data_period_start = Column(DateTime(timezone=True), nullable=False)
    data_period_end = Column(DateTime(timezone=True), nullable=False)
    data_points_analyzed = Column(Integer, default=0, nullable=False)

    # Model metadata
    model_version = Column(String(50), nullable=False)
    algorithm_used = Column(String(100), nullable=False)
    requires_review = Column(Boolean, default=False, nullable=False)
    reviewed_by = Column(Integer, nullable=True)  # Counselor/admin ID
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    review_notes = Column(Text, nullable=True)

    # Relationships
    user = relationship("User", back_populates="risk_scores")
