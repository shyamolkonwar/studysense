from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class AlertType(str, enum.Enum):
    RISK_SPIKE = "risk_spike"
    NEGATIVE_TREND = "negative_trend"
    CRISIS_INDICATOR = "crisis_indicator"
    ACADEMIC_STRESS = "academic_stress"
    SOCIAL_WITHDRAWAL = "social_withdrawal"
    SLEEP_DISRUPTION = "sleep_disruption"
    DEADLINE_OVERLOAD = "deadline_overload"
    ANOMALY_DETECTED = "anomaly_detected"


class AlertSeverity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, enum.Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"
    ESCALATED = "escalated"


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    risk_score_id = Column(Integer, ForeignKey("risk_scores.id"), nullable=True)

    # Alert classification
    alert_type = Column(Enum(AlertType), nullable=False)
    severity = Column(Enum(AlertSeverity), nullable=False)
    status = Column(Enum(AlertStatus), default=AlertStatus.ACTIVE, nullable=False)

    # Alert content
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    recommendation = Column(Text, nullable=True)
    evidence_summary = Column(Text, nullable=True)  # Why this alert was triggered

    # Triggering conditions
    threshold_value = Column(String(100), nullable=True)
    actual_value = Column(String(100), nullable=True)
    trigger_rules = Column(Text, nullable=True)  # JSON array of rules that triggered

    # Contextual information
    context_data = Column(Text, nullable=True)  # JSON with relevant context
    contributing_factors = Column(Text, nullable=True)  # JSON array of factors
    related_events = Column(Text, nullable=True)  # JSON array of event IDs

    # Actions and outcomes
    recommended_actions = Column(Text, nullable=True)  # JSON array
    suggested_resources = Column(Text, nullable=True)  # JSON array of resource IDs
    escalation_path = Column(String(255), nullable=True)

    # Delivery and tracking
    delivery_channels = Column(Text, nullable=True)  # JSON: email, sms, push, in-app
    sent_at = Column(DateTime(timezone=True), nullable=True)
    read_at = Column(DateTime(timezone=True), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    acknowledged_by = Column(Integer, nullable=True)  # User who acknowledged

    # Resolution
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(Integer, nullable=True)
    resolution_method = Column(String(100), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    was_helpful = Column(Boolean, nullable=True)

    # Feedback and learning
    user_feedback = Column(Integer, nullable=True)  # 1-5 rating
    feedback_notes = Column(Text, nullable=True)
    false_positive = Column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # Analytics
    response_time_minutes = Column(Integer, nullable=True)  # Time to acknowledgment
    resolution_time_minutes = Column(Integer, nullable=True)  # Time to resolution
    follow_up_required = Column(Boolean, default=False, nullable=False)
    follow_up_scheduled_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="alerts")
    risk_score = relationship("RiskScore", foreign_keys=[risk_score_id])