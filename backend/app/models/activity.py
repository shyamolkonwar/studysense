from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, JSON, Boolean, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class ActivityType(str, enum.Enum):
    STUDY_SESSION = "study_session"
    ASSIGNMENT_SUBMIT = "assignment_submit"
    CALENDAR_EVENT = "calendar_event"
    SLEEP_PERIOD = "sleep_period"
    EXERCISE = "exercise"
    SOCIAL_INTERACTION = "social_interaction"
    APP_USAGE = "app_usage"
    DEADLINE = "deadline"


class Activity(Base):
    __tablename__ = "activities"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    integration_id = Column(Integer, ForeignKey("integration_accounts.id"), nullable=True)

    # Activity classification
    activity_type = Column(Enum(ActivityType), nullable=False)
    category = Column(String(100), nullable=True)  # Sub-category (e.g., "exam", "homework")

    # Temporal information
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration_minutes = Column(Integer, nullable=True)

    # Academic context
    course_id = Column(String(100), nullable=True)
    assignment_id = Column(String(100), nullable=True)
    deadline_proximity_hours = Column(Integer, nullable=True)
    academic_load_score = Column(Float, nullable=True)  # 0-1 based on concurrent deadlines

    # Behavioral patterns
    time_of_day = Column(Integer, nullable=False)  # Hour of day (0-23)
    is_weekend = Column(Boolean, default=False, nullable=False)
    is_night_owl = Column(Boolean, default=False, nullable=False)  # Activity after 11 PM
    is_early_bird = Column(Boolean, default=False, nullable=False)  # Activity before 6 AM

    # Study patterns (for study sessions)
    focus_score = Column(Float, nullable=True)  # 0-1 based on activity consistency
    break_frequency = Column(Float, nullable=True)  # Breaks per hour
    session_quality = Column(Float, nullable=True)  # Quality metric based on patterns

    # Social and communication patterns
    interaction_count = Column(Integer, default=0, nullable=False)
    group_size = Column(Integer, nullable=True)
    is_collaborative = Column(Boolean, default=False, nullable=False)

    # Wellness indicators
    stress_markers = Column(JSON, nullable=True)  # Various stress-related features
    sleep_quality = Column(Float, nullable=True)  # 0-1 if sleep activity
    exercise_intensity = Column(String(50), nullable=True)  # low/medium/high

    # Anomaly detection
    is_anomaly = Column(Boolean, default=False, nullable=False)
    anomaly_score = Column(Float, default=0.0, nullable=False)
    anomaly_reason = Column(String(255), nullable=True)

    # Processing metadata
    processed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    embedding_ref = Column(String(255), nullable=True)  # Reference to vector store
    risk_features = Column(JSON, nullable=True)  # Feature vector for risk scoring

    # Raw data reference (for reprocessing)
    raw_data = Column(JSON, nullable=True)  # Original activity data from source
    raw_data_hash = Column(String(64), nullable=True)  # For deduplication

    # Relationships
    user = relationship("User", back_populates="activities")
    integration = relationship("IntegrationAccount", back_populates="activities")