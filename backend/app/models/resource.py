from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean, JSON, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class ResourceType(str, enum.Enum):
    ARTICLE = "article"
    VIDEO = "video"
    PODCAST = "podcast"
    EXERCISE = "exercise"
    WORKSHEET = "worksheet"
    APP = "app"
    HELPLINE = "helpline"
    WEBSITE = "website"
    BOOK = "book"
    COURSE = "course"


class ResourceCategory(str, enum.Enum):
    STRESS_MANAGEMENT = "stress_management"
    MINDFULNESS = "mindfulness"
    STUDY_SKILLS = "study_skills"
    SLEEP_HYGIENE = "sleep_hygiene"
    EXERCISE_FITNESS = "exercise_fitness"
    NUTRITION = "nutrition"
    SOCIAL_CONNECTION = "social_connection"
    TIME_MANAGEMENT = "time_management"
    CRISIS_SUPPORT = "crisis_support"
    MENTAL_HEALTH_EDUCATION = "mental_health_education"


class Resource(Base):
    __tablename__ = "resources"

    id = Column(Integer, primary_key=True, index=True)

    # Basic information
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)  # AI-generated summary
    resource_type = Column(Enum(ResourceType), nullable=False)
    category = Column(Enum(ResourceCategory), nullable=False)

    # Content and access
    content_url = Column(String(500), nullable=True)
    content_text = Column(Text, nullable=True)  # Full text content for search
    file_path = Column(String(500), nullable=True)  # Local file reference
    external_url = Column(String(500), nullable=True)  # External resource link
    access_level = Column(String(20), default="public", nullable=False)  # public/institutional/private

    # Metadata
    author = Column(String(255), nullable=True)
    publisher = Column(String(255), nullable=True)
    publication_date = Column(DateTime(timezone=True), nullable=True)
    language = Column(String(10), default="en", nullable=False)
    reading_time_minutes = Column(Integer, nullable=True)
    difficulty_level = Column(String(20), nullable=True)  # beginner/intermediate/advanced

    # Classification and tagging
    tags = Column(JSON, nullable=True)  # JSON array of tags
    target_audience = Column(JSON, nullable=True)  # students/counselors/admins
    suitable_risk_levels = Column(JSON, nullable=True)  # Risk levels this helps with
    topics_covered = Column(JSON, nullable=True)  # JSON array of topics
    therapeutic_approaches = Column(JSON, nullable=True)  # CBT/DBT/mindfulness/etc

    # Evidence and quality
    evidence_based = Column(Boolean, default=False, nullable=False)
    clinical_reviewed = Column(Boolean, default=False, nullable=False)
    quality_rating = Column(Integer, nullable=True)  # 1-5 internal rating
    user_rating = Column(Float, default=0.0, nullable=False)  # Average user rating
    rating_count = Column(Integer, default=0, nullable=False)

    # Usage and engagement
    view_count = Column(Integer, default=0, nullable=False)
    bookmark_count = Column(Integer, default=0, nullable=False)
    share_count = Column(Integer, default=0, nullable=False)
    completion_rate = Column(Float, default=0.0, nullable=False)
    average_time_spent = Column(Integer, nullable=True)  # minutes

    # Institution-specific
    institution_id = Column(Integer, nullable=True)  # For institutional resources
    campus_resource = Column(Boolean, default=False, nullable=False)
    location_details = Column(Text, nullable=True)  # Physical location if applicable
    contact_info = Column(JSON, nullable=True)  # Phone/email/address

    # Availability and scheduling
    available_24_7 = Column(Boolean, default=False, nullable=False)
    operating_hours = Column(JSON, nullable=True)  # JSON schedule
    appointment_required = Column(Boolean, default=False, nullable=False)
    booking_url = Column(String(500), nullable=True)

    # Crisis support specific
    is_crisis_resource = Column(Boolean, default=False, nullable=False)
    crisis_types = Column(JSON, nullable=True)  # Types of crises handled
    response_time承诺 = Column(String(100), nullable=True)  # Response time promise

    # RAG and search
    embedding_id = Column(String(255), nullable=True)  # Reference to vector store
    search_keywords = Column(JSON, nullable=True)  # Keywords for search
    related_resources = Column(JSON, nullable=True)  # IDs of related resources

    # Content processing
    last_indexed = Column(DateTime(timezone=True), nullable=True)
    processing_status = Column(String(50), default="pending", nullable=False)
    processing_error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=False)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # Moderation and compliance
    status = Column(String(20), default="active", nullable=False)  # active/inactive/archived
    moderated = Column(Boolean, default=False, nullable=False)
    moderation_notes = Column(Text, nullable=True)
    compliance_flags = Column(JSON, nullable_TRUE)  # Any compliance concerns