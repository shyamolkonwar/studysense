from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Enum, Float, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class EmbeddingType(str, enum.Enum):
    MESSAGE = "message"
    ACTIVITY = "activity"
    RESOURCE = "resource"
    USER_CONTEXT = "user_context"
    KNOWLEDGE_BASE = "knowledge_base"
    CONVERSATION = "conversation"


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Nullable for KB resources

    # Source information
    embedding_type = Column(Enum(EmbeddingType), nullable=False)
    source_id = Column(String(255), nullable=False)  # ID of source record
    source_table = Column(String(100), nullable=False)  # Table name of source

    # Embedding metadata
    model_name = Column(String(100), nullable=False)  # e.g., "text-embedding-ada-002"
    model_version = Column(String(50), nullable=False)
    embedding_dimension = Column(Integer, nullable=False)
    chunk_index = Column(Integer, default=0, nullable=False)  # For chunked content
    total_chunks = Column(Integer, default=1, nullable=False)

    # Content (for search and retrieval)
    original_text = Column(Text, nullable=False)
    processed_text = Column(Text, nullable=False)  # After preprocessing
    text_hash = Column(String(64), nullable=False)  # SHA-256 for deduplication
    chunk_metadata = Column(Text, nullable=True)  # JSON with chunking info

    # Vector reference
    vector_store_id = Column(String(255), nullable=False)  # ID in ChromaDB/Pinecone/etc
    collection_name = Column(String(100), nullable=False)
    similarity_threshold = Column(Float, default=0.7, nullable=False)

    # Usage and performance
    retrieval_count = Column(Integer, default=0, nullable=False)
    last_retrieved_at = Column(DateTime(timezone=True), nullable=True)
    average_relevance_score = Column(Float, nullable=True)
    user_feedback_count = Column(Integer, default=0, nullable=False)

    # Indexing and search optimization
    search_keywords = Column(Text, nullable=True)  # JSON array of keywords
    entity_tags = Column(Text, nullable=True)  # JSON array of entities
    topic_tags = Column(Text, nullable=True)  # JSON array of topics
    language_detected = Column(String(10), nullable=True)

    # Quality metrics
    embedding_quality_score = Column(Float, nullable=True)
    content_length = Column(Integer, nullable=False)
    semantic_density = Column(Float, nullable=True)
    requires_update = Column(Boolean, default=False, nullable=False)

    # Temporal information
    context_window_start = Column(DateTime(timezone=True), nullable=True)
    context_window_end = Column(DateTime(timezone=True), nullable=True)
    temporal_weight = Column(Float, default=1.0, nullable=False)

    # Processing metadata
    processed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    processing_version = Column(String(50), nullable=False)
    processing_errors = Column(Text, nullable=True)
    batch_id = Column(String(100), nullable=True)  # For batch processing

    # Privacy and access control
    access_level = Column(String(20), default="private", nullable=False)  # private/shared/public
    sharing_allowed = Column(Boolean, default=False, nullable=False)
    research_use_only = Column(Boolean, default=False, nullable=False)

    # Retention and lifecycle
    expires_at = Column(DateTime(timezone=True), nullable=True)
    archived_at = Column(DateTime(timezone=True), nullable=True)
    deletion_scheduled_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User")

    # Indexes for performance
    __table_args__ = (
        {"schema": "public"},
    )
