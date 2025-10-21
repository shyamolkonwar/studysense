from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class MessageType(str, Enum):
    SENT = "sent"
    RECEIVED = "received"


class MessageChannel(str, Enum):
    SLACK = "slack"
    TEAMS = "teams"
    EMAIL = "email"
    SMS = "sms"
    CHAT = "chat"


class MessageBase(BaseModel):
    """Base message schema with core fields"""
    external_id: str = Field(..., description="ID in source system")
    source: str = Field(..., max_length=100, description="Integration provider")
    channel: MessageChannel = Field(..., description="Message channel")
    thread_id: Optional[str] = Field(None, max_length=255, description="Thread ID for grouping")
    timestamp: datetime = Field(..., description="Message timestamp")
    message_type: MessageType = Field(..., description="Message direction")

    class Config:
        json_schema_extra = {
            "example": {
                "external_id": "msg_123456",
                "source": "slack",
                "channel": "slack",
                "thread_id": "thread_789",
                "timestamp": "2024-01-01T10:30:00Z",
                "message_type": "sent"
            }
        }


class MessageCreate(MessageBase):
    """Schema for creating a new message"""
    user_id: int = Field(..., description="User ID")
    integration_id: Optional[int] = Field(None, description="Integration account ID")
    content: Optional[str] = Field(None, max_length=10000, description="Message content")
    word_count: Optional[int] = Field(None, ge=0, description="Word count")

    # NLP and sentiment fields
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1, description="VADER compound score")
    sentiment_label: Optional[str] = Field(None, description="Sentiment label")
    emotions: Optional[Dict[str, float]] = Field(None, description="Emotion scores")
    stress_indicators: Optional[float] = Field(None, ge=0, le=1, description="Stress indicators")

    # Linguistic features
    urgency_indicators: int = Field(default=0, ge=0, description="Urgency indicators count")
    question_count: int = Field(default=0, ge=0, description="Question count")
    exclamation_count: int = Field(default=0, ge=0, description="Exclamation count")
    caps_ratio: float = Field(default=0.0, ge=0, le=1, description="Capitalization ratio")

    # Contextual features
    deadline_proximity_hours: Optional[int] = Field(None, ge=0, description="Hours to nearest deadline")
    time_of_day: int = Field(..., ge=0, le=23, description="Hour of day")
    is_weekend: bool = Field(default=False, description="Is weekend")
    response_time_seconds: Optional[int] = Field(None, ge=0, description="Response time in seconds")


class MessageUpdate(BaseModel):
    """Schema for updating message metadata"""
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1)
    sentiment_label: Optional[str] = None
    emotions: Optional[Dict[str, float]] = None
    stress_indicators: Optional[float] = Field(None, ge=0, le=1)
    urgency_indicators: Optional[int] = Field(None, ge=0)
    question_count: Optional[int] = Field(None, ge=0)
    exclamation_count: Optional[int] = Field(None, ge=0)
    caps_ratio: Optional[float] = Field(None, ge=0, le=1)
    deadline_proximity_hours: Optional[int] = Field(None, ge=0)
    response_time_seconds: Optional[int] = Field(None, ge=0)
    is_pii_detected: Optional[bool] = None
    content_redacted: Optional[str] = None
    embedding_ref: Optional[str] = None
    risk_features: Optional[Dict[str, Any]] = None


class MessageResponse(MessageBase):
    """Schema for message response"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    integration_id: Optional[int]
    content_hash: Optional[str]
    content_redacted: Optional[str]
    word_count: Optional[int]

    # NLP and sentiment analysis
    sentiment_score: Optional[float]
    sentiment_label: Optional[str]
    emotions: Optional[Dict[str, float]]
    stress_indicators: Optional[float]

    # Linguistic features
    urgency_indicators: int
    question_count: int
    exclamation_count: int
    caps_ratio: float

    # Contextual features
    deadline_proximity_hours: Optional[int]
    time_of_day: int
    is_weekend: bool
    response_time_seconds: Optional[int]

    # Processing metadata
    processed_at: datetime
    embedding_ref: Optional[str]
    risk_features: Optional[Dict[str, Any]]

    # Privacy and compliance
    is_pii_detected: bool
    retention_expires_at: Optional[datetime]


class MessageSummary(BaseModel):
    """Summary of message statistics"""
    model_config = ConfigDict(from_attributes=True)

    total_messages: int
    sentiment_distribution: Dict[str, int]
    stress_level_avg: Optional[float]
    urgency_indicators_total: int
    response_time_avg_seconds: Optional[float]
    peak_activity_hours: List[int]
    recent_trend: str  # improving/stable/declining

    class Config:
        json_schema_extra = {
            "example": {
                "total_messages": 1250,
                "sentiment_distribution": {
                    "positive": 625,
                    "neutral": 450,
                    "negative": 175
                },
                "stress_level_avg": 0.35,
                "urgency_indicators_total": 89,
                "response_time_avg_seconds": 180,
                "peak_activity_hours": [14, 15, 20, 21],
                "recent_trend": "improving"
            }
        }


class MessageBatch(BaseModel):
    """Schema for batch message processing"""
    messages: List[MessageCreate] = Field(..., max_items=1000, description="Batch of messages")
    processing_options: Optional[Dict[str, Any]] = Field(None, description="Processing options")

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [],
                "processing_options": {
                    "skip_nlp": False,
                    "batch_size": 100,
                    "parallel_processing": True
                }
            }
        }


class MessageSearch(BaseModel):
    """Schema for message search parameters"""
    query: Optional[str] = Field(None, description="Text search query")
    sentiment_filter: Optional[List[str]] = Field(None, description="Sentiment filters")
    date_from: Optional[datetime] = Field(None, description="Start date")
    date_to: Optional[datetime] = Field(None, description="End date")
    channels: Optional[List[MessageChannel]] = Field(None, description="Channel filters")
    min_stress_level: Optional[float] = Field(None, ge=0, le=1, description="Minimum stress level")
    include_content: bool = Field(default=False, description="Include message content")
    limit: int = Field(default=100, ge=1, le=1000, description="Result limit")
    offset: int = Field(default=0, ge=0, description="Result offset")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "deadline",
                "sentiment_filter": ["negative"],
                "date_from": "2024-01-01T00:00:00Z",
                "date_to": "2024-01-31T23:59:59Z",
                "channels": ["slack", "email"],
                "min_stress_level": 0.6,
                "include_content": False,
                "limit": 50,
                "offset": 0
            }
        }