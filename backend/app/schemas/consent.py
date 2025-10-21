from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ConsentType(str, Enum):
    MESSAGES_METADATA = "messages_metadata"
    MESSAGES_CONTENT = "messages_content"
    CALENDAR_EVENTS = "calendar_events"
    LMS_DATA = "lms_data"
    ACTIVITY_PATTERNS = "activity_patterns"
    LOCATION_DATA = "location_data"


class ConsentStatus(str, Enum):
    PENDING = "pending"
    GRANTED = "granted"
    REVOKED = "revoked"
    EXPIRED = "expired"


class ConsentBase(BaseModel):
    """Base consent schema"""
    consent_type: ConsentType = Field(..., description="Type of consent")
    purpose: str = Field(..., description="Plain language description of consent purpose")
    data_retention_days: Optional[int] = Field(None, ge=1, le=2555, description="Data retention period in days")
    scope_details: Optional[Dict[str, Any]] = Field(None, description="Specific data fields scope")
    legal_version: str = Field(..., description="Version of privacy policy/T&Cs")

    class Config:
        json_schema_extra = {
            "example": {
                "consent_type": "messages_metadata",
                "purpose": "Analyze message metadata patterns to detect stress indicators while preserving content privacy",
                "data_retention_days": 90,
                "scope_details": {
                    "included_fields": ["timestamp", "frequency", "sentiment", "response_times"],
                    "excluded_fields": ["content", "attachments"]
                },
                "legal_version": "1.0"
            }
        }


class ConsentCreate(ConsentBase):
    """Schema for creating new consent"""
    pass


class ConsentUpdate(BaseModel):
    """Schema for updating consent"""
    status: Optional[ConsentStatus] = None
    data_retention_days: Optional[int] = Field(None, ge=1, le=2555)
    scope_details: Optional[Dict[str, Any]] = None
    expires_at: Optional[datetime] = None


class ConsentResponse(ConsentBase):
    """Schema for consent response"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    status: ConsentStatus
    granted_at: Optional[datetime]
    revoked_at: Optional[datetime]
    expires_at: Optional[datetime]
    created_at: datetime
    updated_at: Optional[datetime]
    ip_address: Optional[str]
    user_agent: Optional[str]

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "user_id": 1,
                "consent_type": "messages_metadata",
                "status": "granted",
                "purpose": "Analyze message metadata patterns...",
                "data_retention_days": 90,
                "scope_details": {"included_fields": ["timestamp"]},
                "legal_version": "1.0",
                "granted_at": "2024-01-01T10:00:00Z",
                "revoked_at": None,
                "expires_at": "2024-04-01T10:00:00Z",
                "created_at": "2024-01-01T10:00:00Z",
                "updated_at": "2024-01-01T10:00:00Z",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0..."
            }
        }


class ConsentSummary(BaseModel):
    """Summary of user's consent status"""
    model_config = ConfigDict(from_attributes=True)

    user_id: int
    total_consents: int
    granted_consents: int
    revoked_consents: int
    expired_consents: int
    pending_consents: int
    consents: List[ConsentResponse]

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "total_consents": 5,
                "granted_consents": 3,
                "revoked_consents": 1,
                "expired_consents": 1,
                "pending_consents": 0,
                "consents": []
            }
        }


class ConsentRequest(BaseModel):
    """Schema for requesting user consent"""
    consent_types: List[ConsentType] = Field(..., description="List of consent types to request")
    legal_version: str = Field(..., description="Legal version to use")
    redirect_url: Optional[str] = Field(None, description="URL to redirect after consent")

    class Config:
        json_schema_extra = {
            "example": {
                "consent_types": ["messages_metadata", "calendar_events"],
                "legal_version": "1.0",
                "redirect_url": "/dashboard"
            }
        }