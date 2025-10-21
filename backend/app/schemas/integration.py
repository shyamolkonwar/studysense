from pydantic import BaseModel, Field, ConfigDict, HttpUrl
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class IntegrationProvider(str, Enum):
    GOOGLE_CALENDAR = "google_calendar"
    SLACK = "slack"
    MICROSOFT_TEAMS = "microsoft_teams"
    CANVAS = "canvas"
    MOODLE = "moodle"
    GOOGLE_FIT = "google_fit"
    APPLE_HEALTH = "apple_health"


class IntegrationStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    EXPIRED = "expired"


class IntegrationBase(BaseModel):
    """Base integration schema"""
    provider: IntegrationProvider = Field(..., description="Integration provider")
    external_user_id: Optional[str] = Field(None, max_length=255, description="External system user ID")
    external_account_name: Optional[str] = Field(None, max_length=255, description="External account display name")
    sync_frequency_minutes: int = Field(default=60, ge=5, le=1440, description="Sync frequency in minutes")
    config: Optional[Dict[str, Any]] = Field(None, description="Provider-specific settings")

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "google_calendar",
                "external_user_id": "user@gmail.com",
                "external_account_name": "John Doe",
                "sync_frequency_minutes": 30,
                "config": {
                    "calendar_ids": ["primary", "work"],
                    "sync_past_days": 30
                }
            }
        }


class IntegrationCreate(IntegrationBase):
    """Schema for creating a new integration"""
    access_token: Optional[str] = Field(None, description="OAuth access token")
    refresh_token: Optional[str] = Field(None, description="OAuth refresh token")
    scope: Optional[str] = Field(None, description="OAuth scope string")
    token_expires_at: Optional[datetime] = Field(None, description="Token expiration time")


class IntegrationUpdate(BaseModel):
    """Schema for updating integration settings"""
    sync_frequency_minutes: Optional[int] = Field(None, ge=5, le=1440)
    config: Optional[Dict[str, Any]] = None
    status: Optional[IntegrationStatus] = None
    error_message: Optional[str] = Field(None, max_length=1000)


class IntegrationResponse(IntegrationBase):
    """Schema for integration response"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    status: IntegrationStatus
    error_message: Optional[str]
    last_sync_at: Optional[datetime]
    created_at: datetime
    updated_at: Optional[datetime]
    token_expires_at: Optional[datetime]

    # Sensitive data excluded for security
    # access_token, refresh_token are not returned

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "user_id": 1,
                "provider": "google_calendar",
                "external_user_id": "user@gmail.com",
                "external_account_name": "John Doe",
                "status": "connected",
                "error_message": None,
                "sync_frequency_minutes": 30,
                "config": {"calendar_ids": ["primary"]},
                "last_sync_at": "2024-01-01T10:00:00Z",
                "created_at": "2024-01-01T09:00:00Z",
                "updated_at": "2024-01-01T10:00:00Z",
                "token_expires_at": "2024-02-01T09:00:00Z"
            }
        }


class IntegrationAuth(BaseModel):
    """Schema for OAuth authentication request"""
    provider: IntegrationProvider = Field(..., description="Integration provider")
    auth_code: str = Field(..., description="Authorization code from OAuth provider")
    redirect_uri: Optional[str] = Field(None, description="Redirect URI used in OAuth flow")
    state: Optional[str] = Field(None, description="State parameter for CSRF protection")

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "slack",
                "auth_code": "xoxb-1234567890-abcdef123456",
                "redirect_uri": "http://localhost:3000/auth/slack/callback",
                "state": "random_state_string"
            }
        }


class IntegrationOAuthUrl(BaseModel):
    """Schema for OAuth authorization URL response"""
    authorization_url: str = Field(..., description="OAuth authorization URL")
    state: str = Field(..., description="State parameter for CSRF protection")
    provider: IntegrationProvider = Field(..., description="Integration provider")

    class Config:
        json_schema_extra = {
            "example": {
                "authorization_url": "https://slack.com/oauth/v2/authorize?client_id=...&state=...",
                "state": "random_state_string",
                "provider": "slack"
            }
        }


class IntegrationSync(BaseModel):
    """Schema for triggering manual sync"""
    integration_ids: Optional[List[int]] = Field(None, description="Specific integration IDs to sync")
    force_full_sync: bool = Field(default=False, description="Force full historical sync")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range for sync")

    class Config:
        json_schema_extra = {
            "example": {
                "integration_ids": [1, 2],
                "force_full_sync": True,
                "date_range": {
                    "start": "2024-01-01T00:00:00Z",
                    "end": "2024-01-31T23:59:59Z"
                }
            }
        }


class IntegrationStatus(BaseModel):
    """Schema for integration status response"""
    integration_id: int
    provider: IntegrationProvider
    status: IntegrationStatus
    last_sync_at: Optional[datetime]
    next_sync_at: Optional[datetime]
    sync_success_count: int
    sync_error_count: int
    last_error_message: Optional[str]
    data_points_synced: int
    is_healthy: bool


class IntegrationHealth(BaseModel):
    """Schema for overall integration health"""
    total_integrations: int
    healthy_integrations: int
    unhealthy_integrations: int
    last_global_sync: Optional[datetime]
    sync_queue_size: int
    integration_statuses: List[IntegrationStatus]

    class Config:
        json_schema_extra = {
            "example": {
                "total_integrations": 4,
                "healthy_integrations": 3,
                "unhealthy_integrations": 1,
                "last_global_sync": "2024-01-01T10:00:00Z",
                "sync_queue_size": 2,
                "integration_statuses": []
            }
        }