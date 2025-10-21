from pydantic import BaseModel, EmailStr, ConfigDict, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    STUDENT = "student"
    COUNSELOR = "counselor"
    ADMIN = "admin"


class UserBase(BaseModel):
    """Base user schema with common fields"""
    email: EmailStr = Field(..., description="User's email address")
    full_name: str = Field(..., min_length=1, max_length=255, description="Full name")
    role: UserRole = Field(default=UserRole.STUDENT, description="User role")
    institution: Optional[str] = Field(None, max_length=255, description="Educational institution")
    timezone: str = Field(default="UTC", description="User's timezone")
    language: str = Field(default="en", max_length=10, description="Preferred language")
    notification_preferences: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON string of notification preferences"
    )


class UserCreate(UserBase):
    """Schema for creating a new user"""
    password: str = Field(..., min_length=8, max_length=255, description="Password")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "student@university.edu",
                "full_name": "John Doe",
                "password": "securepassword123",
                "role": "student",
                "institution": "University Name",
                "timezone": "America/New_York",
                "language": "en"
            }
        }


class UserUpdate(BaseModel):
    """Schema for updating user information"""
    full_name: Optional[str] = Field(None, min_length=1, max_length=255)
    institution: Optional[str] = Field(None, max_length=255)
    timezone: Optional[str] = None
    language: Optional[str] = Field(None, max_length=10)
    notification_preferences: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "full_name": "John Smith",
                "timezone": "America/Los_Angeles",
                "notification_preferences": {
                    "email_alerts": True,
                    "push_notifications": False,
                    "weekly_reports": True
                }
            }
        }


class UserResponse(UserBase):
    """Schema for user response (excluding sensitive data)"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "email": "student@university.edu",
                "full_name": "John Doe",
                "role": "student",
                "institution": "University Name",
                "is_active": True,
                "is_verified": True,
                "timezone": "America/New_York",
                "language": "en",
                "notification_preferences": {"email_alerts": True},
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        }


class UserPublic(BaseModel):
    """Minimal user information for public display"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    role: UserRole

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "full_name": "John Doe",
                "role": "student"
            }
        }


class UserWithIntegrations(UserResponse):
    """User schema with related integrations count"""
    integrations_count: int = Field(default=0, description="Number of connected integrations")
    last_sync_at: Optional[datetime] = Field(None, description="Last data sync timestamp")


class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr
    password: str

    class Config:
        json_schema_extra = {
            "example": {
                "email": "student@university.edu",
                "password": "securepassword123"
            }
        }


class Token(BaseModel):
    """Schema for JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token expiration time in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800
            }
        }


class TokenData(BaseModel):
    """Schema for token payload"""
    user_id: Optional[int] = None
    email: Optional[str] = None
    role: Optional[UserRole] = None