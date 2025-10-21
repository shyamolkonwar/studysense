from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from supabase import create_client, Client


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "StudySense"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"

    # Database (Supabase)
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:54322/postgres"
    SUPABASE_URL: str = "http://localhost:54321"
    SUPABASE_ANON_KEY: Optional[str] = None
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None

    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Vector Database (ChromaDB)
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"

    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    DEFAULT_LLM_PROVIDER: str = "openai"
    DEFAULT_MODEL: str = "gpt-3.5-turbo"

    # Embedding Configuration
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    EMBEDDING_DIMENSION: int = 1536

    # Redis/Celery for background tasks
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Email configuration
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[str] = None
    EMAILS_FROM_NAME: Optional[str] = None

    # OAuth providers
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None

    SLACK_CLIENT_ID: Optional[str] = None
    SLACK_CLIENT_SECRET: Optional[str] = None

    MICROSOFT_CLIENT_ID: Optional[str] = None
    MICROSOFT_CLIENT_SECRET: Optional[str] = None

    # File storage
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB

    # Object storage (Supabase)
    STORAGE_BUCKET_NAME: str = "studysense-data"

    # Monitoring and logging
    SENTRY_DSN: Optional[str] = None
    LOG_LEVEL: str = "INFO"

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000

    # Data retention
    DEFAULT_DATA_RETENTION_DAYS: int = 365
    MESSAGE_RETENTION_DAYS: int = 90
    AUDIT_RETENTION_DAYS: int = 2555  # 7 years

    # Risk scoring configuration
    RISK_CALCULATION_INTERVAL_MINUTES: int = 60
    ALERT_COOLDOWN_MINUTES: int = 30

    # Integration settings
    INTEGRATION_SYNC_INTERVAL_MINUTES: int = 30
    MAX_INTEGRATION_RETRIES: int = 3

    # Privacy and compliance
    COOKIE_SECURE: bool = True
    COOKIE_SAMESITE: str = "lax"
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # Environment
    ENVIRONMENT: str = "development"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Supabase client initialization
supabase: Optional[Client] = None

def get_supabase_client() -> Optional[Client]:
    """Get Supabase client if credentials are configured"""
    global supabase

    if supabase is None and settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY:
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

    return supabase