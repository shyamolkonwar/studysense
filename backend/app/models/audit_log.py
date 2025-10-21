from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean, Enum, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class AuditAction(str, enum.Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    EXPORT = "export"
    INTEGRATION_CONNECT = "integration_connect"
    INTEGRATION_DISCONNECT = "integration_disconnect"
    ALERT_TRIGGER = "alert_trigger"
    ALERT_ACKNOWLEDGE = "alert_acknowledge"
    CONSENT_GRANT = "consent_grant"
    CONSENT_REVOKE = "consent_revoke"
    RISK_CALCULATE = "risk_calculate"
    DATA_PROCESS = "data_process"
    AI_GENERATE = "ai_generate"
    ESCALATE = "escalate"


class AuditSeverity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Nullable for system actions
    session_id = Column(String(255), nullable=True)

    # Action information
    action = Column(Enum(AuditAction), nullable=False)
    resource_type = Column(String(100), nullable=False)  # User, Message, Alert, etc.
    resource_id = Column(String(255), nullable=True)  # ID of affected resource
    severity = Column(Enum(AuditSeverity), default=AuditSeverity.LOW, nullable=False)

    # Description and context
    description = Column(Text, nullable=False)
    action_details = Column(Text, nullable=True)  # JSON with specific details
    old_values = Column(Text, nullable=True)  # JSON with previous state
    new_values = Column(Text, nullable=True)  # JSON with new state

    # Request information
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    endpoint = Column(String(255), nullable=True)
    http_method = Column(String(10), nullable=True)
    request_id = Column(String(100), nullable=True)

    # System and AI actions
    is_system_action = Column(Boolean, default=False, nullable=False)
    is_ai_action = Column(Boolean, default=False, nullable=False)
    ai_model_used = Column(String(100), nullable=True)
    ai_decision_reason = Column(Text, nullable=True)
    ai_confidence_score = Column(Float, nullable=True)

    # Privacy and compliance
    involves_pii = Column(Boolean, default=False, nullable=False)
    data_accessed = Column(Text, nullable=True)  # JSON with data types accessed
    legal_basis = Column(String(255), nullable=True)  # Legal basis for processing
    consent_reference = Column(String(255), nullable=True)

    # Security and authentication
    auth_method = Column(String(50), nullable=True)  # password/sso/oauth/etc
    mfa_verified = Column(Boolean, default=False, nullable=False)
    privilege_level = Column(String(50), nullable=True)
    sudo_used = Column(Boolean, default=False, nullable=False)

    # Performance and impact
    execution_time_ms = Column(Integer, nullable=True)
    records_affected = Column(Integer, nullable=True)
    cascade_impact = Column(Text, nullable=True)  # JSON with cascade effects

    # Error handling
    success = Column(Boolean, default=True, nullable=False)
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    recovery_action = Column(Text, nullable=True)

    # Location and context
    country = Column(String(2), nullable=True)  # ISO 3166-1 alpha-2
    region = Column(String(100), nullable=True)
    device_type = Column(String(50), nullable=True)
    integration_provider = Column(String(100), nullable=True)  # For integration actions

    # Timestamps
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    session_start = Column(DateTime(timezone=True), nullable=True)

    # Retention and lifecycle
    retention_period_days = Column(Integer, default=365, nullable=False)
    archived_at = Column(DateTime(timezone=True), nullable=True)
    deletion_scheduled_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User")

    # Indexes for performance
    __table_args__ = (
        {"schema": "public"},
    )