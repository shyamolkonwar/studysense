"""
Phase 5: Alerts, Escalation, and Notifications System
Implements policy-driven rules, multi-channel notifications, and escalation protocols
as specified in the MVP requirements.
"""

from .alert_engine import (
    AlertEngine,
    AlertPolicy,
    AlertRule,
    Alert,
    AlertSeverity,
    AlertStatus,
    alert_engine
)

from .escalation_manager import (
    EscalationManager,
    EscalationPolicy,
    EscalationLevel,
    EscalationProtocol,
    escalation_manager
)

from .notification_service import (
    NotificationService,
    NotificationChannel,
    NotificationMessage,
    NotificationTemplate,
    notification_service
)

__all__ = [
    "AlertEngine",
    "AlertPolicy",
    "AlertRule",
    "Alert",
    "AlertSeverity",
    "AlertStatus",
    "alert_engine",
    "EscalationManager",
    "EscalationPolicy",
    "EscalationLevel",
    "EscalationProtocol",
    "escalation_manager",
    "NotificationService",
    "NotificationChannel",
    "NotificationMessage",
    "NotificationTemplate",
    "notification_service"
]