"""
Phase 5: Notifications Service
Multi-channel notification delivery system with templates and preferences
"""

from .notification_service import (
    NotificationService,
    NotificationChannel,
    NotificationMessage,
    NotificationTemplate,
    NotificationStatus,
    notification_service
)

from .notification_channels import (
    EmailChannel,
    SMSChannel,
    PushNotificationChannel,
    InAppChannel
)

from .template_manager import (
    TemplateManager,
    template_manager
)

__all__ = [
    "NotificationService",
    "NotificationChannel",
    "NotificationMessage",
    "NotificationTemplate",
    "NotificationStatus",
    "notification_service",
    "EmailChannel",
    "SMSChannel",
    "PushNotificationChannel",
    "InAppChannel",
    "TemplateManager",
    "template_manager"
]