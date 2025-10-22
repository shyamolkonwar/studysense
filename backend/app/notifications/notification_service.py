"""
Phase 5: Notification Service
Central notification delivery system with multi-channel support
"""

from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import uuid

logger = logging.getLogger(__name__)

class NotificationStatus(str, Enum):
    """Notification delivery status"""
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"

class NotificationPriority(str, Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class NotificationMessage:
    """Notification message structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    channel: str = ""
    priority: NotificationPriority = NotificationPriority.MEDIUM
    subject: str = ""
    content: str = ""
    template_id: Optional[str] = None
    template_data: Dict[str, Any] = field(default_factory=dict)
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: NotificationStatus = NotificationStatus.PENDING
    delivery_attempts: int = 0
    last_attempt: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class NotificationTemplate:
    """Notification template structure"""
    id: str
    name: str
    channel: str
    language: str
    subject_template: str
    content_template: str
    variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeliveryReport:
    """Notification delivery report"""
    notification_id: str
    channel: str
    status: NotificationStatus
    sent_at: datetime
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class NotificationService:
    """
    Central notification service for managing multi-channel notifications
    with template support, scheduling, and delivery tracking.
    """

    def __init__(self):
        """Initialize notification service"""

        # Notification channels registry
        self.channels: Dict[str, Any] = {}

        # Template storage
        self.templates: Dict[str, NotificationTemplate] = {}

        # Message queues and tracking
        self.pending_messages: List[NotificationMessage] = []
        self.delivery_history: List[DeliveryReport] = []

        # User preferences
        self.user_preferences: Dict[str, Dict[str, Any]] = {}

        # Rate limiting
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}

        # Notification processors
        self.message_processors: List[callable] = []

        logger.info("Notification Service initialized")

    def register_channel(self, channel_type: str, channel_instance: Any):
        """Register a notification channel"""
        self.channels[channel_type] = channel_instance
        logger.info(f"Registered notification channel: {channel_type}")

    def add_template(self, template: NotificationTemplate):
        """Add a notification template"""
        self.templates[template.id] = template
        logger.info(f"Added notification template: {template.name}")

    async def send_notification(
        self,
        user_id: str,
        channel: str,
        subject: str,
        content: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        template_id: Optional[str] = None,
        template_data: Optional[Dict[str, Any]] = None,
        scheduled_at: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send a notification"""

        # Create notification message
        message = NotificationMessage(
            user_id=user_id,
            channel=channel,
            priority=priority,
            subject=subject,
            content=content,
            template_id=template_id,
            template_data=template_data or {},
            scheduled_at=scheduled_at,
            expires_at=expires_at,
            metadata=metadata or {}
        )

        # Check user preferences
        if not self._is_channel_enabled(user_id, channel):
            logger.info(f"Channel {channel} disabled for user {user_id}")
            message.status = NotificationStatus.CANCELLED
            return message.id

        # Apply template if specified
        if template_id and template_id in self.templates:
            message = await self._apply_template(message, template_id)

        # Process message through processors
        for processor in self.message_processors:
            try:
                message = await processor(message)
            except Exception as e:
                logger.error(f"Error in message processor: {e}")

        # Schedule or send immediately
        if scheduled_at and scheduled_at > datetime.now():
            self.pending_messages.append(message)
            logger.info(f"Notification {message.id} scheduled for {scheduled_at}")
        else:
            await self._deliver_message(message)

        return message.id

    async def send_bulk_notifications(
        self,
        notifications: List[Dict[str, Any]]
    ) -> List[str]:
        """Send multiple notifications in bulk"""

        notification_ids = []

        for notification_data in notifications:
            try:
                notification_id = await self.send_notification(**notification_data)
                notification_ids.append(notification_id)
            except Exception as e:
                logger.error(f"Failed to send bulk notification: {e}")

        return notification_ids

    async def _deliver_message(self, message: NotificationMessage):
        """Deliver a notification message"""

        if message.channel not in self.channels:
            logger.error(f"Unknown notification channel: {message.channel}")
            message.status = NotificationStatus.FAILED
            message.error_message = f"Unknown channel: {message.channel}"
            return

        # Check rate limiting
        if self._is_rate_limited(message.user_id, message.channel):
            logger.info(f"Rate limited for user {message.user_id} on channel {message.channel}")
            # Requeue for later delivery
            message.scheduled_at = datetime.now() + timedelta(minutes=5)
            self.pending_messages.append(message)
            return

        # Check expiration
        if message.expires_at and message.expires_at < datetime.now():
            logger.info(f"Notification {message.id} expired")
            message.status = NotificationStatus.CANCELLED
            return

        try:
            message.status = NotificationStatus.SENDING
            message.delivery_attempts += 1
            message.last_attempt = datetime.now()

            # Get channel instance
            channel = self.channels[message.channel]

            # Send notification
            success = await channel.send(
                user_id=message.user_id,
                subject=message.subject,
                content=message.content,
                metadata=message.metadata
            )

            if success:
                message.status = NotificationStatus.SENT
                message.delivered_at = datetime.now()

                # Create delivery report
                report = DeliveryReport(
                    notification_id=message.id,
                    channel=message.channel,
                    status=NotificationStatus.DELIVERED,
                    sent_at=message.last_attempt,
                    delivered_at=message.delivered_at,
                    metadata=message.metadata
                )
                self.delivery_history.append(report)

                logger.info(f"Notification {message.id} delivered via {message.channel}")
            else:
                message.status = NotificationStatus.FAILED
                message.error_message = "Channel delivery failed"

                # Retry logic
                if message.delivery_attempts < 3:
                    logger.info(f"Retrying notification {message.id} (attempt {message.delivery_attempts})")
                    await asyncio.sleep(2 ** message.delivery_attempts)  # Exponential backoff
                    await self._deliver_message(message)
                else:
                    logger.error(f"Notification {message.id} failed after {message.delivery_attempts} attempts")

        except Exception as e:
            logger.error(f"Error delivering notification {message.id}: {e}")
            message.status = NotificationStatus.FAILED
            message.error_message = str(e)

    async def _apply_template(self, message: NotificationMessage, template_id: str) -> NotificationMessage:
        """Apply template to notification message"""

        template = self.templates[template_id]

        try:
            # Merge template data with message metadata
            template_vars = {**message.template_data, **message.metadata}

            # Apply subject template
            if template.subject_template:
                message.subject = template.subject_template.format(**template_vars)

            # Apply content template
            if template.content_template:
                message.content = template.content_template.format(**template_vars)

            logger.info(f"Applied template {template_id} to notification {message.id}")

        except Exception as e:
            logger.error(f"Error applying template {template_id}: {e}")
            # Keep original subject/content if template fails

        return message

    def _is_channel_enabled(self, user_id: str, channel: str) -> bool:
        """Check if a notification channel is enabled for a user"""

        preferences = self.user_preferences.get(user_id, {})
        channel_preferences = preferences.get("channels", {})

        # Default to enabled if not specified
        return channel_preferences.get(channel, {}).get("enabled", True)

    def _is_rate_limited(self, user_id: str, channel: str) -> bool:
        """Check if user is rate limited for a channel"""

        rate_key = f"{user_id}_{channel}"

        if rate_key not in self.rate_limiters:
            return False

        rate_limiter = self.rate_limiters[rate_key]
        now = datetime.now()

        # Check different rate limits
        if "per_minute" in rate_limiter:
            last_minute = now - timedelta(minutes=1)
            if rate_limiter["last_sent"] > last_minute:
                if rate_limiter["count"] >= rate_limiter["per_minute"]:
                    return True

        if "per_hour" in rate_limiter:
            last_hour = now - timedelta(hours=1)
            if rate_limiter["last_sent"] > last_hour:
                if rate_limiter["count"] >= rate_limiter["per_hour"]:
                    return True

        return False

    def set_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Set user notification preferences"""
        self.user_preferences[user_id] = preferences
        logger.info(f"Updated notification preferences for user {user_id}")

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user notification preferences"""
        return self.user_preferences.get(user_id, {
            "channels": {
                "email": {"enabled": True, "quiet_hours": {"start": "22:00", "end": "08:00"}},
                "sms": {"enabled": True, "critical_only": True},
                "push": {"enabled": True},
                "in_app": {"enabled": True}
            },
            "frequency": {
                "max_per_hour": 10,
                "max_per_day": 50
            }
        })

    async def process_scheduled_messages(self):
        """Process scheduled messages for delivery"""

        now = datetime.now()
        messages_to_send = []

        # Find messages scheduled for now or past
        for message in self.pending_messages:
            if message.scheduled_at and message.scheduled_at <= now:
                messages_to_send.append(message)

        # Remove from pending and deliver
        for message in messages_to_send:
            self.pending_messages.remove(message)
            await self._deliver_message(message)

        if messages_to_send:
            logger.info(f"Processed {len(messages_to_send)} scheduled messages")

    def get_delivery_stats(self, user_id: Optional[str] = None,
                          time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get notification delivery statistics"""

        reports = self.delivery_history

        # Filter by user if specified
        if user_id:
            # This would need message_id to user_id mapping
            # For now, return overall stats
            pass

        # Filter by time range if specified
        if time_range:
            cutoff_time = datetime.now() - time_range
            reports = [r for r in reports if r.sent_at >= cutoff_time]

        # Calculate statistics
        total = len(reports)
        if total == 0:
            return {"total": 0}

        delivered = len([r for r in reports if r.status == NotificationStatus.DELIVERED])
        failed = len([r for r in reports if r.status == NotificationStatus.FAILED])

        # Channel breakdown
        channel_stats = {}
        for report in reports:
            channel = report.channel
            if channel not in channel_stats:
                channel_stats[channel] = {"total": 0, "delivered": 0, "failed": 0}

            channel_stats[channel]["total"] += 1
            if report.status == NotificationStatus.DELIVERED:
                channel_stats[channel]["delivered"] += 1
            elif report.status == NotificationStatus.FAILED:
                channel_stats[channel]["failed"] += 1

        return {
            "total": total,
            "delivered": delivered,
            "failed": failed,
            "delivery_rate": delivered / total if total > 0 else 0,
            "channel_breakdown": channel_stats,
            "time_range_hours": time_range.total_seconds() / 3600 if time_range else None
        }

    def add_message_processor(self, processor: callable):
        """Add a message processor function"""
        self.message_processors.append(processor)

    async def cleanup_old_messages(self, days_to_keep: int = 30):
        """Clean up old messages and reports"""

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        # Clean up delivery history
        self.delivery_history = [
            report for report in self.delivery_history
            if report.sent_at > cutoff_date
        ]

        # Clean up old pending messages
        self.pending_messages = [
            message for message in self.pending_messages
            if message.created_at > cutoff_date
        ]

        logger.info(f"Cleaned up messages older than {days_to_keep} days")

# Global notification service instance
notification_service = NotificationService()