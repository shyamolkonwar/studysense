"""
Phase 5: Notification Channels
Implementation of different notification delivery channels
"""

from typing import Dict, Any, Optional
import logging
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)

class NotificationChannel(ABC):
    """Base class for notification channels"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)

    @abstractmethod
    async def send(self, user_id: str, subject: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Send notification through this channel"""
        pass

    @abstractmethod
    async def validate_config(self) -> bool:
        """Validate channel configuration"""
        pass

class EmailChannel(NotificationChannel):
    """Email notification channel"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smtp_server = config.get("smtp_server", "localhost")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.from_email = config.get("from_email", "noreply@example.com")
        self.use_tls = config.get("use_tls", True)

    async def send(self, user_id: str, subject: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Send email notification"""

        if not self.enabled:
            logger.info("Email channel disabled")
            return False

        try:
            # Get user email from metadata
            to_email = metadata.get("email")
            if not to_email:
                logger.error(f"No email address found for user {user_id}")
                return False

            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = subject

            # Add HTML and plain text versions
            html_content = f"""
            <html>
            <body>
                <h2>{subject}</h2>
                <div>{content.replace('\n', '<br>')}</div>
                <br>
                <hr>
                <p><small>This is an automated message from the Mental Health & Study-Stress Analyzer.</small></p>
                <p><small>If you're experiencing a crisis, please call 988 or text HOME to 741741.</small></p>
            </body>
            </html>
            """

            msg.attach(MIMEText(content, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Email sent successfully to {to_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email to user {user_id}: {e}")
            return False

    async def validate_config(self) -> bool:
        """Validate email configuration"""
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
            return True
        except Exception as e:
            logger.error(f"Email configuration validation failed: {e}")
            return False

class SMSChannel(NotificationChannel):
    """SMS notification channel"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = config.get("provider", "twilio")
        self.api_key = config.get("api_key", "")
        self.api_secret = config.get("api_secret", "")
        self.from_number = config.get("from_number", "")
        self.account_sid = config.get("account_sid", "")

    async def send(self, user_id: str, subject: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Send SMS notification"""

        if not self.enabled:
            logger.info("SMS channel disabled")
            return False

        try:
            # Get user phone number from metadata
            to_number = metadata.get("phone")
            if not to_number:
                logger.error(f"No phone number found for user {user_id}")
                return False

            # Format SMS content (limit to 160 characters)
            sms_content = f"{subject}: {content}"
            if len(sms_content) > 160:
                sms_content = sms_content[:157] + "..."

            # Send SMS based on provider
            if self.provider.lower() == "twilio":
                success = await self._send_twilio_sms(to_number, sms_content)
            else:
                # Generic SMS provider implementation
                success = await self._send_generic_sms(to_number, sms_content)

            if success:
                logger.info(f"SMS sent successfully to {to_number}")
            return success

        except Exception as e:
            logger.error(f"Failed to send SMS to user {user_id}: {e}")
            return False

    async def _send_twilio_sms(self, to_number: str, content: str) -> bool:
        """Send SMS via Twilio"""

        try:
            from twilio.rest import Client
            from twilio.base.exceptions import TwilioRestException

            client = Client(self.account_sid, self.api_secret)

            message = client.messages.create(
                body=content,
                from_=self.from_number,
                to=to_number
            )

            logger.info(f"Twilio SMS sent with SID: {message.sid}")
            return True

        except ImportError:
            logger.error("Twilio library not installed")
            return False
        except TwilioRestException as e:
            logger.error(f"Twilio SMS error: {e}")
            return False

    async def _send_generic_sms(self, to_number: str, content: str) -> bool:
        """Send SMS via generic provider (placeholder)"""

        # This would be implemented based on the specific SMS provider
        logger.info(f"Generic SMS implementation would send to {to_number}: {content}")
        return True

    async def validate_config(self) -> bool:
        """Validate SMS configuration"""
        return bool(self.api_key and self.from_number)

class PushNotificationChannel(NotificationChannel):
    """Push notification channel"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = config.get("provider", "firebase")
        self.server_key = config.get("server_key", "")
        self.fcm_url = "https://fcm.googleapis.com/fcm/send"

    async def send(self, user_id: str, subject: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Send push notification"""

        if not self.enabled:
            logger.info("Push notification channel disabled")
            return False

        try:
            # Get device tokens from metadata
            device_tokens = metadata.get("device_tokens", [])
            if not device_tokens:
                logger.error(f"No device tokens found for user {user_id}")
                return False

            # Create push notification payload
            payload = {
                "notification": {
                    "title": subject,
                    "body": content,
                    "sound": "default",
                    "badge": 1
                },
                "data": {
                    "user_id": user_id,
                    "type": metadata.get("type", "general"),
                    "priority": metadata.get("priority", "medium")
                }
            }

            # Send push notifications
            if self.provider.lower() == "firebase":
                success = await self._send_fcm_notification(device_tokens, payload)
            else:
                success = await self._send_generic_push(device_tokens, payload)

            if success:
                logger.info(f"Push notification sent to {len(device_tokens)} devices")
            return success

        except Exception as e:
            logger.error(f"Failed to send push notification to user {user_id}: {e}")
            return False

    async def _send_fcm_notification(self, device_tokens: list, payload: dict) -> bool:
        """Send notification via Firebase Cloud Messaging"""

        try:
            import httpx

            headers = {
                "Authorization": f"key={self.server_key}",
                "Content-Type": "application/json"
            }

            # Send to each device token (batch sending could be optimized)
            success_count = 0
            for token in device_tokens:
                payload["to"] = token

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.fcm_url,
                        headers=headers,
                        json=payload,
                        timeout=10.0
                    )

                    if response.status_code == 200:
                        success_count += 1
                    else:
                        logger.error(f"FCM notification failed: {response.text}")

            return success_count > 0

        except ImportError:
            logger.error("httpx library not installed")
            return False
        except Exception as e:
            logger.error(f"FCM notification error: {e}")
            return False

    async def _send_generic_push(self, device_tokens: list, payload: dict) -> bool:
        """Send push notification via generic provider"""

        # Generic push notification implementation
        logger.info(f"Generic push notification would send to {len(device_tokens)} devices")
        return True

    async def validate_config(self) -> bool:
        """Validate push notification configuration"""
        return bool(self.server_key)

class InAppChannel(NotificationChannel):
    """In-app notification channel"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.storage = {}  # In-memory storage (would use database in production)

    async def send(self, user_id: str, subject: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Store in-app notification"""

        if not self.enabled:
            logger.info("In-app channel disabled")
            return False

        try:
            import uuid
            from datetime import datetime

            notification = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "subject": subject,
                "content": content,
                "metadata": metadata,
                "created_at": datetime.now().isoformat(),
                "read": False
            }

            # Store notification (would use database in production)
            if user_id not in self.storage:
                self.storage[user_id] = []
            self.storage[user_id].append(notification)

            # Keep only last 50 notifications per user
            if len(self.storage[user_id]) > 50:
                self.storage[user_id] = self.storage[user_id][-50:]

            logger.info(f"In-app notification stored for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store in-app notification for user {user_id}: {e}")
            return False

    async def validate_config(self) -> bool:
        """Validate in-app channel configuration"""
        return True

    def get_user_notifications(self, user_id: str, unread_only: bool = False) -> list:
        """Get notifications for a user"""

        if user_id not in self.storage:
            return []

        notifications = self.storage[user_id]

        if unread_only:
            notifications = [n for n in notifications if not n["read"]]

        return sorted(notifications, key=lambda x: x["created_at"], reverse=True)

    def mark_notification_read(self, user_id: str, notification_id: str) -> bool:
        """Mark a notification as read"""

        if user_id not in self.storage:
            return False

        for notification in self.storage[user_id]:
            if notification["id"] == notification_id:
                notification["read"] = True
                return True

        return False

    def mark_all_read(self, user_id: str) -> int:
        """Mark all notifications as read for a user"""

        if user_id not in self.storage:
            return 0

        count = 0
        for notification in self.storage[user_id]:
            if not notification["read"]:
                notification["read"] = True
                count += 1

        return count

class WebhookChannel(NotificationChannel):
    """Webhook notification channel"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get("webhook_url", "")
        self.secret = config.get("secret", "")
        self.timeout = config.get("timeout", 10)

    async def send(self, user_id: str, subject: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Send webhook notification"""

        if not self.enabled:
            logger.info("Webhook channel disabled")
            return False

        try:
            import httpx
            import hashlib
            import hmac

            # Prepare webhook payload
            payload = {
                "user_id": user_id,
                "subject": subject,
                "content": content,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }

            # Generate signature if secret is provided
            headers = {"Content-Type": "application/json"}
            if self.secret:
                signature = hmac.new(
                    self.secret.encode(),
                    json.dumps(payload).encode(),
                    hashlib.sha256
                ).hexdigest()
                headers["X-Webhook-Signature"] = f"sha256={signature}"

            # Send webhook
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.webhook_url,
                    headers=headers,
                    json=payload
                )

                if response.status_code in [200, 201, 202]:
                    logger.info(f"Webhook sent successfully to {self.webhook_url}")
                    return True
                else:
                    logger.error(f"Webhook failed with status {response.status_code}: {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Failed to send webhook for user {user_id}: {e}")
            return False

    async def validate_config(self) -> bool:
        """Validate webhook configuration"""
        return bool(self.webhook_url)

# Channel factory function
def create_channel(channel_type: str, config: Dict[str, Any]) -> Optional[NotificationChannel]:
    """Create a notification channel instance"""

    channels = {
        "email": EmailChannel,
        "sms": SMSChannel,
        "push": PushNotificationChannel,
        "in_app": InAppChannel,
        "webhook": WebhookChannel
    }

    channel_class = channels.get(channel_type.lower())
    if not channel_class:
        logger.error(f"Unknown channel type: {channel_type}")
        return None

    try:
        return channel_class(config)
    except Exception as e:
        logger.error(f"Failed to create channel {channel_type}: {e}")
        return None