"""
Phase 5: Integration Service
Integrates alerts, notifications, and escalation with the analysis engine
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime
import asyncio

from .alert_engine import alert_engine, Alert
from .escalation_manager import escalation_manager
from ..notifications.notification_service import notification_service
from ..analysis.risk_scoring_engine import ComprehensiveRiskAssessment

logger = logging.getLogger(__name__)

class AlertIntegrationService:
    """
    Integration service that connects the analysis engine with alerts,
    notifications, and escalation systems.
    """

    def __init__(self):
        """Initialize integration service"""

        # Register alert callbacks
        alert_engine.add_alert_callback(self._on_alert_triggered)

        # Register escalation callbacks
        escalation_manager.add_escalation_callback(self._on_escalation_completed)

        # Configure notification channels
        self._setup_notification_channels()

        # Initialize alert processing
        self._initialize_alert_processing()

        logger.info("Alert Integration Service initialized")

    def _setup_notification_channels(self):
        """Setup notification channels"""

        # Import channel configurations
        try:
            from ..core.config import get_config
            config = get_config()

            # Email channel
            email_config = {
                "enabled": True,
                "smtp_server": config.get("SMTP_SERVER", "localhost"),
                "smtp_port": config.get("SMTP_PORT", 587),
                "username": config.get("SMTP_USERNAME", ""),
                "password": config.get("SMTP_PASSWORD", ""),
                "from_email": config.get("FROM_EMAIL", "noreply@studysense.com"),
                "use_tls": True
            }

            from ..notifications.notification_channels import create_channel
            email_channel = create_channel("email", email_config)
            if email_channel:
                notification_service.register_channel("email", email_channel)

            # SMS channel
            sms_config = {
                "enabled": config.get("SMS_ENABLED", False),
                "provider": config.get("SMS_PROVIDER", "twilio"),
                "api_key": config.get("SMS_API_KEY", ""),
                "api_secret": config.get("SMS_API_SECRET", ""),
                "account_sid": config.get("TWILIO_ACCOUNT_SID", ""),
                "from_number": config.get("TWILIO_FROM_NUMBER", "")
            }

            sms_channel = create_channel("sms", sms_config)
            if sms_channel:
                notification_service.register_channel("sms", sms_channel)

            # Push notification channel
            push_config = {
                "enabled": True,
                "provider": "firebase",
                "server_key": config.get("FCM_SERVER_KEY", "")
            }

            push_channel = create_channel("push", push_config)
            if push_channel:
                notification_service.register_channel("push", push_channel)

            # In-app channel
            in_app_config = {"enabled": True}
            in_app_channel = create_channel("in_app", in_app_config)
            if in_app_channel:
                notification_service.register_channel("in_app", in_app_channel)

            logger.info("Notification channels configured")

        except Exception as e:
            logger.error(f"Error setting up notification channels: {e}")

    def _initialize_alert_processing(self):
        """Initialize alert processing"""

        # Add notification templates
        self._add_notification_templates()

        logger.info("Alert processing initialized")

    def _add_notification_templates(self):
        """Add notification templates"""

        from ..notifications.notification_service import NotificationTemplate

        # Crisis alert templates
        notification_service.add_template(NotificationTemplate(
            id="crisis_alert_email",
            name="Crisis Alert Email",
            channel="email",
            language="en",
            subject_template="🚨 CRITICAL: Immediate Support Required for {user_name}",
            content_template="""Dear {user_name},

We have detected that you may need immediate support. This is a crisis alert triggered by our monitoring system.

IMMEDIATE ACTIONS:
1. Call 988 (Suicide Prevention Lifeline)
2. Text HOME to 741741
3. Call 911 if you are in immediate danger

You are not alone. Help is available 24/7 and completely confidential.

With concern,
StudySense Support Team""",
            variables=["user_name"],
            metadata={"priority": "critical", "type": "crisis"}
        ))

        # High stress alert templates
        notification_service.add_template(NotificationTemplate(
            id="high_stress_push",
            name="High Stress Alert Push",
            channel="push",
            language="en",
            subject_template="High Stress Detected",
            content_template="Your stress levels appear to be elevated. Consider taking a break and using some stress management techniques.",
            variables=[],
            metadata={"priority": "high", "type": "stress"}
        ))

        # Moderate concern templates
        notification_service.add_template(NotificationTemplate(
            id="moderate_concern_email",
            name="Moderate Concern Email",
            channel="email",
            language="en",
            subject_template="Check-in: Your Wellness",
            content_template="""Hi {user_name},

We noticed some patterns in your recent activity that suggest you might be experiencing increased stress.

Here are some resources that might help:
• Campus Counseling Services
• Stress management techniques
• Time management strategies

Remember to take care of yourself and reach out if you need support.

Best regards,
StudySense Team""",
            variables=["user_name"],
            metadata={"priority": "medium", "type": "wellness"}
        ))

    async def process_risk_assessment(self, assessment: ComprehensiveRiskAssessment):
        """Process risk assessment and trigger appropriate alerts"""

        try:
            # Process through alert engine
            await alert_engine.process_risk_assessment(assessment)

            # Send baseline notifications for non-critical assessments
            if assessment.classification.risk_level in ["low", "mild"]:
                await self._send_wellness_notification(assessment)

        except Exception as e:
            logger.error(f"Error processing risk assessment for {assessment.user_id}: {e}")

    async def _send_wellness_notification(self, assessment: ComprehensiveRiskAssessment):
        """Send wellness notification for low-risk assessments"""

        try:
            # Determine notification content based on primary concerns
            primary_concerns = assessment.classification.primary_risk_factors
            content = self._generate_wellness_content(primary_concerns)

            # Send in-app notification
            await notification_service.send_notification(
                user_id=assessment.user_id,
                channel="in_app",
                subject="Wellness Check-in",
                content=content,
                priority="low",
                metadata={
                    "assessment_id": getattr(assessment, 'id', None),
                    "risk_level": assessment.classification.risk_level,
                    "type": "wellness"
                }
            )

        except Exception as e:
            logger.error(f"Error sending wellness notification: {e}")

    def _generate_wellness_content(self, primary_concerns: list) -> str:
        """Generate wellness content based on primary concerns"""

        content = "Here's a gentle reminder to take care of yourself today:\n\n"

        concern_mapping = {
            "academic": ["• Break study sessions into smaller chunks", "• Take regular breaks", "• Practice time management"],
            "stress": ["• Try deep breathing exercises", "• Take a short walk", "• Listen to calming music"],
            "sleep": ["• Maintain a consistent sleep schedule", "• Avoid screens before bedtime", "• Create a relaxing bedtime routine"],
            "social": ["• Reach out to a friend", "• Join a study group", "• Schedule social time"]
        }

        for concern in primary_concerns[:2]:  # Limit to top 2 concerns
            concern_lower = concern.lower()
            for key, suggestions in concern_mapping.items():
                if key in concern_lower:
                    content += f"Regarding {concern}:\n" + "\n".join(suggestions) + "\n\n"
                    break

        content += "Remember, it's okay to take things one day at a time."

        return content

    async def _on_alert_triggered(self, alert: Alert):
        """Handle alert triggered event"""

        try:
            logger.info(f"Alert triggered: {alert.title} for user {alert.user_id}")

            # Send notification for the alert
            await self._send_alert_notification(alert)

            # Process for escalation if needed
            if alert.metadata.get("escalation_enabled", True):
                user_context = {
                    "alert_severity": alert.severity.value,
                    "alert_context": alert.context,
                    "user_preferences": notification_service.get_user_preferences(alert.user_id)
                }

                await escalation_manager.process_alert_for_escalation(alert, user_context)

        except Exception as e:
            logger.error(f"Error handling alert triggered event: {e}")

    async def _send_alert_notification(self, alert: Alert):
        """Send notification for an alert"""

        try:
            # Get notification channels from alert metadata
            channels = alert.metadata.get("notification_channels", ["push"])

            # Determine notification priority
            priority_mapping = {
                "critical": "critical",
                "high": "high",
                "medium": "medium",
                "low": "low",
                "info": "low"
            }
            priority = priority_mapping.get(alert.severity.value, "medium")

            # Get user contact information
            user_preferences = notification_service.get_user_preferences(alert.user_id)
            user_metadata = {
                "email": user_preferences.get("email", f"user_{alert.user_id}@example.com"),
                "phone": user_preferences.get("phone", None),
                "device_tokens": user_preferences.get("device_tokens", [])
            }

            # Send notifications through all configured channels
            for channel in channels:
                try:
                    # Check if channel is enabled for this user
                    if not self._is_channel_enabled_for_user(alert.user_id, channel, alert.severity.value):
                        continue

                    await notification_service.send_notification(
                        user_id=alert.user_id,
                        channel=channel,
                        subject=alert.title,
                        content=alert.message,
                        priority=priority,
                        metadata={
                            "alert_id": alert.id,
                            "severity": alert.severity.value,
                            "risk_score": alert.risk_score,
                            "risk_level": alert.risk_level,
                            "type": "alert",
                            **user_metadata
                        }
                    )

                except Exception as e:
                    logger.error(f"Failed to send {channel} notification for alert {alert.id}: {e}")

        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")

    def _is_channel_enabled_for_user(self, user_id: str, channel: str, severity: str) -> bool:
        """Check if a channel is enabled for a user based on their preferences and alert severity"""

        try:
            preferences = notification_service.get_user_preferences(user_id)
            channel_prefs = preferences.get("channels", {}).get(channel, {})

            # Check if channel is enabled
            if not channel_prefs.get("enabled", True):
                return False

            # Check quiet hours
            quiet_hours = channel_prefs.get("quiet_hours")
            if quiet_hours:
                # Simple quiet hours check (in production would be more sophisticated)
                current_hour = datetime.now().hour
                start = int(quiet_hours.get("start", "22:00").split(":")[0])
                end = int(quiet_hours.get("end", "08:00").split(":")[0])

                if start <= end:  # Same day range
                    if start <= current_hour <= end:
                        # Allow critical alerts during quiet hours
                        return severity in ["critical", "high"]
                else:  # Crosses midnight
                    if current_hour >= start or current_hour <= end:
                        return severity in ["critical", "high"]

            # Check critical-only setting
            if channel_prefs.get("critical_only", False):
                return severity in ["critical", "high"]

            return True

        except Exception as e:
            logger.error(f"Error checking channel preferences: {e}")
            return True  # Default to enabled

    async def _on_escalation_completed(self, escalation_event):
        """Handle escalation completed event"""

        try:
            logger.info(f"Escalation completed: {escalation_event.id} for user {escalation_event.user_id}")

            # Send notification about escalation completion
            await notification_service.send_notification(
                user_id=escalation_event.user_id,
                channel="in_app",
                subject="Support Process Completed",
                content="The support process has been completed. We hope you're feeling better. Remember, support is always available when you need it.",
                priority="low",
                metadata={
                    "escalation_id": escalation_event.id,
                    "type": "escalation_completed"
                }
            )

        except Exception as e:
            logger.error(f"Error handling escalation completed event: {e}")

    async def process_scheduled_tasks(self):
        """Process scheduled tasks like notification delivery and cleanup"""

        try:
            # Process scheduled notifications
            await notification_service.process_scheduled_messages()

            # Clean up old messages and reports
            await notification_service.cleanup_old_messages(days_to_keep=30)

            # Process any pending escalations
            await self._process_pending_escalations()

        except Exception as e:
            logger.error(f"Error processing scheduled tasks: {e}")

    async def _process_pending_escalations(self):
        """Process pending escalations that need next steps"""

        try:
            current_time = datetime.now()

            for escalation in escalation_manager.active_escalations.values():
                if (escalation.status.value == "in_progress" and
                    escalation.next_step_time and
                    escalation.next_step_time <= current_time):

                    # Find the protocol and execute next step
                    protocol = escalation_manager.protocols.get(escalation.protocol_id)
                    if protocol:
                        next_step_index = len(escalation.steps_completed)
                        if next_step_index < len(protocol.steps):
                            await escalation_manager._execute_escalation_step(
                                escalation, protocol, next_step_index
                            )

        except Exception as e:
            logger.error(f"Error processing pending escalations: {e}")

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration service statistics"""

        return {
            "alert_engine_stats": alert_engine.get_policy_stats(),
            "escalation_manager_stats": escalation_manager.get_escalation_stats(),
            "notification_stats": notification_service.get_delivery_stats(),
            "active_integrations": {
                "alert_engine": bool(alert_engine.policies),
                "escalation_manager": bool(escalation_manager.protocols),
                "notification_service": bool(notification_service.channels)
            }
        }

# Global integration service instance
alert_integration_service = AlertIntegrationService()