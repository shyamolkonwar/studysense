"""
Phase 5: Escalation Manager
Handles escalation protocols and crisis management
"""

from typing import List, Dict, Any, Optional, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json

from .alert_engine import Alert, AlertSeverity, AlertStatus
from ..notifications.notification_service import notification_service

logger = logging.getLogger(__name__)

class EscalationLevel(str, Enum):
    """Escalation levels"""
    LEVEL_1 = "level_1"  # Self-help resources
    LEVEL_2 = "level_2"  # Peer support
    LEVEL_3 = "level_3"  # Professional support
    LEVEL_4 = "level_4"  # Emergency services
    CRISIS = "crisis"    # Immediate crisis intervention

class EscalationStatus(str, Enum):
    """Escalation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class EscalationStep:
    """Individual escalation step"""
    level: EscalationLevel
    action: str
    target: str
    timeframe: int  # Minutes
    notification_channels: List[str]
    message_template: str
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EscalationProtocol:
    """Escalation protocol definition"""
    id: str
    name: str
    description: str
    trigger_conditions: List[str]
    steps: List[EscalationStep]
    max_escalation_level: EscalationLevel
    auto_escalate: bool = True
    quiet_hours: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EscalationPolicy:
    """Escalation policy configuration"""
    id: str
    name: str
    protocols: List[EscalationProtocol]
    user_consent_required: bool = True
    notification_channels: List[str] = field(default_factory=list)
    emergency_contacts: List[Dict[str, Any]] = field(default_factory=list)
    institutional_contacts: List[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class EscalationEvent:
    """Escalation event record"""
    id: str
    alert_id: str
    user_id: str
    protocol_id: str
    current_level: EscalationLevel
    status: EscalationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    steps_completed: List[str] = field(default_factory=list)
    next_step_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class EscalationManager:
    """
    Escalation manager for handling alert escalation and crisis management
    according to defined protocols and policies.
    """

    def __init__(self):
        """Initialize escalation manager"""

        # Escalation policies and protocols
        self.policies: Dict[str, EscalationPolicy] = {}
        self.protocols: Dict[str, EscalationProtocol] = {}

        # Active escalations
        self.active_escalations: Dict[str, EscalationEvent] = {}
        self.escalation_history: Dict[str, List[EscalationEvent]] = {}

        # Escalation callbacks
        self.escalation_callbacks: List[Callable[[EscalationEvent], None]] = []

        # Crisis resources
        self.crisis_resources = {
            "hotlines": [
                {"name": "National Suicide Prevention Lifeline", "number": "988"},
                {"name": "Crisis Text Line", "text": "HOME to 741741"},
                {"name": "Emergency", "number": "911"}
            ],
            "resources": [
                {
                    "name": "Find a Helpline",
                    "url": "https://findahelpline.com/",
                    "description": "Find crisis helplines in your country"
                },
                {
                    "name": "Counseling Resources",
                    "url": "https://www.counseling.org/find-a-counselor",
                    "description": "Find professional counseling services"
                }
            ]
        }

        # Initialize default escalation protocols
        self._initialize_default_protocols()

        logger.info("Escalation Manager initialized")

    def _initialize_default_protocols(self):
        """Initialize default escalation protocols"""

        # Crisis escalation protocol
        crisis_steps = [
            EscalationStep(
                level=EscalationLevel.CRISIS,
                action="immediate_intervention",
                target="user",
                timeframe=0,  # Immediate
                notification_channels=["sms", "email", "push"],
                message_template="CRISIS: Immediate support required. Please call 988 or text HOME to 741741.",
                required=True
            ),
            EscalationStep(
                level=EscalationLevel.CRISIS,
                action="emergency_contact",
                target="emergency_contacts",
                timeframe=5,  # 5 minutes
                notification_channels=["email", "sms"],
                message_template="CRISIS ALERT: {user_name} requires immediate attention. Please check on them immediately.",
                required=True
            ),
            EscalationStep(
                level=EscalationLevel.CRISIS,
                action="institutional_notification",
                target="institution",
                timeframe=15,  # 15 minutes
                notification_channels=["email", "webhook"],
                message_template="CRISIS ESCALATION: Student {user_id} requires immediate crisis intervention.",
                required=False  # Depends on user consent
            )
        ]

        self.add_protocol(EscalationProtocol(
            id="crisis_protocol",
            name="Crisis Intervention Protocol",
            description="Immediate escalation for crisis-level alerts",
            trigger_conditions=["risk_score >= 0.9", "crisis_keywords_detected", "suicidal_ideation"],
            steps=crisis_steps,
            max_escalation_level=EscalationLevel.CRISIS,
            auto_escalate=True,
            metadata={"priority": "critical", "bypass_quiet_hours": True}
        ))

        # High stress escalation protocol
        high_stress_steps = [
            EscalationStep(
                level=EscalationLevel.LEVEL_1,
                action="self_help_resources",
                target="user",
                timeframe=0,
                notification_channels=["push", "in_app"],
                message_template="High stress detected. Here are some self-help resources that may help: {resources}",
                required=True
            ),
            EscalationStep(
                level=EscalationLevel.LEVEL_2,
                action="peer_support_suggestion",
                target="user",
                timeframe=60,  # 1 hour
                notification_channels=["push", "email"],
                message_template="Consider reaching out to peer support or study groups. You're not alone in this.",
                required=False
            ),
            EscalationStep(
                level=EscalationLevel.LEVEL_3,
                action="professional_support",
                target="user",
                timeframe=240,  # 4 hours
                notification_channels=["email", "push"],
                message_template="Professional support is available. Consider scheduling a meeting with counseling services.",
                required=False
            )
        ]

        self.add_protocol(EscalationProtocol(
            id="high_stress_protocol",
            name="High Stress Escalation Protocol",
            description="Escalation for high-stress situations",
            trigger_conditions=["risk_score >= 0.75", "rapid_deterioration", "severe_sleep_disruption"],
            steps=high_stress_steps,
            max_escalation_level=EscalationLevel.LEVEL_3,
            auto_escalate=True,
            quiet_hours={"start": "22:00", "end": "08:00", "timezone": "UTC"}
        ))

        # Moderate concern escalation protocol
        moderate_steps = [
            EscalationStep(
                level=EscalationLevel.LEVEL_1,
                action="wellness_resources",
                target="user",
                timeframe=0,
                notification_channels=["push", "in_app"],
                message_template="Here are some wellness resources that might help: {resources}",
                required=True
            ),
            EscalationStep(
                level=EscalationLevel.LEVEL_2,
                action="check_in_reminder",
                target="user",
                timeframe=1440,  # 24 hours
                notification_channels=["push"],
                message_template="Just checking in to see how you're doing. Remember to take care of yourself.",
                required=False
            )
        ]

        self.add_protocol(EscalationProtocol(
            id="moderate_concern_protocol",
            name="Moderate Concern Protocol",
            description="Gentle escalation for moderate concerns",
            trigger_conditions=["risk_score >= 0.55", "social_withdrawal", "academic_pressure"],
            steps=moderate_steps,
            max_escalation_level=EscalationLevel.LEVEL_2,
            auto_escalate=False,  # Manual escalation only
            quiet_hours={"start": "21:00", "end": "09:00", "timezone": "UTC"}
        ))

        # Add default escalation policy
        self.add_policy(EscalationPolicy(
            id="default_escalation_policy",
            name="Default Escalation Policy",
            protocols=list(self.protocols.values()),
            user_consent_required=True,
            notification_channels=["email", "sms", "push"],
            emergency_contacts=[],
            institutional_contacts=[
                {
                    "type": "counseling_center",
                    "name": "Campus Counseling Services",
                    "email": "counseling@university.edu",
                    "phone": "555-0123"
                }
            ]
        ))

    def add_protocol(self, protocol: EscalationProtocol):
        """Add an escalation protocol"""
        self.protocols[protocol.id] = protocol
        logger.info(f"Added escalation protocol: {protocol.name}")

    def add_policy(self, policy: EscalationPolicy):
        """Add an escalation policy"""
        self.policies[policy.id] = policy
        logger.info(f"Added escalation policy: {policy.name}")

    async def process_alert_for_escalation(self, alert: Alert, user_context: Optional[Dict[str, Any]] = None):
        """Process an alert for potential escalation"""

        try:
            user_id = alert.user_id

            # Check if escalation is already active for this alert
            active_escalation = self._get_active_escalation_for_alert(alert.id)
            if active_escalation:
                logger.info(f"Escalation already active for alert {alert.id}")
                return

            # Find matching protocols
            matching_protocols = self._find_matching_protocols(alert, user_context)
            if not matching_protocols:
                logger.info(f"No escalation protocols match alert {alert.id}")
                return

            # Select the most appropriate protocol (highest priority)
            protocol = matching_protocols[0]

            # Check user consent if required
            policy = self._get_policy_for_protocol(protocol.id)
            if policy and policy.user_consent_required:
                if not self._has_user_consent(user_id, protocol.id):
                    logger.info(f"User {user_id} has not consented to escalation protocol {protocol.id}")
                    return

            # Start escalation
            await self._start_escalation(alert, protocol, user_context)

        except Exception as e:
            logger.error(f"Error processing alert {alert.id} for escalation: {e}")

    async def _start_escalation(self, alert: Alert, protocol: EscalationProtocol, user_context: Optional[Dict[str, Any]]):
        """Start an escalation process"""

        import uuid

        escalation_id = str(uuid.uuid4())

        escalation = EscalationEvent(
            id=escalation_id,
            alert_id=alert.id,
            user_id=alert.user_id,
            protocol_id=protocol.id,
            current_level=EscalationLevel.LEVEL_1,
            status=EscalationStatus.PENDING,
            started_at=datetime.now(),
            metadata={
                "alert_severity": alert.severity.value,
                "alert_title": alert.title,
                "protocol_name": protocol.name,
                "user_context": user_context or {}
            }
        )

        # Store escalation
        self.active_escalations[escalation_id] = escalation

        if alert.user_id not in self.escalation_history:
            self.escalation_history[alert.user_id] = []
        self.escalation_history[alert.user_id].append(escalation)

        # Start escalation process
        await self._execute_escalation_step(escalation, protocol, 0)

        logger.info(f"Started escalation {escalation_id} for alert {alert.id}")

    async def _execute_escalation_step(self, escalation: EscalationEvent, protocol: EscalationProtocol, step_index: int):
        """Execute an escalation step"""

        if step_index >= len(protocol.steps):
            await self._complete_escalation(escalation)
            return

        step = protocol.steps[step_index]
        escalation.status = EscalationStatus.IN_PROGRESS

        try:
            # Execute the step action
            success = await self._execute_step_action(escalation, step)

            if success:
                escalation.steps_completed.append(step.action)

                # Check if we should auto-escalate to next level
                if protocol.auto_escalate and step_index < len(protocol.steps) - 1:
                    next_step = protocol.steps[step_index + 1]
                    escalation.next_step_time = datetime.now() + timedelta(minutes=next_step.timeframe)
                    escalation.current_level = next_step.level

                    # Schedule next step
                    asyncio.create_task(
                        self._schedule_next_step(escalation, protocol, step_index + 1, next_step.timeframe)
                    )
                else:
                    await self._complete_escalation(escalation)
            else:
                escalation.status = EscalationStatus.FAILED
                logger.error(f"Escalation step {step.action} failed for escalation {escalation.id}")

        except Exception as e:
            logger.error(f"Error executing escalation step {step.action}: {e}")
            escalation.status = EscalationStatus.FAILED

    async def _execute_step_action(self, escalation: EscalationEvent, step: EscalationStep) -> bool:
        """Execute a specific escalation step action"""

        try:
            if step.action == "immediate_intervention":
                return await self._send_immediate_intervention(escalation, step)
            elif step.action == "emergency_contact":
                return await self._notify_emergency_contacts(escalation, step)
            elif step.action == "institutional_notification":
                return await self._notify_institutional_contacts(escalation, step)
            elif step.action == "self_help_resources":
                return await self._send_self_help_resources(escalation, step)
            elif step.action == "peer_support_suggestion":
                return await self._suggest_peer_support(escalation, step)
            elif step.action == "professional_support":
                return await self._offer_professional_support(escalation, step)
            elif step.action == "wellness_resources":
                return await self._send_wellness_resources(escalation, step)
            elif step.action == "check_in_reminder":
                return await self._schedule_check_in(escalation, step)
            else:
                logger.warning(f"Unknown escalation action: {step.action}")
                return False

        except Exception as e:
            logger.error(f"Error executing step action {step.action}: {e}")
            return False

    async def _send_immediate_intervention(self, escalation: EscalationEvent, step: EscalationStep) -> bool:
        """Send immediate crisis intervention"""

        # Send crisis resources to user
        crisis_message = self._build_crisis_message()

        success_count = 0
        for channel in step.notification_channels:
            try:
                await notification_service.send_notification(
                    user_id=escalation.user_id,
                    channel=channel,
                    subject="🚨 CRITICAL: Immediate Support Required",
                    content=crisis_message,
                    priority="critical",
                    metadata={
                        "escalation_id": escalation.id,
                        "type": "crisis_intervention"
                    }
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to send crisis intervention via {channel}: {e}")

        return success_count > 0

    async def _notify_emergency_contacts(self, escalation: EscalationEvent, step: EscalationStep) -> bool:
        """Notify emergency contacts"""

        policy = self._get_policy_for_protocol(escalation.protocol_id)
        if not policy or not policy.emergency_contacts:
            logger.warning("No emergency contacts configured")
            return False

        success_count = 0
        for contact in policy.emergency_contacts:
            try:
                message = self._build_emergency_contact_message(escalation, contact)

                # Send notification based on contact type
                if "email" in contact:
                    await notification_service.send_notification(
                        user_id=contact["email"],  # Using email as user_id for external contacts
                        channel="email",
                        subject=f"🚨 Crisis Alert: {escalation.metadata.get('alert_title', 'Student Support')}",
                        content=message,
                        priority="critical",
                        metadata={
                            "escalation_id": escalation.id,
                            "contact_type": "emergency",
                            "student_id": escalation.user_id
                        }
                    )
                    success_count += 1

                if "phone" in contact:
                    await notification_service.send_notification(
                        user_id=contact["phone"],
                        channel="sms",
                        subject="CRISIS ALERT",
                        content=message,
                        priority="critical",
                        metadata={
                            "escalation_id": escalation.id,
                            "contact_type": "emergency",
                            "student_id": escalation.user_id,
                            "phone": contact["phone"]
                        }
                    )
                    success_count += 1

            except Exception as e:
                logger.error(f"Failed to notify emergency contact {contact}: {e}")

        return success_count > 0

    async def _notify_institutional_contacts(self, escalation: EscalationEvent, step: EscalationStep) -> bool:
        """Notify institutional contacts"""

        policy = self._get_policy_for_protocol(escalation.protocol_id)
        if not policy or not policy.institutional_contacts:
            logger.warning("No institutional contacts configured")
            return False

        success_count = 0
        for contact in policy.institutional_contacts:
            try:
                message = self._build_institutional_notification_message(escalation, contact)

                if "email" in contact:
                    await notification_service.send_notification(
                        user_id=contact["email"],
                        channel="email",
                        subject=f"Student Crisis Escalation: {escalation.user_id}",
                        content=message,
                        priority="high",
                        metadata={
                            "escalation_id": escalation.id,
                            "contact_type": "institutional",
                            "student_id": escalation.user_id
                        }
                    )
                    success_count += 1

            except Exception as e:
                logger.error(f"Failed to notify institutional contact {contact}: {e}")

        return success_count > 0

    async def _send_self_help_resources(self, escalation: EscalationEvent, step: EscalationStep) -> bool:
        """Send self-help resources"""

        resources = [
            "• Practice deep breathing: 4 counts in, 4 counts out, 4 counts hold",
            "• Try the 5-4-3-2-1 grounding technique",
            "• Take a short walk or stretch",
            "• Listen to calming music",
            "• Write down your thoughts in a journal"
        ]

        message = f"Here are some immediate self-help strategies:\n\n" + "\n".join(resources)

        try:
            await notification_service.send_notification(
                user_id=escalation.user_id,
                channel=step.notification_channels[0],
                subject="Self-Help Resources for Stress",
                content=message,
                priority="medium",
                metadata={
                    "escalation_id": escalation.id,
                    "type": "self_help"
                }
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send self-help resources: {e}")
            return False

    async def _suggest_peer_support(self, escalation: EscalationEvent, step: EscalationStep) -> bool:
        """Suggest peer support options"""

        message = """Consider connecting with others who understand:

• Join a study group for academic support
• Reach out to classmates you trust
• Talk to a friend or family member
• Join campus clubs or organizations
• Consider peer mentoring programs

Remember, you're not alone in this journey."""

        try:
            await notification_service.send_notification(
                user_id=escalation.user_id,
                channel=step.notification_channels[0],
                subject="Peer Support Options",
                content=message,
                priority="medium",
                metadata={
                    "escalation_id": escalation.id,
                    "type": "peer_support"
                }
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send peer support suggestion: {e}")
            return False

    async def _offer_professional_support(self, escalation: EscalationEvent, step: EscalationStep) -> bool:
        """Offer professional support options"""

        message = """Professional support is available and can make a real difference:

• Campus Counseling Services: Free and confidential
• Academic Advisors: Help with academic stress
• Health Services: Medical and mental health support
• Online therapy options: Flexible and accessible
• Support groups: Connect with others facing similar challenges

Taking this step shows strength, not weakness."""

        try:
            await notification_service.send_notification(
                user_id=escalation.user_id,
                channel=step.notification_channels[0],
                subject="Professional Support Available",
                content=message,
                priority="medium",
                metadata={
                    "escalation_id": escalation.id,
                    "type": "professional_support"
                }
            )
            return True
        except Exception as e:
            logger.error(f"Failed to offer professional support: {e}")
            return False

    async def _send_wellness_resources(self, escalation: EscalationEvent, step: EscalationStep) -> bool:
        """Send general wellness resources"""

        message = """Here are some wellness resources to support your mental health:

📚 Educational Resources:
• Mental health websites and articles
• Apps for meditation and mindfulness
• Online stress management courses

🏃‍♂️ Physical Wellness:
• Exercise and movement benefits
• Nutrition and mental health
• Sleep hygiene tips

🧘‍♀️ Mindfulness Practices:
• Guided meditation apps
• Breathing exercises
• Progressive muscle relaxation

💬 Support Options:
• Peer support groups
• Online communities
• Crisis helplines"""

        try:
            await notification_service.send_notification(
                user_id=escalation.user_id,
                channel=step.notification_channels[0],
                subject="Wellness Resources",
                content=message,
                priority="low",
                metadata={
                    "escalation_id": escalation.id,
                    "type": "wellness_resources"
                }
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send wellness resources: {e}")
            return False

    async def _schedule_check_in(self, escalation: EscalationEvent, step: EscalationStep) -> bool:
        """Schedule a check-in reminder"""

        try:
            await notification_service.send_notification(
                user_id=escalation.user_id,
                channel=step.notification_channels[0],
                subject="Just Checking In",
                content="How are you doing today? Remember to take things one day at a time and be kind to yourself.",
                priority="low",
                scheduled_at=datetime.now() + timedelta(hours=24),
                metadata={
                    "escalation_id": escalation.id,
                    "type": "check_in"
                }
            )
            return True
        except Exception as e:
            logger.error(f"Failed to schedule check-in: {e}")
            return False

    async def _schedule_next_step(self, escalation: EscalationEvent, protocol: EscalationProtocol, step_index: int, delay_minutes: int):
        """Schedule the next escalation step"""

        await asyncio.sleep(delay_minutes * 60)  # Convert minutes to seconds

        if escalation.id in self.active_escalations and escalation.status == EscalationStatus.IN_PROGRESS:
            await self._execute_escalation_step(escalation, protocol, step_index)

    async def _complete_escalation(self, escalation: EscalationEvent):
        """Complete an escalation process"""

        escalation.status = EscalationStatus.COMPLETED
        escalation.completed_at = datetime.now()

        # Remove from active escalations
        if escalation.id in self.active_escalations:
            del self.active_escalations[escalation.id]

        # Trigger callbacks
        for callback in self.escalation_callbacks:
            try:
                await self._run_escalation_callback(callback, escalation)
            except Exception as e:
                logger.error(f"Error in escalation callback: {e}")

        logger.info(f"Completed escalation {escalation.id}")

    async def _run_escalation_callback(self, callback: Callable[[EscalationEvent], None], escalation: EscalationEvent):
        """Run escalation callback safely"""
        if asyncio.iscoroutinefunction(callback):
            await callback(escalation)
        else:
            callback(escalation)

    def _find_matching_protocols(self, alert: Alert, user_context: Optional[Dict[str, Any]]) -> List[EscalationProtocol]:
        """Find protocols that match the alert conditions"""

        matching_protocols = []

        for protocol in self.protocols.values():
            if self._protocol_matches_alert(protocol, alert, user_context):
                matching_protocols.append(protocol)

        # Sort by priority (crisis protocols first)
        priority_order = {
            "crisis_protocol": 0,
            "high_stress_protocol": 1,
            "moderate_concern_protocol": 2
        }

        return sorted(matching_protocols, key=lambda p: priority_order.get(p.id, 999))

    def _protocol_matches_alert(self, protocol: EscalationProtocol, alert: Alert, user_context: Optional[Dict[str, Any]]) -> bool:
        """Check if a protocol matches the alert conditions"""

        # Simple condition matching - in production would be more sophisticated
        conditions = protocol.trigger_conditions

        for condition in conditions:
            if "risk_score" in condition:
                # Extract threshold from condition like "risk_score >= 0.9"
                if ">=" in condition:
                    threshold = float(condition.split(">=")[1].strip())
                    if alert.risk_score >= threshold:
                        return True
                elif ">" in condition:
                    threshold = float(condition.split(">")[1].strip())
                    if alert.risk_score > threshold:
                        return True

            # Check for crisis keywords in alert metadata
            if "crisis_keywords" in condition and alert.metadata:
                alert_context = alert.metadata.get("context", {})
                if "crisis_detected" in alert_context:
                    return True

            # Check alert severity
            if "severity" in condition and alert.severity.value in condition:
                return True

        return False

    def _get_policy_for_protocol(self, protocol_id: str) -> Optional[EscalationPolicy]:
        """Get the policy that contains a protocol"""
        for policy in self.policies.values():
            if any(p.id == protocol_id for p in policy.protocols):
                return policy
        return None

    def _get_active_escalation_for_alert(self, alert_id: str) -> Optional[EscalationEvent]:
        """Get active escalation for an alert"""
        for escalation in self.active_escalations.values():
            if escalation.alert_id == alert_id:
                return escalation
        return None

    def _has_user_consent(self, user_id: str, protocol_id: str) -> bool:
        """Check if user has consented to escalation protocol"""
        # In production, this would check user preferences
        # For now, assume consent is granted
        return True

    def _build_crisis_message(self) -> str:
        """Build crisis intervention message"""
        hotlines = self.crisis_resources["hotlines"]

        message = "🚨 IMMEDIATE SUPPORT REQUIRED\n\n"
        message += "If you're in crisis, please reach out now:\n\n"

        for hotline in hotlines:
            if "number" in hotline:
                message += f"• {hotline['name']}: {hotline['number']}\n"
            if "text" in hotline:
                message += f"• {hotline['name']}: Text {hotline['text']}\n"

        message += "\nYour life matters. Help is available 24/7."

        return message

    def _build_emergency_contact_message(self, escalation: EscalationEvent, contact: Dict[str, Any]) -> str:
        """Build message for emergency contacts"""
        return f"""URGENT ALERT: Student Support Needed

Student ID: {escalation.user_id}
Alert: {escalation.metadata.get('alert_title', 'High Risk Detected')}
Time: {escalation.started_at.strftime('%Y-%m-%d %H:%M')}

The student has triggered a crisis alert and may need immediate support. Please check on them as soon as possible.

If you believe they are in immediate danger, call emergency services (911).

Campus Counseling: 555-0123
Emergency Services: 911"""

    def _build_institutional_notification_message(self, escalation: EscalationEvent, contact: Dict[str, Any]) -> str:
        """Build message for institutional contacts"""
        return f"""STUDENT CRISIS ESCALATION NOTIFICATION

Student ID: {escalation.user_id}
Alert: {escalation.metadata.get('alert_title', 'High Risk Detected')}
Escalation Started: {escalation.started_at.strftime('%Y-%m-%d %H:%M')}
Current Level: {escalation.current_level.value}

This is an automated notification following a crisis alert escalation. The student has been provided with immediate resources and their emergency contacts have been notified.

Please follow institutional protocols for student crisis intervention.

Confidentiality: This notification is for official use only."""

    def get_active_escalations(self, user_id: Optional[str] = None) -> List[EscalationEvent]:
        """Get active escalations, optionally filtered by user"""
        escalations = list(self.active_escalations.values())

        if user_id:
            escalations = [e for e in escalations if e.user_id == user_id]

        return sorted(escalations, key=lambda x: x.started_at, reverse=True)

    def get_escalation_history(self, user_id: str, limit: int = 50) -> List[EscalationEvent]:
        """Get escalation history for a user"""
        if user_id not in self.escalation_history:
            return []

        history = self.escalation_history[user_id]
        return sorted(history, key=lambda x: x.started_at, reverse=True)[:limit]

    def cancel_escalation(self, escalation_id: str, cancelled_by: str = "system") -> bool:
        """Cancel an active escalation"""
        if escalation_id not in self.active_escalations:
            return False

        escalation = self.active_escalations[escalation_id]
        escalation.status = EscalationStatus.CANCELLED
        escalation.completed_at = datetime.now()

        # Remove from active escalations
        del self.active_escalations[escalation_id]

        logger.info(f"Escalation {escalation_id} cancelled by {cancelled_by}")
        return True

    def add_escalation_callback(self, callback: Callable[[EscalationEvent], None]):
        """Add callback for escalation events"""
        self.escalation_callbacks.append(callback)

    def get_escalation_stats(self) -> Dict[str, Any]:
        """Get escalation statistics"""

        stats = {
            "total_protocols": len(self.protocols),
            "total_policies": len(self.policies),
            "active_escalations": len(self.active_escalations),
            "escalations_by_level": {},
            "escalations_by_status": {}
        }

        # Count by level and status
        for escalation in self.active_escalations.values():
            level = escalation.current_level.value
            status = escalation.status.value

            stats["escalations_by_level"][level] = stats["escalations_by_level"].get(level, 0) + 1
            stats["escalations_by_status"][status] = stats["escalations_by_status"].get(status, 0) + 1

        return stats

# Global escalation manager instance
escalation_manager = EscalationManager()