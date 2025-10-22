"""
Phase 5: Alert Engine
Implements policy-driven alert rules and threshold management
"""

from typing import List, Dict, Any, Optional, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json

from ..analysis.risk_scoring_engine import ComprehensiveRiskAssessment

logger = logging.getLogger(__name__)

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    """Alert status tracking"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    SUPPRESSED = "suppressed"

@dataclass
class AlertRule:
    """Individual alert rule definition"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str  # Condition expression
    threshold: float
    time_window: int  # Minutes
    cooldown_period: int  # Minutes
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertPolicy:
    """Alert policy configuration"""
    id: str
    name: str
    description: str
    rules: List[AlertRule]
    notification_channels: List[str]
    escalation_enabled: bool = True
    rate_limiting: Dict[str, Any] = field(default_factory=dict)
    quiet_hours: Dict[str, Any] = field(default_factory=dict)
    user_consent_required: bool = True

@dataclass
class Alert:
    """Alert instance"""
    id: str
    rule_id: str
    user_id: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    message: str
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalated_at: Optional[datetime] = None
    risk_score: float = 0.0
    risk_level: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AlertEngine:
    """
    Alert engine for monitoring risk assessments and triggering alerts
    based on policy-driven rules and thresholds.
    """

    def __init__(self):
        """Initialize alert engine"""

        # Default alert policies
        self.policies: Dict[str, AlertPolicy] = {}
        self.rules: Dict[str, AlertRule] = {}

        # Active alerts tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: Dict[str, List[Alert]] = {}

        # Rate limiting and cooldowns
        self.rule_cooldowns: Dict[str, datetime] = {}
        self.notification_timestamps: Dict[str, datetime] = {}

        # Alert processing callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Initialize default policies
        self._initialize_default_policies()

        logger.info("Alert Engine initialized")

    def _initialize_default_policies(self):
        """Initialize default alert policies"""

        # Crisis detection policy
        crisis_rules = [
            AlertRule(
                id="crisis_risk_score",
                name="Crisis Risk Score",
                description="Trigger when risk score reaches crisis level",
                severity=AlertSeverity.CRITICAL,
                condition="risk_score >= 0.9",
                threshold=0.9,
                time_window=0,
                cooldown_period=60,  # 1 hour cooldown
                tags=["crisis", "immediate"]
            ),
            AlertRule(
                id="crisis_keywords",
                name="Crisis Keywords Detected",
                description="Trigger when crisis language is detected",
                severity=AlertSeverity.CRITICAL,
                condition="crisis_keywords_detected == true",
                threshold=1.0,
                time_window=0,
                cooldown_period=30,
                tags=["crisis", "language"]
            )
        ]

        self.add_policy(AlertPolicy(
            id="crisis_detection",
            name="Crisis Detection Policy",
            description="Immediate alert for crisis-level risk indicators",
            rules=crisis_rules,
            notification_channels=["email", "sms", "push"],
            escalation_enabled=True,
            user_consent_required=True
        ))

        # High stress policy
        high_stress_rules = [
            AlertRule(
                id="high_risk_score",
                name="High Risk Score",
                description="Trigger when risk score is high",
                severity=AlertSeverity.HIGH,
                condition="risk_score >= 0.75",
                threshold=0.75,
                time_window=15,  # 15 minutes
                cooldown_period=120,  # 2 hours
                tags=["high_stress", "academic"]
            ),
            AlertRule(
                id="rapid_deterioration",
                name="Rapid Risk Deterioration",
                description="Trigger when risk score increases rapidly",
                severity=AlertSeverity.HIGH,
                condition="risk_score_increase >= 0.3 AND risk_score >= 0.6",
                threshold=0.3,
                time_window=60,  # 1 hour
                cooldown_period=180,  # 3 hours
                tags=["deterioration", "trend"]
            ),
            AlertRule(
                id="severe_sleep_disruption",
                name="Severe Sleep Disruption",
                description="Trigger when sleep patterns are severely disrupted",
                severity=AlertSeverity.HIGH,
                condition="sleep_disruption >= 0.8 AND sleep_duration < 5",
                threshold=0.8,
                time_window=48,  # 48 hours
                cooldown_period=240,  # 4 hours
                tags=["sleep", "behavioral"]
            )
        ]

        self.add_policy(AlertPolicy(
            id="high_stress_monitoring",
            name="High Stress Monitoring Policy",
            description="Alert for high-stress situations requiring attention",
            rules=high_stress_rules,
            notification_channels=["email", "push"],
            escalation_enabled=True,
            rate_limiting={"max_per_hour": 3, "max_per_day": 10},
            user_consent_required=True
        ))

        # Moderate concern policy
        moderate_rules = [
            AlertRule(
                id="moderate_risk_score",
                name="Moderate Risk Score",
                description="Trigger when risk score is moderately elevated",
                severity=AlertSeverity.MEDIUM,
                condition="risk_score >= 0.55",
                threshold=0.55,
                time_window=30,  # 30 minutes
                cooldown_period=360,  # 6 hours
                tags=["moderate_stress"]
            ),
            AlertRule(
                id="social_withdrawal",
                name="Social Withdrawal Detected",
                description="Trigger when social withdrawal patterns are detected",
                severity=AlertSeverity.MEDIUM,
                condition="social_withdrawal >= 0.7 AND social_frequency < 0.5",
                threshold=0.7,
                time_window=72,  # 3 days
                cooldown_period=480,  # 8 hours
                tags=["social", "behavioral"]
            ),
            AlertRule(
                id="academic_pressure",
                name="High Academic Pressure",
                description="Trigger when academic pressure is consistently high",
                severity=AlertSeverity.MEDIUM,
                condition="academic_pressure >= 0.8 AND deadline_proximity <= 3",
                threshold=0.8,
                time_window=24,  # 1 day
                cooldown_period=360,  # 6 hours
                tags=["academic", "deadline"]
            )
        ]

        self.add_policy(AlertPolicy(
            id="moderate_concern_monitoring",
            name="Moderate Concern Monitoring Policy",
            description="Alert for moderate concerns requiring attention",
            rules=moderate_rules,
            notification_channels=["push", "email"],
            escalation_enabled=False,
            rate_limiting={"max_per_hour": 2, "max_per_day": 6},
            user_consent_required=True,
            quiet_hours={"start": "22:00", "end": "08:00", "timezone": "UTC"}
        ))

        # Wellness check policy
        wellness_rules = [
            AlertRule(
                id="low_activity",
                name="Low Activity Pattern",
                description="Trigger when activity patterns are unusually low",
                severity=AlertSeverity.LOW,
                condition="activity_regularity < 0.3 AND message_frequency < 0.2",
                threshold=0.3,
                time_window=48,  # 2 days
                cooldown_period=720,  # 12 hours
                tags=["wellness", "activity"]
            )
        ]

        self.add_policy(AlertPolicy(
            id="wellness_monitoring",
            name="Wellness Monitoring Policy",
            description="Gentle alerts for wellness check-ins",
            rules=wellness_rules,
            notification_channels=["push"],
            escalation_enabled=False,
            rate_limiting={"max_per_day": 2},
            user_consent_required=True,
            quiet_hours={"start": "21:00", "end": "09:00", "timezone": "UTC"}
        ))

    def add_policy(self, policy: AlertPolicy):
        """Add an alert policy"""
        self.policies[policy.id] = policy
        for rule in policy.rules:
            self.rules[rule.id] = rule
        logger.info(f"Added alert policy: {policy.name}")

    def remove_policy(self, policy_id: str):
        """Remove an alert policy"""
        if policy_id in self.policies:
            policy = self.policies[policy_id]
            for rule in policy.rules:
                self.rules.pop(rule.id, None)
            del self.policies[policy_id]
            logger.info(f"Removed alert policy: {policy.name}")

    async def process_risk_assessment(self, assessment: ComprehensiveRiskAssessment):
        """Process risk assessment and trigger alerts if needed"""

        try:
            user_id = assessment.user_id

            # Evaluate all active policies
            for policy in self.policies.values():
                if not policy.rules:
                    continue

                await self._evaluate_policy(policy, assessment)

        except Exception as e:
            logger.error(f"Error processing risk assessment for {assessment.user_id}: {e}")

    async def _evaluate_policy(self, policy: AlertPolicy, assessment: ComprehensiveRiskAssessment):
        """Evaluate a single policy against the assessment"""

        user_id = assessment.user_id

        for rule in policy.rules:
            if not rule.enabled:
                continue

            # Check cooldown period
            if self._is_rule_in_cooldown(rule.id):
                continue

            # Check quiet hours
            if self._is_in_quiet_hours(policy):
                continue

            # Check rate limiting
            if self._is_rate_limited(policy, user_id):
                continue

            # Evaluate rule condition
            if await self._evaluate_rule_condition(rule, assessment):
                await self._trigger_alert(rule, assessment, policy)

    async def _evaluate_rule_condition(self, rule: AlertRule, assessment: ComprehensiveRiskAssessment) -> bool:
        """Evaluate if a rule condition is met"""

        try:
            # Create evaluation context
            context = {
                "risk_score": assessment.classification.risk_score,
                "risk_level": assessment.classification.risk_level,
                "user_id": assessment.user_id,
                "timestamp": assessment.assessment_timestamp,
                # Component scores
                "sentiment_score": assessment.component_scores.get("sentiment", 0.0),
                "behavioral_score": assessment.component_scores.get("behavioral", 0.0),
                "academic_score": assessment.component_scores.get("academic", 0.0),
                "contextual_score": assessment.component_scores.get("contextual", 0.0),
            }

            # Add behavioral features if available
            if hasattr(assessment, 'behavioral_features') and assessment.behavioral_features:
                bf = assessment.behavioral_features
                context.update({
                    "sleep_disruption": bf.risk_indicators.get("sleep_disruption", 0.0),
                    "social_withdrawal": bf.risk_indicators.get("social_withdrawal", 0.0),
                    "social_frequency": bf.social.social_frequency if hasattr(bf, 'social') else 0.0,
                    "sleep_duration": bf.sleep.avg_sleep_duration if hasattr(bf, 'sleep') else 0.0,
                    "academic_pressure": bf.risk_indicators.get("academic_pressure", 0.0),
                    "deadline_proximity": bf.risk_indicators.get("deadline_pressure", 0.0),
                    "activity_regularity": bf.activity_patterns.activity_regularity if hasattr(bf, 'activity_patterns') else 0.0,
                    "message_frequency": 0.0  # Would need message history data
                })

            # Check for crisis keywords in text processing
            if hasattr(assessment, 'text_processing') and assessment.text_processing:
                tp = assessment.text_processing
                crisis_keywords = ["suicide", "kill myself", "end it all", "can't go on", "want to die"]
                text_content = " ".join([tp.stress_lexicon.stress_keywords])

                context["crisis_keywords_detected"] = any(
                    keyword in text_content.lower() for keyword in crisis_keywords
                )

            # Calculate trend-based features
            context["risk_score_increase"] = self._calculate_risk_increase(assessment)

            # Evaluate the condition
            return self._evaluate_condition_expression(rule.condition, context)

        except Exception as e:
            logger.error(f"Error evaluating rule condition for {rule.id}: {e}")
            return False

    def _evaluate_condition_expression(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition expression against context"""

        try:
            # Simple condition evaluator for basic expressions
            # In production, this would use a more robust expression parser

            # Replace variables with values
            eval_condition = condition
            for var, value in context.items():
                if isinstance(value, str):
                    eval_condition = eval_condition.replace(var, f"'{value}'")
                else:
                    eval_condition = eval_condition.replace(var, str(value))

            # Simple evaluation of basic conditions
            # This is a simplified version - production would use ast.parse with safety checks
            if ">=" in eval_condition:
                left, right = eval_condition.split(">=")
                return float(left.strip()) >= float(right.strip())
            elif "<=" in eval_condition:
                left, right = eval_condition.split("<=")
                return float(left.strip()) <= float(right.strip())
            elif ">" in eval_condition:
                left, right = eval_condition.split(">")
                return float(left.strip()) > float(right.strip())
            elif "<" in eval_condition:
                left, right = eval_condition.split("<")
                return float(left.strip()) < float(right.strip())
            elif "==" in eval_condition:
                left, right = eval_condition.split("==")
                return left.strip().strip("'\"") == right.strip().strip("'\"")
            elif "AND" in eval_condition:
                parts = eval_condition.split("AND")
                return all(self._evaluate_simple_condition(part.strip(), context) for part in parts)
            elif "OR" in eval_condition:
                parts = eval_condition.split("OR")
                return any(self._evaluate_simple_condition(part.strip(), context) for part in parts)

            return eval(eval_condition)

        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

    def _evaluate_simple_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a simple condition without logical operators"""
        return self._evaluate_condition_expression(condition, context)

    def _calculate_risk_increase(self, assessment: ComprehensiveRiskAssessment) -> float:
        """Calculate recent risk score increase"""

        user_id = assessment.user_id
        current_score = assessment.classification.risk_score

        # Get historical assessments
        if user_id in self.alert_history and len(self.alert_history[user_id]) > 0:
            # Get the most recent previous alert
            previous_alerts = sorted(
                self.alert_history[user_id],
                key=lambda x: x.triggered_at,
                reverse=True
            )

            for alert in previous_alerts[:5]:  # Check last 5 alerts
                if alert.risk_score > 0:
                    return current_score - alert.risk_score

        return 0.0

    async def _trigger_alert(self, rule: AlertRule, assessment: ComprehensiveRiskAssessment, policy: AlertPolicy):
        """Trigger an alert"""

        try:
            # Create alert
            alert_id = f"{assessment.user_id}_{rule.id}_{int(datetime.now().timestamp())}"

            alert = Alert(
                id=alert_id,
                rule_id=rule.id,
                user_id=assessment.user_id,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                title=self._generate_alert_title(rule, assessment),
                message=self._generate_alert_message(rule, assessment),
                triggered_at=datetime.now(),
                risk_score=assessment.classification.risk_score,
                risk_level=assessment.classification.risk_level,
                context={
                    "assessment_id": getattr(assessment, 'id', None),
                    "policy_id": policy.id,
                    "rule_name": rule.name,
                    "primary_concerns": assessment.classification.primary_risk_factors
                },
                metadata={
                    "tags": rule.tags,
                    "threshold": rule.threshold,
                    "notification_channels": policy.notification_channels,
                    "escalation_enabled": policy.escalation_enabled
                }
            )

            # Store alert
            self.active_alerts[alert_id] = alert

            if assessment.user_id not in self.alert_history:
                self.alert_history[assessment.user_id] = []
            self.alert_history[assessment.user_id].append(alert)

            # Set cooldown
            self.rule_cooldowns[rule.id] = datetime.now() + timedelta(minutes=rule.cooldown_period)

            # Trigger callbacks
            for callback in self.alert_callbacks:
                try:
                    await self._run_alert_callback(callback, alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

            logger.info(f"Alert triggered: {alert.title} for user {assessment.user_id}")

        except Exception as e:
            logger.error(f"Error triggering alert for rule {rule.id}: {e}")

    async def _run_alert_callback(self, callback: Callable[[Alert], None], alert: Alert):
        """Run alert callback safely"""
        if asyncio.iscoroutinefunction(callback):
            await callback(alert)
        else:
            callback(alert)

    def _generate_alert_title(self, rule: AlertRule, assessment: ComprehensiveRiskAssessment) -> str:
        """Generate alert title"""

        base_titles = {
            AlertSeverity.CRITICAL: "🚨 CRITICAL ALERT",
            AlertSeverity.HIGH: "⚠️ High Priority Alert",
            AlertSeverity.MEDIUM: "📋 Attention Required",
            AlertSeverity.LOW: "ℹ️ Wellness Check",
            AlertSeverity.INFO: "📊 Information"
        }

        prefix = base_titles.get(rule.severity, "🔔 Alert")

        if rule.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            return f"{prefix}: {rule.name}"
        else:
            return f"{prefix}: {rule.name}"

    def _generate_alert_message(self, rule: AlertRule, assessment: ComprehensiveRiskAssessment) -> str:
        """Generate alert message"""

        risk_level = assessment.classification.risk_level
        risk_score = assessment.classification.risk_score
        primary_concerns = assessment.classification.primary_risk_factors[:3]

        message = f"Risk Level: {risk_level.upper()} (Score: {risk_score:.2f})\n\n"

        if primary_concerns:
            message += f"Primary Concerns: {', '.join(primary_concerns)}\n\n"

        # Add specific guidance based on severity
        if rule.severity == AlertSeverity.CRITICAL:
            message += "Immediate attention is required. Please reach out to support services or emergency contacts."
        elif rule.severity == AlertSeverity.HIGH:
            message += "Please consider reaching out to counseling services or support resources."
        elif rule.severity == AlertSeverity.MEDIUM:
            message += "Consider reviewing your current stress management strategies and seeking support if needed."
        else:
            message += "This is a gentle reminder to check in on your well-being."

        return message

    def _is_rule_in_cooldown(self, rule_id: str) -> bool:
        """Check if rule is in cooldown period"""
        if rule_id not in self.rule_cooldowns:
            return False
        return datetime.now() < self.rule_cooldowns[rule_id]

    def _is_in_quiet_hours(self, policy: AlertPolicy) -> bool:
        """Check if current time is in quiet hours"""
        if not policy.quiet_hours:
            return False

        try:
            quiet_hours = policy.quiet_hours
            current_time = datetime.now()

            # Parse quiet hours
            start_time = datetime.strptime(quiet_hours["start"], "%H:%M").time()
            end_time = datetime.strptime(quiet_hours["end"], "%H:%M").time()
            current_time_only = current_time.time()

            # Check if current time is in quiet hours range
            if start_time <= end_time:
                # Same day range (e.g., 22:00 to 08:00 crosses midnight)
                return start_time <= current_time_only or current_time_only <= end_time
            else:
                # Normal range
                return start_time <= current_time_only <= end_time

        except Exception as e:
            logger.error(f"Error checking quiet hours: {e}")
            return False

    def _is_rate_limited(self, policy: AlertPolicy, user_id: str) -> bool:
        """Check if notifications are rate limited for this user"""

        if not policy.rate_limiting:
            return False

        try:
            rate_limits = policy.rate_limiting
            user_key = f"{policy.id}_{user_id}"

            if user_key not in self.notification_timestamps:
                return False

            last_notification = self.notification_timestamps[user_key]
            now = datetime.now()

            # Check hourly limit
            if "max_per_hour" in rate_limits:
                hour_ago = now - timedelta(hours=1)
                if last_notification > hour_ago:
                    # Count notifications in the last hour
                    # This is simplified - production would maintain proper counters
                    return True

            # Check daily limit
            if "max_per_day" in rate_limits:
                day_ago = now - timedelta(days=1)
                if last_notification > day_ago:
                    # Count notifications in the last day
                    # This is simplified - production would maintain proper counters
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking rate limiting: {e}")
            return False

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert"""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()

        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True

    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert"""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()

        # Move from active to history
        del self.active_alerts[alert_id]

        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True

    def get_active_alerts(self, user_id: Optional[str] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by user"""
        alerts = list(self.active_alerts.values())

        if user_id:
            alerts = [alert for alert in alerts if alert.user_id == user_id]

        return sorted(alerts, key=lambda x: x.triggered_at, reverse=True)

    def get_alert_history(self, user_id: str, limit: int = 50) -> List[Alert]:
        """Get alert history for a user"""
        if user_id not in self.alert_history:
            return []

        history = self.alert_history[user_id]
        return sorted(history, key=lambda x: x.triggered_at, reverse=True)[:limit]

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)

    def get_policy_stats(self) -> Dict[str, Any]:
        """Get policy and alert statistics"""

        stats = {
            "total_policies": len(self.policies),
            "total_rules": len(self.rules),
            "active_alerts": len(self.active_alerts),
            "policies": {}
        }

        for policy_id, policy in self.policies.items():
            policy_alerts = [alert for alert in self.active_alerts.values()
                           if alert.context.get("policy_id") == policy_id]

            stats["policies"][policy_id] = {
                "name": policy.name,
                "enabled_rules": len([r for r in policy.rules if r.enabled]),
                "total_rules": len(policy.rules),
                "active_alerts": len(policy_alerts),
                "severity_breakdown": {}
            }

            # Severity breakdown
            for alert in policy_alerts:
                severity = alert.severity.value
                stats["policies"][policy_id]["severity_breakdown"][severity] = \
                    stats["policies"][policy_id]["severity_breakdown"].get(severity, 0) + 1

        return stats

# Global alert engine instance
alert_engine = AlertEngine()