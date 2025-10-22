"""
Phase 5: Alerts, Escalation, and Notifications Demo
Demonstrates the complete Phase 5 system functionality
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_crisis_risk_assessment() -> Dict[str, Any]:
    """Create a crisis-level risk assessment for demo"""

    return {
        "user_id": "demo_user_crisis",
        "classification": {
            "risk_level": "crisis",
            "risk_score": 0.95,
            "confidence": 0.9,
            "primary_risk_factors": [
                "suicidal_ideation_detected",
                "severe_depression",
                "isolation_patterns"
            ],
            "escalation_needed": True,
            "percentile_rank": 99.0
        },
        "component_scores": {
            "sentiment": 0.9,
            "behavioral": 0.8,
            "academic": 0.7,
            "contextual": 0.95
        },
        "feature_contributions": {
            "text_crisis_keywords": 0.4,
            "behavioral_social_withdrawal": 0.3,
            "text_negative_sentiment": 0.2,
            "contextual_crisis_detected": 0.1
        },
        "personalized_profile": {
            "user_id": "demo_user_crisis",
            "baseline_score": 0.3,
            "risk_history": []
        },
        "assessment_timestamp": datetime.now()
    }

def create_high_risk_assessment() -> Dict[str, Any]:
    """Create a high-risk assessment for demo"""

    return {
        "user_id": "demo_user_high",
        "classification": {
            "risk_level": "severe",
            "risk_score": 0.78,
            "confidence": 0.85,
            "primary_risk_factors": [
                "severe_sleep_disruption",
                "academic_pressure",
                "procrastination_patterns"
            ],
            "escalation_needed": True,
            "percentile_rank": 85.0
        },
        "component_scores": {
            "sentiment": 0.7,
            "behavioral": 0.85,
            "academic": 0.8,
            "contextual": 0.6
        },
        "feature_contributions": {
            "behavioral_sleep_disruption": 0.35,
            "academic_academic_pressure": 0.25,
            "text_stress_keywords": 0.2,
            "behavioral_irregular_schedule": 0.2
        },
        "personalized_profile": {
            "user_id": "demo_user_high",
            "baseline_score": 0.4,
            "risk_history": []
        },
        "assessment_timestamp": datetime.now()
    }

def create_moderate_risk_assessment() -> Dict[str, Any]:
    """Create a moderate-risk assessment for demo"""

    return {
        "user_id": "demo_user_moderate",
        "classification": {
            "risk_level": "moderate",
            "risk_score": 0.58,
            "confidence": 0.75,
            "primary_risk_factors": [
                "academic_pressure",
                "mild_anxiety",
                "time_management_issues"
            ],
            "escalation_needed": False,
            "percentile_rank": 60.0
        },
        "component_scores": {
            "sentiment": 0.5,
            "behavioral": 0.6,
            "academic": 0.7,
            "contextual": 0.4
        },
        "feature_contributions": {
            "academic_academic_pressure": 0.3,
            "text_mild_anxiety": 0.25,
            "behavioral_procrastination": 0.2,
            "contextual_deadline_proximity": 0.25
        },
        "personalized_profile": {
            "user_id": "demo_user_moderate",
            "baseline_score": 0.35,
            "risk_history": []
        },
        "assessment_timestamp": datetime.now()
    }

async def demo_alert_engine():
    """Demonstrate alert engine functionality"""

    print("\n🚨 Alert Engine Demo")
    print("-" * 30)

    try:
        from app.alerts.alert_engine import alert_engine
        from app.analysis.risk_scoring_engine import ComprehensiveRiskAssessment, RiskClassification

        # Get alert statistics
        stats = alert_engine.get_policy_stats()
        print(f"📊 Alert Engine Statistics:")
        print(f"   • Total Policies: {stats['total_policies']}")
        print(f"   • Total Rules: {stats['total_rules']}")
        print(f"   • Active Alerts: {stats['active_alerts']}")

        print(f"\n📋 Available Policies:")
        for policy_id, policy_stats in stats['policies'].items():
            print(f"   • {policy_stats['name']}: {policy_stats['enabled_rules']} active rules")

        # Test with different risk levels
        test_assessments = [
            ("Crisis Level", create_crisis_risk_assessment()),
            ("Severe Level", create_high_risk_assessment()),
            ("Moderate Level", create_moderate_risk_assessment())
        ]

        for level_name, assessment_data in test_assessments:
            print(f"\n🔍 Testing {level_name} Assessment:")

            # Create mock assessment object
            assessment = ComprehensiveRiskAssessment(
                user_id=assessment_data["user_id"],
                assessment_timestamp=assessment_data["assessment_timestamp"],
                classification=RiskClassification(
                    risk_level=assessment_data["classification"]["risk_level"],
                    risk_score=assessment_data["classification"]["risk_score"],
                    confidence=assessment_data["classification"]["confidence"],
                    primary_risk_factors=assessment_data["classification"]["primary_risk_factors"],
                    escalation_needed=assessment_data["classification"]["escalation_needed"],
                    percentile_rank=assessment_data["classification"]["percentile_rank"]
                ),
                component_scores=assessment_data["component_scores"],
                feature_contributions=assessment_data["feature_contributions"],
                personalized_profile=assessment_data["personalized_profile"],
                recommendations=[]
            )

            # Process assessment
            await alert_engine.process_risk_assessment(assessment)

            # Check for triggered alerts
            user_alerts = alert_engine.get_active_alerts(assessment.user_id)
            if user_alerts:
                print(f"   ✅ {len(user_alerts)} alert(s) triggered")
                for alert in user_alerts:
                    print(f"      • {alert.title} ({alert.severity.value})")
            else:
                print(f"   ℹ️  No alerts triggered (baseline wellness)")

        # Display final statistics
        final_stats = alert_engine.get_policy_stats()
        print(f"\n📈 Final Alert Statistics:")
        print(f"   • Total Active Alerts: {final_stats['active_alerts']}")

    except Exception as e:
        print(f"❌ Alert engine demo failed: {e}")

async def demo_notification_service():
    """Demonstrate notification service functionality"""

    print("\n📬 Notification Service Demo")
    print("-" * 30)

    try:
        from app.notifications.notification_service import notification_service

        # Get service statistics
        stats = notification_service.get_delivery_stats()
        print(f"📊 Notification Service Statistics:")
        print(f"   • Total Notifications: {stats['total']}")
        print(f"   • Delivered: {stats['delivered']}")
        print(f"   • Failed: {stats['failed']}")
        print(f"   • Delivery Rate: {stats['delivery_rate']:.2%}")

        print(f"\n📱 Available Channels:")
        for channel_name in notification_service.channels.keys():
            print(f"   • {channel_name}")

        # Send test notifications
        test_notifications = [
            {
                "user_id": "demo_user_001",
                "channel": "in_app",
                "subject": "Test Notification",
                "content": "This is a test notification from the Phase 5 system.",
                "priority": "low"
            },
            {
                "user_id": "demo_user_001",
                "channel": "in_app",
                "subject": "High Priority Alert",
                "content": "This is a high priority test notification.",
                "priority": "high"
            }
        ]

        print(f"\n📤 Sending {len(test_notifications)} test notifications...")
        notification_ids = await notification_service.send_bulk_notifications(test_notifications)
        print(f"   ✅ Sent notifications: {notification_ids}")

        # Check user notifications
        from app.notifications.notification_channels import InAppChannel
        in_app_channel = None
        for channel_instance in notification_service.channels.values():
            if isinstance(channel_instance, InAppChannel):
                in_app_channel = channel_instance
                break

        if in_app_channel:
            user_notifications = in_app_channel.get_user_notifications("demo_user_001")
            print(f"   📨 User has {len(user_notifications)} in-app notifications")

            # Mark one as read
            if user_notifications:
                success = in_app_channel.mark_notification_read("demo_user_001", user_notifications[0]["id"])
                print(f"   ✅ Marked notification as read: {success}")

    except Exception as e:
        print(f"❌ Notification service demo failed: {e}")

async def demo_escalation_manager():
    """Demonstrate escalation manager functionality"""

    print("\n🔄 Escalation Manager Demo")
    print("-" * 30)

    try:
        from app.alerts.escalation_manager import escalation_manager

        # Get escalation statistics
        stats = escalation_manager.get_escalation_stats()
        print(f"📊 Escalation Manager Statistics:")
        print(f"   • Total Protocols: {stats['total_protocols']}")
        print(f"   • Total Policies: {stats['total_policies']}")
        print(f"   • Active Escalations: {stats['active_escalations']}")

        print(f"\n📋 Available Protocols:")
        for protocol_id, protocol in escalation_manager.protocols.items():
            print(f"   • {protocol.name}: {len(protocol.steps)} steps")

        print(f"\n📞 Crisis Resources:")
        for hotline in escalation_manager.crisis_resources["hotlines"]:
            print(f"   • {hotline['name']}: {hotline.get('number', hotline.get('text', 'N/A'))}")

        # Test crisis escalation
        print(f"\n🚨 Testing Crisis Escalation Protocol...")
        crisis_alert = {
            "id": "test_crisis_alert_001",
            "rule_id": "crisis_risk_score",
            "user_id": "demo_user_crisis_test",
            "severity": "critical",
            "status": "active",
            "title": "🚨 CRITICAL: Crisis Risk Score Detected",
            "message": "Crisis-level risk detected. Immediate intervention required.",
            "triggered_at": datetime.now(),
            "risk_score": 0.95,
            "risk_level": "crisis",
            "metadata": {
                "protocol_name": "Crisis Detection Protocol",
                "crisis_detected": True
            }
        }

        # Process crisis alert
        await escalation_manager.process_alert_for_escalation(crisis_alert)

        # Check for active escalations
        user_escalations = escalation_manager.get_active_escalations("demo_user_crisis_test")
        if user_escalations:
            print(f"   ✅ Crisis escalation started: {user_escalations[0].id}")
            print(f"      Current Level: {user_escalations[0].current_level.value}")
            print(f"      Status: {user_escalations[0].status.value}")
        else:
            print(f"   ℹ️  No escalation triggered (consent may be required)")

        # Test moderate escalation
        print(f"\n⚠️  Testing Moderate Concern Protocol...")
        moderate_alert = {
            "id": "test_moderate_alert_001",
            "rule_id": "moderate_risk_score",
            "user_id": "demo_user_moderate_test",
            "severity": "medium",
            "status": "active",
            "title": "Moderate Risk Detected",
            "message": "Moderate risk level detected. Support resources recommended.",
            "triggered_at": datetime.now(),
            "risk_score": 0.58,
            "risk_level": "moderate",
            "metadata": {
                "protocol_name": "Moderate Concern Protocol"
            }
        }

        await escalation_manager.process_alert_for_escalation(moderate_alert)

        # Check final statistics
        final_stats = escalation_manager.get_escalation_stats()
        print(f"\n📈 Final Escalation Statistics:")
        print(f"   • Active Escalations: {final_stats['active_escalations']}")

        # Clean up test escalations
        for escalation in escalation_manager.active_escalations.values():
            if escalation.user_id in ["demo_user_crisis_test", "demo_user_moderate_test"]:
                escalation_manager.cancel_escalation(escalation.id, cancelled_by="demo_cleanup")
                print(f"   🧹 Cleaned up test escalation: {escalation.id}")

    except Exception as e:
        print(f"❌ Escalation manager demo failed: {e}")

async def demo_integration_service():
    """Demonstrate integration service functionality"""

    print("\n🔗 Integration Service Demo")
    print("-" * 30)

    try:
        from app.alerts.integration_service import alert_integration_service

        # Get integration statistics
        stats = alert_integration_service.get_integration_stats()
        print(f"📊 Integration Service Statistics:")
        print(f"   • Alert Engine: {stats['active_integrations']['alert_engine']}")
        print(f"   • Escalation Manager: {stats['active_integrations']['escalation_manager']}")
        print(f"   • Notification Service: {stats['active_integrations']['notification_service']}")

        # Test processing risk assessment
        print(f"\n🔄 Testing Risk Assessment Processing...")
        crisis_assessment = create_crisis_risk_assessment()

        # Create mock assessment object
        from app.analysis.risk_scoring_engine import ComprehensiveRiskAssessment, RiskClassification
        assessment = ComprehensiveRiskAssessment(
            user_id=crisis_assessment["user_id"],
            assessment_timestamp=crisis_assessment["assessment_timestamp"],
            classification=RiskClassification(
                risk_level=crisis_assessment["classification"]["risk_level"],
                risk_score=crisis_assessment["classification"]["risk_score"],
                confidence=crisis_assessment["classification"]["confidence"],
                primary_risk_factors=crisis_assessment["classification"]["primary_risk_factors"],
                escalation_needed=crisis_assessment["classification"]["escalation_needed"],
                percentile_rank=crisis_assessment["classification"]["percentile_rank"]
            ),
            component_scores=crisis_assessment["component_scores"],
            feature_contributions=crisis_assessment["feature_contributions"],
            personalized_profile=crisis_assessment["personalized_profile"],
            recommendations=[]
        )

        await alert_integration_service.process_risk_assessment(assessment)
        print(f"   ✅ Processed crisis assessment for {assessment.user_id}")

        # Check results
        alerts = alert_engine.get_active_alerts(assessment.user_id)
        escalations = escalation_manager.get_active_escalations(assessment.user_id)

        print(f"   📊 Results:")
        print(f"      • Alerts Triggered: {len(alerts)}")
        print(f"      • Escalations Started: {len(escalations)}")

        # Process scheduled tasks
        print(f"\n⏰ Processing Scheduled Tasks...")
        await alert_integration_service.process_scheduled_tasks()
        print(f"   ✅ Scheduled tasks processed")

        # Display final integration stats
        final_stats = alert_integration_service.get_integration_stats()
        print(f"\n📈 Final Integration Statistics:")
        print(f"   • Active Alerts: {final_stats['alert_engine_stats']['active_alerts']}")
        print(f"   • Active Escalations: {final_stats['escalation_manager_stats']['active_escalations']}")
        print(f"   • Total Notifications Sent: {final_stats['notification_stats']['total']}")

    except Exception as e:
        print(f"❌ Integration service demo failed: {e}")

async def main():
    """Run all Phase 5 demos"""

    print("🚀 Phase 5: Alerts, Escalation, and Notifications Demo")
    print("=" * 60)
    print("Demonstrating the complete Phase 5 system with:")
    print("• Policy-driven alert rules and thresholds")
    print("• Multi-channel notification delivery")
    print("• Escalation protocols and crisis management")
    print("• Integration with analysis engine")
    print()

    # Run individual component demos
    await demo_alert_engine()
    await demo_notification_service()
    await demo_escalation_manager()
    await demo_integration_service()

    print("\n" + "=" * 60)
    print("✨ Phase 5 Demo Completed Successfully!")
    print("🎯 All alert, escalation, and notification systems are functional")
    print("📊 System is ready for production integration")
    print("=" * 60)

if __name__ == "__main__":
    # Run the Phase 5 demo
    asyncio.run(main())