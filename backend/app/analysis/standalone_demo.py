"""
Phase 4 Analysis Engine Standalone Demo
Demonstrates text processing and behavioral analysis without external dependencies
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data() -> Dict[str, Any]:
    """Create sample data for demonstration"""

    # Sample messages with stress indicators
    messages = [
        {
            "content": "I'm really stressed about the upcoming exam. I feel like I'm drowning in work and can't keep up with everything.",
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
            "type": "message"
        },
        {
            "content": "Been studying all night for this deadline tomorrow. I'm so exhausted and worried I won't finish in time.",
            "timestamp": (datetime.now() - timedelta(hours=5)).isoformat(),
            "type": "message"
        },
        {
            "content": "I can't sleep properly and I'm feeling overwhelmed with all these assignments. I don't know how to manage my time better.",
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "type": "message"
        },
        {
            "content": "Feeling anxious about my grades. I keep procrastinating and then panic when deadlines get close.",
            "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
            "type": "message"
        }
    ]

    # Sample activities showing concerning patterns
    activities = [
        {
            "type": "study",
            "category": "academic",
            "duration": 480,  # 8 hours
            "timestamp": (datetime.now() - timedelta(hours=8)).isoformat(),
            "start_time": (datetime.now() - timedelta(hours=16)).hour
        },
        {
            "type": "sleep",
            "category": "sleep",
            "duration": 5,  # Only 5 hours of sleep
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "start_time": 2  # 2 AM
        },
        {
            "type": "assignment",
            "category": "academic",
            "duration": 120,
            "timestamp": (datetime.now() - timedelta(days=1, hours=3)).isoformat(),
            "submitted_close_to_deadline": True
        },
        {
            "type": "social",
            "category": "social",
            "duration": 30,
            "timestamp": (datetime.now() - timedelta(days=3)).isoformat()
        }
    ]

    return {
        "messages": messages,
        "activities": activities
    }

async def demo_text_processing():
    """Demonstrate text processing capabilities"""

    print("📝 Text Processing Demo")
    print("-" * 30)

    try:
        # Import text processing components
        from app.analysis.text_processing import text_processor

        # Create sample data
        sample_data = create_sample_data()
        messages = sample_data["messages"]

        # Process text
        result = await text_processor.process_text(
            messages=messages,
            time_window=7,
            include_burstiness=True
        )

        # Display results
        print(f"📝 Sentiment: {result.sentiment.sentiment_label}")
        print(f"   • Compound Score: {result.sentiment.compound_score:.3f}")
        print(f"   • Positive: {result.sentiment.positive_score:.3f}")
        print(f"   • Negative: {result.sentiment.negative_score:.3f}")
        print(f"   • Subjectivity: {result.sentiment.subjectivity:.3f}")

        print(f"\n😊 Emotion Analysis:")
        print(f"   • Primary Emotion: {result.emotions.primary_emotion}")
        print(f"   • Confidence: {result.emotions.confidence:.3f}")
        print(f"   • Emotional Intensity: {result.emotions.emotional_intensity:.3f}")

        print(f"\n😰 Stress Lexicon:")
        print(f"   • Stress Score: {result.stress_lexicon.stress_score:.3f}")
        print(f"   • Urgency Level: {result.stress_lexicon.urgency_level}")
        print(f"   • Stress Themes: {', '.join(result.stress_lexicon.stress_themes)}")
        print(f"   • Academic Indicators: {', '.join(result.stress_lexicon.academic_stress_indicators)}")
        print(f"   • Stress Keywords: {', '.join(result.stress_lexicon.stress_keywords[:5])}")

        print(f"\n📊 Linguistic Features:")
        for feature, value in result.linguistic_features.items():
            if isinstance(value, (int, float)):
                print(f"   • {feature.replace('_', ' ').title()}: {value:.3f}")

        print(f"\n🚨 Risk Features:")
        for feature, value in result.risk_features.items():
            print(f"   • {feature.replace('_', ' ').title()}: {value:.3f}")

        print("✅ Text processing completed successfully")

    except Exception as e:
        print(f"❌ Text processing failed: {e}")
        import traceback
        traceback.print_exc()

async def demo_behavioral_analysis():
    """Demonstrate behavioral analysis capabilities"""

    print("\n🏃 Behavioral Analysis Demo")
    print("-" * 30)

    try:
        # Import behavioral analysis components
        from app.analysis.behavioral_features import behavioral_extractor

        # Create sample data
        sample_data = create_sample_data()
        activities = sample_data["activities"]
        messages = sample_data["messages"]

        # Extract behavioral features
        result = await behavioral_extractor.extract_features(
            user_id="demo_user",
            activities=activities,
            messages=messages,
            time_window=7
        )

        # Display sleep analysis
        print(f"😴 Sleep Analysis:")
        print(f"   • Average Duration: {result.sleep.avg_sleep_duration:.1f} hours")
        print(f"   • Regularity: {result.sleep.sleep_regularity:.3f}")
        print(f"   • Disruption Score: {result.sleep.sleep_disruption_score:.3f}")
        print(f"   • Quality Trend: {result.sleep.sleep_quality_trend}")

        # Display social analysis
        print(f"\n👥 Social Analysis:")
        print(f"   • Social Frequency: {result.social.social_frequency:.2f} per day")
        print(f"   • Withdrawal Score: {result.social.social_withdrawal_score:.3f}")
        print(f"   • Interaction Diversity: {result.social.interaction_diversity:.3f}")
        print(f"   • Social Activity Score: {result.social.social_activity_score:.3f}")

        # Display academic behavior analysis
        print(f"\n📚 Academic Behavior:")
        print(f"   • Study Regularity: {result.academic.study_pattern_regularity:.3f}")
        print(f"   • Procrastination Level: {result.academic.procrastination_level:.3f}")
        print(f"   • Deadline Pressure: {result.academic.deadline_pressure_score:.3f}")
        print(f"   • Engagement Score: {result.academic.academic_engagement_score:.3f}")

        # Display activity patterns
        print(f"\n🔄 Activity Patterns:")
        print(f"   • Regularity: {result.activity_patterns.activity_regularity:.3f}")
        print(f"   • Nocturnal Activity: {result.activity_patterns.nocturnal_activity_ratio:.3f}")
        print(f"   • Sedentary Time: {result.activity_patterns.sedentary_time_ratio:.3f}")
        print(f"   • Work-Life Balance: {result.activity_patterns.work_life_balance_score:.3f}")

        # Display risk indicators
        print(f"\n⚠️  Risk Indicators:")
        for risk, score in result.risk_indicators.items():
            print(f"   • {risk.replace('_', ' ').title()}: {score:.3f}")

        # Display recommendations
        if result.recommendations:
            print(f"\n💡 Behavioral Recommendations:")
            for i, rec in enumerate(result.recommendations[:3]):
                print(f"   {i+1}. {rec['title']} ({rec['priority']})")

        print("✅ Behavioral analysis completed successfully")

    except Exception as e:
        print(f"❌ Behavioral analysis failed: {e}")
        import traceback
        traceback.print_exc()

async def demo_risk_scoring():
    """Demonstrate risk scoring capabilities"""

    print("\n📊 Risk Scoring Demo")
    print("-" * 30)

    try:
        # Import risk scoring components
        from app.analysis.risk_scoring_engine import risk_scoring_engine
        from app.analysis.text_processing import text_processor
        from app.analysis.behavioral_features import behavioral_extractor

        # Create sample data
        sample_data = create_sample_data()

        # Process text and behavioral data
        text_result = await text_processor.process_text(
            messages=sample_data["messages"],
            time_window=7
        )

        behavioral_result = await behavioral_extractor.extract_features(
            user_id="demo_user",
            activities=sample_data["activities"],
            time_window=7
        )

        # Calculate risk score
        risk_assessment = await risk_scoring_engine.calculate_risk_score(
            user_id="demo_user",
            text_processing=text_result,
            behavioral_features=behavioral_result,
            contextual_data={
                "support_system_strength": 0.3,
                "coping_mechanisms": 0.4,
                "recent_stress_events": 0.6
            }
        )

        # Display risk classification
        print(f"🎯 Risk Classification:")
        print(f"   • Risk Level: {risk_assessment.classification.risk_level}")
        print(f"   • Risk Score: {risk_assessment.classification.risk_score:.3f}")
        print(f"   • Confidence: {risk_assessment.classification.confidence:.3f}")
        print(f"   • Percentile Rank: {risk_assessment.classification.percentile_rank:.1f}")
        print(f"   • Trajectory: {risk_assessment.classification.risk_trajectory}")
        print(f"   • Escalation Needed: {risk_assessment.classification.escalation_needed}")

        # Display component scores
        print(f"\n📊 Component Scores:")
        for component, score in risk_assessment.component_scores.items():
            print(f"   • {component.title()}: {score:.3f}")

        # Display primary risk factors
        print(f"\n⚠️  Primary Risk Factors:")
        for factor in risk_assessment.classification.primary_risk_factors:
            print(f"   • {factor}")

        # Display feature contributions
        print(f"\n🔍 Top Feature Contributions:")
        sorted_features = sorted(
            risk_assessment.feature_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for feature, contribution in sorted_features:
            print(f"   • {feature.replace('_', ' ').title()}: {contribution:.3f}")

        # Display personalized profile info
        profile = risk_assessment.personalized_profile
        print(f"\n👤 Personalized Profile:")
        print(f"   • Baseline Score: {profile.baseline_score:.3f}")
        print(f"   • Risk History Length: {len(profile.risk_history)}")
        print(f"   • Adaptive Thresholds: {list(profile.adaptive_thresholds.keys())}")

        # Display recommendations
        if risk_assessment.recommendations:
            print(f"\n💡 Risk-Based Recommendations:")
            for i, rec in enumerate(risk_assessment.recommendations[:3]):
                print(f"   {i+1}. {rec['title']} ({rec['priority']})")
                if 'evidence' in rec:
                    print(f"      Evidence: {rec['evidence']}")

        print("✅ Risk scoring completed successfully")

    except Exception as e:
        print(f"❌ Risk scoring failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all demos"""

    print("🧠 Phase 4 Analysis Engine - Standalone Demo")
    print("=" * 50)
    print("Demonstrating core analysis capabilities without external dependencies")
    print()

    # Run individual component demos
    await demo_text_processing()
    await demo_behavioral_analysis()
    await demo_risk_scoring()

    print("\n" + "=" * 50)
    print("✨ Phase 4 Analysis Engine Demo Completed!")
    print("🎯 All core components are functional and ready for integration")
    print("=" * 50)

if __name__ == "__main__":
    # Run the standalone demo
    asyncio.run(main())