"""
Phase 4 Analysis Engine Demo
Demonstrates the complete Phase 4 system functionality
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import Phase 4 components
from app.analysis.phase4_integration import (
    Phase4AnalysisEngine,
    Phase4AnalysisRequest
)

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

    # Sample calendar events
    calendar_events = [
        {
            "title": "Final Exam - Calculus",
            "type": "exam",
            "date": (datetime.now() + timedelta(days=2)).isoformat(),
            "stress_score": 0.8
        },
        {
            "title": "Project Deadline",
            "type": "deadline",
            "date": (datetime.now() + timedelta(days=1)).isoformat(),
            "stress_score": 0.7
        },
        {
            "title": "Study Group",
            "type": "social",
            "date": (datetime.now() + timedelta(days=3)).isoformat()
        }
    ]

    # Sample user context
    user_context = {
        "preferences": {
            "learning_style": "visual",
            "time_availability": "limited",
            "stress_management": "beginner"
        },
        "history": {
            "previous_counseling": False,
            "academic_performance": "declining",
            "sleep_issues": True
        },
        "support_system": 0.3,  # Low support system
        "coping_skills": 0.4    # Limited coping skills
    }

    return {
        "messages": messages,
        "activities": activities,
        "calendar_events": calendar_events,
        "user_context": user_context
    }

def create_moderate_risk_data() -> Dict[str, Any]:
    """Create sample data with moderate risk indicators"""

    messages = [
        {
            "content": "Feeling a bit overwhelmed with classes this semester, but managing to keep up with the workload.",
            "timestamp": (datetime.now() - timedelta(hours=6)).isoformat(),
            "type": "message"
        },
        {
            "content": "Had a productive study session today. The group project is going well, though deadlines are approaching.",
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "type": "message"
        },
        {
            "content": "Sometimes I worry about balancing everything, but I'm trying to maintain a good schedule and sleep routine.",
            "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
            "type": "message"
        }
    ]

    activities = [
        {
            "type": "study",
            "category": "academic",
            "duration": 180,  # 3 hours
            "timestamp": (datetime.now() - timedelta(hours=4)).isoformat()
        },
        {
            "type": "sleep",
            "category": "sleep",
            "duration": 7,  # 7 hours of sleep
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "start_time": 23  # 11 PM
        },
        {
            "type": "exercise",
            "category": "wellness",
            "duration": 45,
            "timestamp": (datetime.now() - timedelta(hours=8)).isoformat()
        },
        {
            "type": "social",
            "category": "social",
            "duration": 120,
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat()
        }
    ]

    user_context = {
        "support_system": 0.7,  # Good support system
        "coping_skills": 0.6,   # Moderate coping skills
        "preferences": {
            "learning_style": "mixed",
            "time_availability": "moderate"
        }
    }

    return {
        "messages": messages,
        "activities": activities,
        "user_context": user_context
    }

async def demo_phase4_analysis():
    """Demonstrate Phase 4 analysis with different scenarios"""

    print("🧠 Phase 4 Analysis Engine Demo")
    print("=" * 50)

    # Initialize the analysis engine
    engine = Phase4AnalysisEngine()
    print("✅ Phase 4 Analysis Engine initialized")

    # Demo 1: High-risk scenario
    print("\n📊 Demo 1: High-Risk Scenario Analysis")
    print("-" * 40)

    high_risk_data = create_sample_data()

    request1 = Phase4AnalysisRequest(
        user_id="demo_user_001",
        messages=high_risk_data["messages"],
        activities=high_risk_data["activities"],
        calendar_events=high_risk_data["calendar_events"],
        user_context=high_risk_data["user_context"],
        time_window=7,
        include_recommendations=True,
        personalization_enabled=True
    )

    try:
        result1 = await engine.analyze_comprehensive(request1)

        print(f"📈 Risk Level: {result1.risk_assessment.classification.risk_level}")
        print(f"📊 Risk Score: {result1.risk_assessment.classification.risk_score:.3f}")
        print(f"🎯 Confidence: {result1.risk_assessment.classification.confidence:.3f}")
        print(f"📋 Primary Concerns: {', '.join(result1.risk_assessment.classification.primary_risk_factors[:3])}")

        # Component scores
        print("\n📊 Component Scores:")
        for component, score in result1.risk_assessment.component_scores.items():
            print(f"  • {component.title()}: {score:.3f}")

        # Top recommendations
        if result1.recommendations:
            print(f"\n💡 Top {min(3, len(result1.recommendations.recommendations))} Recommendations:")
            for i, rec in enumerate(result1.recommendations.recommendations[:3]):
                print(f"  {i+1}. {rec.title}")
                print(f"     Priority: {rec.priority} | Impact: {rec.estimated_impact}")
                print(f"     Time: {rec.time_commitment} | Difficulty: {rec.difficulty_level}")

        # Processing summary
        print(f"\n⏱️  Processing completed in {result1.analysis_metadata['processing_time_seconds']:.2f} seconds")
        print(f"📋 Data quality score: {result1.processing_summary['data_quality']['overall_quality']:.3f}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")

    # Demo 2: Moderate-risk scenario
    print("\n\n📊 Demo 2: Moderate-Risk Scenario Analysis")
    print("-" * 40)

    moderate_risk_data = create_moderate_risk_data()

    request2 = Phase4AnalysisRequest(
        user_id="demo_user_002",
        messages=moderate_risk_data["messages"],
        activities=moderate_risk_data["activities"],
        user_context=moderate_risk_data["user_context"],
        time_window=7,
        include_recommendations=True,
        personalization_enabled=True
    )

    try:
        result2 = await engine.analyze_comprehensive(request2)

        print(f"📈 Risk Level: {result2.risk_assessment.classification.risk_level}")
        print(f"📊 Risk Score: {result2.risk_assessment.classification.risk_score:.3f}")
        print(f"🎯 Confidence: {result2.risk_assessment.classification.confidence:.3f}")
        print(f"📋 Primary Concerns: {', '.join(result2.risk_assessment.classification.primary_risk_factors[:3])}")

        # Component scores
        print("\n📊 Component Scores:")
        for component, score in result2.risk_assessment.component_scores.items():
            print(f"  • {component.title()}: {score:.3f}")

        # Top recommendations
        if result2.recommendations:
            print(f"\n💡 Top {min(3, len(result2.recommendations.recommendations))} Recommendations:")
            for i, rec in enumerate(result2.recommendations.recommendations[:3]):
                print(f"  {i+1}. {rec.title}")
                print(f"     Priority: {rec.priority} | Impact: {rec.estimated_impact}")

        # Processing summary
        print(f"\n⏱️  Processing completed in {result2.analysis_metadata['processing_time_seconds']:.2f} seconds")
        print(f"📋 Data quality score: {result2.processing_summary['data_quality']['overall_quality']:.3f}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")

    # Demo 3: Text processing only
    print("\n\n📝 Demo 3: Text Processing Analysis Only")
    print("-" * 40)

    try:
        from app.analysis.text_processing import text_processor

        text_result = await text_processor.process_text(
            messages=high_risk_data["messages"],
            time_window=7,
            include_burstiness=True
        )

        print(f"📝 Sentiment: {text_result.sentiment.sentiment_label} (compound: {text_result.sentiment.compound_score:.3f})")
        print(f"😊 Primary Emotion: {text_result.emotions.primary_emotion}")
        print(f"😰 Stress Score: {text_result.stress_lexicon.stress_score:.3f}")
        print(f"🚨 Urgency Level: {text_result.stress_lexicon.urgency_level}")
        print(f"📊 Stress Themes: {', '.join(text_result.stress_lexicon.stress_themes)}")

        print("\n📊 Linguistic Features:")
        for feature, value in text_result.linguistic_features.items():
            if isinstance(value, (int, float)):
                print(f"  • {feature.replace('_', ' ').title()}: {value:.3f}")

    except Exception as e:
        print(f"❌ Text processing failed: {e}")

    print("\n\n✨ Demo completed! Phase 4 Analysis Engine is fully functional.")
    print("=" * 50)

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_phase4_analysis())