"""
Phase 4 Analysis Engine
Contains text processing, behavioral analysis, and risk scoring components
for the Mental Health & Study-Stress Analyzer MVP.
"""

from .text_processing import (
    TextProcessingPipeline,
    TextProcessingResult,
    SentimentAnalysis,
    EmotionAnalysis,
    StressLexiconAnalysis,
    BurstinessAnalysis,
    text_processor
)

from .behavioral_features import (
    BehavioralFeaturesExtractor,
    BehavioralFeaturesResult,
    SleepAnalysis,
    SocialAnalysis,
    AcademicBehaviorAnalysis,
    ActivityPatternAnalysis,
    behavioral_extractor
)

from .risk_scoring_engine import (
    RiskScoringEngine,
    RiskClassification,
    PersonalizedRiskProfile,
    risk_scoring_engine
)

from .recommendations_engine import (
    RAGRecommendationsEngine,
    RecommendationItem,
    RecommendationSource,
    PersonalizedRecommendations,
    recommendations_engine
)

from .phase4_integration import (
    Phase4AnalysisEngine,
    Phase4AnalysisRequest,
    Phase4AnalysisResult,
    phase4_engine
)

from .config import config, get_config, validate_config

__all__ = [
    "TextProcessingPipeline",
    "TextProcessingResult",
    "SentimentAnalysis",
    "EmotionAnalysis",
    "StressLexiconAnalysis",
    "BurstinessAnalysis",
    "text_processor",
    "BehavioralFeaturesExtractor",
    "BehavioralFeaturesResult",
    "SleepAnalysis",
    "SocialAnalysis",
    "AcademicBehaviorAnalysis",
    "ActivityPatternAnalysis",
    "behavioral_extractor",
    "RiskScoringEngine",
    "RiskClassification",
    "PersonalizedRiskProfile",
    "risk_scoring_engine",
    "RAGRecommendationsEngine",
    "RecommendationItem",
    "RecommendationSource",
    "PersonalizedRecommendations",
    "recommendations_engine",
    "Phase4AnalysisEngine",
    "Phase4AnalysisRequest",
    "Phase4AnalysisResult",
    "phase4_engine",
    "config",
    "get_config",
    "validate_config"
]