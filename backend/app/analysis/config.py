"""
Phase 4 Analysis Engine Configuration
Configuration settings for text processing, behavioral analysis, risk scoring, and recommendations.
"""

from typing import Dict, Any, List
import os
from pathlib import Path

# Base configuration
class AnalysisConfig:
    """Base configuration for Phase 4 analysis engine"""

    # Text processing configuration
    TEXT_PROCESSING = {
        "max_text_length": 10000,  # Maximum text length to process
        "min_message_count": 1,    # Minimum messages for analysis
        "default_time_window": 7,  # Default time window in days
        "sentiment_thresholds": {
            "positive": 0.05,
            "negative": -0.05,
            "neutral_range": 0.1
        },
        "stress_keywords_threshold": 0.3,  # Minimum stress keyword ratio
        "burstiness_enabled": True,
        "linguistic_features_enabled": True
    }

    # Behavioral analysis configuration
    BEHAVIORAL_ANALYSIS = {
        "min_activity_count": 1,
        "sleep_analysis": {
            "optimal_duration": 8.0,     # hours
            "min_duration": 6.0,
            "max_duration": 10.0,
            "regularity_weight": 0.7
        },
        "social_analysis": {
            "healthy_frequency": 2.0,   # interactions per day
            "response_time_threshold": 4.0,  # hours
            "withdrawal_threshold": 0.6
        },
        "academic_analysis": {
            "healthy_study_hours": 6.0,  # per day
            "max_study_hours": 12.0,
            "procrastination_threshold": 0.3
        },
        "activity_patterns": {
            "night_hours": list(range(22, 24)) + list(range(0, 6)),
            "day_hours": list(range(6, 22)),
            "sedentary_threshold": 0.8
        }
    }

    # Risk scoring configuration
    RISK_SCORING = {
        "risk_levels": ["low", "mild", "moderate", "severe", "crisis"],
        "base_thresholds": {
            "low": 0.2,
            "mild": 0.35,
            "moderate": 0.55,
            "severe": 0.75,
            "crisis": 0.90
        },
        "component_weights": {
            "sentiment": 0.25,
            "behavioral": 0.25,
            "academic": 0.30,
            "contextual": 0.20
        },
        "sigmoid_parameters": {
            "steepness": 6.0,
            "center_point": 0.3
        },
        "personalization": {
            "enabled": True,
            "baseline_update_alpha": 0.1,  # EMA smoothing factor
            "adaptive_threshold_adjustment": 0.3,
            "min_history_for_adaptation": 5
        }
    }

    # Recommendations configuration
    RECOMMENDATIONS = {
        "max_recommendations_per_type": 3,
        "max_total_recommendations": 15,
        "evidence_threshold": 0.5,
        "recommendation_types": {
            "crisis_support": {
                "priority": "immediate",
                "collections": ["crisis_resources", "emergency_protocols"],
                "max_recommendations": 3
            },
            "academic_stress": {
                "priority": "high",
                "collections": ["academic_resources", "study_techniques"],
                "max_recommendations": 5
            },
            "sleep_improvement": {
                "priority": "high",
                "collections": ["sleep_hygiene", "wellness_resources"],
                "max_recommendations": 4
            },
            "social_support": {
                "priority": "medium",
                "collections": ["social_connection", "peer_support"],
                "max_recommendations": 4
            },
            "stress_management": {
                "priority": "medium",
                "collections": ["stress_management", "mindfulness"],
                "max_recommendations": 5
            },
            "wellness_practices": {
                "priority": "low",
                "collections": ["general_wellness", "self_care"],
                "max_recommendations": 4
            }
        }
    }

    # Integration configuration
    INTEGRATION = {
        "max_concurrent_processes": 4,
        "processing_timeout_seconds": 300,  # 5 minutes
        "data_quality_thresholds": {
            "min_completeness": 0.3,
            "high_confidence": 0.8,
            "min_messages_for_text": 1,
            "min_activities_for_behavioral": 1
        },
        "caching": {
            "enabled": True,
            "cache_ttl_seconds": 3600,  # 1 hour
            "max_cache_size": 1000
        }
    }

    # Logging configuration
    LOGGING = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_path": "logs/phase4_analysis.log",
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5
    }

    # Resource paths
    RESOURCE_PATHS = {
        "stress_lexicon": "data/lexicons/stress_keywords.json",
        "emotion_lexicon": "data/lexicons/emotions.json",
        "academic_lexicon": "data/lexicons/academic.json",
        "knowledge_base": "data/knowledge_base/",
        "models_dir": "models/"
    }

    # API configuration
    API = {
        "rate_limiting": {
            "enabled": True,
            "requests_per_minute": 60,
            "burst_size": 10
        },
        "response_caching": {
            "enabled": True,
            "ttl_seconds": 300  # 5 minutes
        }
    }

class DevelopmentConfig(AnalysisConfig):
    """Development environment configuration"""

    LOGGING = {
        **AnalysisConfig.LOGGING,
        "level": "DEBUG"
    }

    INTEGRATION = {
        **AnalysisConfig.INTEGRATION,
        "processing_timeout_seconds": 600,  # 10 minutes for development
        "max_concurrent_processes": 2
    }

class ProductionConfig(AnalysisConfig):
    """Production environment configuration"""

    LOGGING = {
        **AnalysisConfig.LOGGING,
        "level": "WARNING"
    }

    INTEGRATION = {
        **AnalysisConfig.INTEGRATION,
        "processing_timeout_seconds": 180,  # 3 minutes for production
        "max_concurrent_processes": 8
    }

    API = {
        **AnalysisConfig.API,
        "rate_limiting": {
            "enabled": True,
            "requests_per_minute": 120,
            "burst_size": 20
        }
    }

class TestingConfig(AnalysisConfig):
    """Testing environment configuration"""

    TEXT_PROCESSING = {
        **AnalysisConfig.TEXT_PROCESSING,
        "default_time_window": 1  # Short window for testing
    }

    INTEGRATION = {
        **AnalysisConfig.INTEGRATION,
        "processing_timeout_seconds": 60,  # 1 minute for tests
        "max_concurrent_processes": 1
    }

def get_config() -> AnalysisConfig:
    """Get configuration based on environment"""

    env = os.getenv("ANALYSIS_ENV", "development").lower()

    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()

# Global configuration instance
config = get_config()

def validate_config() -> List[str]:
    """Validate configuration and return list of issues"""

    issues = []

    # Check resource paths
    for path_name, path_value in config.RESOURCE_PATHS.items():
        if path_value and not Path(path_value).exists():
            issues.append(f"Resource path does not exist: {path_name} -> {path_value}")

    # Check numeric ranges
    if config.TEXT_PROCESSING["max_text_length"] <= 0:
        issues.append("max_text_length must be positive")

    if config.RISK_SCORING["component_weights"]["sentiment"] + \
       config.RISK_SCORING["component_weights"]["behavioral"] + \
       config.RISK_SCORING["component_weights"]["academic"] + \
       config.RISK_SCORING["component_weights"]["contextual"] != 1.0:
        issues.append("Component weights must sum to 1.0")

    # Check thresholds are in valid range
    for threshold_name, threshold_value in config.RISK_SCORING["base_thresholds"].items():
        if not 0 <= threshold_value <= 1:
            issues.append(f"Risk threshold {threshold_name} must be in [0, 1] range")

    return issues

# Initialize and validate configuration
if __name__ == "__main__":
    issues = validate_config()
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid")