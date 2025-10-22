from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RiskLevel(str, Enum):
    """Risk severity levels"""
    LOW = "low"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRISIS = "crisis"

@dataclass
class RiskFactors:
    """Represents various risk factors for mental health assessment"""

    # Academic factors
    academic_pressure: float = 0.0          # 0-1 scale
    deadline_proximity: float = 0.0          # 0-1 scale
    grade_performance: float = 0.0           # 0-1 scale (higher = worse)
    workload_intensity: float = 0.0          # 0-1 scale

    # Behavioral factors
    sleep_disruption: float = 0.0            # 0-1 scale
    social_withdrawal: float = 0.0           # 0-1 scale
    activity_irregularity: float = 0.0        # 0-1 scale
    procrastination_level: float = 0.0          # 0-1 scale

    # Emotional factors
    negative_sentiment: float = 0.0          # 0-1 scale
    anxiety_indicators: float = 0.0           # 0-1 scale
    depression_indicators: float = 0.0         # 0-1 scale
    stress_keywords: float = 0.0              # 0-1 scale

    # Contextual factors
    recent_stress_events: float = 0.0         # 0-1 scale
    support_system_strength: float = 0.0         # 0-1 scale (higher = better)
    coping_mechanisms: float = 0.0             # 0-1 scale (higher = better)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization"""
        return {
            "academic_pressure": self.academic_pressure,
            "deadline_proximity": self.deadline_proximity,
            "grade_performance": self.grade_performance,
            "workload_intensity": self.workload_intensity,
            "sleep_disruption": self.sleep_disruption,
            "social_withdrawal": self.social_withdrawal,
            "activity_irregularity": self.activity_irregularity,
            "procrastination_level": self.procrastination_level,
            "negative_sentiment": self.negative_sentiment,
            "anxiety_indicators": self.anxiety_indicators,
            "depression_indicators": self.depression_indicators,
            "stress_keywords": self.stress_keywords,
            "recent_stress_events": self.recent_stress_events,
            "support_system_strength": self.support_system_strength,
            "coping_mechanisms": self.coping_mechanisms
        }

@dataclass
class RiskScore:
    """Comprehensive risk assessment result"""

    # Core metrics
    user_id: str
    overall_score: float                    # 0-1 scale
    risk_level: RiskLevel
    confidence: float                        # 0-1 scale

    # Component scores
    academic_risk: float                    # 0-1 scale
    behavioral_risk: float                  # 0-1 scale
    emotional_risk: float                   # 0-1 scale
    contextual_risk: float                  # 0-1 scale

    # Risk factors breakdown
    factors: RiskFactors

    # Metadata
    timestamp: datetime
    data_sources: List[str]                 # Sources used for scoring
    personalized_weights: Optional[Dict[str, float]] = None

    # Recommendations
    primary_concerns: Optional[List[str]] = None     # Top risk factors
    recommended_actions: Optional[List[Dict[str, Any]]] = None
    escalation_threshold_met: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "user_id": self.user_id,
            "overall_score": self.overall_score,
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "academic_risk": self.academic_risk,
            "behavioral_risk": self.behavioral_risk,
            "emotional_risk": self.emotional_risk,
            "contextual_risk": self.contextual_risk,
            "factors": self.factors.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "data_sources": self.data_sources,
            "personalized_weights": self.personalized_weights,
            "primary_concerns": self.primary_concerns,
            "recommended_actions": self.recommended_actions,
            "escalation_threshold_met": self.escalation_threshold_met
        }

@dataclass
class RiskTrend:
    """Represents risk score trends over time"""

    user_id: str
    time_window: int                        # Number of days analyzed
    trend_direction: str                     # "improving", "stable", "worsening"
    trend_strength: float                    # 0-1 scale

    # Statistics
    current_score: float
    average_score: float
    min_score: float
    max_score: float
    score_volatility: float                 # Standard deviation

    # Data points
    daily_scores: List[Dict[str, Any]]       # [{date: date, score: score}]
    significant_events: List[Dict[str, Any]]   # Notable changes or events

    # Metadata
    last_updated: datetime
    data_quality: float                     # 0-1 scale based on data completeness

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "user_id": self.user_id,
            "time_window": self.time_window,
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "current_score": self.current_score,
            "average_score": self.average_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "score_volatility": self.score_volatility,
            "daily_scores": self.daily_scores,
            "significant_events": self.significant_events,
            "last_updated": self.last_updated.isoformat(),
            "data_quality": self.data_quality
        }

class RiskThresholds:
    """Configurable thresholds for risk assessment"""

    # Default risk level thresholds (can be personalized per user)
    RISK_THRESHOLDS = {
        RiskLevel.LOW: 0.2,
        RiskLevel.MILD: 0.35,
        RiskLevel.MODERATE: 0.55,
        RiskLevel.SEVERE: 0.75,
        RiskLevel.CRISIS: 0.90
    }

    # Escalation thresholds
    CRISIS_ESCALATION_THRESHOLD = 0.9
    SEVERE_ESCALATION_THRESHOLD = 0.75
    TREND_DETERIORATION_THRESHOLD = 0.2  # 20% increase over baseline

    # Weight configurations for different user types
    DEFAULT_WEIGHTS = {
        "academic": {
            "academic_pressure": 0.25,
            "deadline_proximity": 0.20,
            "grade_performance": 0.15,
            "workload_intensity": 0.15
        },
        "behavioral": {
            "sleep_disruption": 0.25,
            "social_withdrawal": 0.20,
            "activity_irregularity": 0.15,
            "procrastination_level": 0.15
        },
        "emotional": {
            "negative_sentiment": 0.25,
            "anxiety_indicators": 0.25,
            "depression_indicators": 0.25,
            "stress_keywords": 0.15
        },
        "contextual": {
            "recent_stress_events": 0.20,
            "support_system_strength": -0.15,  # Negative weight (protective factor)
            "coping_mechanisms": -0.15        # Negative weight (protective factor)
        }
    }

    # Category weights
    CATEGORY_WEIGHTS = {
        "academic": 0.30,
        "behavioral": 0.25,
        "emotional": 0.30,
        "contextual": 0.15
    }

class RiskValidator:
    """Validates risk scores and factors"""

    @staticmethod
    def validate_factors(factors: RiskFactors) -> List[str]:
        """Validate risk factors and return list of issues"""
        issues = []

        # Check value ranges
        for attr_name, value in factors.to_dict().items():
            if not isinstance(value, (int, float)):
                issues.append(f"Invalid type for {attr_name}: {type(value)}")
            elif value < 0 or value > 1:
                issues.append(f"Value out of range for {attr_name}: {value}")

        return issues

    @staticmethod
    def validate_risk_score(risk_score: RiskScore) -> List[str]:
        """Validate complete risk score"""
        issues = []

        # Check overall score
        if not 0 <= risk_score.overall_score <= 1:
            issues.append(f"Overall score out of range: {risk_score.overall_score}")

        # Check component scores
        components = [
            ("academic_risk", risk_score.academic_risk),
            ("behavioral_risk", risk_score.behavioral_risk),
            ("emotional_risk", risk_score.emotional_risk),
            ("contextual_risk", risk_score.contextual_risk)
        ]

        for name, score in components:
            if not 0 <= score <= 1:
                issues.append(f"{name} out of range: {score}")

        # Check confidence
        if not 0 <= risk_score.confidence <= 1:
            issues.append(f"Confidence out of range: {risk_score.confidence}")

        # Check required fields
        if not risk_score.user_id:
            issues.append("Missing user_id")

        if not risk_score.risk_level:
            issues.append("Missing risk_level")

        return issues

class RiskCalculator:
    """Core risk calculation utilities"""

    @staticmethod
    def calculate_category_score(factors: RiskFactors, category: str, weights: Dict[str, float]) -> float:
        """Calculate risk score for a specific category"""
        category_weights = weights.get(category, {})
        factor_values = factors.to_dict()

        weighted_sum = 0.0
        total_weight = 0.0

        for factor_name, weight in category_weights.items():
            if factor_name in factor_values:
                value = factor_values[factor_name]
                weighted_sum += value * weight
                total_weight += abs(weight)  # Use absolute value for protective factors

        if total_weight == 0:
            return 0.0

        # Normalize by total weight
        normalized_score = weighted_sum / total_weight

        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, normalized_score))

    @staticmethod
    def calculate_overall_risk(category_scores: Dict[str, float], category_weights: Dict[str, float]) -> float:
        """Calculate overall risk from category scores"""
        weighted_sum = 0.0
        total_weight = 0.0

        for category, score in category_scores.items():
            weight = category_weights.get(category, 0.0)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        overall_score = weighted_sum / total_weight

        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, overall_score))

    @staticmethod
    def determine_risk_level(score: float, thresholds: Dict[RiskLevel, float]) -> RiskLevel:
        """Determine risk level from score using provided thresholds"""
        # Check thresholds in descending order
        for risk_level, threshold in sorted(thresholds.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return risk_level

        return RiskLevel.LOW

    @staticmethod
    def calculate_trend(scores: List[float]) -> Dict[str, float]:
        """Calculate trend statistics from a list of scores"""
        if len(scores) < 2:
            return {
                "direction": "stable",
                "strength": 0.0,
                "slope": 0.0
            }

        # Simple linear regression
        n = len(scores)
        x_values = list(range(n))

        sum_x = sum(x_values)
        sum_y = sum(scores)
        sum_xy = sum(x * y for x, y in zip(x_values, scores))
        sum_x2 = sum(x * x for x in x_values)

        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # Determine direction and strength
        if abs(slope) < 0.01:
            direction = "stable"
            strength = 0.0
        elif slope > 0:
            direction = "worsening"
            strength = min(abs(slope) * 10, 1.0)
        else:
            direction = "improving"
            strength = min(abs(slope) * 10, 1.0)

        return {
            "direction": direction,
            "strength": strength,
            "slope": slope
        }
