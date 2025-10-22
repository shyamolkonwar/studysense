"""
Phase 4: Risk Scoring Engine
Implements the comprehensive risk scoring formula and classification system
as specified in the MVP requirements.

Risk Score Formula:
R_t = σ(w_s·S_t + w_e·E_t + w_b·B_t + w_d·D_t + b)

Where:
- S_t = Sentiment and emotional features
- E_t = Behavioral features
- B_t = Academic/Study features
- D_t = Contextual/Deadline features
- w_* = Adaptive weights
- b = Personalized baseline
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import asyncio
import json

from .text_processing import TextProcessingResult
from .behavioral_features import BehavioralFeaturesResult
from ..rag.chroma_client import chroma_client

logger = logging.getLogger(__name__)

@dataclass
class RiskClassification:
    """Risk classification result"""
    risk_level: str                 # "low", "mild", "moderate", "severe", "crisis"
    risk_score: float              # 0-1 normalized score
    confidence: float              # 0-1 confidence in classification
    primary_risk_factors: List[str]  # Top contributing factors
    risk_trajectory: str           # "improving", "stable", "worsening"
    escalation_needed: bool        # Whether immediate escalation is required
    percentile_rank: float         # Population percentile (if available)

@dataclass
class PersonalizedRiskProfile:
    """Personalized risk profile for adaptive scoring"""
    user_id: str
    baseline_score: float          # Personal baseline risk level
    sensitivity_weights: Dict[str, float]  # Individual sensitivity to different factors
    adaptive_thresholds: Dict[str, float]   # Personalized risk thresholds
    protective_factors: List[str]   # Personal protective factors
    risk_history: List[Dict[str, Any]]  # Historical risk assessments
    last_updated: datetime

@dataclass
class ComprehensiveRiskAssessment:
    """Complete risk assessment result"""
    user_id: str
    assessment_timestamp: datetime
    time_window_days: int
    classification: RiskClassification
    component_scores: Dict[str, float]  # Scores for each component
    feature_contributions: Dict[str, float]  # Individual feature contributions
    personalized_profile: PersonalizedRiskProfile
    recommendations: List[Dict[str, Any]]
    evidence_summary: Dict[str, Any]

class RiskScoringEngine:
    """
    Advanced risk scoring engine implementing the Phase 4 specifications.
    Uses adaptive weights, personalized baselines, and percentile-based thresholds.
    """

    def __init__(self):
        """Initialize risk scoring engine"""

        # Risk level thresholds (percentile-based adaptive thresholds)
        self.base_thresholds = {
            "low": 0.2,
            "mild": 0.35,
            "moderate": 0.55,
            "severe": 0.75,
            "crisis": 0.90
        }

        # Default weights for risk components
        self.default_weights = {
            "sentiment": 0.25,      # w_s - Sentiment and emotional features
            "behavioral": 0.25,     # w_e - Behavioral features
            "academic": 0.30,       # w_b - Academic/Study features
            "contextual": 0.20      # w_d - Contextual/Deadline features
        }

        # Component feature mappings
        self.sentiment_features = [
            "negative_sentiment", "fear_intensity", "sadness_intensity",
            "anger_intensity", "emotional_intensity", "stress_lexicon_score"
        ]

        self.behavioral_features = [
            "sleep_disruption", "irregular_sleep", "insufficient_sleep",
            "social_withdrawal", "social_isolation", "irregular_schedule",
            "nocturnal_activity", "sedentary_behavior"
        ]

        self.academic_features = [
            "procrastination", "academic_pressure", "poor_study_patterns",
            "low_engagement", "academic_stress_focus", "deadline_proximity"
        ]

        self.contextual_features = [
            "recent_stress_events", "support_system_strength",
            "coping_mechanisms", "urgency_level"
        ]

        # User profiles storage
        self.user_profiles: Dict[str, PersonalizedRiskProfile] = {}

        # Risk level colors for UI
        self.risk_colors = {
            "low": "#10b981",      # Green
            "mild": "#84cc16",     # Light green
            "moderate": "#f59e0b", # Yellow
            "severe": "#f97316",   # Orange
            "crisis": "#dc2626"    # Red
        }

    async def calculate_risk_score(
        self,
        user_id: str,
        text_processing: TextProcessingResult,
        behavioral_features: BehavioralFeaturesResult,
        contextual_data: Optional[Dict[str, Any]] = None,
        time_window: int = 7
    ) -> ComprehensiveRiskAssessment:
        """
        Calculate comprehensive risk score using the Phase 4 formula

        Args:
            user_id: Unique user identifier
            text_processing: Text processing analysis results
            behavioral_features: Behavioral features analysis results
            contextual_data: Additional contextual information
            time_window: Analysis time window in days

        Returns:
            Complete ComprehensiveRiskAssessment
        """

        try:
            # Get or create personalized profile
            user_profile = await self._get_or_create_profile(user_id)

            # Extract component scores
            sentiment_score = self._calculate_sentiment_score(text_processing)
            behavioral_score = self._calculate_behavioral_score(behavioral_features)
            academic_score = self._calculate_academic_score(behavioral_features)
            contextual_score = self._calculate_contextual_score(
                text_processing, behavioral_features, contextual_data
            )

            # Get personalized weights
            weights = user_profile.sensitivity_weights

            # Apply the risk scoring formula: R_t = σ(w_s·S_t + w_e·E_t + w_b·B_t + w_d·D_t + b)
            weighted_sum = (
                weights["sentiment"] * sentiment_score +
                weights["behavioral"] * behavioral_score +
                weights["academic"] * academic_score +
                weights["contextual"] * contextual_score +
                user_profile.baseline_score
            )

            # Apply sigmoid normalization: σ(x) = 1 / (1 + e^(-x))
            # Adjusted to keep range approximately 0-1
            raw_score = self._apply_adaptive_sigmoid(weighted_sum)

            # Calculate feature contributions
            feature_contributions = self._calculate_feature_contributions(
                text_processing, behavioral_features, contextual_data
            )

            # Determine risk classification
            classification = await self._classify_risk_level(
                raw_score, user_profile, feature_contributions
            )

            # Generate recommendations
            recommendations = await self._generate_risk_recommendations(
                classification, feature_contributions, user_profile
            )

            # Create evidence summary
            evidence_summary = self._create_evidence_summary(
                text_processing, behavioral_features, feature_contributions
            )

            # Update user profile
            await self._update_user_profile(user_id, raw_score, classification)

            return ComprehensiveRiskAssessment(
                user_id=user_id,
                assessment_timestamp=datetime.now(),
                time_window_days=time_window,
                classification=classification,
                component_scores={
                    "sentiment": sentiment_score,
                    "behavioral": behavioral_score,
                    "academic": academic_score,
                    "contextual": contextual_score
                },
                feature_contributions=feature_contributions,
                personalized_profile=user_profile,
                recommendations=recommendations,
                evidence_summary=evidence_summary
            )

        except Exception as e:
            logger.error(f"Risk scoring failed for user {user_id}: {e}")
            return self._create_default_assessment(user_id, time_window)

    def _calculate_sentiment_score(self, text_processing: TextProcessingResult) -> float:
        """Calculate sentiment and emotional component score (S_t)"""

        features = text_processing.risk_features
        score = 0.0
        total_weight = 0.0

        # Weight different sentiment features
        feature_weights = {
            "negative_sentiment": 0.3,
            "fear_intensity": 0.2,
            "sadness_intensity": 0.2,
            "anger_intensity": 0.15,
            "emotional_intensity": 0.1,
            "stress_lexicon_score": 0.25
        }

        for feature, weight in feature_weights.items():
            if feature in features:
                score += features[feature] * weight
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0

    def _calculate_behavioral_score(self, behavioral_features: BehavioralFeaturesResult) -> float:
        """Calculate behavioral component score (E_t)"""

        risks = behavioral_features.risk_indicators
        score = 0.0
        total_weight = 0.0

        # Weight different behavioral features
        feature_weights = {
            "sleep_disruption": 0.25,
            "irregular_sleep": 0.15,
            "social_withdrawal": 0.20,
            "social_isolation": 0.15,
            "irregular_schedule": 0.10,
            "nocturnal_activity": 0.10,
            "sedentary_behavior": 0.05
        }

        for feature, weight in feature_weights.items():
            if feature in risks:
                score += risks[feature] * weight
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0

    def _calculate_academic_score(self, behavioral_features: BehavioralFeaturesResult) -> float:
        """Calculate academic/study component score (B_t)"""

        risks = behavioral_features.risk_indicators
        score = 0.0
        total_weight = 0.0

        # Weight different academic features
        feature_weights = {
            "procrastination": 0.25,
            "academic_pressure": 0.30,
            "poor_study_patterns": 0.20,
            "low_engagement": 0.15,
            "deadline_pressure_score": 0.10
        }

        for feature, weight in feature_weights.items():
            if feature in risks:
                score += risks[feature] * weight
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0

    def _calculate_contextual_score(
        self,
        text_processing: TextProcessingResult,
        behavioral_features: BehavioralFeaturesResult,
        contextual_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate contextual component score (D_t)"""

        score = 0.0
        total_weight = 0.0

        # Extract contextual features from text processing
        text_features = text_processing.risk_features

        # Text-based contextual features
        contextual_text_features = {
            "urgency_level": 0.4,
            "academic_stress_focus": 0.3
        }

        for feature, weight in contextual_text_features.items():
            if feature in text_features:
                score += text_features[feature] * weight
                total_weight += weight

        # Add external contextual data if available
        if contextual_data:
            external_features = {
                "recent_stress_events": 0.2,
                "support_system_strength": -0.15,  # Negative weight (protective)
                "coping_mechanisms": -0.15          # Negative weight (protective)
            }

            for feature, weight in external_features.items():
                if feature in contextual_data:
                    score += contextual_data[feature] * weight
                    total_weight += abs(weight)

        return max(0, min(score / total_weight if total_weight > 0 else 0.0, 1.0))

    def _apply_adaptive_sigmoid(self, x: float) -> float:
        """Apply adaptive sigmoid function to normalize scores

        Modified sigmoid: σ(x) = 1 / (1 + e^(-k(x - x0)))
        where k controls steepness and x0 controls center point
        """

        # Adaptive parameters based on expected score distribution
        k = 6.0  # Steepness parameter
        x0 = 0.3  # Center point (slightly above 0 for positive bias)

        # Apply sigmoid
        sigmoid_value = 1.0 / (1.0 + np.exp(-k * (x - x0)))

        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, sigmoid_value))

    def _calculate_feature_contributions(
        self,
        text_processing: TextProcessingResult,
        behavioral_features: BehavioralFeaturesResult,
        contextual_data: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate individual feature contributions to overall risk"""

        contributions = {}

        # Text processing contributions
        for feature, value in text_processing.risk_features.items():
            contributions[f"text_{feature}"] = value

        # Behavioral feature contributions
        for feature, value in behavioral_features.risk_indicators.items():
            contributions[f"behavioral_{feature}"] = value

        # Contextual contributions
        if contextual_data:
            for feature, value in contextual_data.items():
                if isinstance(value, (int, float)):
                    contributions[f"contextual_{feature}"] = value

        # Normalize contributions to sum to 1.0
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            contributions = {
                feature: value / total_contribution
                for feature, value in contributions.items()
            }

        return contributions

    async def _classify_risk_level(
        self,
        raw_score: float,
        user_profile: PersonalizedRiskProfile,
        feature_contributions: Dict[str, float]
    ) -> RiskClassification:
        """Classify risk level using personalized adaptive thresholds"""

        # Get personalized thresholds
        thresholds = user_profile.adaptive_thresholds

        # Determine risk level
        risk_level = "low"
        for level, threshold in sorted(thresholds.items(), key=lambda x: x[1], reverse=True):
            if raw_score >= threshold:
                risk_level = level
                break

        # Calculate confidence based on data quality and consistency
        confidence = self._calculate_classification_confidence(
            raw_score, feature_contributions, user_profile
        )

        # Identify primary risk factors (top contributing features)
        primary_factors = sorted(
            feature_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 factors

        primary_risk_factors = [
            feature.replace("_", " ").title() for feature, _ in primary_factors
        ]

        # Determine risk trajectory
        risk_trajectory = self._calculate_risk_trajectory(user_profile)

        # Check if escalation is needed
        escalation_needed = (
            risk_level in ["severe", "crisis"] or
            raw_score >= thresholds.get("crisis", 0.9) * 0.95
        )

        # Calculate percentile rank (simplified - would use population data in production)
        percentile_rank = self._estimate_percentile_rank(raw_score, user_profile)

        return RiskClassification(
            risk_level=risk_level,
            risk_score=raw_score,
            confidence=confidence,
            primary_risk_factors=primary_risk_factors,
            risk_trajectory=risk_trajectory,
            escalation_needed=escalation_needed,
            percentile_rank=percentile_rank
        )

    def _calculate_classification_confidence(
        self,
        raw_score: float,
        feature_contributions: Dict[str, float],
        user_profile: PersonalizedRiskProfile
    ) -> float:
        """Calculate confidence in risk classification"""

        # Base confidence from score extremity
        if raw_score < 0.2 or raw_score > 0.8:
            score_confidence = 0.9  # High confidence at extremes
        elif 0.3 <= raw_score <= 0.7:
            score_confidence = 0.6  # Lower confidence in middle range
        else:
            score_confidence = 0.75

        # Data quality confidence
        data_quality = min(len(feature_contributions) / 10.0, 1.0)

        # Historical consistency confidence
        if len(user_profile.risk_history) > 0:
            recent_scores = [entry.get("score", 0) for entry in user_profile.risk_history[-5:]]
            if recent_scores:
                score_variance = np.var(recent_scores)
                consistency_confidence = max(0, 1.0 - score_variance * 2)
            else:
                consistency_confidence = 0.5
        else:
            consistency_confidence = 0.3

        # Overall confidence
        overall_confidence = (
            score_confidence * 0.4 +
            data_quality * 0.3 +
            consistency_confidence * 0.3
        )

        return min(overall_confidence, 1.0)

    def _calculate_risk_trajectory(self, user_profile: PersonalizedRiskProfile) -> str:
        """Calculate risk trajectory based on historical data"""

        if len(user_profile.risk_history) < 3:
            return "stable"

        # Get recent scores
        recent_scores = [
            entry.get("score", 0) for entry in user_profile.risk_history[-5:]
        ]

        if len(recent_scores) < 2:
            return "stable"

        # Calculate trend
        x = np.arange(len(recent_scores))
        slope, _ = np.polyfit(x, recent_scores, 1)

        # Determine trajectory
        if slope > 0.05:
            return "worsening"
        elif slope < -0.05:
            return "improving"
        else:
            return "stable"

    def _estimate_percentile_rank(
        self,
        raw_score: float,
        user_profile: PersonalizedRiskProfile
    ) -> float:
        """Estimate percentile rank (simplified version)"""

        # In production, this would use actual population data
        # For now, use a simplified mapping based on score
        if raw_score < 0.1:
            return raw_score * 200  # Bottom 20%
        elif raw_score < 0.3:
            return 20 + (raw_score - 0.1) * 200  # 20-60%
        elif raw_score < 0.7:
            return 60 + (raw_score - 0.3) * 100  # 60-100%
        else:
            return 90 + (raw_score - 0.7) * 33  # 90-100%

    async def _generate_risk_recommendations(
        self,
        classification: RiskClassification,
        feature_contributions: Dict[str, float],
        user_profile: PersonalizedRiskProfile
    ) -> List[Dict[str, Any]]:
        """Generate personalized risk-based recommendations"""

        recommendations = []

        # Crisis recommendations
        if classification.risk_level == "crisis":
            recommendations.append({
                "type": "crisis_intervention",
                "priority": "immediate",
                "title": "Immediate Crisis Support",
                "content": "Please reach out to crisis services immediately. Call 988 or text HOME to 741741",
                "action_required": True,
                "contact_info": {
                    "hotline": "988",
                    "text": "HOME to 741741",
                    "emergency": "911"
                },
                "evidence": f"Crisis risk level detected (score: {classification.risk_score:.2f})"
            })

        # High-priority recommendations for severe risk
        if classification.risk_level == "severe":
            recommendations.append({
                "type": "professional_support",
                "priority": "high",
                "title": "Seek Professional Support",
                "content": "Consider reaching out to counseling services or mental health professionals",
                "action_required": True,
                "evidence": f"Severe risk level detected (score: {classification.risk_score:.2f})"
            })

        # Feature-specific recommendations
        top_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)[:3]

        for feature, contribution in top_features:
            if "sleep" in feature.lower() and contribution > 0.15:
                recommendations.append({
                    "type": "sleep_improvement",
                    "priority": "medium",
                    "title": "Improve Sleep Patterns",
                    "content": "Focus on consistent sleep schedule and sleep hygiene practices",
                    "evidence": f"Sleep factors contribute {contribution:.1%} to risk score"
                })

            elif "social" in feature.lower() and contribution > 0.15:
                recommendations.append({
                    "type": "social_connection",
                    "priority": "medium",
                    "title": "Increase Social Connection",
                    "content": "Consider joining study groups or scheduling regular social activities",
                    "evidence": f"Social factors contribute {contribution:.1%} to risk score"
                })

            elif "academic" in feature.lower() and contribution > 0.15:
                recommendations.append({
                    "type": "academic_support",
                    "priority": "medium",
                    "title": "Academic Stress Management",
                    "content": "Consider meeting with academic advisors or using campus tutoring services",
                    "evidence": f"Academic factors contribute {contribution:.1%} to risk score"
                })

        # General wellness recommendations
        if classification.risk_level in ["moderate", "mild"]:
            recommendations.append({
                "type": "wellness_practices",
                "priority": "low",
                "title": "Wellness and Self-Care",
                "content": "Practice regular stress management techniques and maintain healthy routines",
                "evidence": f"Preventive care for {classification.risk_level} risk level"
            })

        return recommendations

    def _create_evidence_summary(
        self,
        text_processing: TextProcessingResult,
        behavioral_features: BehavioralFeaturesResult,
        feature_contributions: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create summary of evidence supporting the risk assessment"""

        summary = {
            "data_sources_used": [],
            "key_findings": [],
            "top_risk_factors": [],
            "protective_factors": [],
            "data_quality_indicators": {}
        }

        # Data sources
        if text_processing.text_hash != "empty":
            summary["data_sources_used"].append("text_messages")

        if behavioral_features.user_id:
            summary["data_sources_used"].append("behavioral_data")

        # Key findings from text processing
        if text_processing.sentiment.negative_score > 0.6:
            summary["key_findings"].append("High negative sentiment detected")

        if text_processing.stress_lexicon.urgency_level in ["high", "crisis"]:
            summary["key_findings"].append(f"High urgency stress indicators: {text_processing.stress_lexicon.urgency_level}")

        # Key findings from behavioral features
        if behavioral_features.sleep.sleep_disruption_score > 0.6:
            summary["key_findings"].append("Significant sleep disruption detected")

        if behavioral_features.social.social_withdrawal_score > 0.6:
            summary["key_findings"].append("Social withdrawal patterns detected")

        # Top risk factors
        top_factors = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)[:5]
        summary["top_risk_factors"] = [
            {"factor": factor.replace("_", " ").title(), "contribution": f"{contribution:.1%}"}
            for factor, contribution in top_factors
        ]

        # Data quality indicators
        summary["data_quality_indicators"] = {
            "text_processing_available": text_processing.text_hash != "empty",
            "behavioral_data_available": len(behavioral_features.risk_indicators) > 0,
            "feature_count": len(feature_contributions),
            "analysis_timestamp": datetime.now().isoformat()
        }

        return summary

    async def _get_or_create_profile(self, user_id: str) -> PersonalizedRiskProfile:
        """Get existing user profile or create new one"""

        if user_id in self.user_profiles:
            return self.user_profiles[user_id]

        # Create new profile with defaults
        new_profile = PersonalizedRiskProfile(
            user_id=user_id,
            baseline_score=0.0,  # Will be updated over time
            sensitivity_weights=self.default_weights.copy(),
            adaptive_thresholds=self.base_thresholds.copy(),
            protective_factors=[],
            risk_history=[],
            last_updated=datetime.now()
        )

        self.user_profiles[user_id] = new_profile
        return new_profile

    async def _update_user_profile(
        self,
        user_id: str,
        risk_score: float,
        classification: RiskClassification
    ):
        """Update user profile with new assessment data"""

        if user_id not in self.user_profiles:
            return

        profile = self.user_profiles[user_id]

        # Add to risk history
        profile.risk_history.append({
            "timestamp": datetime.now().isoformat(),
            "score": risk_score,
            "risk_level": classification.risk_level,
            "primary_factors": classification.primary_risk_factors
        })

        # Keep only last 90 days of history
        cutoff_date = datetime.now() - timedelta(days=90)
        profile.risk_history = [
            entry for entry in profile.risk_history
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_date
        ]

        # Update baseline score (exponential moving average)
        if len(profile.risk_history) > 1:
            alpha = 0.1  # Smoothing factor
            profile.baseline_score = (
                alpha * risk_score +
                (1 - alpha) * profile.baseline_score
            )

        # Adaptive threshold adjustment based on history
        if len(profile.risk_history) > 5:
            await self._adjust_adaptive_thresholds(profile)

        profile.last_updated = datetime.now()

    async def _adjust_adaptive_thresholds(self, profile: PersonalizedRiskProfile):
        """Adjust thresholds based on user's historical patterns"""

        if len(profile.risk_history) < 5:
            return

        # Calculate personal baseline and variance
        scores = [entry["score"] for entry in profile.risk_history[-10:]]
        personal_mean = np.mean(scores)
        personal_std = np.std(scores)

        # Adjust thresholds based on personal patterns
        adjustment_factor = personal_mean - 0.5  # Deviation from population mean

        for level in profile.adaptive_thresholds:
            base_threshold = self.base_thresholds[level]
            # Personalize thresholds
            profile.adaptive_thresholds[level] = max(0.1, min(0.95,
                base_threshold + adjustment_factor * 0.3
            ))

    def _create_default_assessment(self, user_id: str, time_window: int) -> ComprehensiveRiskAssessment:
        """Create default assessment when analysis fails"""

        default_classification = RiskClassification(
            risk_level="low",
            risk_score=0.0,
            confidence=0.0,
            primary_risk_factors=["analysis_error"],
            risk_trajectory="stable",
            escalation_needed=False,
            percentile_rank=0.0
        )

        default_profile = PersonalizedRiskProfile(
            user_id=user_id,
            baseline_score=0.0,
            sensitivity_weights=self.default_weights,
            adaptive_thresholds=self.base_thresholds,
            protective_factors=[],
            risk_history=[],
            last_updated=datetime.now()
        )

        return ComprehensiveRiskAssessment(
            user_id=user_id,
            assessment_timestamp=datetime.now(),
            time_window_days=time_window,
            classification=default_classification,
            component_scores={"sentiment": 0.0, "behavioral": 0.0, "academic": 0.0, "contextual": 0.0},
            feature_contributions={},
            personalized_profile=default_profile,
            recommendations=[{
                "type": "system",
                "priority": "medium",
                "title": "Analysis Unavailable",
                "content": "Unable to complete risk assessment. Please try again later."
            }],
            evidence_summary={"error": "analysis_failed"}
        )

# Global risk scoring engine instance
risk_scoring_engine = RiskScoringEngine()