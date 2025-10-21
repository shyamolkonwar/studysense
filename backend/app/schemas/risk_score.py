from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRISIS = "crisis"


class RiskScoreBase(BaseModel):
    """Base risk score schema"""
    risk_level: RiskLevel = Field(..., description="Risk classification level")
    overall_score: float = Field(..., ge=0, le=1, description="Overall risk score (0-1)")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in assessment")

    # Component scores
    sentiment_score: float = Field(..., ge=0, le=1, description="Sentiment-based risk score")
    emotion_score: float = Field(..., ge=0, le=1, description="Emotion-based risk score")
    behavioral_score: float = Field(..., ge=0, le=1, description="Behavioral pattern risk score")
    academic_score: float = Field(..., ge=0, le=1, description="Academic pressure risk score")
    social_score: float = Field(..., ge=0, le=1, description="Social interaction risk score")
    sleep_score: Optional[float] = Field(None, ge=0, le=1, description="Sleep pattern risk score")

    class Config:
        json_schema_extra = {
            "example": {
                "risk_level": "mild",
                "overall_score": 0.42,
                "confidence_score": 0.87,
                "sentiment_score": 0.35,
                "emotion_score": 0.28,
                "behavioral_score": 0.51,
                "academic_score": 0.48,
                "social_score": 0.39,
                "sleep_score": 0.45
            }
        }


class RiskScoreCreate(RiskScoreBase):
    """Schema for creating a new risk score calculation"""
    user_id: int = Field(..., description="User ID")
    feature_vector: Dict[str, float] = Field(..., description="Normalized feature vector")
    feature_weights: Dict[str, float] = Field(..., description="Weights used for calculation")
    academic_calendar_phase: Optional[str] = Field(None, description="Academic calendar context")
    deadline_density: int = Field(default=0, ge=0, description="Number of deadlines in next 7 days")
    recent_stress_events: Optional[List[Dict[str, Any]]] = Field(None, description="Recent stress triggers")
    model_version: str = Field(..., description="Model version used")
    algorithm_used: str = Field(..., description="Algorithm used for calculation")
    data_period_start: datetime = Field(..., description="Analysis period start")
    data_period_end: datetime = Field(..., description="Analysis period end")
    data_points_analyzed: int = Field(default=0, ge=0, description="Number of data points analyzed")


class RiskScoreUpdate(BaseModel):
    """Schema for updating risk score metadata"""
    requires_review: Optional[bool] = None
    reviewed_by: Optional[int] = None
    reviewed_at: Optional[datetime] = None
    review_notes: Optional[str] = None
    alert_triggered: Optional[bool] = None
    alert_reason: Optional[str] = None
    alert_threshold_breached: Optional[str] = None


class RiskScoreResponse(RiskScoreBase):
    """Schema for risk score response"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int

    # Trend analysis
    previous_score: Optional[float]
    score_change_24h: Optional[float]
    score_change_7d: Optional[float]
    trend_direction: Optional[str]

    # Contextual factors
    academic_calendar_phase: Optional[str]
    deadline_density: int
    recent_stress_events: Optional[List[Dict[str, Any]]]

    # Alert information
    alert_triggered: bool
    alert_reason: Optional[str]
    alert_threshold_breached: Optional[str]

    # Temporal information
    calculated_at: datetime
    data_period_start: datetime
    data_period_end: datetime
    data_points_analyzed: int

    # Model metadata
    model_version: str
    algorithm_used: str
    requires_review: bool
    reviewed_by: Optional[int]
    reviewed_at: Optional[datetime]
    review_notes: Optional[str]

    # Feature data (excluded from public responses for privacy)
    # feature_vector: Dict[str, float]
    # feature_weights: Dict[str, float]

    class Config:
        json_schema_extra = {
            "example": {
                "id": 123,
                "user_id": 1,
                "risk_level": "mild",
                "overall_score": 0.42,
                "confidence_score": 0.87,
                "sentiment_score": 0.35,
                "emotion_score": 0.28,
                "behavioral_score": 0.51,
                "academic_score": 0.48,
                "social_score": 0.39,
                "sleep_score": 0.45,
                "previous_score": 0.38,
                "score_change_24h": 0.04,
                "score_change_7d": 0.08,
                "trend_direction": "improving",
                "academic_calendar_phase": "exam_preparation",
                "deadline_density": 3,
                "recent_stress_events": [
                    {"type": "deadline", "severity": "medium", "timestamp": "2024-01-01T10:00:00Z"}
                ],
                "alert_triggered": False,
                "alert_reason": None,
                "alert_threshold_breached": None,
                "calculated_at": "2024-01-01T12:00:00Z",
                "data_period_start": "2024-01-01T00:00:00Z",
                "data_period_end": "2024-01-01T23:59:59Z",
                "data_points_analyzed": 156,
                "model_version": "v1.2.0",
                "algorithm_used": "weighted_ensemble_v2",
                "requires_review": False,
                "reviewed_by": None,
                "reviewed_at": None,
                "review_notes": None
            }
        }


class RiskScoreTrend(BaseModel):
    """Schema for risk score trend analysis"""
    user_id: int
    current_score: RiskScoreResponse
    historical_scores: List[RiskScoreResponse]
    trend_analysis: Dict[str, Any]
    risk_trajectory: str  # improving/stable/declining/fluctuating
    key_contributing_factors: List[str]
    recommendations: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "current_score": {},
                "historical_scores": [],
                "trend_analysis": {
                    "slope_7d": -0.02,
                    "volatility": 0.15,
                    "pattern_recognition": "stress_spike_before_deadlines"
                },
                "risk_trajectory": "improving",
                "key_contributing_factors": [
                    "academic_pressure",
                    "sleep_disruption",
                    "social_withdrawal"
                ],
                "recommendations": [
                    "Schedule study breaks",
                    "Practice mindfulness",
                    "Connect with study group"
                ]
            }
        }


class RiskScoreBatch(BaseModel):
    """Schema for batch risk score calculation"""
    user_ids: List[int] = Field(..., max_items=100, description="User IDs to calculate scores for")
    calculation_options: Optional[Dict[str, Any]] = Field(None, description="Calculation options")
    force_recalculate: bool = Field(default=False, description="Force recalculation even if recent scores exist")


class RiskScoreComparison(BaseModel):
    """Schema for comparing risk scores across users or time periods"""
    comparison_type: str = Field(..., description="Type of comparison: users, time_periods, demographics")
    parameters: Dict[str, Any] = Field(..., description="Comparison parameters")
    results: List[Dict[str, Any]] = Field(..., description="Comparison results")
    statistical_significance: Optional[float] = Field(None, description="Statistical significance level")
    insights: List[str] = Field(..., description="Key insights from comparison")