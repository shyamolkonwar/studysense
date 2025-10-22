"""
Analysis Pipeline Integration
Integrates all analysis components into a unified analysis pipeline
as specified in the MVP requirements.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import json

from .text_processing import TextProcessingPipeline, TextProcessingResult, text_processor
from .behavioral_features import BehavioralFeaturesExtractor, BehavioralFeaturesResult, behavioral_extractor
from .risk_scoring_engine import RiskScoringEngine, ComprehensiveRiskAssessment, risk_scoring_engine
from .recommendations_engine import RAGRecommendationsEngine, PersonalizedRecommendations, recommendations_engine

logger = logging.getLogger(__name__)

@dataclass
class AnalysisRequest:
    """Request structure for comprehensive analysis"""
    user_id: str
    messages: List[Dict[str, Any]]
    activities: Optional[List[Dict[str, Any]]] = None
    calendar_events: Optional[List[Dict[str, Any]]] = None
    user_context: Optional[Dict[str, Any]] = None
    time_window: int = 7
    include_recommendations: bool = True
    personalization_enabled: bool = True

@dataclass
class AnalysisResult:
    """Complete analysis result"""
    user_id: str
    analysis_timestamp: datetime
    time_window_days: int
    text_processing: TextProcessingResult
    behavioral_features: BehavioralFeaturesResult
    risk_assessment: ComprehensiveRiskAssessment
    recommendations: Optional[PersonalizedRecommendations]
    analysis_metadata: Dict[str, Any]
    processing_summary: Dict[str, Any]

class AnalysisPipeline:
    """
    Main integration pipeline for comprehensive analysis.
    Coordinates text processing, behavioral analysis, risk scoring, and recommendations.
    """

    def __init__(self):
        """Initialize analysis pipeline with all components"""

        self.text_processor = text_processor
        self.behavioral_extractor = behavioral_extractor
        self.risk_scoring_engine = risk_scoring_engine
        self.recommendations_engine = recommendations_engine

        # Processing configuration
        self.max_concurrent_processes = 4
        self.timeout_seconds = 300  # 5 minutes max processing time

        # Analysis quality thresholds
        self.min_data_quality_threshold = 0.3
        self.high_confidence_threshold = 0.8

        logger.info("Analysis Pipeline initialized")

    async def analyze_comprehensive(
        self,
        request: AnalysisRequest
    ) -> AnalysisResult:
        """
        Perform comprehensive analysis

        Args:
            request: Complete analysis request with all data

        Returns:
            Complete AnalysisResult with all analyses
        """

        start_time = datetime.now()
        analysis_id = f"{request.user_id}_{start_time.strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting comprehensive analysis: {analysis_id}")

        try:
            # Validate request data
            validation_result = self._validate_request(request)
            if not validation_result["valid"]:
                raise ValueError(f"Invalid request: {validation_result['errors']}")

            # Execute analysis components concurrently
            text_processing_task = self._process_text_data(request)
            behavioral_analysis_task = self._process_behavioral_data(request)

            # Wait for both analyses to complete
            text_result, behavioral_result = await asyncio.gather(
                text_processing_task,
                behavioral_analysis_task,
                return_exceptions=True
            )

            # Handle exceptions
            if isinstance(text_result, Exception):
                logger.error(f"Text processing failed: {text_result}")
                text_result = self._create_default_text_result(request.user_id)

            if isinstance(behavioral_result, Exception):
                logger.error(f"Behavioral analysis failed: {behavioral_result}")
                behavioral_result = self._create_default_behavioral_result(request.user_id, request.time_window)

            # Calculate comprehensive risk assessment
            risk_assessment = await self._calculate_risk_assessment(
                request, text_result, behavioral_result
            )

            # Generate recommendations if requested
            recommendations = None
            if request.include_recommendations:
                recommendations = await self._generate_recommendations(
                    request, risk_assessment
                )

            # Create processing metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            analysis_metadata = self._create_analysis_metadata(
                analysis_id, request, processing_time
            )

            # Create processing summary
            processing_summary = self._create_processing_summary(
                text_result, behavioral_result, risk_assessment, recommendations
            )

            result = AnalysisResult(
                user_id=request.user_id,
                analysis_timestamp=start_time,
                time_window_days=request.time_window,
                text_processing=text_result,
                behavioral_features=behavioral_result,
                risk_assessment=risk_assessment,
                recommendations=recommendations,
                analysis_metadata=analysis_metadata,
                processing_summary=processing_summary
            )

            logger.info(f"Analysis completed: {analysis_id} in {processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Analysis failed for {request.user_id}: {e}")
            return self._create_error_result(request, str(e), start_time)

    async def _process_text_data(self, request: AnalysisRequest) -> TextProcessingResult:
        """Process text data with comprehensive analysis"""

        if not request.messages:
            return self._create_default_text_result(request.user_id)

        return await self.text_processor.process_text(
            messages=request.messages,
            time_window=request.time_window,
            include_burstiness=True
        )

    async def _process_behavioral_data(self, request: AnalysisRequest) -> BehavioralFeaturesResult:
        """Process behavioral data with comprehensive analysis"""

        if not request.activities:
            return self._create_default_behavioral_result(request.user_id, request.time_window)

        return await self.behavioral_extractor.extract_features(
            user_id=request.user_id,
            activities=request.activities,
            messages=request.messages,
            calendar_events=request.calendar_events,
            time_window=request.time_window
        )

    async def _calculate_risk_assessment(
        self,
        request: AnalysisRequest,
        text_result: TextProcessingResult,
        behavioral_result: BehavioralFeaturesResult
    ) -> ComprehensiveRiskAssessment:
        """Calculate comprehensive risk assessment"""

        # Prepare contextual data
        contextual_data = self._prepare_contextual_data(request)

        return await self.risk_scoring_engine.calculate_risk_score(
            user_id=request.user_id,
            text_processing=text_result,
            behavioral_features=behavioral_result,
            contextual_data=contextual_data,
            time_window=request.time_window
        )

    async def _generate_recommendations(
        self,
        request: AnalysisRequest,
        risk_assessment: ComprehensiveRiskAssessment
    ) -> PersonalizedRecommendations:
        """Generate personalized recommendations"""

        # Prepare risk context for recommendations
        risk_context = {
            "risk_level": risk_assessment.classification.risk_level,
            "risk_score": risk_assessment.classification.risk_score,
            "primary_concerns": risk_assessment.classification.primary_risk_factors,
            "escalation_needed": risk_assessment.classification.escalation_needed,
            "component_scores": risk_assessment.component_scores,
            "feature_contributions": risk_assessment.feature_contributions
        }

        return await self.recommendations_engine.generate_recommendations(
            user_id=request.user_id,
            risk_assessment=risk_context,
            user_context=request.user_context,
            limit_per_type=3
        )

    def _validate_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Validate analysis request"""

        errors = []

        # Check required fields
        if not request.user_id:
            errors.append("user_id is required")

        if not isinstance(request.time_window, int) or request.time_window < 1:
            errors.append("time_window must be a positive integer")

        if request.time_window > 90:
            errors.append("time_window cannot exceed 90 days")

        # Check data quality
        if not request.messages and not request.activities:
            errors.append("At least messages or activities must be provided")

        # Check message format
        if request.messages:
            for i, msg in enumerate(request.messages[:5]):  # Check first 5 messages
                if not isinstance(msg, dict):
                    errors.append(f"Message {i} must be a dictionary")
                elif "content" not in msg:
                    errors.append(f"Message {i} must contain 'content' field")

        # Check activity format
        if request.activities:
            for i, activity in enumerate(request.activities[:5]):  # Check first 5 activities
                if not isinstance(activity, dict):
                    errors.append(f"Activity {i} must be a dictionary")
                elif "type" not in activity and "category" not in activity:
                    errors.append(f"Activity {i} must contain 'type' or 'category' field")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": []  # Could add warnings for suboptimal data
        }

    def _prepare_contextual_data(self, request: AnalysisRequest) -> Optional[Dict[str, Any]]:
        """Prepare contextual data for risk assessment"""

        contextual = {}

        # Add calendar event context
        if request.calendar_events:
            upcoming_deadlines = [
                event for event in request.calendar_events
                if event.get("type") in ["deadline", "exam"] and
                   self._is_event_upcoming(event, 7)
            ]
            contextual["deadline_proximity"] = min(len(upcoming_deadlines) / 5.0, 1.0)

        # Add user context
        if request.user_context:
            contextual.update({
                "support_system_strength": request.user_context.get("support_system", 0.5),
                "coping_mechanisms": request.user_context.get("coping_skills", 0.5),
                "recent_stress_events": request.user_context.get("recent_events", 0.0)
            })

        return contextual if contextual else None

    def _is_event_upcoming(self, event: Dict[str, Any], days: int) -> bool:
        """Check if event is upcoming within specified days"""
        try:
            event_date = event.get("date")
            if not event_date:
                return False

            if isinstance(event_date, str):
                event_date = datetime.fromisoformat(event_date.replace('Z', '+00:00'))

            cutoff_date = datetime.now() + timedelta(days=days)
            return datetime.now() <= event_date <= cutoff_date

        except Exception:
            return False

    def _create_analysis_metadata(
        self,
        analysis_id: str,
        request: AnalysisRequest,
        processing_time: float
    ) -> Dict[str, Any]:
        """Create analysis metadata"""

        return {
            "analysis_id": analysis_id,
            "engine_version": "4.0.0",
            "processing_time_seconds": processing_time,
            "data_sources": {
                "messages": len(request.messages) if request.messages else 0,
                "activities": len(request.activities) if request.activities else 0,
                "calendar_events": len(request.calendar_events) if request.calendar_events else 0
            },
            "configuration": {
                "time_window_days": request.time_window,
                "include_recommendations": request.include_recommendations,
                "personalization_enabled": request.personalization_enabled
            },
            "quality_metrics": {
                "data_completeness": self._calculate_data_completeness(request),
                "temporal_coverage": self._calculate_temporal_coverage(request),
                "multimodal_richness": self._calculate_multimodal_richness(request)
            }
        }

    def _create_processing_summary(
        self,
        text_result: TextProcessingResult,
        behavioral_result: BehavioralFeaturesResult,
        risk_assessment: ComprehensiveRiskAssessment,
        recommendations: Optional[PersonalizedRecommendations]
    ) -> Dict[str, Any]:
        """Create processing summary"""

        summary = {
            "components_completed": {
                "text_processing": text_result.text_hash != "empty",
                "behavioral_analysis": behavioral_result.user_id != "",
                "risk_scoring": risk_assessment.classification.risk_level is not None,
                "recommendations": recommendations is not None
            },
            "key_insights": {
                "risk_level": risk_assessment.classification.risk_level,
                "risk_score": risk_assessment.classification.risk_score,
                "confidence": risk_assessment.classification.confidence,
                "escalation_needed": risk_assessment.classification.escalation_needed
            },
            "data_quality": {
                "text_analysis_quality": 0.8 if text_result.text_hash != "empty" else 0.0,
                "behavioral_analysis_quality": 0.8 if behavioral_result.user_id != "" else 0.0,
                "overall_quality": risk_assessment.classification.confidence
            }
        }

        if recommendations:
            summary["recommendations_summary"] = {
                "total_recommendations": len(recommendations.recommendations),
                "immediate_actions": len(recommendations.categorized_recommendations.get("immediate_actions", [])),
                "high_priority": len(recommendations.categorized_recommendations.get("high_priority", []))
            }

        return summary

    def _calculate_data_completeness(self, request: AnalysisRequest) -> float:
        """Calculate data completeness score"""

        scores = []

        # Message data
        if request.messages:
            messages_with_content = sum(1 for msg in request.messages if msg.get("content"))
            scores.append(messages_with_content / len(request.messages))
        else:
            scores.append(0.0)

        # Activity data
        if request.activities:
            activities_with_type = sum(1 for act in request.activities if act.get("type") or act.get("category"))
            scores.append(activities_with_type / len(request.activities))
        else:
            scores.append(0.0)

        # Calendar data
        if request.calendar_events:
            scores.append(1.0)
        else:
            scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def _calculate_temporal_coverage(self, request: AnalysisRequest) -> float:
        """Calculate temporal coverage score"""

        if not request.messages and not request.activities:
            return 0.0

        # Check if data spans the requested time window
        all_timestamps = []

        for msg in request.messages or []:
            if msg.get("timestamp"):
                all_timestamps.append(msg["timestamp"])

        for activity in request.activities or []:
            if activity.get("timestamp"):
                all_timestamps.append(activity["timestamp"])

        if len(all_timestamps) < 2:
            return 0.5  # Some data but limited temporal coverage

        # Convert to datetime objects
        dates = []
        for ts in all_timestamps:
            try:
                if isinstance(ts, str):
                    dates.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                elif isinstance(ts, (int, float)):
                    dates.append(datetime.fromtimestamp(ts))
            except Exception:
                continue

        if len(dates) < 2:
            return 0.5

        # Calculate coverage
        time_span = (max(dates) - min(dates)).days
        expected_span = request.time_window

        return min(time_span / expected_span, 1.0)

    def _calculate_multimodal_richness(self, request: AnalysisRequest) -> float:
        """Calculate multimodal data richness score"""

        modalities = 0

        if request.messages:
            modalities += 1

        if request.activities:
            modalities += 1

        if request.calendar_events:
            modalities += 1

        return modalities / 3.0

    def _create_default_text_result(self, user_id: str) -> TextProcessingResult:
        """Create default text processing result"""

        from .text_processing import (
            SentimentAnalysis, EmotionAnalysis, StressLexiconAnalysis,
            BurstinessAnalysis
        )

        return TextProcessingResult(
            text_hash="empty",
            processing_timestamp=datetime.now(),
            sentiment=SentimentAnalysis(0.0, 0.0, 0.0, 1.0, "neutral", 0.0, 0.0),
            emotions=EmotionAnalysis("neutral", {}, 0.0, 0.0),
            stress_lexicon=StressLexiconAnalysis(0.0, [], [], [], "low"),
            burstiness=BurstinessAnalysis(0.0, 0.0, 0.0, 0.0),
            linguistic_features={},
            risk_features={}
        )

    def _create_default_behavioral_result(self, user_id: str, time_window: int) -> BehavioralFeaturesResult:
        """Create default behavioral features result"""

        from .behavioral_features import (
            SleepAnalysis, SocialAnalysis, AcademicBehaviorAnalysis,
            ActivityPatternAnalysis
        )

        return BehavioralFeaturesResult(
            user_id=user_id,
            analysis_timestamp=datetime.now(),
            time_window_days=time_window,
            sleep=SleepAnalysis(0.0, 0.0, 0.5, 0.0, 0.0, "stable"),
            social=SocialAnalysis(0.0, 0.5, 0.0, 0.0, 0.5),
            academic=AcademicBehaviorAnalysis(0.0, 0.0, 0.0, {}, 0.5),
            activity_patterns=ActivityPatternAnalysis(0.0, 0.0, 0.0, 0.0, 0.5),
            risk_indicators={},
            recommendations=[]
        )

    def _create_error_result(
        self,
        request: AnalysisRequest,
        error_message: str,
        start_time: datetime
    ) -> AnalysisResult:
        """Create error result when analysis fails"""

        processing_time = (datetime.now() - start_time).total_seconds()

        return AnalysisResult(
            user_id=request.user_id,
            analysis_timestamp=start_time,
            time_window_days=request.time_window,
            text_processing=self._create_default_text_result(request.user_id),
            behavioral_features=self._create_default_behavioral_result(request.user_id, request.time_window),
            risk_assessment=self.risk_scoring_engine._create_default_assessment(
                request.user_id, request.time_window
            ),
            recommendations=None,
            analysis_metadata={
                "analysis_id": "error",
                "engine_version": "4.0.0",
                "processing_time_seconds": processing_time,
                "error": error_message
            },
            processing_summary={
                "components_completed": {
                    "text_processing": False,
                    "behavioral_analysis": False,
                    "risk_scoring": False,
                    "recommendations": False
                },
                "error": error_message
            }
        )

# Global analysis pipeline instance
analysis_pipeline = AnalysisPipeline()
