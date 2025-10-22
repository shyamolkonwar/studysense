from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio

from .tool_manager import tool_manager, ToolCall, ToolResult
from ..llm.llm_service import LLMService, LLMMessage, LLMConfig, llm_service
from ..rag.retrieval import retrieval_pipeline

logger = logging.getLogger(__name__)

@dataclass
class StressAnalysis:
    """Represents stress analysis results"""
    overall_score: float  # 0-1 scale
    severity_level: str   # "low", "moderate", "high", "severe", "crisis"
    contributing_factors: List[str]
    recommendations: List[Dict[str, Any]]
    confidence: float     # 0-1 scale
    analysis_date: datetime
    user_id: str

class StressAnalyzerAgent:
    """
    Agent for analyzing student stress levels using multiple data sources
    and providing evidence-based recommendations
    """

    def __init__(self, llm_service: LLMService = None):
        """
        Initialize stress analyzer agent

        Args:
            llm_service: LLM service for analysis
        """
        self.llm_service = llm_service or llm_service
        self.tool_manager = tool_manager
        self.retrieval_pipeline = retrieval_pipeline

        # Stress analysis thresholds
        self.severity_thresholds = {
            "low": 0.3,
            "moderate": 0.5,
            "high": 0.7,
            "severe": 0.85,
            "crisis": 0.95
        }

        logger.info("Stress Analyzer Agent initialized")

    async def analyze_student_stress(
        self,
        user_id: str,
        messages: List[Dict[str, Any]],
        activities: Optional[List[Dict[str, Any]]] = None,
        calendar_events: Optional[List[Dict[str, Any]]] = None,
        time_window: int = 7
    ) -> StressAnalysis:
        """
        Analyze student's stress level from multiple data sources

        Args:
            user_id: Unique identifier for the student
            messages: Recent messages/text data
            activities: Recent activities and behaviors
            calendar_events: Upcoming academic events
            time_window: Number of days to analyze

        Returns:
            StressAnalysis with comprehensive results
        """
        try:
            # Step 1: Analyze textual content for stress indicators
            text_analysis = await self._analyze_textual_stress_indicators(messages, time_window)

            # Step 2: Analyze behavioral patterns
            behavior_analysis = await self._analyze_behavioral_patterns(activities, time_window)

            # Step 3: Analyze academic context
            academic_analysis = await self._analyze_academic_context(calendar_events, time_window)

            # Step 4: Retrieve relevant stress management resources
            resource_analysis = await self._retrieve_stress_resources(text_analysis, behavior_analysis)

            # Step 5: Synthesize analysis using LLM
            comprehensive_analysis = await self._synthesize_stress_analysis(
                text_analysis=text_analysis,
                behavior_analysis=behavior_analysis,
                academic_analysis=academic_analysis,
                resource_analysis=resource_analysis,
                user_id=user_id
            )

            return comprehensive_analysis

        except Exception as e:
            logger.error(f"Stress analysis failed for user {user_id}: {e}")
            # Return safe default analysis
            return StressAnalysis(
                overall_score=0.0,
                severity_level="unknown",
                contributing_factors=["analysis_error"],
                recommendations=[{"type": "system", "content": "Unable to complete stress analysis. Please try again later."}],
                confidence=0.0,
                analysis_date=datetime.now(),
                user_id=user_id
            )

    async def _analyze_textual_stress_indicators(
        self,
        messages: List[Dict[str, Any]],
        time_window: int
    ) -> Dict[str, Any]:
        """Analyze textual content for stress indicators"""

        if not messages:
            return {"score": 0.0, "indicators": [], "sentiment": "neutral"}

        # Combine recent messages
        recent_messages = [
            msg for msg in messages
            if self._is_within_time_window(msg.get("timestamp"), time_window)
        ]

        if not recent_messages:
            return {"score": 0.0, "indicators": [], "sentiment": "neutral"}

        # Create analysis prompt
        combined_text = " ".join([msg.get("content", "") for msg in recent_messages[-10:]])

        analysis_prompt = f"""
        Analyze the following text for stress indicators in an academic context:

        Text: "{combined_text}"

        Provide analysis in JSON format with:
        - stress_score (0-1): Overall stress level
        - indicators: List of stress indicators found
        - sentiment: "positive", "neutral", or "negative"
        - themes: List of main stress themes
        - urgency: "low", "medium", "high", or "crisis"

        Focus on academic stress patterns like:
        - Exam/deadline anxiety
        - Overwhelm from workload
        - Time management concerns
        - Performance pressure
        - Social/academic balance issues
        """

        try:
            messages_for_llm = [
                LLMMessage(
                    role="system",
                    content="You are a stress analysis expert specializing in student mental health. Provide objective, evidence-based analysis."
                ),
                LLMMessage(
                    role="user",
                    content=analysis_prompt
                )
            ]

            response = await self.llm_service.generate(
                messages=messages_for_llm,
                config=LLMConfig(
                    provider=self.llm_service.default_config.provider,
                    model=self.llm_service.default_config.model,
                    temperature=0.3,  # Lower temperature for consistent analysis
                    max_tokens=500
                )
            )

            # Parse JSON response
            import json
            try:
                analysis_result = json.loads(response.content)
                return analysis_result
            except json.JSONDecodeError:
                # Fallback to text parsing
                return {
                    "score": 0.5,
                    "indicators": ["text_analysis_fallback"],
                    "sentiment": "neutral",
                    "themes": ["analysis_uncertain"],
                    "urgency": "medium"
                }

        except Exception as e:
            logger.error(f"Text analysis error: {e}")
            return {"score": 0.0, "indicators": ["text_error"], "sentiment": "neutral"}

    async def _analyze_behavioral_patterns(
        self,
        activities: Optional[List[Dict[str, Any]]],
        time_window: int
    ) -> Dict[str, Any]:
        """Analyze behavioral patterns for stress indicators"""

        if not activities:
            return {"score": 0.0, "patterns": [], "anomalies": []}

        # Filter recent activities
        recent_activities = [
            activity for activity in activities
            if self._is_within_time_window(activity.get("timestamp"), time_window)
        ]

        if not recent_activities:
            return {"score": 0.0, "patterns": [], "anomalies": []}

        # Analyze different behavioral dimensions
        sleep_analysis = self._analyze_sleep_patterns(recent_activities)
        activity_analysis = self._analyze_activity_levels(recent_activities)
        social_analysis = self._analyze_social_patterns(recent_activities)
        academic_analysis = self._analyze_academic_behavior(recent_activities)

        # Calculate overall behavioral stress score
        behavioral_score = (
            sleep_analysis["score"] * 0.3 +
            activity_analysis["score"] * 0.2 +
            social_analysis["score"] * 0.2 +
            academic_analysis["score"] * 0.3
        )

        return {
            "score": min(behavioral_score, 1.0),
            "patterns": [
                {"type": "sleep", "score": sleep_analysis["score"], "details": sleep_analysis},
                {"type": "activity", "score": activity_analysis["score"], "details": activity_analysis},
                {"type": "social", "score": social_analysis["score"], "details": social_analysis},
                {"type": "academic", "score": academic_analysis["score"], "details": academic_analysis}
            ],
            "anomalies": self._detect_behavioral_anomalies(recent_activities)
        }

    def _analyze_sleep_patterns(self, activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sleep patterns from activities"""
        sleep_activities = [
            activity for activity in activities
            if activity.get("type") == "sleep"
        ]

        if not sleep_activities:
            return {"score": 0.0, "pattern": "insufficient_data"}

        # Calculate average sleep duration
        avg_duration = sum(
            activity.get("duration", 0) for activity in sleep_activities
        ) / len(sleep_activities)

        # Calculate sleep regularity
        sleep_times = [
            activity.get("start_time").hour
            for activity in sleep_activities
            if activity.get("start_time")
        ]
        regularity = 1.0 - (abs(max(sleep_times) - min(sleep_times)) / 24) if sleep_times else 0.0

        # Score based on duration (target: 7-9 hours) and regularity
        duration_score = 1.0 - abs(avg_duration - 8) / 8
        sleep_score = (duration_score * 0.7 + regularity * 0.3)

        return {
            "score": max(0, min(1, sleep_score)),
            "avg_duration": avg_duration,
            "regularity": regularity,
            "pattern": "optimal" if sleep_score > 0.8 else "suboptimal"
        }

    def _analyze_activity_levels(self, activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze activity levels and patterns"""
        study_activities = [
            activity for activity in activities
            if activity.get("category") == "academic"
        ]

        if not study_activities:
            return {"score": 0.0, "pattern": "insufficient_data"}

        # Calculate study intensity
        total_study_time = sum(
            activity.get("duration", 0) for activity in study_activities
        )
        daily_average = total_study_time / 7  # Weekly average

        # Score based on balanced study time (not too little, not too much)
        if daily_average < 2:  # Less than 2 hours/day
            intensity_score = daily_average / 2
        elif daily_average > 10:  # More than 10 hours/day
            intensity_score = max(0, 1 - (daily_average - 10) / 10)
        else:  # 2-10 hours/day
            intensity_score = 0.8

        return {
            "score": intensity_score,
            "daily_average": daily_average,
            "total_weekly": total_study_time,
            "pattern": "balanced" if 2 <= daily_average <= 10 else "unbalanced"
        }

    def _analyze_social_patterns(self, activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze social interaction patterns"""
        social_activities = [
            activity for activity in activities
            if activity.get("category") == "social"
        ]

        if not social_activities:
            return {"score": 0.0, "pattern": "insufficient_data"}

        # Social interaction frequency
        social_frequency = len(social_activities) / 7  # Per day

        # Score based on healthy social interaction
        if social_frequency < 0.5:  # Less than once every 2 days
            social_score = 0.3
        elif social_frequency > 5:  # More than 5 times per day
            social_score = 0.6  # Might indicate avoidance through excessive socializing
        else:
            social_score = 0.8

        return {
            "score": social_score,
            "frequency": social_frequency,
            "pattern": "healthy" if 0.5 <= social_frequency <= 5 else "unbalanced"
        }

    def _analyze_academic_behavior(self, activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze academic-related behaviors"""
        academic_activities = [
            activity for activity in activities
            if activity.get("category") in ["academic", "study", "assignment"]
        ]

        if not academic_activities:
            return {"score": 0.0, "pattern": "insufficient_data"}

        # Procrastination indicators (last-minute work)
        late_assignments = [
            activity for activity in academic_activities
            if activity.get("submitted_close_to_deadline", False)
        ]

        procrastination_ratio = len(late_assignments) / len(academic_activities)
        procrastination_score = 1.0 - procrastination_ratio

        return {
            "score": procrastination_score,
            "procrastination_ratio": procrastination_ratio,
            "late_assignments": len(late_assignments),
            "pattern": "proactive" if procrastination_ratio < 0.3 else "procrastinating"
        }

    def _detect_behavioral_anomalies(self, activities: List[Dict[str, Any]]) -> List[str]:
        """Detect anomalies in behavioral patterns"""
        anomalies = []

        # Check for sudden changes in patterns
        if len(activities) < 10:
            return ["insufficient_data"]

        # Add anomaly detection logic here
        # For now, return empty list
        return anomalies

    async def _analyze_academic_context(
        self,
        calendar_events: Optional[List[Dict[str, Any]]],
        time_window: int
    ) -> Dict[str, Any]:
        """Analyze academic context from calendar events"""

        if not calendar_events:
            return {"score": 0.0, "upcoming_stressors": [], "academic_load": "normal"}

        # Filter upcoming events
        upcoming_events = [
            event for event in calendar_events
            if self._is_event_upcoming(event, time_window)
        ]

        if not upcoming_events:
            return {"score": 0.0, "upcoming_stressors": [], "academic_load": "light"}

        # Analyze stress levels of different event types
        stress_scores = {
            "exam": 0.8,
            "deadline": 0.7,
            "presentation": 0.6,
            "assignment": 0.4,
            "class": 0.2
        }

        upcoming_stressors = []
        total_stress = 0.0

        for event in upcoming_events:
            event_type = event.get("type", "class").lower()
            stress_score = stress_scores.get(event_type, 0.3)

            upcoming_stressors.append({
                "title": event.get("title", "Event"),
                "type": event_type,
                "date": event.get("date"),
                "stress_score": stress_score,
                "days_until": self._days_until(event)
            })

            total_stress += stress_score

        # Calculate academic stress score
        academic_score = min(total_stress / 5, 1.0)  # Normalize to 0-1

        # Determine academic load level
        if academic_score < 0.3:
            load_level = "light"
        elif academic_score < 0.6:
            load_level = "moderate"
        elif academic_score < 0.8:
            load_level = "heavy"
        else:
            load_level = "overwhelming"

        return {
            "score": academic_score,
            "upcoming_stressors": upcoming_stressors,
            "academic_load": load_level,
            "total_events": len(upcoming_events)
        }

    async def _retrieve_stress_resources(
        self,
        text_analysis: Dict[str, Any],
        behavior_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Retrieve relevant stress management resources"""

        try:
            # Extract key themes and indicators
            themes = text_analysis.get("themes", [])
            if not themes:
                themes = ["general_stress"]

            # Build search query
            query = f"stress management techniques for {', '.join(themes)}"

            # Retrieve resources
            response = await self.retrieval_pipeline.retrieve(
                query=query,
                k=5,
                collections=["kb_global"]
            )

            # Format resources
            resources = []
            for result in response.results:
                resources.append({
                    "title": result.metadata.get("section_heading", "Resource"),
                    "content": result.text,
                    "relevance": result.score,
                    "source": result.metadata.get("document_id", "Knowledge Base")
                })

            return {
                "resources": resources,
                "query": query,
                "total_resources": len(resources)
            }

        except Exception as e:
            logger.error(f"Resource retrieval error: {e}")
            return {"resources": [], "query": "error", "total_resources": 0}

    async def _synthesize_stress_analysis(
        self,
        text_analysis: Dict[str, Any],
        behavior_analysis: Dict[str, Any],
        academic_analysis: Dict[str, Any],
        resource_analysis: Dict[str, Any],
        user_id: str
    ) -> StressAnalysis:
        """Synthesize all analyses into comprehensive stress assessment"""

        # Calculate overall stress score
        overall_score = (
            text_analysis.get("score", 0.0) * 0.4 +
            behavior_analysis.get("score", 0.0) * 0.3 +
            academic_analysis.get("score", 0.0) * 0.3
        )

        # Determine severity level
        severity_level = self._determine_severity_level(overall_score)

        # Collect contributing factors
        contributing_factors = []

        # Text-based factors
        if text_analysis.get("score", 0) > 0.5:
            contributing_factors.extend(text_analysis.get("indicators", []))

        # Behavioral factors
        if behavior_analysis.get("score", 0) > 0.6:
            for pattern in behavior_analysis.get("patterns", []):
                if pattern.get("score", 0) > 0.7:
                    contributing_factors.append(f"{pattern['type']}_pattern_concern")

        # Academic factors
        if academic_analysis.get("score", 0) > 0.6:
            contributing_factors.append("high_academic_load")

        # Generate recommendations using LLM
        recommendations = await self._generate_recommendations(
            overall_score=overall_score,
            severity_level=severity_level,
            contributing_factors=contributing_factors,
            resources=resource_analysis.get("resources", [])
        )

        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(
            text_analysis,
            behavior_analysis,
            academic_analysis
        )

        return StressAnalysis(
            overall_score=overall_score,
            severity_level=severity_level,
            contributing_factors=contributing_factors[:5],  # Top 5 factors
            recommendations=recommendations,
            confidence=confidence,
            analysis_date=datetime.now(),
            user_id=user_id
        )

    def _determine_severity_level(self, score: float) -> str:
        """Determine severity level from stress score"""
        for level, threshold in sorted(self.severity_thresholds.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return level
        return "low"

    async def _generate_recommendations(
        self,
        overall_score: float,
        severity_level: str,
        contributing_factors: List[str],
        resources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate personalized recommendations based on analysis"""

        recommendations = []

        # Immediate recommendations based on severity
        if severity_level in ["crisis", "severe"]:
            crisis_tool = ToolCall(
                tool_name="get_crisis_support",
                parameters={"crisis_type": "general"}
            )
            crisis_result = await self.tool_manager.execute_tool(crisis_tool)
            if crisis_result.success:
                recommendations.append({
                    "type": "crisis",
                    "priority": "immediate",
                    "content": crisis_result.data,
                    "evidence": "severity_based"
                })

        # Academic stress recommendations
        if "high_academic_load" in contributing_factors:
            calendar_tool = ToolCall(
                tool_name="get_calendar_events",
                parameters={"days_ahead": 7, "event_type": "exam"}
            )
            calendar_result = await self.tool_manager.execute_tool(calendar_tool)
            if calendar_result.success:
                recommendations.append({
                    "type": "academic_planning",
                    "priority": "high",
                    "content": "Review upcoming deadlines and create a structured study schedule",
                    "evidence": "academic_stress_detected"
                })

        # General stress management recommendations
        if overall_score > 0.4:
            # Add top knowledge base resources
            for resource in resources[:3]:
                recommendations.append({
                    "type": "self_help",
                    "priority": "medium",
                    "content": resource["content"][:200] + "...",
                    "title": resource["title"],
                    "source": resource["source"],
                    "relevance": resource["relevance"],
                    "evidence": "knowledge_base_match"
                })

        return recommendations

    def _calculate_confidence(
        self,
        text_analysis: Dict[str, Any],
        behavior_analysis: Dict[str, Any],
        academic_analysis: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the analysis based on data quality"""

        # Check for insufficient data indicators
        text_quality = 0.0 if text_analysis.get("score") == 0.0 else 1.0
        behavior_quality = 0.0 if behavior_analysis.get("score") == 0.0 else 1.0
        academic_quality = 0.0 if academic_analysis.get("score") == 0.0 else 1.0

        # Weight by importance
        confidence = (
            text_quality * 0.4 +
            behavior_quality * 0.3 +
            academic_quality * 0.3
        )

        return min(confidence, 1.0)

    def _is_within_time_window(self, timestamp: Any, days: int) -> bool:
        """Check if a timestamp is within the specified time window"""
        if not timestamp:
            return False

        try:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)

            cutoff_date = datetime.now() - timedelta(days=days)
            return timestamp >= cutoff_date

        except Exception:
            return False

    def _is_event_upcoming(self, event: Dict[str, Any], days: int) -> bool:
        """Check if an event is upcoming within specified days"""
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

    def _days_until(self, event: Dict[str, Any]) -> int:
        """Calculate days until an event"""
        try:
            event_date = event.get("date")
            if not event_date:
                return 0

            if isinstance(event_date, str):
                event_date = datetime.fromisoformat(event_date.replace('Z', '+00:00'))

            delta = event_date - datetime.now()
            return max(0, delta.days)

        except Exception:
            return 0

# Global stress analyzer instance
stress_analyzer = StressAnalyzerAgent()