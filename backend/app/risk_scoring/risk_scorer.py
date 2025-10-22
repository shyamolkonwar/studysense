from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import asyncio

from .risk_models import (
    RiskScore, RiskFactors, RiskTrend, RiskLevel,
    RiskThresholds, RiskValidator, RiskCalculator
)
from ..agents.stress_analyzer import stress_analyzer
from ..rag.chroma_client import chroma_client

logger = logging.getLogger(__name__)

class RiskScorer:
    """
    Advanced risk scoring engine that analyzes multiple data sources
    to calculate comprehensive mental health risk assessments
    """

    def __init__(self):
        """Initialize risk scorer with default configuration"""
        self.chroma_client = chroma_client
        self.stress_analyzer = stress_analyzer

        # Personalized weights storage (per user)
        self.personalized_weights: Dict[str, Dict[str, float]] = {}

        # Risk history for trend analysis
        self.risk_history: Dict[str, List[RiskScore]] = {}

        logger.info("Risk Scorer initialized")

    async def calculate_risk_score(
        self,
        user_id: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        activities: Optional[List[Dict[str, Any]]] = None,
        calendar_events: Optional[List[Dict[str, Any]]] = None,
        time_window: int = 7,
        include_personalization: bool = True
    ) -> RiskScore:
        """
        Calculate comprehensive risk score for a user

        Args:
            user_id: Unique identifier for the user
            messages: Recent messages/text data
            activities: Recent activity data
            calendar_events: Upcoming calendar events
            time_window: Time window in days for analysis
            include_personalization: Whether to use personalized weights

        Returns:
            Comprehensive RiskScore with all components
        """
        try:
            # Step 1: Extract risk factors from data sources
            risk_factors = await self._extract_risk_factors(
                user_id, messages, activities, calendar_events, time_window
            )

            # Step 2: Calculate category scores
            personalized_weights = self.personalized_weights.get(user_id) if include_personalization else None
            category_scores = self._calculate_category_scores(risk_factors, personalized_weights)

            # Step 3: Calculate overall risk score
            overall_score = RiskCalculator.calculate_overall_risk(
                category_scores,
                RiskThresholds.CATEGORY_WEIGHTS
            )

            # Step 4: Determine risk level and confidence
            risk_level = RiskCalculator.determine_risk_level(
                overall_score,
                RiskThresholds.RISK_THRESHOLDS
            )

            confidence = self._calculate_confidence(messages, activities, calendar_events)

            # Step 5: Generate recommendations and primary concerns
            primary_concerns = self._identify_primary_concerns(risk_factors, category_scores)
            recommended_actions = await self._generate_recommendations(
                risk_level, primary_concerns, user_id, risk_factors
            )

            # Step 6: Check escalation thresholds
            escalation_threshold_met = self._check_escalation_thresholds(overall_score, risk_level)

            # Step 7: Create risk score object
            risk_score = RiskScore(
                user_id=user_id,
                overall_score=overall_score,
                risk_level=risk_level,
                confidence=confidence,
                academic_risk=category_scores.get("academic", 0.0),
                behavioral_risk=category_scores.get("behavioral", 0.0),
                emotional_risk=category_scores.get("emotional", 0.0),
                contextual_risk=category_scores.get("contextual", 0.0),
                factors=risk_factors,
                timestamp=datetime.now(),
                data_sources=self._identify_data_sources(messages, activities, calendar_events),
                personalized_weights=personalized_weights,
                primary_concerns=primary_concerns,
                recommended_actions=recommended_actions,
                escalation_threshold_met=escalation_threshold_met
            )

            # Validate risk score
            validation_issues = RiskValidator.validate_risk_score(risk_score)
            if validation_issues:
                logger.warning(f"Risk score validation issues: {validation_issues}")
                # Still return the score but note the issues in confidence
                risk_score.confidence *= 0.8

            # Step 8: Store in history
            self._store_risk_score(risk_score)

            logger.info(f"Risk score calculated for {user_id}: {risk_level} ({overall_score:.2f})")
            return risk_score

        except Exception as e:
            logger.error(f"Risk score calculation failed for {user_id}: {e}")
            # Return safe default
            return self._create_default_risk_score(user_id)

    async def calculate_risk_trend(
        self,
        user_id: str,
        time_window: int = 30
    ) -> RiskTrend:
        """
        Calculate risk score trends over time

        Args:
            user_id: Unique identifier for the user
            time_window: Number of days to analyze for trends

        Returns:
            RiskTrend with trend analysis
        """
        try:
            # Get historical risk scores
            historical_scores = self.risk_history.get(user_id, [])

            # Filter scores within time window
            cutoff_date = datetime.now() - timedelta(days=time_window)
            recent_scores = [
                score for score in historical_scores
                if score.timestamp >= cutoff_date
            ]

            if len(recent_scores) < 2:
                return self._create_default_risk_trend(user_id, time_window)

            # Calculate trend statistics
            scores_list = [score.overall_score for score in recent_scores]
            current_score = scores_list[-1] if scores_list else 0.0
            average_score = sum(scores_list) / len(scores_list)
            min_score = min(scores_list)
            max_score = max(scores_list)

            # Calculate trend
            trend_result = RiskCalculator.calculate_trend(scores_list)

            # Calculate volatility (standard deviation)
            if len(scores_list) > 1:
                mean = average_score
                variance = sum((x - mean) ** 2 for x in scores_list) / len(scores_list)
                score_volatility = variance ** 0.5
            else:
                score_volatility = 0.0

            # Identify significant events
            significant_events = self._identify_significant_events(recent_scores)

            # Calculate data quality based on score frequency
            ideal_frequency = time_window / 7  # Weekly scoring
            actual_frequency = len(recent_scores)
            data_quality = min(actual_frequency / ideal_frequency, 1.0)

            # Create daily scores for visualization
            daily_scores = self._create_daily_scores(recent_scores, time_window)

            return RiskTrend(
                user_id=user_id,
                time_window=time_window,
                trend_direction=trend_result["direction"],
                trend_strength=trend_result["strength"],
                current_score=current_score,
                average_score=average_score,
                min_score=min_score,
                max_score=max_score,
                score_volatility=score_volatility,
                daily_scores=daily_scores,
                significant_events=significant_events,
                last_updated=datetime.now(),
                data_quality=data_quality
            )

        except Exception as e:
            logger.error(f"Risk trend calculation failed for {user_id}: {e}")
            return self._create_default_risk_trend(user_id, time_window)

    async def personalize_weights(
        self,
        user_id: str,
        feedback_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Personalize risk scoring weights based on user feedback

        Args:
            user_id: Unique identifier for the user
            feedback_data: Historical feedback on risk assessments

        Returns:
            Updated personalized weights
        """
        try:
            if not feedback_data:
                return {"status": "no_feedback", "weights": self.personalized_weights.get(user_id, {})}

            # Analyze feedback patterns
            feedback_analysis = self._analyze_feedback_patterns(feedback_data)

            # Adjust weights based on feedback
            base_weights = RiskThresholds.DEFAULT_WEIGHTS.copy()
            personalized_weights = self._adjust_weights(base_weights, feedback_analysis)

            # Store personalized weights
            self.personalized_weights[user_id] = personalized_weights

            logger.info(f"Personalized weights updated for {user_id}")
            return {
                "status": "success",
                "weights": personalized_weights,
                "feedback_analysis": feedback_analysis
            }

        except Exception as e:
            logger.error(f"Weight personalization failed for {user_id}: {e}")
            return {"status": "error", "error": str(e)}

    async def _extract_risk_factors(
        self,
        user_id: str,
        messages: Optional[List[Dict[str, Any]]],
        activities: Optional[List[Dict[str, Any]]],
        calendar_events: Optional[List[Dict[str, Any]]],
        time_window: int
    ) -> RiskFactors:
        """Extract risk factors from various data sources"""

        factors = RiskFactors()

        # Process messages for emotional factors
        if messages:
            emotional_factors = await self._analyze_messages(messages, time_window)
            factors.negative_sentiment = emotional_factors.get("negative_sentiment", 0.0)
            factors.anxiety_indicators = emotional_factors.get("anxiety", 0.0)
            factors.depression_indicators = emotional_factors.get("depression", 0.0)
            factors.stress_keywords = emotional_factors.get("stress_keywords", 0.0)

        # Process activities for behavioral factors
        if activities:
            behavioral_factors = self._analyze_activities(activities, time_window)
            factors.sleep_disruption = behavioral_factors.get("sleep_disruption", 0.0)
            factors.social_withdrawal = behavioral_factors.get("social_withdrawal", 0.0)
            factors.activity_irregularity = behavioral_factors.get("irregularity", 0.0)
            factors.procrastination_level = behavioral_factors.get("procrastination", 0.0)

        # Process calendar events for academic factors
        if calendar_events:
            academic_factors = self._analyze_calendar_events(calendar_events)
            factors.academic_pressure = academic_factors.get("academic_pressure", 0.0)
            factors.deadline_proximity = academic_factors.get("deadline_proximity", 0.0)
            factors.workload_intensity = academic_factors.get("workload_intensity", 0.0)

        # Get contextual factors from historical data
        contextual_factors = await self._get_contextual_factors(user_id, time_window)
        factors.recent_stress_events = contextual_factors.get("recent_stress_events", 0.0)
        factors.support_system_strength = contextual_factors.get("support_system", 0.5)
        factors.coping_mechanisms = contextual_factors.get("coping_mechanisms", 0.5)

        return factors

    async def _analyze_messages(self, messages: List[Dict[str, Any]], time_window: int) -> Dict[str, float]:
        """Analyze messages for emotional risk factors"""

        # Filter recent messages
        recent_messages = [
            msg for msg in messages
            if self._is_within_time_window(msg.get("timestamp"), time_window)
        ]

        if not recent_messages:
            return {"negative_sentiment": 0.0, "anxiety": 0.0, "depression": 0.0, "stress_keywords": 0.0}

        # Combine message content
        combined_text = " ".join([msg.get("content", "") for msg in recent_messages])

        # Use stress analyzer for message analysis
        try:
            stress_analysis = await self.stress_analyzer._analyze_textual_stress_indicators(
                messages, time_window
            )

            # Map stress analysis to risk factors
            return {
                "negative_sentiment": 1.0 if stress_analysis.get("sentiment") == "negative" else 0.5,
                "anxiety": stress_analysis.get("score", 0.0) * 0.8,
                "depression": 0.3 if "depression" in stress_analysis.get("themes", []) else 0.1,
                "stress_keywords": min(stress_analysis.get("score", 0.0) * 1.2, 1.0)
            }

        except Exception as e:
            logger.error(f"Message analysis error: {e}")
            return {"negative_sentiment": 0.3, "anxiety": 0.3, "depression": 0.1, "stress_keywords": 0.2}

    def _analyze_activities(self, activities: List[Dict[str, Any]], time_window: int) -> Dict[str, float]:
        """Analyze activities for behavioral risk factors"""

        # Filter recent activities
        recent_activities = [
            activity for activity in activities
            if self._is_within_time_window(activity.get("timestamp"), time_window)
        ]

        if not recent_activities:
            return {
                "sleep_disruption": 0.0,
                "social_withdrawal": 0.0,
                "irregularity": 0.0,
                "procrastination": 0.0
            }

        # Analyze sleep patterns
        sleep_activities = [
            activity for activity in recent_activities
            if activity.get("type") == "sleep"
        ]

        if sleep_activities:
            avg_sleep_duration = sum(
                activity.get("duration", 0) for activity in sleep_activities
            ) / len(sleep_activities)

            # Score based on deviation from optimal (8 hours)
            sleep_disruption = min(abs(avg_sleep_duration - 8) / 4, 1.0)
        else:
            sleep_disruption = 0.5  # Unknown - moderate risk

        # Analyze social activities
        social_activities = [
            activity for activity in recent_activities
            if activity.get("category") == "social"
        ]

        social_frequency = len(social_activities) / time_window
        social_withdrawal = max(0, 1.0 - (social_frequency / 2.0))

        # Analyze activity regularity
        academic_activities = [
            activity for activity in recent_activities
            if activity.get("category") == "academic"
        ]

        if len(academic_activities) > 0:
            # Check for irregular patterns (e.g., all-nighters)
            irregularity_score = 0.0
            for activity in academic_activities:
                start_hour = activity.get("start_time", {}).get("hour", 12)
                if start_hour < 6 or start_hour > 23:  # Very early or very late
                    irregularity_score += 0.2

            irregularity_score = min(irregularity_score, 1.0)
        else:
            irregularity_score = 0.3

        # Analyze procrastination (last-minute work)
        procrastination_indicators = [
            activity for activity in academic_activities
            if activity.get("submitted_close_to_deadline", False)
        ]

        procrastination_ratio = (
            len(procrastination_indicators) / len(academic_activities)
            if academic_activities else 0.0
        )

        return {
            "sleep_disruption": sleep_disruption,
            "social_withdrawal": social_withdrawal,
            "irregularity": irregularity_score,
            "procrastination": procrastination_ratio
        }

    def _analyze_calendar_events(self, calendar_events: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze calendar events for academic risk factors"""

        if not calendar_events:
            return {"academic_pressure": 0.0, "deadline_proximity": 0.0, "workload_intensity": 0.0}

        upcoming_events = [
            event for event in calendar_events
            if self._is_event_upcoming(event, 14)  # Next 2 weeks
        ]

        if not upcoming_events:
            return {"academic_pressure": 0.0, "deadline_proximity": 0.0, "workload_intensity": 0.0}

        # Calculate academic pressure based on event types
        stress_weights = {
            "exam": 0.8,
            "deadline": 0.7,
            "presentation": 0.6,
            "assignment": 0.4,
            "class": 0.2
        }

        total_pressure = 0.0
        for event in upcoming_events:
            event_type = event.get("type", "class").lower()
            weight = stress_weights.get(event_type, 0.3)
            days_until = self._days_until(event)

            # Closer events have higher pressure
            proximity_factor = max(0, 1 - (days_until / 14))
            total_pressure += weight * proximity_factor

        academic_pressure = min(total_pressure / 3, 1.0)

        # Calculate deadline proximity
        deadline_events = [
            event for event in upcoming_events
            if event.get("type") == "deadline"
        ]

        if deadline_events:
            closest_deadline = min(
                self._days_until(event) for event in deadline_events
            )
            deadline_proximity = max(0, 1 - (closest_deadline / 7))
        else:
            deadline_proximity = 0.0

        # Calculate workload intensity
        workload_events = [
            event for event in upcoming_events
            if event.get("type") in ["assignment", "exam", "deadline"]
        ]

        workload_intensity = min(len(workload_events) / 5, 1.0)

        return {
            "academic_pressure": academic_pressure,
            "deadline_proximity": deadline_proximity,
            "workload_intensity": workload_intensity
        }

    async def _get_contextual_factors(self, user_id: str, time_window: int) -> Dict[str, float]:
        """Get contextual factors from historical data"""

        try:
            # Get user context from ChromaDB
            context_result = self.chroma_client.query_user_context(
                user_id=user_id,
                query="recent stress events support coping",
                n_results=5
            )

            # Extract contextual information
            recent_stress_events = 0.0
            support_system = 0.5  # Default
            coping_mechanisms = 0.5  # Default

            if context_result.get("documents"):
                # This is a simplified analysis - in production, this would be more sophisticated
                recent_stress_events = min(len(context_result["documents"][0]) / 100, 1.0)

            return {
                "recent_stress_events": recent_stress_events,
                "support_system": support_system,
                "coping_mechanisms": coping_mechanisms
            }

        except Exception as e:
            logger.error(f"Contextual factors retrieval error: {e}")
            return {
                "recent_stress_events": 0.0,
                "support_system": 0.5,
                "coping_mechanisms": 0.5
            }

    def _calculate_category_scores(
        self,
        risk_factors: RiskFactors,
        personalized_weights: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate risk scores for each category"""

        # Use personalized weights if available, otherwise use defaults
        weights = personalized_weights or RiskThresholds.DEFAULT_WEIGHTS

        category_scores = {}

        # Calculate academic risk
        category_scores["academic"] = RiskCalculator.calculate_category_score(
            risk_factors, "academic", weights
        )

        # Calculate behavioral risk
        category_scores["behavioral"] = RiskCalculator.calculate_category_score(
            risk_factors, "behavioral", weights
        )

        # Calculate emotional risk
        category_scores["emotional"] = RiskCalculator.calculate_category_score(
            risk_factors, "emotional", weights
        )

        # Calculate contextual risk
        category_scores["contextual"] = RiskCalculator.calculate_category_score(
            risk_factors, "contextual", weights
        )

        return category_scores

    def _calculate_confidence(
        self,
        messages: Optional[List[Dict[str, Any]]],
        activities: Optional[List[Dict[str, Any]]],
        calendar_events: Optional[List[Dict[str, Any]]]
    ) -> float:
        """Calculate confidence in the risk assessment based on data availability"""

        data_quality_scores = []

        # Message data quality
        if messages and len(messages) > 0:
            message_quality = min(len(messages) / 10, 1.0)  # More messages = higher confidence
            data_quality_scores.append(message_quality)
        else:
            data_quality_scores.append(0.0)

        # Activity data quality
        if activities and len(activities) > 0:
            activity_quality = min(len(activities) / 20, 1.0)
            data_quality_scores.append(activity_quality)
        else:
            data_quality_scores.append(0.0)

        # Calendar data quality
        if calendar_events and len(calendar_events) > 0:
            calendar_quality = min(len(calendar_events) / 5, 1.0)
            data_quality_scores.append(calendar_quality)
        else:
            data_quality_scores.append(0.0)

        # Overall confidence is average of data quality scores
        return sum(data_quality_scores) / len(data_quality_scores)

    def _identify_primary_concerns(
        self,
        risk_factors: RiskFactors,
        category_scores: Dict[str, float]
    ) -> List[str]:
        """Identify the top risk factors/concerns"""

        concerns = []

        # Check individual factor thresholds (0.7 = significant concern)
        factor_threshold = 0.7

        factors_dict = risk_factors.to_dict()
        for factor_name, value in factors_dict.items():
            if value >= factor_threshold:
                # Map factor names to human-readable concerns
                concern_map = {
                    "academic_pressure": "High academic pressure",
                    "deadline_proximity": "Approaching deadlines",
                    "sleep_disruption": "Sleep pattern disruption",
                    "anxiety_indicators": "High anxiety levels",
                    "social_withdrawal": "Social withdrawal",
                    "procrastination_level": "Procrastination patterns",
                    "negative_sentiment": "Negative emotional state"
                }
                concern = concern_map.get(factor_name, f"High {factor_name}")
                if concern not in concerns:
                    concerns.append(concern)

        # Add category-level concerns
        for category, score in category_scores.items():
            if score >= 0.6:
                category_concern = f"High {category} risk"
                if category_concern not in concerns:
                    concerns.append(category_concern)

        # Return top 5 concerns
        return concerns[:5]

    async def _generate_recommendations(
        self,
        risk_level: RiskLevel,
        primary_concerns: List[str],
        user_id: str,
        risk_factors: RiskFactors
    ) -> List[Dict[str, Any]]:
        """Generate personalized recommendations based on risk assessment"""

        recommendations = []

        # Crisis recommendations
        if risk_level == RiskLevel.CRISIS:
            recommendations.append({
                "type": "crisis",
                "priority": "immediate",
                "title": "Immediate Crisis Support",
                "content": "Please seek immediate help: Call 988 (Suicide Prevention Lifeline) or text HOME to 741741",
                "action_required": True,
                "contact_info": {
                    "hotline": "988",
                    "text": "HOME to 741741",
                    "emergency": "911"
                }
            })

        # Academic stress recommendations
        if any("academic" in concern.lower() or "deadline" in concern.lower() for concern in primary_concerns):
            recommendations.append({
                "type": "academic_support",
                "priority": "high",
                "title": "Academic Stress Management",
                "content": "Consider meeting with academic advisor or using campus tutoring services",
                "suggested_resources": ["academic_advising", "tutoring", "time_management_workshop"]
            })

        # Sleep recommendations
        if risk_factors.sleep_disruption > 0.6:
            recommendations.append({
                "type": "sleep_hygiene",
                "priority": "high",
                "title": "Sleep Pattern Improvement",
                "content": "Focus on consistent sleep schedule and sleep hygiene practices",
                "suggested_resources": ["sleep_hygiene_guide", "relaxation_techniques"]
            })

        # Social support recommendations
        if risk_factors.social_withdrawal > 0.6:
            recommendations.append({
                "type": "social_support",
                "priority": "medium",
                "title": "Social Connection",
                "content": "Consider joining study groups or campus organizations",
                "suggested_resources": ["student_organizations", "support_groups", "peer_mentoring"]
            })

        # General stress management (for moderate risk)
        if risk_level in [RiskLevel.MODERATE, RiskLevel.SEVERE]:
            recommendations.append({
                "type": "stress_management",
                "priority": "medium",
                "title": "Stress Management Techniques",
                "content": "Practice regular stress management and relaxation techniques",
                "suggested_resources": ["mindfulness", "breathing_exercises", "progressive_muscle_relaxation"]
            })

        return recommendations

    def _check_escalation_thresholds(self, overall_score: float, risk_level: RiskLevel) -> bool:
        """Check if escalation thresholds are met"""

        # Crisis threshold
        if overall_score >= RiskThresholds.CRISIS_ESCALATION_THRESHOLD:
            return True

        # Severe threshold with rapid increase
        if risk_level == RiskLevel.SEVERE and overall_score >= RiskThresholds.SEVERE_ESCALATION_THRESHOLD:
            return True

        return False

    def _identify_data_sources(self, *data_sources) -> List[str]:
        """Identify which data sources were used in assessment"""
        sources = []

        if data_sources[0]:  # messages
            sources.append("text_messages")
        if data_sources[1]:  # activities
            sources.append("activity_data")
        if data_sources[2]:  # calendar_events
            sources.append("calendar_data")

        return sources

    def _store_risk_score(self, risk_score: RiskScore):
        """Store risk score in history"""
        if risk_score.user_id not in self.risk_history:
            self.risk_history[risk_score.user_id] = []

        self.risk_history[risk_score.user_id].append(risk_score)

        # Keep only last 90 days of history
        cutoff_date = datetime.now() - timedelta(days=90)
        self.risk_history[risk_score.user_id] = [
            score for score in self.risk_history[risk_score.user_id]
            if score.timestamp >= cutoff_date
        ]

    def _create_default_risk_score(self, user_id: str) -> RiskScore:
        """Create a default risk score when calculation fails"""
        return RiskScore(
            user_id=user_id,
            overall_score=0.0,
            risk_level=RiskLevel.LOW,
            confidence=0.0,
            academic_risk=0.0,
            behavioral_risk=0.0,
            emotional_risk=0.0,
            contextual_risk=0.0,
            factors=RiskFactors(),
            timestamp=datetime.now(),
            data_sources=["error"],
            primary_concerns=["analysis_error"],
            recommended_actions=[{
                "type": "system",
                "content": "Unable to calculate risk score. Please try again later."
            }],
            escalation_threshold_met=False
        )

    def _create_default_risk_trend(self, user_id: str, time_window: int) -> RiskTrend:
        """Create default risk trend when insufficient data"""
        return RiskTrend(
            user_id=user_id,
            time_window=time_window,
            trend_direction="stable",
            trend_strength=0.0,
            current_score=0.0,
            average_score=0.0,
            min_score=0.0,
            max_score=0.0,
            score_volatility=0.0,
            daily_scores=[],
            significant_events=[],
            last_updated=datetime.now(),
            data_quality=0.0
        )

    def _is_within_time_window(self, timestamp: Any, days: int) -> bool:
        """Check if timestamp is within specified time window"""
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
        """Check if event is within specified days"""
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
        """Calculate days until event"""
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

    def _identify_significant_events(self, risk_scores: List[RiskScore]) -> List[Dict[str, Any]]:
        """Identify significant events in risk score history"""
        events = []

        for i, score in enumerate(risk_scores):
            # Look for risk level changes
            if i > 0:
                prev_score = risk_scores[i-1]
                if score.risk_level != prev_score.risk_level:
                    events.append({
                        "date": score.timestamp.isoformat(),
                        "type": "risk_level_change",
                        "from_level": prev_score.risk_level.value,
                        "to_level": score.risk_level.value,
                        "score_change": score.overall_score - prev_score.overall_score
                    })

            # Look for rapid increases
            if i > 2:
                recent_scores = [s.overall_score for s in risk_scores[i-3:i]]
                avg_recent = sum(recent_scores) / len(recent_scores)
                if score.overall_score > avg_recent + 0.2:  # 20% increase
                    events.append({
                        "date": score.timestamp.isoformat(),
                        "type": "rapid_increase",
                        "score": score.overall_score,
                        "baseline": avg_recent
                    })

        return events

    def _create_daily_scores(self, risk_scores: List[RiskScore], time_window: int) -> List[Dict[str, Any]]:
        """Create daily score list for visualization"""
        daily_scores = []

        # Group scores by date
        scores_by_date = {}
        for score in risk_scores:
            date_str = score.timestamp.date().isoformat()
            if date_str not in scores_by_date:
                scores_by_date[date_str] = []
            scores_by_date[date_str].append(score.overall_score)

        # Create daily entries
        current_date = datetime.now().date()
        for days_back in range(time_window - 1, -1, -1):
            check_date = current_date - timedelta(days=days_back)
            date_str = check_date.isoformat()

            if date_str in scores_by_date:
                daily_scores.append({
                    "date": date_str,
                    "score": sum(scores_by_date[date_str]) / len(scores_by_date[date_str])
                })
            else:
                daily_scores.append({
                    "date": date_str,
                    "score": None
                })

        return daily_scores

# Global risk scorer instance
risk_scorer = RiskScorer()