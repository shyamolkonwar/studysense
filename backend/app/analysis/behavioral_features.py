"""
Phase 4: Behavioral Features Extraction System
Analyzes sleep patterns, study behaviors, social interactions, and activity patterns
as specified in the MVP requirements.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class SleepAnalysis:
    """Sleep pattern analysis results"""
    avg_sleep_duration: float        # Average hours of sleep
    sleep_regularity: float          # Regularity score (0-1)
    sleep_disruption_score: float    # Disruption indicator (0-1)
    bedtime_variance: float          # Variance in bedtime
    wake_time_variance: float        # Variance in wake time
    sleep_quality_trend: str         # "improving", "stable", "declining"

@dataclass
class SocialAnalysis:
    """Social interaction analysis results"""
    social_frequency: float          # Interactions per day
    social_withdrawal_score: float   # Withdrawal indicator (0-1)
    interaction_diversity: float     # Different interaction types
    response_time_avg: float         # Average response time
    social_activity_score: float     # Overall social health score

@dataclass
class AcademicBehaviorAnalysis:
    """Academic behavior analysis results"""
    study_pattern_regularity: float  # Regular study schedule score
    procrastination_level: float     # Procrastination indicator (0-1)
    deadline_pressure_score: float   # Deadline-related stress
    productivity_patterns: Dict[str, float]  # Time-based productivity
    academic_engagement_score: float # Overall engagement level

@dataclass
class ActivityPatternAnalysis:
    """General activity pattern analysis"""
    activity_regularity: float       # Regular activity patterns score
    nocturnal_activity_ratio: float  # Night vs day activity ratio
    sedentary_time_ratio: float      # Sedentary behavior indicator
    physical_activity_score: float   # Physical activity level
    work_life_balance_score: float   # Balance between different activities

@dataclass
class BehavioralFeaturesResult:
    """Complete behavioral features analysis result"""
    user_id: str
    analysis_timestamp: datetime
    time_window_days: int
    sleep: SleepAnalysis
    social: SocialAnalysis
    academic: AcademicBehaviorAnalysis
    activity_patterns: ActivityPatternAnalysis
    risk_indicators: Dict[str, float]
    recommendations: List[Dict[str, Any]]

class BehavioralFeaturesExtractor:
    """
    Extracts behavioral features from activity data for mental health analysis.
    Implements study patterns, sleep analysis, and social interaction tracking.
    """

    def __init__(self):
        """Initialize behavioral features extractor"""

        # Sleep health thresholds
        self.optimal_sleep_duration = 8.0  # hours
        self.min_sleep_duration = 6.0
        self.max_sleep_duration = 10.0

        # Social interaction thresholds
        self.healthy_social_frequency = 2.0  # interactions per day
        self.concerning_response_time = 4.0  # hours

        # Academic behavior thresholds
        self.healthy_study_hours = 6.0  # hours per day
        self.max_study_hours = 12.0
        self.procrastination_threshold = 0.3  # ratio of last-minute work

        # Activity pattern thresholds
        self.night_hours = list(range(22, 24)) + list(range(0, 6))
        self.day_hours = list(range(6, 22))

    async def extract_features(
        self,
        user_id: str,
        activities: List[Dict[str, Any]],
        messages: Optional[List[Dict[str, Any]]] = None,
        calendar_events: Optional[List[Dict[str, Any]]] = None,
        time_window: int = 7
    ) -> BehavioralFeaturesResult:
        """
        Extract comprehensive behavioral features

        Args:
            user_id: Unique user identifier
            activities: List of activity events with timestamps and metadata
            messages: Optional message data for social analysis
            calendar_events: Optional calendar events for academic context
            time_window: Analysis time window in days

        Returns:
            Complete BehavioralFeaturesResult
        """

        try:
            # Filter recent activities
            recent_activities = self._filter_activities_by_time(activities, time_window)

            if not recent_activities:
                return self._create_empty_result(user_id, time_window)

            # Analyze different behavioral dimensions
            sleep_analysis = await self._analyze_sleep_patterns(recent_activities)

            social_analysis = await self._analyze_social_interactions(
                messages, recent_activities, time_window
            )

            academic_analysis = await self._analyze_academic_behaviors(
                recent_activities, calendar_events
            )

            activity_analysis = await self._analyze_activity_patterns(recent_activities)

            # Calculate risk indicators
            risk_indicators = self._calculate_behavioral_risks(
                sleep_analysis, social_analysis, academic_analysis, activity_analysis
            )

            # Generate behavioral recommendations
            recommendations = self._generate_behavioral_recommendations(
                sleep_analysis, social_analysis, academic_analysis, activity_analysis
            )

            return BehavioralFeaturesResult(
                user_id=user_id,
                analysis_timestamp=datetime.now(),
                time_window_days=time_window,
                sleep=sleep_analysis,
                social=social_analysis,
                academic=academic_analysis,
                activity_patterns=activity_analysis,
                risk_indicators=risk_indicators,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Behavioral features extraction failed for user {user_id}: {e}")
            return self._create_empty_result(user_id, time_window)

    async def _analyze_sleep_patterns(self, activities: List[Dict[str, Any]]) -> SleepAnalysis:
        """Analyze sleep patterns from activity data"""

        # Filter sleep activities
        sleep_activities = [
            activity for activity in activities
            if activity.get("type") == "sleep" or activity.get("category") == "sleep"
        ]

        if not sleep_activities:
            return SleepAnalysis(0.0, 0.0, 0.5, 0.0, 0.0, "stable")

        # Extract sleep durations and times
        sleep_durations = []
        bedtimes = []
        wake_times = []

        for activity in sleep_activities:
            duration = activity.get("duration", 0)
            start_time = activity.get("start_time")
            end_time = activity.get("end_time")

            if duration > 0:
                sleep_durations.append(duration)

            if start_time:
                if isinstance(start_time, datetime):
                    bedtimes.append(start_time.hour + start_time.minute / 60)
                elif isinstance(start_time, str):
                    time_obj = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    bedtimes.append(time_obj.hour + time_obj.minute / 60)

            if end_time:
                if isinstance(end_time, datetime):
                    wake_times.append(end_time.hour + end_time.minute / 60)
                elif isinstance(end_time, str):
                    time_obj = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    wake_times.append(time_obj.hour + time_obj.minute / 60)

        # Calculate average sleep duration
        avg_sleep_duration = np.mean(sleep_durations) if sleep_durations else 0.0

        # Calculate sleep regularity (consistency of sleep duration)
        if len(sleep_durations) > 1:
            duration_std = np.std(sleep_durations)
            sleep_regularity = max(0, 1 - (duration_std / self.optimal_sleep_duration))
        else:
            sleep_regularity = 0.5

        # Calculate sleep disruption score
        deviation_from_optimal = abs(avg_sleep_duration - self.optimal_sleep_duration)
        sleep_disruption_score = min(deviation_from_optimal / 4.0, 1.0)  # 4-hour max deviation

        # Calculate bedtime and wake time variance
        bedtime_variance = np.std(bedtimes) if len(bedtimes) > 1 else 0.0
        wake_time_variance = np.std(wake_times) if len(wake_times) > 1 else 0.0

        # Normalize variance scores (max 4-hour variance)
        bedtime_variance = min(bedtime_variance / 4.0, 1.0)
        wake_time_variance = min(wake_time_variance / 4.0, 1.0)

        # Determine sleep quality trend (simplified)
        if len(sleep_durations) > 3:
            recent_avg = np.mean(sleep_durations[-3:])
            earlier_avg = np.mean(sleep_durations[:-3]) if len(sleep_durations) > 3 else recent_avg
            if recent_avg > earlier_avg + 0.5:
                trend = "improving"
            elif recent_avg < earlier_avg - 0.5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return SleepAnalysis(
            avg_sleep_duration=avg_sleep_duration,
            sleep_regularity=sleep_regularity,
            sleep_disruption_score=sleep_disruption_score,
            bedtime_variance=bedtime_variance,
            wake_time_variance=wake_time_variance,
            sleep_quality_trend=trend
        )

    async def _analyze_social_interactions(
        self,
        messages: Optional[List[Dict[str, Any]]],
        activities: List[Dict[str, Any]],
        time_window: int
    ) -> SocialAnalysis:
        """Analyze social interaction patterns"""

        # Extract social interactions from messages and activities
        social_interactions = []

        # Process messages for social data
        if messages:
            for msg in messages:
                if msg.get("type") in ["message", "chat", "social"]:
                    social_interactions.append({
                        "timestamp": msg.get("timestamp"),
                        "type": "message",
                        "response_time": msg.get("response_time"),
                        "participants": msg.get("participants", 1)
                    })

        # Process activities for social data
        social_activities = [
            activity for activity in activities
            if activity.get("category") == "social" or activity.get("type") in ["meeting", "call", "social"]
        ]

        for activity in social_activities:
            social_interactions.append({
                "timestamp": activity.get("timestamp"),
                "type": activity.get("type", "social"),
                "duration": activity.get("duration", 0),
                "participants": activity.get("participants", 1)
            })

        if not social_interactions:
            return SocialAnalysis(0.0, 0.5, 0.0, 0.0, 0.5)

        # Calculate social frequency (interactions per day)
        social_frequency = len(social_interactions) / max(time_window, 1)

        # Calculate social withdrawal score
        healthy_frequency = self.healthy_social_frequency
        if social_frequency < healthy_frequency * 0.5:
            withdrawal_score = 1.0
        elif social_frequency < healthy_frequency:
            withdrawal_score = 1.0 - (social_frequency / healthy_frequency)
        else:
            withdrawal_score = 0.0

        # Calculate interaction diversity
        interaction_types = set(interaction["type"] for interaction in social_interactions)
        interaction_diversity = len(interaction_types) / 5.0  # Normalize to 0-1

        # Calculate average response time
        response_times = [
            interaction["response_time"]
            for interaction in social_interactions
            if interaction.get("response_time") is not None
        ]

        response_time_avg = np.mean(response_times) if response_times else 0.0

        # Calculate overall social activity score
        frequency_score = min(social_frequency / healthy_frequency, 1.0)
        response_score = 1.0 - min(response_time_avg / self.concerning_response_time, 1.0)
        social_activity_score = (frequency_score + response_score + interaction_diversity) / 3.0

        return SocialAnalysis(
            social_frequency=social_frequency,
            social_withdrawal_score=withdrawal_score,
            interaction_diversity=interaction_diversity,
            response_time_avg=response_time_avg,
            social_activity_score=social_activity_score
        )

    async def _analyze_academic_behaviors(
        self,
        activities: List[Dict[str, Any]],
        calendar_events: Optional[List[Dict[str, Any]]]
    ) -> AcademicBehaviorAnalysis:
        """Analyze academic-related behaviors"""

        # Filter academic activities
        academic_activities = [
            activity for activity in activities
            if activity.get("category") in ["academic", "study", "assignment", "work"]
        ]

        if not academic_activities:
            return AcademicBehaviorAnalysis(
                0.0, 0.0, 0.0, {}, 0.0
            )

        # Analyze study patterns
        study_hours_by_day = defaultdict(float)
        study_sessions = []

        for activity in academic_activities:
            duration = activity.get("duration", 0)
            timestamp = activity.get("timestamp")

            if duration > 0 and timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

                day_key = timestamp.strftime("%A")
                study_hours_by_day[day_key] += duration
                study_sessions.append({
                    "start_time": timestamp,
                    "duration": duration,
                    "hour": timestamp.hour
                })

        # Calculate study pattern regularity
        if len(study_hours_by_day) > 0:
            study_values = list(study_hours_by_day.values())
            study_regularity = 1.0 - (np.std(study_values) / max(np.mean(study_values), 1))
            study_regularity = max(0, min(study_regularity, 1.0))
        else:
            study_regularity = 0.0

        # Calculate procrastination level
        late_assignments = [
            activity for activity in academic_activities
            if activity.get("submitted_close_to_deadline", False) or
               activity.get("type") == "last_minute_work"
        ]

        total_academic_activities = len(academic_activities)
        procrastination_level = (
            len(late_assignments) / max(total_academic_activities, 1)
        )

        # Calculate deadline pressure score
        if calendar_events:
            upcoming_deadlines = [
                event for event in calendar_events
                if event.get("type") in ["deadline", "exam", "assignment"] and
                   self._is_event_upcoming(event, 7)
            ]

            deadline_pressure = min(len(upcoming_deadlines) / 5.0, 1.0)
        else:
            deadline_pressure = 0.0

        # Calculate productivity patterns by hour
        productivity_by_hour = defaultdict(float)
        for session in study_sessions:
            hour = session.get("hour", 12)
            productivity_by_hour[hour] += session.get("duration", 0)

        # Normalize productivity scores
        total_study_time = sum(productivity_by_hour.values())
        if total_study_time > 0:
            productivity_patterns = {
                f"hour_{hour}": time / total_study_time
                for hour, time in productivity_by_hour.items()
            }
        else:
            productivity_patterns = {}

        # Calculate academic engagement score
        daily_study_avg = sum(study_hours_by_day.values()) / max(len(study_hours_by_day), 1)

        if daily_study_avg < 2:  # Less than 2 hours/day
            engagement_score = daily_study_avg / 2.0
        elif daily_study_avg > self.max_study_hours:
            engagement_score = max(0, 1 - (daily_study_avg - self.max_study_hours) / 6.0)
        else:
            engagement_score = 0.8

        engagement_score = engagement_score * (1.0 - procrastination_level)

        return AcademicBehaviorAnalysis(
            study_pattern_regularity=study_regularity,
            procrastination_level=procrastination_level,
            deadline_pressure_score=deadline_pressure,
            productivity_patterns=productivity_patterns,
            academic_engagement_score=engagement_score
        )

    async def _analyze_activity_patterns(self, activities: List[Dict[str, Any]]) -> ActivityPatternAnalysis:
        """Analyze general activity patterns"""

        if not activities:
            return ActivityPatternAnalysis(0.0, 0.0, 0.0, 0.0, 0.0)

        # Categorize activities by time of day
        day_activities = []
        night_activities = []

        for activity in activities:
            timestamp = activity.get("timestamp")
            if not timestamp:
                continue

            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            hour = timestamp.hour

            if hour in self.day_hours:
                day_activities.append(activity)
            else:
                night_activities.append(activity)

        # Calculate nocturnal activity ratio
        total_day_time = sum(a.get("duration", 0) for a in day_activities)
        total_night_time = sum(a.get("duration", 0) for a in night_activities)
        total_activity_time = total_day_time + total_night_time

        if total_activity_time > 0:
            nocturnal_activity_ratio = total_night_time / total_activity_time
        else:
            nocturnal_activity_ratio = 0.0

        # Calculate activity regularity
        activities_by_day = defaultdict(int)
        for activity in activities:
            timestamp = activity.get("timestamp")
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                day_key = timestamp.date()
                activities_by_day[day_key] += 1

        if len(activities_by_day) > 1:
            activity_counts = list(activities_by_day.values())
            activity_regularity = 1.0 - (np.std(activity_counts) / max(np.mean(activity_counts), 1))
            activity_regularity = max(0, min(activity_regularity, 1.0))
        else:
            activity_regularity = 0.0

        # Calculate sedentary time ratio
        sedentary_activities = [
            activity for activity in activities
            if activity.get("type") in ["sitting", "studying", "computer", "screen"]
        ]

        sedentary_time = sum(a.get("duration", 0) for a in sedentary_activities)
        sedentary_time_ratio = sedentary_time / max(total_activity_time, 1)

        # Calculate physical activity score
        physical_activities = [
            activity for activity in activities
            if activity.get("type") in ["exercise", "walking", "sports", "gym"]
        ]

        physical_time = sum(a.get("duration", 0) for a in physical_activities)
        recommended_physical_time = 1.0  # 1 hour per day
        physical_activity_score = min(physical_time / (recommended_physical_time * 7), 1.0)

        # Calculate work-life balance score
        work_activities = [
            activity for activity in activities
            if activity.get("category") in ["academic", "work", "study"]
        ]

        leisure_activities = [
            activity for activity in activities
            if activity.get("category") in ["social", "entertainment", "relaxation"]
        ]

        work_time = sum(a.get("duration", 0) for a in work_activities)
        leisure_time = sum(a.get("duration", 0) for a in leisure_activities)

        if work_time + leisure_time > 0:
            balance_ratio = leisure_time / (work_time + leisure_time)
            # Optimal balance is around 0.3-0.4 (30-40% leisure)
            optimal_balance = 0.35
            work_life_balance_score = 1.0 - abs(balance_ratio - optimal_balance) / optimal_balance
            work_life_balance_score = max(0, min(work_life_balance_score, 1.0))
        else:
            work_life_balance_score = 0.5

        return ActivityPatternAnalysis(
            activity_regularity=activity_regularity,
            nocturnal_activity_ratio=nocturnal_activity_ratio,
            sedentary_time_ratio=sedentary_time_ratio,
            physical_activity_score=physical_activity_score,
            work_life_balance_score=work_life_balance_score
        )

    def _calculate_behavioral_risks(
        self,
        sleep: SleepAnalysis,
        social: SocialAnalysis,
        academic: AcademicBehaviorAnalysis,
        activity: ActivityPatternAnalysis
    ) -> Dict[str, float]:
        """Calculate behavioral risk indicators"""

        risks = {}

        # Sleep-related risks
        risks["sleep_disruption"] = sleep.sleep_disruption_score
        risks["irregular_sleep"] = 1.0 - sleep.sleep_regularity
        risks["insufficient_sleep"] = max(0, (self.min_sleep_duration - sleep.avg_sleep_duration) / self.min_sleep_duration)

        # Social-related risks
        risks["social_withdrawal"] = social.social_withdrawal_score
        risks["social_isolation"] = 1.0 - social.social_activity_score
        risks["delayed_responses"] = min(social.response_time_avg / 24.0, 1.0)

        # Academic-related risks
        risks["procrastination"] = academic.procrastination_level
        risks["academic_pressure"] = academic.deadline_pressure_score
        risks["poor_study_patterns"] = 1.0 - academic.study_pattern_regularity
        risks["low_engagement"] = 1.0 - academic.academic_engagement_score

        # Activity-related risks
        risks["irregular_schedule"] = 1.0 - activity.activity_regularity
        risks["nocturnal_activity"] = activity.nocturnal_activity_ratio
        risks["sedentary_behavior"] = activity.sedentary_time_ratio
        risks["poor_work_life_balance"] = 1.0 - activity.work_life_balance_score

        return risks

    def _generate_behavioral_recommendations(
        self,
        sleep: SleepAnalysis,
        social: SocialAnalysis,
        academic: AcademicBehaviorAnalysis,
        activity: ActivityPatternAnalysis
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on behavioral analysis"""

        recommendations = []

        # Sleep recommendations
        if sleep.sleep_disruption_score > 0.6:
            recommendations.append({
                "type": "sleep_improvement",
                "priority": "high",
                "title": "Improve Sleep Quality",
                "content": "Focus on consistent sleep schedule and sleep hygiene practices",
                "evidence": f"Sleep disruption score: {sleep.sleep_disruption_score:.2f}"
            })

        if sleep.avg_sleep_duration < self.min_sleep_duration:
            recommendations.append({
                "type": "sleep_duration",
                "priority": "high",
                "title": "Increase Sleep Duration",
                "content": f"Try to get at least {self.min_sleep_duration} hours of sleep per night",
                "evidence": f"Current average: {sleep.avg_sleep_duration:.1f} hours"
            })

        # Social recommendations
        if social.social_withdrawal_score > 0.6:
            recommendations.append({
                "type": "social_engagement",
                "priority": "medium",
                "title": "Increase Social Interaction",
                "content": "Consider joining study groups or scheduling regular social activities",
                "evidence": f"Social withdrawal score: {social.social_withdrawal_score:.2f}"
            })

        # Academic recommendations
        if academic.procrastination_level > self.procrastination_threshold:
            recommendations.append({
                "type": "procrastination_management",
                "priority": "medium",
                "title": "Address Procrastination",
                "content": "Break tasks into smaller steps and use time management techniques",
                "evidence": f"Procrastination level: {academic.procrastination_level:.2f}"
            })

        if academic.deadline_pressure_score > 0.7:
            recommendations.append({
                "type": "academic_planning",
                "priority": "high",
                "title": "Academic Stress Management",
                "content": "Create a structured study schedule and seek academic support",
                "evidence": f"Deadline pressure: {academic.deadline_pressure_score:.2f}"
            })

        # Activity recommendations
        if activity.nocturnal_activity_ratio > 0.4:
            recommendations.append({
                "type": "sleep_schedule",
                "priority": "medium",
                "title": "Reduce Nighttime Activity",
                "content": "Try to maintain more regular daytime activity patterns",
                "evidence": f"Nocturnal activity ratio: {activity.nocturnal_activity_ratio:.2f}"
            })

        if activity.sedentary_time_ratio > 0.8:
            recommendations.append({
                "type": "physical_activity",
                "priority": "medium",
                "title": "Increase Physical Activity",
                "content": "Incorporate regular exercise and movement breaks",
                "evidence": f"Sedentary time ratio: {activity.sedentary_time_ratio:.2f}"
            })

        return recommendations

    def _filter_activities_by_time(
        self,
        activities: List[Dict[str, Any]],
        days: int
    ) -> List[Dict[str, Any]]:
        """Filter activities within specified time window"""

        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_activities = []

        for activity in activities:
            timestamp = activity.get("timestamp")
            if not timestamp:
                continue

            try:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                elif isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp)

                if timestamp >= cutoff_date:
                    filtered_activities.append(activity)

            except Exception as e:
                logger.warning(f"Invalid timestamp format: {timestamp}, error: {e}")
                continue

        return filtered_activities

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

    def _create_empty_result(self, user_id: str, time_window: int) -> BehavioralFeaturesResult:
        """Create empty result when no data is available"""

        empty_sleep = SleepAnalysis(0.0, 0.0, 0.5, 0.0, 0.0, "stable")
        empty_social = SocialAnalysis(0.0, 0.5, 0.0, 0.0, 0.5)
        empty_academic = AcademicBehaviorAnalysis(0.0, 0.0, 0.0, {}, 0.5)
        empty_activity = ActivityPatternAnalysis(0.0, 0.0, 0.0, 0.0, 0.5)

        return BehavioralFeaturesResult(
            user_id=user_id,
            analysis_timestamp=datetime.now(),
            time_window_days=time_window,
            sleep=empty_sleep,
            social=empty_social,
            academic=empty_academic,
            activity_patterns=empty_activity,
            risk_indicators={},
            recommendations=[]
        )

# Global behavioral features extractor instance
behavioral_extractor = BehavioralFeaturesExtractor()