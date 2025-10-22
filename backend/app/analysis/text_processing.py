"""
Phase 4: Text Processing Pipeline
Implements sentiment analysis, emotion tagging, and stress lexicon features
as specified in the MVP requirements.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import Counter

# NLP Libraries
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import spacy

logger = logging.getLogger(__name__)

# Download required NLTK data (ensure these are available)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

@dataclass
class SentimentAnalysis:
    """Sentiment analysis results"""
    compound_score: float          # VADER compound score (-1 to 1)
    positive_score: float          # VADER positive score (0 to 1)
    negative_score: float          # VADER negative score (0 to 1)
    neutral_score: float           # VADER neutral score (0 to 1)
    sentiment_label: str           # "positive", "negative", "neutral"
    subjectivity: float            # TextBlob subjectivity (0 to 1)
    polarity: float                # TextBlob polarity (-1 to 1)

@dataclass
class EmotionAnalysis:
    """Emotion tagging results"""
    primary_emotion: str           # Main emotion detected
    emotion_scores: Dict[str, float]  # All emotion scores
    confidence: float              # Confidence in emotion detection
    emotional_intensity: float     # Overall emotional intensity

@dataclass
class StressLexiconAnalysis:
    """Stress lexicon analysis results"""
    stress_score: float            # Overall stress score (0 to 1)
    stress_keywords: List[str]     # Detected stress keywords
    stress_themes: List[str]       # Identified stress themes
    academic_stress_indicators: List[str]  # Academic-specific indicators
    urgency_level: str             # "low", "medium", "high", "crisis"

@dataclass
class BurstinessAnalysis:
    """Message burstiness and frequency analysis"""
    message_frequency: float       # Messages per time period
    burstiness_score: float        # Variance from normal patterns
    active_hours_variance: float   # Irregular timing patterns
    response_latency_variance: float  # Inconsistent response times

@dataclass
class TextProcessingResult:
    """Complete text processing analysis result"""
    text_hash: str                 # Hash of analyzed text
    processing_timestamp: datetime
    sentiment: SentimentAnalysis
    emotions: EmotionAnalysis
    stress_lexicon: StressLexiconAnalysis
    burstiness: BurstinessAnalysis
    linguistic_features: Dict[str, Any]
    risk_features: Dict[str, float]

class TextProcessingPipeline:
    """
    Advanced text processing pipeline for mental health analysis.
    Implements VADER sentiment, emotion tagging, and stress lexicon analysis.
    """

    def __init__(self):
        """Initialize text processing pipeline with all required models"""

        # Initialize VADER sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # Initialize spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Some features will be limited.")
            self.nlp = None

        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # Academic stress lexicon
        self.academic_stress_keywords = {
            'exam_stress': [
                'exam', 'test', 'midterm', 'final', 'quiz', 'assessment',
                'studying', 'cramming', 'all-nighter', 'review', 'prep'
            ],
            'deadline_pressure': [
                'deadline', 'due', 'submission', 'turn in', 'submit',
                'late', 'overdue', 'extension', 'procrastinate', 'rushed'
            ],
            'academic_overwhelm': [
                'overwhelmed', 'swamped', 'buried', 'drowning', 'too much',
                'cant keep up', 'falling behind', 'struggling', 'difficult'
            ],
            'performance_anxiety': [
                'fail', 'failing', 'grade', 'gpa', 'performance', 'score',
                'worried', 'anxious', 'nervous', 'pressure', 'expectations'
            ],
            'time_management': [
                'time', 'schedule', 'busy', 'no time', 'running out',
                'behind schedule', 'deadline', 'manage time', 'organized'
            ]
        }

        # General stress lexicon
        self.stress_keywords = {
            'high_stress': [
                'stress', 'stressed', 'overwhelmed', 'anxious', 'worried',
                'panic', 'freaking out', 'losing mind', 'cant handle',
                'breaking point', 'at my limit', 'exhausted', 'burnout'
            ],
            'moderate_stress': [
                'tired', 'busy', 'pressure', 'challenge', 'difficult',
                'hard', 'struggling', 'trying', 'managing', 'coping'
            ],
            'coping_indicators': [
                'break', 'rest', 'relax', 'breathe', 'meditate', 'exercise',
                'talking', 'sharing', 'support', 'help', 'therapy'
            ]
        }

        # Emotion lexicon (simplified Plutchik-inspired)
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'pleased', 'satisfied', 'content'],
            'trust': ['confident', 'secure', 'comfortable', 'safe', 'supported'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious'],
            'surprise': ['shocked', 'amazed', 'surprised', 'astonished', 'unexpected'],
            'sadness': ['sad', 'depressed', 'down', 'unhappy', 'miserable'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'frustrated'],
            'anticipation': ['excited', 'eager', 'looking forward', 'expectant', 'hopeful']
        }

    async def process_text(
        self,
        messages: List[Dict[str, Any]],
        time_window: int = 7,
        include_burstiness: bool = True
    ) -> TextProcessingResult:
        """
        Process text data with comprehensive analysis pipeline

        Args:
            messages: List of message dictionaries with 'content' and 'timestamp'
            time_window: Time window in days for analysis
            include_burstiness: Whether to include burstiness analysis

        Returns:
            Complete TextProcessingResult with all analyses
        """

        if not messages:
            return self._create_empty_result()

        # Filter messages by time window
        recent_messages = self._filter_messages_by_time(messages, time_window)

        if not recent_messages:
            return self._create_empty_result()

        # Extract text content
        text_content = " ".join([msg.get("content", "") for msg in recent_messages])

        # Create text hash for caching/deduplication
        text_hash = self._create_text_hash(text_content)

        try:
            # Sentiment analysis (VADER + TextBlob)
            sentiment = await self._analyze_sentiment(text_content)

            # Emotion analysis
            emotions = await self._analyze_emotions(text_content)

            # Stress lexicon analysis
            stress_lexicon = await self._analyze_stress_lexicon(text_content)

            # Burstiness analysis (if requested and sufficient data)
            burstiness = BurstinessAnalysis(0.0, 0.0, 0.0, 0.0)
            if include_burstiness and len(recent_messages) > 1:
                burstiness = await self._analyze_burstiness(recent_messages, time_window)

            # Linguistic features
            linguistic_features = await self._extract_linguistic_features(text_content)

            # Calculate risk features
            risk_features = self._calculate_risk_features(
                sentiment, emotions, stress_lexicon, burstiness
            )

            return TextProcessingResult(
                text_hash=text_hash,
                processing_timestamp=datetime.now(),
                sentiment=sentiment,
                emotions=emotions,
                stress_lexicon=stress_lexicon,
                burstiness=burstiness,
                linguistic_features=linguistic_features,
                risk_features=risk_features
            )

        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return self._create_empty_result()

    async def _analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Analyze sentiment using VADER and TextBlob"""

        # VADER analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)

        # TextBlob analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Determine sentiment label
        compound = vader_scores['compound']
        if compound >= 0.05:
            sentiment_label = "positive"
        elif compound <= -0.05:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        return SentimentAnalysis(
            compound_score=compound,
            positive_score=vader_scores['pos'],
            negative_score=vader_scores['neg'],
            neutral_score=vader_scores['neu'],
            sentiment_label=sentiment_label,
            subjectivity=subjectivity,
            polarity=polarity
        )

    async def _analyze_emotions(self, text: str) -> EmotionAnalysis:
        """Analyze emotions using keyword-based approach"""

        # Tokenize and lowercase text
        tokens = [token.lower() for token in word_tokenize(text)]

        # Count emotion keywords
        emotion_scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for token in tokens if token in keywords)
            emotion_scores[emotion] = count / len(tokens) if tokens else 0.0

        # Find primary emotion
        if emotion_scores:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            max_score = emotion_scores[primary_emotion]
        else:
            primary_emotion = "neutral"
            max_score = 0.0

        # Calculate confidence based on score distribution
        if max_score > 0:
            total_scores = sum(emotion_scores.values())
            confidence = max_score / total_scores if total_scores > 0 else 0.0
        else:
            confidence = 0.0

        # Calculate emotional intensity
        emotional_intensity = sum(emotion_scores.values()) / len(emotion_scores)

        return EmotionAnalysis(
            primary_emotion=primary_emotion,
            emotion_scores=emotion_scores,
            confidence=confidence,
            emotional_intensity=emotional_intensity
        )

    async def _analyze_stress_lexicon(self, text: str) -> StressLexiconAnalysis:
        """Analyze stress using academic and general stress lexicons"""

        text_lower = text.lower()

        # Count academic stress indicators
        academic_indicators = []
        academic_stress_count = 0

        for theme, keywords in self.academic_stress_keywords.items():
            theme_count = sum(1 for keyword in keywords if keyword in text_lower)
            if theme_count > 0:
                academic_indicators.append(theme)
                academic_stress_count += theme_count

        # Count general stress keywords
        stress_keywords_found = []
        high_stress_count = 0
        moderate_stress_count = 0
        coping_count = 0

        for keyword in self.stress_keywords['high_stress']:
            if keyword in text_lower:
                stress_keywords_found.append(keyword)
                high_stress_count += 1

        for keyword in self.stress_keywords['moderate_stress']:
            if keyword in text_lower:
                stress_keywords_found.append(keyword)
                moderate_stress_count += 1

        for keyword in self.stress_keywords['coping_indicators']:
            if keyword in text_lower:
                coping_count += 1

        # Calculate stress score (weighted)
        stress_score = (
            high_stress_count * 0.4 +
            moderate_stress_count * 0.2 +
            academic_stress_count * 0.3 -
            coping_count * 0.1
        )

        # Normalize to 0-1 range
        total_text_length = len(text.split())
        stress_score = min(stress_score / max(total_text_length * 0.1, 1), 1.0)

        # Determine stress themes
        stress_themes = []
        if academic_stress_count > 0:
            stress_themes.extend(academic_indicators)
        if high_stress_count > 2:
            stress_themes.append("high_general_stress")
        if moderate_stress_count > 3:
            stress_themes.append("moderate_stress_accumulation")

        # Determine urgency level
        if stress_score >= 0.8:
            urgency = "crisis"
        elif stress_score >= 0.6:
            urgency = "high"
        elif stress_score >= 0.3:
            urgency = "medium"
        else:
            urgency = "low"

        return StressLexiconAnalysis(
            stress_score=stress_score,
            stress_keywords=stress_keywords_found[:10],  # Top 10 keywords
            stress_themes=stress_themes,
            academic_stress_indicators=academic_indicators,
            urgency_level=urgency
        )

    async def _analyze_burstiness(
        self,
        messages: List[Dict[str, Any]],
        time_window: int
    ) -> BurstinessAnalysis:
        """Analyze message burstiness and frequency patterns"""

        if len(messages) < 2:
            return BurstinessAnalysis(0.0, 0.0, 0.0, 0.0)

        # Calculate message frequency
        time_span = self._calculate_time_span(messages)
        message_frequency = len(messages) / max(time_span, 1)  # Messages per hour

        # Calculate burstiness (variance from expected frequency)
        expected_frequency = len(messages) / (time_window * 24)  # Expected per hour
        burstiness_score = abs(message_frequency - expected_frequency) / max(expected_frequency, 1)
        burstiness_score = min(burstiness_score, 1.0)

        # Calculate active hours variance
        message_hours = []
        for msg in messages:
            timestamp = msg.get("timestamp")
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                message_hours.append(timestamp.hour)

        if message_hours:
            hour_variance = len(set(message_hours)) / 24  # Hours with messages
            active_hours_variance = 1.0 - hour_variance  # Irregularity score
        else:
            active_hours_variance = 0.0

        # For response latency, we need conversation structure
        # This is a simplified version
        response_latency_variance = 0.0

        return BurstinessAnalysis(
            message_frequency=message_frequency,
            burstiness_score=burstiness_score,
            active_hours_variance=active_hours_variance,
            response_latency_variance=response_latency_variance
        )

    async def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features for analysis"""

        features = {}

        # Basic text statistics
        words = word_tokenize(text)
        sentences = sent_tokenize(text)

        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = len(words) / max(len(sentences), 1)

        # Readability metrics (simplified)
        features['avg_word_length'] = sum(len(word) for word in words) / max(len(words), 1)

        # Punctuation analysis
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['punctuation_ratio'] = (
            sum(1 for char in text if not char.isalnum() and not char.isspace()) /
            max(len(text), 1)
        )

        # Personal pronouns (indicators of self-focus)
        first_person_pronouns = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ourselves']
        features['first_person_ratio'] = (
            sum(1 for word in words if word.lower() in first_person_pronouns) /
            max(len(words), 1)
        )

        # Negation words
        negation_words = ['not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nor']
        features['negation_ratio'] = (
            sum(1 for word in words if word.lower() in negation_words) /
            max(len(words), 1)
        )

        # If spaCy is available, add more advanced features
        if self.nlp:
            doc = self.nlp(text)

            # Part-of-speech tags
            pos_counts = Counter([token.pos_ for token in doc])
            total_tokens = len(doc)

            features['noun_ratio'] = pos_counts.get('NOUN', 0) / max(total_tokens, 1)
            features['verb_ratio'] = pos_counts.get('VERB', 0) / max(total_tokens, 1)
            features['adj_ratio'] = pos_counts.get('ADJ', 0) / max(total_tokens, 1)
            features['adv_ratio'] = pos_counts.get('ADV', 0) / max(total_tokens, 1)

            # Named entities
            features['named_entity_count'] = len(doc.ents)

            # Dependency parsing features
            features['avg_dependency_depth'] = (
                sum([token.head.i - token.i for token in doc if token.head.i > token.i]) /
                max(len([token for token in doc if token.head.i > token.i]), 1)
            )

        return features

    def _calculate_risk_features(
        self,
        sentiment: SentimentAnalysis,
        emotions: EmotionAnalysis,
        stress_lexicon: StressLexiconAnalysis,
        burstiness: BurstinessAnalysis
    ) -> Dict[str, float]:
        """Calculate risk features from all analyses"""

        risk_features = {}

        # Sentiment-based risks
        risk_features['negative_sentiment'] = sentiment.negative_score
        risk_features['low_subjectivity'] = 1.0 - sentiment.subjectivity  # Can indicate withdrawal

        # Emotion-based risks
        risk_features['fear_intensity'] = emotions.emotion_scores.get('fear', 0.0)
        risk_features['sadness_intensity'] = emotions.emotion_scores.get('sadness', 0.0)
        risk_features['anger_intensity'] = emotions.emotion_scores.get('anger', 0.0)
        risk_features['emotional_intensity'] = emotions.emotional_intensity

        # Stress-based risks
        risk_features['stress_lexicon_score'] = stress_lexicon.stress_score
        risk_features['academic_stress'] = len(stress_lexicon.academic_stress_indicators) / 5.0
        risk_features['urgency_level'] = {
            'low': 0.0, 'medium': 0.3, 'high': 0.6, 'crisis': 1.0
        }.get(stress_lexicon.urgency_level, 0.0)

        # Behavioral risks from text patterns
        risk_features['message_burstiness'] = burstiness.burstiness_score
        risk_features['irregular_timing'] = burstiness.active_hours_variance

        # Combined risk indicators
        high_risk_emotions = ['fear', 'sadness', 'anger']
        negative_emotion_score = sum(
            emotions.emotion_scores.get(emotion, 0.0) for emotion in high_risk_emotions
        ) / len(high_risk_emotions)
        risk_features['negative_emotion_score'] = negative_emotion_score

        # Academic stress focus
        academic_focus = len(stress_lexicon.academic_stress_indicators) > 0
        risk_features['academic_stress_focus'] = 1.0 if academic_focus else 0.0

        return risk_features

    def _filter_messages_by_time(
        self,
        messages: List[Dict[str, Any]],
        days: int
    ) -> List[Dict[str, Any]]:
        """Filter messages within specified time window"""

        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_messages = []

        for msg in messages:
            timestamp = msg.get("timestamp")
            if not timestamp:
                continue

            try:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                elif isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp)

                if timestamp >= cutoff_date:
                    filtered_messages.append(msg)

            except Exception as e:
                logger.warning(f"Invalid timestamp format: {timestamp}, error: {e}")
                continue

        return filtered_messages

    def _create_text_hash(self, text: str) -> str:
        """Create hash for text content"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()

    def _calculate_time_span(self, messages: List[Dict[str, Any]]) -> float:
        """Calculate time span in hours between first and last message"""

        if len(messages) < 2:
            return 1.0

        timestamps = []
        for msg in messages:
            timestamp = msg.get("timestamp")
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    elif isinstance(timestamp, (int, float)):
                        timestamp = datetime.fromtimestamp(timestamp)
                    timestamps.append(timestamp)
                except Exception:
                    continue

        if len(timestamps) < 2:
            return 1.0

        time_span = max(timestamps) - min(timestamps)
        return max(time_span.total_seconds() / 3600, 1.0)  # Convert to hours

    def _create_empty_result(self) -> TextProcessingResult:
        """Create empty result when no data is available"""

        empty_sentiment = SentimentAnalysis(0.0, 0.0, 0.0, 1.0, "neutral", 0.0, 0.0)
        empty_emotions = EmotionAnalysis("neutral", {}, 0.0, 0.0)
        empty_stress = StressLexiconAnalysis(0.0, [], [], [], "low")
        empty_burstiness = BurstinessAnalysis(0.0, 0.0, 0.0, 0.0)

        return TextProcessingResult(
            text_hash="empty",
            processing_timestamp=datetime.now(),
            sentiment=empty_sentiment,
            emotions=empty_emotions,
            stress_lexicon=empty_stress,
            burstiness=empty_burstiness,
            linguistic_features={},
            risk_features={}
        )

# Global text processing pipeline instance
text_processor = TextProcessingPipeline()