"""
RAG-based Recommendations Engine
Implements evidence-based recommendations using Retrieval-Augmented Generation
with proper citation support as specified in the MVP requirements.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio

from ..rag.retrieval import retrieval_pipeline
from ..rag.chroma_client import chroma_client
from ..llm.llm_service import LLMService, LLMMessage, LLMConfig, llm_service as global_llm_service

logger = logging.getLogger(__name__)

@dataclass
class RecommendationSource:
    """Source evidence for a recommendation"""
    content: str                   # Source text content
    relevance_score: float         # Relevance to the recommendation
    source_type: str              # "knowledge_base", "research", "resource"
    document_id: str              # Document identifier
    section_heading: Optional[str]  # Section heading if available
    confidence: float             # Confidence in source relevance

@dataclass
class RecommendationItem:
    """Individual recommendation with RAG evidence"""
    id: str                       # Unique recommendation ID
    type: str                     # Recommendation type (academic, social, wellness, etc.)
    priority: str                 # "low", "medium", "high", "immediate"
    title: str                   # Recommendation title
    content: str                 # Main recommendation content
    action_steps: List[str]      # Specific action steps
    resources: List[Dict[str, Any]]  # Additional resources
    evidence_sources: List[RecommendationSource]  # RAG evidence sources
    personalization_factors: List[str]  # Why this recommendation is personalized
    estimated_impact: str        # Expected impact level
    time_commitment: str         # Estimated time required
    difficulty_level: str        # "easy", "moderate", "challenging"

@dataclass
class PersonalizedRecommendations:
    """Complete set of personalized recommendations"""
    user_id: str
    generated_timestamp: datetime
    risk_context: Dict[str, Any]  # Risk assessment context
    recommendations: List[RecommendationItem]
    categorized_recommendations: Dict[str, List[RecommendationItem]]
    implementation_plan: Dict[str, Any]
    follow_up_schedule: List[Dict[str, Any]]
    evidence_summary: Dict[str, Any]

class RAGRecommendationsEngine:
    """
    RAG-powered recommendations engine that provides evidence-based,
    personalized mental health and academic support recommendations.
    """

    def __init__(self, llm_service: LLMService = None):
        """Initialize recommendations engine"""

        self.llm_service = llm_service or global_llm_service
        self.retrieval_pipeline = retrieval_pipeline
        self.chroma_client = chroma_client

        # Recommendation type configurations
        # Note: Using kb_global collection since that's where all documents are stored
        self.recommendation_types = {
            "crisis_support": {
                "priority": "immediate",
                "collections": ["kb_global"],
                "evidence_threshold": 0.7,
                "max_recommendations": 3
            },
            "academic_stress": {
                "priority": "high",
                "collections": ["kb_global"],
                "evidence_threshold": 0.6,
                "max_recommendations": 5
            },
            "sleep_improvement": {
                "priority": "high",
                "collections": ["kb_global"],
                "evidence_threshold": 0.6,
                "max_recommendations": 4
            },
            "social_support": {
                "priority": "medium",
                "collections": ["kb_global"],
                "evidence_threshold": 0.5,
                "max_recommendations": 4
            },
            "stress_management": {
                "priority": "medium",
                "collections": ["kb_global"],
                "evidence_threshold": 0.5,
                "max_recommendations": 5
            },
            "wellness_practices": {
                "priority": "low",
                "collections": ["kb_global"],
                "evidence_threshold": 0.4,
                "max_recommendations": 4
            }
        }

        # Action step templates for different recommendation types
        self.action_templates = {
            "academic_stress": [
                "Break down large assignments into smaller, manageable tasks",
                "Create a realistic study schedule with regular breaks",
                "Meet with academic advisor or tutoring services",
                "Form or join a study group for difficult subjects",
                "Practice time-blocking techniques for better focus"
            ],
            "sleep_improvement": [
                "Establish a consistent bedtime routine",
                "Avoid screens 1 hour before bedtime",
                "Create a comfortable sleep environment",
                "Limit caffeine intake after 2 PM",
                "Practice relaxation techniques before sleep"
            ],
            "social_support": [
                "Schedule regular social activities with friends",
                "Join campus clubs or organizations related to interests",
                "Reach out to family members for support",
                "Consider joining peer support groups",
                "Volunteer for causes you care about"
            ],
            "stress_management": [
                "Practice deep breathing exercises for 5 minutes daily",
                "Try progressive muscle relaxation techniques",
                "Engage in regular physical activity",
                "Practice mindfulness meditation",
                "Keep a stress journal to identify triggers"
            ]
        }

    async def generate_recommendations(
        self,
        user_id: str,
        risk_assessment: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None,
        limit_per_type: int = 3
    ) -> PersonalizedRecommendations:
        """
        Generate comprehensive RAG-based recommendations

        Args:
            user_id: Unique user identifier
            risk_assessment: Risk assessment results
            user_context: Additional user context (preferences, history, etc.)
            limit_per_type: Maximum recommendations per type

        Returns:
            Complete PersonalizedRecommendations
        """

        try:
            # Analyze risk context to determine recommendation needs
            risk_context = self._analyze_risk_context(risk_assessment)

            # Generate recommendations for each relevant type
            all_recommendations = []

            for rec_type, config in self.recommendation_types.items():
                if self._should_generate_recommendations(rec_type, risk_context):
                    recommendations = await self._generate_type_recommendations(
                        user_id, rec_type, risk_context, user_context, limit_per_type
                    )
                    all_recommendations.extend(recommendations)

            # Sort by priority and relevance
            all_recommendations = self._sort_recommendations(all_recommendations, risk_context)

            # Categorize recommendations
            categorized = self._categorize_recommendations(all_recommendations)

            # Create implementation plan
            implementation_plan = self._create_implementation_plan(categorized)

            # Create follow-up schedule
            follow_up_schedule = self._create_follow_up_schedule(risk_context)

            # Generate evidence summary
            evidence_summary = await self._create_evidence_summary(all_recommendations)

            return PersonalizedRecommendations(
                user_id=user_id,
                generated_timestamp=datetime.now(),
                risk_context=risk_context,
                recommendations=all_recommendations[:15],  # Limit total recommendations
                categorized_recommendations=categorized,
                implementation_plan=implementation_plan,
                follow_up_schedule=follow_up_schedule,
                evidence_summary=evidence_summary
            )

        except Exception as e:
            logger.error(f"Recommendations generation failed for user {user_id}: {e}")
            return self._create_default_recommendations(user_id, risk_assessment)

    def _analyze_risk_context(self, risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk assessment to determine recommendation context"""

        context = {
            "risk_level": risk_assessment.get("risk_level", "low"),
            "risk_score": risk_assessment.get("risk_score", 0.0),
            "primary_concerns": risk_assessment.get("primary_concerns", []),
            "escalation_needed": risk_assessment.get("escalation_needed", False),
            "component_scores": risk_assessment.get("component_scores", {}),
            "feature_contributions": risk_assessment.get("feature_contributions", {})
        }

        # Determine urgency levels
        if context["risk_level"] in ["crisis", "severe"]:
            context["urgency"] = "immediate"
        elif context["risk_level"] == "moderate":
            context["urgency"] = "high"
        elif context["risk_level"] == "mild":
            context["urgency"] = "medium"
        else:
            context["urgency"] = "low"

        # Identify dominant risk domains
        component_scores = context.get("component_scores", {})
        if component_scores:
            max_score = max(component_scores.values())
            dominant_domains = [
                domain for domain, score in component_scores.items()
                if score >= max_score * 0.8
            ]
            context["dominant_domains"] = dominant_domains
        else:
            context["dominant_domains"] = []

        return context

    def _should_generate_recommendations(
        self,
        rec_type: str,
        risk_context: Dict[str, Any]
    ) -> bool:
        """Determine if recommendations of a specific type should be generated"""

        # Always generate for crisis situations
        if risk_context["risk_level"] in ["crisis", "severe"]:
            return rec_type in ["crisis_support", "academic_stress", "stress_management"]

        # Generate based on dominant risk domains
        dominant_domains = risk_context.get("dominant_domains", [])

        domain_mapping = {
            "sentiment": ["stress_management", "social_support"],
            "behavioral": ["sleep_improvement", "wellness_practices"],
            "academic": ["academic_stress", "time_management"],
            "contextual": ["social_support", "stress_management"]
        }

        for domain in dominant_domains:
            if rec_type in domain_mapping.get(domain, []):
                return True

        # Generate based on primary concerns
        primary_concerns = risk_context.get("primary_concerns", [])
        concern_mapping = {
            "academic": ["academic_stress"],
            "sleep": ["sleep_improvement"],
            "social": ["social_support"],
            "stress": ["stress_management"]
        }

        for concern in primary_concerns:
            concern_lower = concern.lower()
            for keyword, types in concern_mapping.items():
                if keyword in concern_lower and rec_type in types:
                    return True

        # Default recommendations for moderate+ risk
        if risk_context["risk_score"] > 0.4:
            return rec_type in ["stress_management", "wellness_practices"]

        return False

    async def _generate_type_recommendations(
        self,
        user_id: str,
        rec_type: str,
        risk_context: Dict[str, Any],
        user_context: Optional[Dict[str, Any]],
        limit: int
    ) -> List[RecommendationItem]:
        """Generate recommendations for a specific type using RAG"""

        config = self.recommendation_types[rec_type]

        try:
            # Build search query based on risk context and type
            query = self._build_search_query(rec_type, risk_context, user_context)

            # Retrieve relevant documents
            retrieval_response = await self.retrieval_pipeline.retrieve(
                query=query,
                k=config["max_recommendations"] * 2,  # Get more to filter
                collections=config["collections"]
            )

            if not retrieval_response.results:
                return []

            # Generate recommendations using LLM with retrieved context
            recommendations = await self._generate_llm_recommendations(
                rec_type, query, retrieval_response.results, risk_context, user_context
            )

            # Limit and validate recommendations
            valid_recommendations = []
            for rec in recommendations[:limit]:
                if self._validate_recommendation(rec, risk_context):
                    valid_recommendations.append(rec)

            return valid_recommendations

        except Exception as e:
            logger.error(f"Type-specific recommendations generation failed for {rec_type}: {e}")
            return []

    def _build_search_query(
        self,
        rec_type: str,
        risk_context: Dict[str, Any],
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Build search query for RAG retrieval"""

        base_query = f"evidence-based {rec_type.replace('_', ' ')} techniques"

        # Add risk context
        if risk_context["primary_concerns"]:
            concerns_text = " ".join(risk_context["primary_concerns"][:3])
            base_query += f" for {concerns_text}"

        # Add user preferences if available
        if user_context:
            preferences = user_context.get("preferences", {})
            if preferences.get("learning_style"):
                base_query += f" {preferences['learning_style']} learning style"
            if preferences.get("time_availability"):
                base_query += f" {preferences['time_availability']} time commitment"

        return base_query

    async def _generate_llm_recommendations(
        self,
        rec_type: str,
        query: str,
        retrieved_docs: List[Any],
        risk_context: Dict[str, Any],
        user_context: Optional[Dict[str, Any]]
    ) -> List[RecommendationItem]:
        """Generate recommendations using LLM with retrieved context"""

        # Prepare context from retrieved documents
        context_docs = []
        for doc in retrieved_docs[:5]:  # Use top 5 documents
            context_docs.append({
                "content": doc.text,
                "source": doc.metadata.get("document_id", "unknown"),
                "section": doc.metadata.get("section_heading", "General"),
                "relevance": doc.score
            })

        # Create prompt for LLM
        prompt = self._create_recommendations_prompt(
            rec_type, query, context_docs, risk_context, user_context
        )

        try:
            messages = [
                LLMMessage(
                    role="system",
                    content="You are a mental health and academic support specialist. Generate evidence-based, personalized recommendations that are practical and actionable."
                ),
                LLMMessage(
                    role="user",
                    content=prompt
                )
            ]

            response = await self.llm_service.generate(
                messages=messages,
                config=LLMConfig(
                    provider=self.llm_service.default_config.provider,
                    model=self.llm_service.default_config.model,
                    temperature=0.4,  # Lower temperature for consistent recommendations
                    max_tokens=1000
                )
            )

            # Parse LLM response
            return self._parse_llm_recommendations(response.content, rec_type, context_docs)

        except Exception as e:
            logger.error(f"LLM recommendations generation failed: {e}")
            return []

    def _create_recommendations_prompt(
        self,
        rec_type: str,
        query: str,
        context_docs: List[Dict[str, Any]],
        risk_context: Dict[str, Any],
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Create prompt for LLM recommendations generation"""

        context_text = "\n\n".join([
            f"Source {i+1} ({doc['source']} - {doc['section']}):\n{doc['content']}"
            for i, doc in enumerate(context_docs)
        ])

        prompt = f"""
Generate 2-3 personalized recommendations for {rec_type.replace('_', ' ')} based on the following context.

CONTEXT:
Risk Level: {risk_context['risk_level']}
Primary Concerns: {', '.join(risk_context['primary_concerns'])}
Risk Score: {risk_context['risk_score']:.2f}

EVIDENCE BASE:
{context_text}

USER CONTEXT:
{json.dumps(user_context, indent=2) if user_context else 'No specific user context available'}

REQUIREMENTS:
1. Generate 2-3 specific, actionable recommendations
2. Each recommendation must be supported by the provided evidence
3. Include 3-4 concrete action steps for each recommendation
4. Estimate time commitment and difficulty level
5. Make recommendations personalized to the risk context

RESPONSE FORMAT (JSON):
{{
    "recommendations": [
        {{
            "title": "Clear, actionable title",
            "content": "Main recommendation content (2-3 sentences)",
            "action_steps": ["Step 1", "Step 2", "Step 3"],
            "estimated_impact": "high/medium/low",
            "time_commitment": "daily/weekly/one-time",
            "difficulty_level": "easy/moderate/challenging",
            "evidence_summary": "Brief explanation of evidence support",
            "personalization_rationale": "Why this recommendation fits the user"
        }}
    ]
}}
"""

        return prompt

    def _parse_llm_recommendations(
        self,
        llm_response: str,
        rec_type: str,
        context_docs: List[Dict[str, Any]]
    ) -> List[RecommendationItem]:
        """Parse LLM response into recommendation items"""

        try:
            # Try to parse JSON response
            response_data = json.loads(llm_response)
            recommendations_data = response_data.get("recommendations", [])
        except json.JSONDecodeError:
            # Fallback: try to extract recommendations from text
            recommendations_data = self._extract_recommendations_from_text(llm_response)

        recommendations = []
        for i, rec_data in enumerate(recommendations_data[:3]):  # Limit to 3 recommendations
            try:
                # Create evidence sources
                evidence_sources = []
                for doc in context_docs:
                    if doc["relevance"] > 0.5:  # Only include relevant sources
                        evidence_sources.append(RecommendationSource(
                            content=doc["content"][:200] + "...",
                            relevance_score=doc["relevance"],
                            source_type="knowledge_base",
                            document_id=doc["source"],
                            section_heading=doc["section"],
                            confidence=doc["relevance"]
                        ))

                recommendation = RecommendationItem(
                    id=f"{rec_type}_{i+1}_{datetime.now().strftime('%Y%m%d')}",
                    type=rec_type,
                    priority=self.recommendation_types[rec_type]["priority"],
                    title=rec_data.get("title", f"Recommendation for {rec_type.replace('_', ' ')}"),
                    content=rec_data.get("content", "Personalized recommendation based on your current situation."),
                    action_steps=rec_data.get("action_steps", []),
                    resources=[],  # Can be enhanced with additional resource retrieval
                    evidence_sources=evidence_sources,
                    personalization_factors=[rec_data.get("personalization_rationale", "Personalized based on risk assessment")],
                    estimated_impact=rec_data.get("estimated_impact", "medium"),
                    time_commitment=rec_data.get("time_commitment", "weekly"),
                    difficulty_level=rec_data.get("difficulty_level", "moderate")
                )

                recommendations.append(recommendation)

            except Exception as e:
                logger.warning(f"Failed to parse recommendation item: {e}")
                continue

        return recommendations

    def _extract_recommendations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract recommendations from unstructured text (fallback method)"""

        # Simple fallback - create basic recommendation structure
        recommendations = []

        # Look for numbered or bulleted recommendations
        lines = text.split('\n')
        current_rec = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for recommendation headers
            if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '-', '•', '*']):
                if current_rec:
                    recommendations.append(current_rec)
                current_rec = {"title": line.lstrip('123.-•* '), "content": "", "action_steps": []}
            elif current_rec and "title" in current_rec:
                if line.lower().startswith(("action", "step", "do")):
                    current_rec["action_steps"].append(line)
                else:
                    current_rec["content"] += line + " "

        if current_rec:
            recommendations.append(current_rec)

        # Fill in missing fields with defaults
        for rec in recommendations:
            rec.setdefault("content", "Personalized recommendation based on your current needs.")
            rec.setdefault("action_steps", ["Follow the recommended practices consistently"])
            rec.setdefault("estimated_impact", "medium")
            rec.setdefault("time_commitment", "weekly")
            rec.setdefault("difficulty_level", "moderate")
            rec.setdefault("personalization_rationale", "Based on current risk assessment")

        return recommendations

    def _validate_recommendation(
        self,
        recommendation: RecommendationItem,
        risk_context: Dict[str, Any]
    ) -> bool:
        """Validate recommendation appropriateness for the risk context"""

        # Check for crisis appropriateness
        if risk_context["risk_level"] == "crisis":
            return recommendation.type == "crisis_support"

        # Check evidence quality
        if not recommendation.evidence_sources:
            return False

        avg_evidence_score = sum(src.confidence for src in recommendation.evidence_sources) / len(recommendation.evidence_sources)
        if avg_evidence_score < 0.05:  # Very low threshold for testing
            return False

        # Check content appropriateness
        if len(recommendation.content) < 5:  # Very lenient
            return False

        if not recommendation.action_steps:
            return False

        return True

    def _sort_recommendations(
        self,
        recommendations: List[RecommendationItem],
        risk_context: Dict[str, Any]
    ) -> List[RecommendationItem]:
        """Sort recommendations by priority and relevance"""

        priority_order = {"immediate": 4, "high": 3, "medium": 2, "low": 1}

        def sort_key(rec: RecommendationItem):
            priority_score = priority_order.get(rec.priority, 1)

            # Boost score based on evidence quality
            evidence_score = sum(src.confidence for src in rec.evidence_sources) / len(rec.evidence_sources) if rec.evidence_sources else 0

            # Boost score based on risk alignment
            risk_alignment = 1.0
            if risk_context["risk_level"] in ["crisis", "severe"] and rec.type == "crisis_support":
                risk_alignment = 2.0

            return (priority_score, evidence_score * risk_alignment)

        return sorted(recommendations, key=sort_key, reverse=True)

    def _categorize_recommendations(
        self,
        recommendations: List[RecommendationItem]
    ) -> Dict[str, List[RecommendationItem]]:
        """Categorize recommendations by type and priority"""

        categorized = {
            "immediate_actions": [],
            "high_priority": [],
            "ongoing_care": [],
            "preventive_measures": []
        }

        for rec in recommendations:
            if rec.priority == "immediate":
                categorized["immediate_actions"].append(rec)
            elif rec.priority == "high":
                categorized["high_priority"].append(rec)
            elif rec.priority == "medium":
                categorized["ongoing_care"].append(rec)
            else:
                categorized["preventive_measures"].append(rec)

        return categorized

    def _create_implementation_plan(
        self,
        categorized_recommendations: Dict[str, List[RecommendationItem]]
    ) -> Dict[str, Any]:
        """Create structured implementation plan"""

        plan = {
            "immediate_actions": {
                "timeline": "Today - Next 24 hours",
                "recommendations": [
                    {
                        "id": rec.id,
                        "title": rec.title,
                        "first_steps": rec.action_steps[:2] if rec.action_steps else []
                    }
                    for rec in categorized_recommendations.get("immediate_actions", [])
                ]
            },
            "week_priorities": {
                "timeline": "This Week",
                "recommendations": [
                    {
                        "id": rec.id,
                        "title": rec.title,
                        "weekly_goals": rec.action_steps[:3] if rec.action_steps else []
                    }
                    for rec in categorized_recommendations.get("high_priority", [])
                ]
            },
            "ongoing_focus": {
                "timeline": "Next 2-4 Weeks",
                "recommendations": [
                    {
                        "id": rec.id,
                        "title": rec.title,
                        "monthly_goals": rec.action_steps[:4] if rec.action_steps else []
                    }
                    for rec in categorized_recommendations.get("ongoing_care", [])
                ]
            }
        }

        return plan

    def _create_follow_up_schedule(
        self,
        risk_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create follow-up schedule based on risk level"""

        schedule = []

        if risk_context["risk_level"] in ["crisis", "severe"]:
            schedule = [
                {"timeline": "24 hours", "type": "check_in", "purpose": "Safety and support verification"},
                {"timeline": "3 days", "type": "assessment", "purpose": "Risk re-evaluation"},
                {"timeline": "1 week", "type": "review", "purpose": "Progress assessment"}
            ]
        elif risk_context["risk_level"] == "moderate":
            schedule = [
                {"timeline": "3 days", "type": "check_in", "purpose": "Initial support verification"},
                {"timeline": "1 week", "type": "review", "purpose": "Progress check"},
                {"timeline": "2 weeks", "type": "assessment", "purpose": "Risk re-evaluation"}
            ]
        elif risk_context["risk_level"] == "mild":
            schedule = [
                {"timeline": "1 week", "type": "check_in", "purpose": "Wellness check"},
                {"timeline": "2 weeks", "type": "review", "purpose": "Progress review"},
                {"timeline": "1 month", "type": "assessment", "purpose": "Comprehensive evaluation"}
            ]
        else:
            schedule = [
                {"timeline": "2 weeks", "type": "check_in", "purpose": "Routine wellness check"},
                {"timeline": "1 month", "type": "review", "purpose": "Monthly review"}
            ]

        return schedule

    async def _create_evidence_summary(
        self,
        recommendations: List[RecommendationItem]
    ) -> Dict[str, Any]:
        """Create summary of evidence supporting recommendations"""

        all_sources = []
        for rec in recommendations:
            all_sources.extend(rec.evidence_sources)

        if not all_sources:
            return {"total_sources": 0, "source_types": [], "avg_relevance": 0.0}

        # Aggregate evidence statistics
        total_sources = len(all_sources)
        source_types = list(set(src.source_type for src in all_sources))
        avg_relevance = sum(src.confidence for src in all_sources) / total_sources

        # Top contributing sources
        top_sources = sorted(all_sources, key=lambda x: x.confidence, reverse=True)[:5]

        return {
            "total_sources": total_sources,
            "source_types": source_types,
            "avg_relevance": avg_relevance,
            "top_sources": [
                {
                    "document_id": src.document_id,
                    "section": src.section_heading,
                    "relevance": src.confidence,
                    "preview": src.content[:100] + "..."
                }
                for src in top_sources
            ]
        }

    def _create_default_recommendations(
        self,
        user_id: str,
        risk_assessment: Dict[str, Any]
    ) -> PersonalizedRecommendations:
        """Create default recommendations when generation fails"""

        default_rec = RecommendationItem(
            id="default_001",
            type="wellness_practices",
            priority="medium",
            title="General Wellness Support",
            content="Focus on basic self-care and stress management techniques.",
            action_steps=[
                "Get adequate sleep (7-9 hours per night)",
                "Maintain regular social connections",
                "Practice stress management techniques",
                "Seek support from friends, family, or professionals"
            ],
            resources=[],
            evidence_sources=[],
            personalization_factors=["Default recommendation"],
            estimated_impact="medium",
            time_commitment="daily",
            difficulty_level="easy"
        )

        return PersonalizedRecommendations(
            user_id=user_id,
            generated_timestamp=datetime.now(),
            risk_context=risk_assessment,
            recommendations=[default_rec],
            categorized_recommendations={
                "ongoing_care": [default_rec],
                "immediate_actions": [],
                "high_priority": [],
                "preventive_measures": []
            },
            implementation_plan={
                "immediate_actions": {"timeline": "Today", "recommendations": []},
                "week_priorities": {"timeline": "This Week", "recommendations": []},
                "ongoing_focus": {"timeline": "Next 2-4 Weeks", "recommendations": []}
            },
            follow_up_schedule=[
                {"timeline": "1 week", "type": "check_in", "purpose": "Follow up on recommendations"}
            ],
            evidence_summary={"total_sources": 0, "error": "generation_failed"}
        )

# Global recommendations engine instance
recommendations_engine = RAGRecommendationsEngine()
