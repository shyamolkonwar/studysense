from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from .tool_manager import tool_manager, ToolCall, ToolResult
from ..llm.llm_service import llm_service, LLMMessage
from ..core.config import settings

logger = logging.getLogger(__name__)

class StudySenseAgent:
    """
    Agent coordinator for StudySense
    Implements tool calling and RAG-augmented responses
    """

    def __init__(self):
        """Initialize the agent coordinator"""
        logger.info("StudySense Agent initialized")

    async def process_request(
        self,
        user_id: str,
        messages: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user request through the agent workflow

        Args:
            user_id: User identifier
            messages: List of conversation messages
            context: Additional context information

        Returns:
            Agent response with context and tool usage
        """
        try:
            # Extract the latest user message
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if not user_messages:
                return {
                    "response": "I didn't receive a clear message. How can I help you today?",
                    "context_used": {},
                    "tools_used": 0,
                    "tool_calls": [],
                    "timestamp": datetime.now().isoformat()
                }

            latest_message = user_messages[-1].get("content", "")

            # Analyze message for keywords that indicate what tools to use
            context_needs = self._analyze_context_needs(latest_message)

            # Gather context using tools
            gathered_context = await self._gather_context(context_needs, user_id)

            # Build enhanced prompt with context
            system_prompt = self._build_enhanced_system_prompt(gathered_context)

            # Add context to messages
            enhanced_messages = messages.copy()

            # Add system message with context
            if system_prompt:
                enhanced_messages.insert(0, {
                    "role": "system",
                    "content": system_prompt
                })

            # Use LLM service to generate response
            response = await llm_service.chat(
                messages=[LLMMessage(**msg) for msg in enhanced_messages],
                conversation_id=f"agent_{user_id}_{datetime.now().timestamp()}",
                use_rag=True,
                citations=True
            )

            return {
                "response": response.content,
                "context_used": gathered_context,
                "tools_used": len(gathered_context),
                "tool_calls": list(gathered_context.keys()),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Agent processing failed for user {user_id}: {e}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again or contact support.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _analyze_context_needs(self, message: str) -> List[str]:
        """Analyze message to determine what context/tools are needed"""
        needs = []
        message_lower = message.lower()

        # Check for academic stress indicators
        if any(word in message_lower for word in ["exam", "test", "deadline", "assignment", "study", "grade"]):
            needs.extend(["academic_schedule", "lms_deadlines"])

        # Check for emotional distress
        if any(word in message_lower for word in ["stress", "anxiety", "worried", "overwhelmed", "depressed"]):
            needs.extend(["emotional_support", "crisis_resources"])

        # Check for sleep/study patterns
        if any(word in message_lower for word in ["sleep", "tired", "can't sleep", "insomnia"]):
            needs.extend(["sleep_patterns", "study_habits"])

        # Check for social context
        if any(word in message_lower for word in ["friend", "social", "lonely", "support"]):
            needs.extend(["social_context", "peer_support"])

        return list(set(needs))  # Remove duplicates

    async def _gather_context(self, context_needs: List[str], user_id: str) -> Dict[str, Any]:
        """Gather relevant context using tools"""
        gathered_context = {}

        # Execute tools based on context needs
        tool_calls = []

        if "academic_schedule" in context_needs:
            tool_calls.append(ToolCall(
                tool_name="get_calendar_events",
                parameters={"days_ahead": 14}
            ))

        if "lms_deadlines" in context_needs:
            tool_calls.append(ToolCall(
                tool_name="lms_deadlines",
                parameters={"user_id": user_id, "days_ahead": 7}
            ))

        if "emotional_support" in context_needs:
            tool_calls.append(ToolCall(
                tool_name="rag_search",
                parameters={"query": "emotional support techniques", "k": 3}
            ))

        if "crisis_resources" in context_needs:
            tool_calls.append(ToolCall(
                tool_name="get_crisis_support",
                parameters={"crisis_type": "general"}
            ))

        # Execute tools and gather results
        for tool_call in tool_calls:
            try:
                result = await tool_manager.execute_tool(tool_call)
                if result.success:
                    gathered_context[tool_call.tool_name] = result.data
                else:
                    logger.warning(f"Tool {tool_call.tool_name} failed: {result.error}")
            except Exception as e:
                logger.error(f"Tool execution error for {tool_call.tool_name}: {e}")

        return gathered_context

    def _build_enhanced_system_prompt(self, gathered_data: Dict[str, Any]) -> str:
        """Build enhanced system prompt with gathered context"""
        prompt_parts = [
            "You are StudySense, an empathetic AI assistant specializing in student mental health and academic support.",
            "Use the following context information to provide personalized, evidence-based support:"
        ]

        # Add context from different tools
        if "get_calendar_events" in gathered_data:
            events = gathered_data["get_calendar_events"].get("events", [])
            if events:
                prompt_parts.append(f"\nUpcoming academic events: {events}")

        if "lms_deadlines" in gathered_data:
            deadlines = gathered_data["lms_deadlines"].get("deadlines", [])
            if deadlines:
                prompt_parts.append(f"\nCurrent deadlines: {deadlines}")

        if "rag_search" in gathered_data:
            search_results = gathered_data["rag_search"].get("results", [])
            if search_results:
                prompt_parts.append(f"\nRelevant knowledge base information: {search_results}")

        if "get_crisis_support" in gathered_data:
            crisis_resources = gathered_data["get_crisis_support"].get("resources", {})
            if crisis_resources:
                prompt_parts.append(f"\nAvailable crisis resources: {crisis_resources}")

        prompt_parts.extend([
            "\nGuidelines:",
            "- Be empathetic and supportive",
            "- Provide practical, actionable advice",
            "- Include specific citations when referencing information",
            "- Recognize crisis situations and provide appropriate resources",
            "- Maintain professional boundaries",
            "\nRemember: Your primary goal is to support student well-being and academic success."
        ])

        return "\n".join(prompt_parts)

# Global agent coordinator instance
agent_coordinator = StudySenseAgent()
