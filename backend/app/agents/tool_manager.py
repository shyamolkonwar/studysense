from typing import List, Dict, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
import logging
import asyncio
import json
from datetime import datetime
from dataclasses import dataclass

from ..rag.retrieval import retrieval_pipeline, RetrievalResponse
from ..llm.llm_service import LLMService, llm_service
from ..core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class ToolCall:
    """Represents a tool call with parameters"""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: Optional[str] = None
    timestamp: Optional[datetime] = None

@dataclass
class ToolResult:
    """Represents the result of a tool call"""
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None

class BaseTool(ABC):
    """Base class for all agent tools"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters"""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters"""
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate tool parameters"""
        return True

class RAGSearchTool(BaseTool):
    """Tool for searching the RAG knowledge base"""

    def __init__(self):
        super().__init__(
            name="rag_search",
            description="Search the knowledge base for mental health and academic support information"
        )
        self.retrieval_pipeline = retrieval_pipeline

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute RAG search"""
        try:
            query = parameters.get("query", "")
            if not query:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Query parameter is required"
                )

            # Extract optional parameters
            stress_level = parameters.get("stress_level")
            target_audience = parameters.get("target_audience")
            k = parameters.get("k", 5)

            # Perform search
            response = await self.retrieval_pipeline.retrieve(
                query=query,
                k=k,
                stress_level=stress_level,
                target_audience=target_audience,
                collections=["kb_global", "campus_resources"]
            )

            # Format results
            results = []
            for result in response.results:
                results.append({
                    "content": result.text,
                    "source": result.metadata.get("document_id", "Unknown"),
                    "score": result.score,
                    "metadata": result.metadata
                })

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": results,
                    "total_results": len(results)
                }
            )

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for the knowledge base"
                        },
                        "stress_level": {
                            "type": "string",
                            "enum": ["mild", "moderate", "severe", "crisis"],
                            "description": "Filter by stress level (optional)"
                        },
                        "target_audience": {
                            "type": "string",
                            "enum": ["students", "undergraduate", "graduate"],
                            "description": "Filter by target audience (optional)"
                        },
                        "k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5,
                            "description": "Number of results to return"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

class CalendarEventsTool(BaseTool):
    """Tool for retrieving calendar events and deadlines"""

    def __init__(self):
        super().__init__(
            name="get_calendar_events",
            description="Retrieve calendar events, deadlines, and academic schedule"
        )

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute calendar events retrieval"""
        try:
            # This would integrate with Google Calendar, Canvas, etc.
            # For now, return mock data
            days_ahead = parameters.get("days_ahead", 7)
            event_type = parameters.get("event_type", "all")

            # Mock implementation - replace with actual integration
            mock_events = [
                {
                    "title": "Psychology 101 Midterm",
                    "date": "2025-01-20",
                    "type": "exam",
                    "stress_impact": "high"
                },
                {
                    "title": "Research Paper Deadline",
                    "date": "2025-01-25",
                    "type": "deadline",
                    "stress_impact": "high"
                }
            ]

            return ToolResult(
                success=True,
                data={
                    "events": mock_events,
                    "total_events": len(mock_events),
                    "timeframe": f"Next {days_ahead} days"
                }
            )

        except Exception as e:
            logger.error(f"Calendar events error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days_ahead": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 30,
                            "default": 7,
                            "description": "Number of days ahead to look for events"
                        },
                        "event_type": {
                            "type": "string",
                            "enum": ["all", "exam", "deadline", "assignment", "class"],
                            "default": "all",
                            "description": "Filter by event type"
                        }
                    }
                }
            }
        }

class StudyResourcesTool(BaseTool):
    """Tool for recommending study resources and techniques"""

    def __init__(self):
        super().__init__(
            name="get_study_resources",
            description="Get personalized study resources and techniques"
        )

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute study resources retrieval"""
        try:
            subject = parameters.get("subject", "")
            difficulty = parameters.get("difficulty", "moderate")
            learning_style = parameters.get("learning_style", "general")

            # Use RAG to get relevant resources
            query = f"study techniques and resources for {subject}" if subject else "effective study strategies"
            response = await retrieval_pipeline.retrieve(
                query=query,
                k=3,
                collections=["kb_global"]
            )

            # Format results
            resources = []
            for result in response.results:
                resources.append({
                    "title": result.metadata.get("section_heading", "Study Resource"),
                    "content": result.text[:200] + "...",
                    "relevance": result.score,
                    "source": result.metadata.get("document_id", "Knowledge Base")
                })

            return ToolResult(
                success=True,
                data={
                    "subject": subject or "general",
                    "difficulty": difficulty,
                    "learning_style": learning_style,
                    "resources": resources,
                    "total_resources": len(resources)
                }
            )

        except Exception as e:
            logger.error(f"Study resources error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "Subject area for study resources (e.g., 'mathematics', 'psychology')"
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["beginner", "moderate", "advanced"],
                            "default": "moderate",
                            "description": "Difficulty level"
                        },
                        "learning_style": {
                            "type": "string",
                            "enum": ["visual", "auditory", "kinesthetic", "general"],
                            "default": "general",
                            "description": "Preferred learning style"
                        }
                    }
                }
            }
        }

class MessagingFetchTool(BaseTool):
    """Tool for fetching recent messaging data"""

    def __init__(self):
        super().__init__(
            name="messaging_fetch",
            description="Fetch recent messaging data for analysis"
        )

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute messaging data retrieval"""
        try:
            user_id = parameters.get("user_id")
            hours_back = parameters.get("hours_back", 24)
            platform = parameters.get("platform", "all")

            if not user_id:
                return ToolResult(
                    success=False,
                    data=None,
                    error="user_id parameter is required"
                )

            # Mock implementation - in real system would integrate with messaging platforms
            mock_messages = [
                {
                    "platform": "slack",
                    "content": "I'm feeling really stressed about finals",
                    "timestamp": "2025-01-18T14:30:00Z",
                    "sentiment": "negative"
                },
                {
                    "platform": "discord",
                    "content": "Can't sleep, too worried about grades",
                    "timestamp": "2025-01-18T02:15:00Z",
                    "sentiment": "negative"
                }
            ]

            return ToolResult(
                success=True,
                data={
                    "user_id": user_id,
                    "messages": mock_messages,
                    "timeframe": f"Last {hours_back} hours",
                    "platform": platform,
                    "total_messages": len(mock_messages)
                }
            )

        except Exception as e:
            logger.error(f"Messaging fetch error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "User ID to fetch messages for"
                        },
                        "hours_back": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 168,  # 1 week
                            "default": 24,
                            "description": "Hours of message history to fetch"
                        },
                        "platform": {
                            "type": "string",
                            "enum": ["all", "slack", "discord", "teams", "whatsapp"],
                            "default": "all",
                            "description": "Platform to fetch from"
                        }
                    },
                    "required": ["user_id"]
                }
            }
        }

class LMSDdeadlinesTool(BaseTool):
    """Tool for fetching LMS deadlines and assignments"""

    def __init__(self):
        super().__init__(
            name="lms_deadlines",
            description="Fetch upcoming deadlines and assignments from LMS"
        )

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute LMS deadlines retrieval"""
        try:
            user_id = parameters.get("user_id")
            days_ahead = parameters.get("days_ahead", 7)
            include_completed = parameters.get("include_completed", False)

            if not user_id:
                return ToolResult(
                    success=False,
                    data=None,
                    error="user_id parameter is required"
                )

            # Mock implementation - in real system would integrate with Canvas/Moodle APIs
            mock_deadlines = [
                {
                    "title": "Psychology Research Paper",
                    "due_date": "2025-01-25T23:59:00Z",
                    "course": "PSYC 101",
                    "type": "assignment",
                    "stress_impact": "high",
                    "completed": False
                },
                {
                    "title": "Math Quiz 3",
                    "due_date": "2025-01-23T10:00:00Z",
                    "course": "MATH 201",
                    "type": "quiz",
                    "stress_impact": "medium",
                    "completed": False
                },
                {
                    "title": "Computer Science Final Project",
                    "due_date": "2025-01-30T23:59:00Z",
                    "course": "CS 301",
                    "type": "project",
                    "stress_impact": "high",
                    "completed": False
                }
            ]

            # Filter by completion status if requested
            if not include_completed:
                mock_deadlines = [d for d in mock_deadlines if not d["completed"]]

            return ToolResult(
                success=True,
                data={
                    "user_id": user_id,
                    "deadlines": mock_deadlines,
                    "days_ahead": days_ahead,
                    "total_deadlines": len(mock_deadlines),
                    "include_completed": include_completed
                }
            )

        except Exception as e:
            logger.error(f"LMS deadlines error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "User ID to fetch deadlines for"
                        },
                        "days_ahead": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 30,
                            "default": 7,
                            "description": "Days ahead to look for deadlines"
                        },
                        "include_completed": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include completed assignments"
                        }
                    },
                    "required": ["user_id"]
                }
            }
        }

class NotificationsTool(BaseTool):
    """Tool for sending notifications and alerts"""

    def __init__(self):
        super().__init__(
            name="send_notification",
            description="Send notifications and alerts to users"
        )

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute notification sending"""
        try:
            user_id = parameters.get("user_id")
            notification_type = parameters.get("notification_type", "alert")
            message = parameters.get("message", "")
            priority = parameters.get("priority", "normal")
            channels = parameters.get("channels", ["in_app"])

            if not user_id:
                return ToolResult(
                    success=False,
                    data=None,
                    error="user_id parameter is required"
                )

            if not message:
                return ToolResult(
                    success=False,
                    data=None,
                    error="message parameter is required"
                )

            # Mock implementation - in real system would send actual notifications
            notification_record = {
                "user_id": user_id,
                "type": notification_type,
                "message": message,
                "priority": priority,
                "channels": channels,
                "timestamp": "2025-01-18T15:00:00Z",
                "status": "sent"
            }

            return ToolResult(
                success=True,
                data={
                    "notification": notification_record,
                    "channels_attempted": channels,
                    "estimated_delivery": "immediate"
                }
            )

        except Exception as e:
            logger.error(f"Notification error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "User ID to send notification to"
                        },
                        "notification_type": {
                            "type": "string",
                            "enum": ["alert", "reminder", "support", "crisis"],
                            "default": "alert",
                            "description": "Type of notification"
                        },
                        "message": {
                            "type": "string",
                            "description": "Notification message content"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "normal", "high", "urgent"],
                            "default": "normal",
                            "description": "Notification priority level"
                        },
                        "channels": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["in_app", "email", "sms", "push"]
                            },
                            "default": ["in_app"],
                            "description": "Notification channels to use"
                        }
                    },
                    "required": ["user_id", "message"]
                }
            }
        }

class CrisisSupportTool(BaseTool):
    """Tool for providing crisis support and emergency resources"""

    def __init__(self):
        super().__init__(
            name="get_crisis_support",
            description="Provide immediate crisis support resources and emergency contacts"
        )

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute crisis support retrieval"""
        try:
            crisis_type = parameters.get("crisis_type", "general")
            location = parameters.get("location", "general")

            # Crisis resources based on type
            resources = {
                "general": {
                    "hotline": "988 - National Suicide Prevention Lifeline",
                    "text": "Text HOME to 741741 - Crisis Text Line",
                    "emergency": "Call 911 or go to nearest emergency room"
                },
                "anxiety": {
                    "breathing": "Practice 4-7-8 breathing technique",
                    "grounding": "Use 5-4-3-2-1 grounding exercise",
                    "professional": "Contact mental health professional"
                },
                "academic": {
                    "academic_advisor": "Contact academic advisor",
                    "counseling": "Visit campus counseling services",
                    "tutoring": "Seek tutoring services"
                }
            }

            return ToolResult(
                success=True,
                data={
                    "crisis_type": crisis_type,
                    "resources": resources.get(crisis_type, resources["general"]),
                    "immediate_help": True,
                    "location": location
                }
            )

        except Exception as e:
            logger.error(f"Crisis support error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "crisis_type": {
                            "type": "string",
                            "enum": ["general", "anxiety", "depression", "academic", "relationship"],
                            "default": "general",
                            "description": "Type of crisis or support needed"
                        },
                        "location": {
                            "type": "string",
                            "default": "general",
                            "description": "User's location for localized resources"
                        }
                    }
                }
            }
        }

class ToolManager:
    """Manages and executes agent tools"""

    def __init__(self):
        """Initialize tool manager with all available tools"""
        self.tools = {
            "rag_search": RAGSearchTool(),
            "get_calendar_events": CalendarEventsTool(),
            "get_study_resources": StudyResourcesTool(),
            "messaging_fetch": MessagingFetchTool(),
            "lms_deadlines": LMSDdeadlinesTool(),
            "send_notification": NotificationsTool(),
            "get_crisis_support": CrisisSupportTool()
        }
        self.llm_service = llm_service

        # Tool call history for audit purposes
        self.tool_call_history: List[ToolCall] = []

    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a specific tool call"""
        tool_name = tool_call.tool_name

        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not found"
            )

        # Log tool call for auditing
        self.tool_call_history.append(ToolCall(
            tool_name=tool_name,
            parameters=tool_call.parameters,
            call_id=tool_call.call_id,
            timestamp=datetime.now()
        ))

        # Validate parameters
        tool = self.tools[tool_name]
        if not tool.validate_parameters(tool_call.parameters):
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid parameters for tool '{tool_name}'"
            )

        # Execute tool
        start_time = datetime.now()
        try:
            result = await tool.execute(tool_call.parameters)
            result.execution_time = (datetime.now() - start_time).total_seconds()

            if result.success:
                logger.info(f"Tool {tool_name} executed successfully in {result.execution_time:.2f}s")
            else:
                logger.warning(f"Tool {tool_name} failed: {result.error}")

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Tool {tool_name} execution error: {e}")

            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time
            )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all available tools"""
        return [tool.get_schema() for tool in self.tools.values()]

    def get_tool_names(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.tools.keys())

    def get_tool_call_history(self, limit: int = 50) -> List[ToolCall]:
        """Get recent tool call history"""
        return self.tool_call_history[-limit:]

    def clear_tool_call_history(self):
        """Clear tool call history"""
        self.tool_call_history.clear()
        logger.info("Tool call history cleared")

# Global tool manager instance
tool_manager = ToolManager()
