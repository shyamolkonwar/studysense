from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

class ModelProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    LOCAL = "local"

@dataclass
class LLMMessage:
    """Represents a message in the conversation"""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    metadata: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

@dataclass
class LLMResponse:
    """Represents an LLM response"""
    content: str
    role: str = "assistant"
    metadata: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    reasoning: Optional[str] = None  # For models that provide reasoning

@dataclass
class LLMConfig:
    """Configuration for LLM service"""
    provider: ModelProvider
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stream: bool = False
    system_prompt: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
