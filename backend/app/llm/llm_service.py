from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
import json

from .providers import OpenAIProvider, AnthropicProvider, LocalProvider
from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
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

class LLMService:
    """
    Provider-agnostic LLM service supporting multiple providers
    with consistent interface and fallback capabilities
    """

    def __init__(self, default_config: Optional[LLMConfig] = None):
        """
        Initialize LLM service with default configuration

        Args:
            default_config: Default configuration for the service
        """
        self.default_config = default_config or self._create_default_config()
        self.providers = self._initialize_providers()
        self.current_provider = self.default_config.provider

        # Cache for conversation contexts
        self._conversation_cache: Dict[str, List[LLMMessage]] = {}

        logger.info(f"LLM Service initialized with provider: {self.current_provider}")

    def _create_default_config(self) -> LLMConfig:
        """Create default configuration based on available API keys"""

        # Determine best available provider
        if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
            return LLMConfig(
                provider=ModelProvider.OPENAI,
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=2000,
                system_prompt=self._get_default_system_prompt()
            )
        elif hasattr(settings, 'ANTHROPIC_API_KEY') and settings.ANTHROPIC_API_KEY:
            return LLMConfig(
                provider=ModelProvider.ANTHROPIC,
                model="claude-3-haiku-20240307",
                temperature=0.7,
                max_tokens=2000,
                system_prompt=self._get_default_system_prompt()
            )
        else:
            # Fallback to local model
            return LLMConfig(
                provider=ModelProvider.LOCAL,
                model="llama2-7b-chat",
                temperature=0.7,
                max_tokens=2000,
                system_prompt=self._get_default_system_prompt()
            )

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for StudySense"""
        return """You are StudySense, an AI assistant focused on student mental health and academic success. Your role is to provide:

1. **Evidence-based support** - Use RAG-retrieved information to provide accurate, research-backed guidance
2. **Empathetic responses** - Show understanding and validation of student concerns
3. **Actionable advice** - Provide concrete, implementable strategies
4. **Resource referrals** - Suggest appropriate campus and community resources
5. **Safety prioritization** - Recognize and respond appropriately to crisis situations

**Guidelines:**
- Always cite your sources when providing specific information
- If you don't know something, be honest and suggest seeking professional help
- For mental health crises, provide immediate resources and hotlines
- Maintain confidentiality and privacy awareness
- Adapt responses to the student's academic context

**Safety Protocol:**
If you detect any indication of self-harm, suicide risk, or crisis, immediately provide:
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741
- Encourage contacting campus counseling services
- Recommend seeking immediate professional help

Your responses should be supportive, practical, and grounded in research while maintaining appropriate boundaries for an AI assistant."""

    def _initialize_providers(self) -> Dict[ModelProvider, Any]:
        """Initialize all available LLM providers"""
        providers = {}

        try:
            # Initialize OpenAI provider
            if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
                providers[ModelProvider.OPENAI] = OpenAIProvider(
                    api_key=settings.OPENAI_API_KEY
                )
                logger.info("OpenAI provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI provider: {e}")

        try:
            # Initialize Anthropic provider
            if hasattr(settings, 'ANTHROPIC_API_KEY') and settings.ANTHROPIC_API_KEY:
                providers[ModelProvider.ANTHROPIC] = AnthropicProvider(
                    api_key=settings.ANTHROPIC_API_KEY
                )
                logger.info("Anthropic provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic provider: {e}")

        try:
            # Initialize local provider (always available as fallback)
            providers[ModelProvider.LOCAL] = LocalProvider()
            logger.info("Local provider initialized")
        except Exception as e:
            logger.error(f"Failed to initialize local provider: {e}")

        if not providers:
            raise ValueError("No LLM providers could be initialized")

        return providers

    async def generate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
        conversation_id: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate response using specified or default configuration

        Args:
            messages: List of conversation messages
            config: Optional configuration override
            conversation_id: Optional conversation ID for context management

        Returns:
            LLMResponse with generated content and metadata
        """
        config = config or self.default_config
        provider = self._get_provider(config.provider)

        try:
            # Add system prompt if provided and not already in messages
            if config.system_prompt and not any(m.role == "system" for m in messages):
                system_msg = LLMMessage(
                    role="system",
                    content=config.system_prompt,
                    metadata={"type": "system_prompt"}
                )
                messages = [system_msg] + messages

            # Cache conversation if conversation_id provided
            if conversation_id:
                self._update_conversation_cache(conversation_id, messages)

            # Generate response
            response = await provider.generate(
                messages=messages,
                config=config
            )

            # Add provider metadata
            response.provider = config.provider.value
            response.model = config.model

            # Cache response if conversation_id provided
            if conversation_id:
                self._add_to_conversation_cache(conversation_id, response)

            return response

        except Exception as e:
            logger.error(f"Error generating response with {config.provider}: {e}")
            # Try fallback provider
            return await self._fallback_generate(messages, config, conversation_id)

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
        conversation_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response

        Args:
            messages: List of conversation messages
            config: Optional configuration override
            conversation_id: Optional conversation ID

        Yields:
            Content chunks as they are generated
        """
        config = config or self.default_config
        config.stream = True
        provider = self._get_provider(config.provider)

        try:
            # Add system prompt if needed
            if config.system_prompt and not any(m.role == "system" for m in messages):
                system_msg = LLMMessage(
                    role="system",
                    content=config.system_prompt,
                    metadata={"type": "system_prompt"}
                )
                messages = [system_msg] + messages

            # Update conversation cache
            if conversation_id:
                self._update_conversation_cache(conversation_id, messages)

            # Stream response
            async for chunk in provider.generate_stream(messages, config):
                yield chunk

        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            # Fallback to non-streaming
            response = await self.generate(messages, config, conversation_id)
            yield response.content

    async def _fallback_generate(
        self,
        messages: List[LLMMessage],
        original_config: LLMConfig,
        conversation_id: Optional[str] = None
    ) -> LLMResponse:
        """Try fallback providers if primary provider fails"""
        fallback_providers = [
            ModelProvider.OPENAI,
            ModelProvider.ANTHROPIC,
            ModelProvider.LOCAL
        ]

        # Remove the failed provider from fallback list
        if original_config.provider in fallback_providers:
            fallback_providers.remove(original_config.provider)

        for fallback_provider in fallback_providers:
            if fallback_provider in self.providers:
                try:
                    logger.info(f"Trying fallback provider: {fallback_provider}")
                    fallback_config = LLMConfig(
                        provider=fallback_provider,
                        model=self.providers[fallback_provider].get_default_model(),
                        temperature=original_config.temperature,
                        max_tokens=original_config.max_tokens
                    )

                    return await self.providers[fallback_provider].generate(
                        messages=messages,
                        config=fallback_config
                    )
                except Exception as e:
                    logger.warning(f"Fallback provider {fallback_provider} also failed: {e}")
                    continue

        raise RuntimeError("All LLM providers failed")

    def _get_provider(self, provider: ModelProvider):
        """Get provider instance"""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")
        return self.providers[provider]

    def _update_conversation_cache(self, conversation_id: str, messages: List[LLMMessage]):
        """Update conversation cache with new messages"""
        if conversation_id not in self._conversation_cache:
            self._conversation_cache[conversation_id] = []

        # Add new messages (avoiding duplicates)
        for message in messages:
            if message not in self._conversation_cache[conversation_id]:
                self._conversation_cache[conversation_id].append(message)

        # Keep only last 20 messages to manage memory
        if len(self._conversation_cache[conversation_id]) > 20:
            self._conversation_cache[conversation_id] = self._conversation_cache[conversation_id][-20:]

    def _add_to_conversation_cache(self, conversation_id: str, response: LLMResponse):
        """Add assistant response to conversation cache"""
        if conversation_id in self._conversation_cache:
            assistant_msg = LLMMessage(
                role="assistant",
                content=response.content,
                metadata=response.metadata,
                tool_calls=response.tool_calls
            )
            self._conversation_cache[conversation_id].append(assistant_msg)

    def get_conversation_history(self, conversation_id: str) -> List[LLMMessage]:
        """Get cached conversation history"""
        return self._conversation_cache.get(conversation_id, [])

    def clear_conversation_cache(self, conversation_id: str):
        """Clear conversation cache for specific conversation"""
        if conversation_id in self._conversation_cache:
            del self._conversation_cache[conversation_id]

    def switch_provider(self, provider: ModelProvider):
        """Switch default provider"""
        if provider in self.providers:
            self.current_provider = provider
            self.default_config.provider = provider
            logger.info(f"Switched to provider: {provider}")
        else:
            raise ValueError(f"Provider {provider} not available")

    def get_available_providers(self) -> List[ModelProvider]:
        """Get list of available providers"""
        return list(self.providers.keys())

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about current providers"""
        return {
            "current_provider": self.current_provider.value,
            "available_providers": [p.value for p in self.providers.keys()],
            "default_config": {
                "provider": self.default_config.provider.value,
                "model": self.default_config.model,
                "temperature": self.default_config.temperature,
                "max_tokens": self.default_config.max_tokens
            }
        }

# Global LLM service instance
llm_service = LLMService()