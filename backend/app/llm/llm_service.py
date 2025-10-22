from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from abc import ABC, abstractmethod
import logging
import asyncio
import json

from .providers import OpenAIProvider, AnthropicProvider, LocalProvider, MistralProvider
from .types import ModelProvider, LLMMessage, LLMResponse, LLMConfig
from app.core.config import settings

logger = logging.getLogger(__name__)

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
        elif hasattr(settings, 'MISTRAL_API_KEY') and settings.MISTRAL_API_KEY:
            return LLMConfig(
                provider=ModelProvider.MISTRAL,
                model="mistral-large-latest",
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
            # Initialize Mistral provider
            if hasattr(settings, 'MISTRAL_API_KEY') and settings.MISTRAL_API_KEY:
                providers[ModelProvider.MISTRAL] = MistralProvider(
                    api_key=settings.MISTRAL_API_KEY
                )
                logger.info("Mistral AI provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Mistral AI provider: {e}")

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

    async def chat(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
        conversation_id: Optional[str] = None,
        use_rag: bool = True,
        citations: bool = True
    ) -> LLMResponse:
        """
        Chat interface with RAG augmentation and citations

        Args:
            messages: List of conversation messages
            config: Optional configuration override
            conversation_id: Optional conversation ID for context
            use_rag: Whether to use RAG for enhanced responses
            citations: Whether to include citations in response

        Returns:
            LLMResponse with enhanced chat response
        """
        config = config or self.default_config

        # Enhance with RAG if requested
        if use_rag and len(messages) > 0:
            enhanced_messages = await self._enhance_with_rag(messages, citations)
        else:
            enhanced_messages = messages

        # Use generate method with chat-specific configuration
        chat_config = LLMConfig(
            provider=config.provider,
            model=config.model,
            temperature=min(config.temperature, 0.8),  # Slightly more conservative for chat
            max_tokens=config.max_tokens,
            system_prompt=self._get_chat_system_prompt(),
            stream=False
        )

        return await self.generate(enhanced_messages, chat_config, conversation_id)

    async def embed(
        self,
        texts: List[str],
        config: Optional[LLMConfig] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for texts

        Args:
            texts: List of texts to embed
            config: Optional configuration override

        Returns:
            List of embedding vectors
        """
        config = config or self.default_config
        provider = self._get_provider(config.provider)

        try:
            # Use provider's embedding capability
            if hasattr(provider, 'embed'):
                return await provider.embed(texts, config)
            else:
                # Fallback: use a simple text-based embedding (placeholder)
                logger.warning(f"Provider {config.provider} doesn't support embeddings, using fallback")
                return self._fallback_embeddings(texts)

        except Exception as e:
            logger.error(f"Error generating embeddings with {config.provider}: {e}")
            # Try fallback provider
            return await self._fallback_embed(texts, config)

    async def moderate(
        self,
        content: str,
        config: Optional[LLMConfig] = None
    ) -> Dict[str, Any]:
        """
        Moderate content for safety and appropriateness

        Args:
            content: Content to moderate
            config: Optional configuration override

        Returns:
            Moderation results with safety flags and suggestions
        """
        config = config or self.default_config
        provider = self._get_provider(config.provider)

        try:
            # Use provider's moderation capability
            if hasattr(provider, 'moderate'):
                return await provider.moderate(content, config)
            else:
                # Fallback: basic keyword-based moderation
                return self._fallback_moderation(content)

        except Exception as e:
            logger.error(f"Error moderating content with {config.provider}: {e}")
            # Return safe defaults on error
            return {
                "safe": True,
                "categories": {},
                "flagged": False,
                "error": str(e)
            }

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
            ModelProvider.MISTRAL,
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

    def _get_chat_system_prompt(self) -> str:
        """Get specialized system prompt for chat interactions"""
        return """You are StudySense, a compassionate AI assistant specializing in student mental health and academic support. Your role is to provide:

1. **Empathetic Support** - Listen actively and validate student experiences
2. **Evidence-Based Guidance** - Draw from verified knowledge sources when providing advice
3. **Practical Strategies** - Offer concrete, actionable steps students can take
4. **Resource Connection** - Direct students to appropriate campus and community resources
5. **Crisis Recognition** - Immediately identify and respond to mental health emergencies

**Communication Style:**
- Warm, supportive, and non-judgmental
- Use simple, clear language
- Ask clarifying questions when needed
- Provide hope and encouragement
- Maintain appropriate professional boundaries

**Response Structure:**
- Acknowledge the student's feelings/experience
- Provide relevant information or strategies
- Include specific citations when referencing sources
- Suggest next steps or resources
- End with an open invitation for follow-up

**Safety First:**
If you detect any signs of crisis, self-harm, or severe distress, immediately:
- Express concern for their safety
- Provide emergency contact information
- Recommend immediate professional help
- Do not attempt to provide crisis counseling yourself

Remember: You are a supportive tool, not a replacement for professional mental health care."""

    async def _enhance_with_rag(self, messages: List[LLMMessage], citations: bool = True) -> List[LLMMessage]:
        """Enhance messages with RAG context and citations"""
        try:
            from ..rag.retrieval import retrieval_pipeline

            # Extract the latest user message for RAG query
            user_messages = [msg for msg in messages if msg.role == "user"]
            if not user_messages:
                return messages

            latest_message = user_messages[-1].content

            # Perform RAG search
            rag_results = await retrieval_pipeline.retrieve(
                query=latest_message,
                k=3,
                collections=["kb_global", "campus_resources"]
            )

            if rag_results.results:
                # Build context from RAG results
                context_parts = []
                citations_list = []

                for i, result in enumerate(rag_results.results[:3], 1):
                    context_parts.append(f"Source {i}: {result.text}")
                    citations_list.append({
                        "number": i,
                        "source": result.metadata.get("document_id", "Knowledge Base"),
                        "relevance": result.score
                    })

                context = "\n\n".join(context_parts)

                # Create enhanced system message
                rag_prompt = f"""Use the following relevant information to inform your response:

{context}

{'Include citations in your response using [1], [2], etc. format.' if citations else 'Use this information to provide informed guidance.'}

Remember to cite sources when providing specific advice or information."""

                # Add RAG context to system message
                enhanced_messages = []
                system_found = False

                for msg in messages:
                    if msg.role == "system":
                        # Enhance existing system message
                        enhanced_content = f"{msg.content}\n\n{rag_prompt}"
                        enhanced_messages.append(LLMMessage(
                            role="system",
                            content=enhanced_content,
                            metadata={**msg.metadata, "rag_enhanced": True, "citations": citations_list}
                        ))
                        system_found = True
                    else:
                        enhanced_messages.append(msg)

                # Add system message if none exists
                if not system_found:
                    enhanced_messages.insert(0, LLMMessage(
                        role="system",
                        content=f"{self._get_chat_system_prompt()}\n\n{rag_prompt}",
                        metadata={"rag_enhanced": True, "citations": citations_list}
                    ))

                return enhanced_messages
            else:
                return messages

        except Exception as e:
            logger.warning(f"RAG enhancement failed: {e}")
            return messages

    def _fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Simple fallback embedding using basic text features"""
        import hashlib

        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding (not for production)
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            # Convert to float list (normalized to 0-1 range)
            embedding = [b / 255.0 for b in hash_bytes]
            # Pad/truncate to 384 dimensions (common embedding size)
            if len(embedding) < 384:
                embedding.extend([0.0] * (384 - len(embedding)))
            else:
                embedding = embedding[:384]
            embeddings.append(embedding)

        return embeddings

    async def _fallback_embed(self, texts: List[str], config: LLMConfig) -> List[List[float]]:
        """Try fallback providers for embeddings"""
        fallback_providers = [
            ModelProvider.OPENAI,
            ModelProvider.ANTHROPIC,
            ModelProvider.MISTRAL,
            ModelProvider.LOCAL
        ]

        # Remove the failed provider
        if config.provider in fallback_providers:
            fallback_providers.remove(config.provider)

        for fallback_provider in fallback_providers:
            if fallback_provider in self.providers:
                provider = self.providers[fallback_provider]
                if hasattr(provider, 'embed'):
                    try:
                        logger.info(f"Trying embedding fallback provider: {fallback_provider}")
                        return await provider.embed(texts, config)
                    except Exception as e:
                        logger.warning(f"Fallback embedding provider {fallback_provider} also failed: {e}")
                        continue

        # Ultimate fallback
        logger.warning("All embedding providers failed, using hash-based fallback")
        return self._fallback_embeddings(texts)

    def _fallback_moderation(self, content: str) -> Dict[str, Any]:
        """Basic keyword-based content moderation"""
        # Define sensitive keywords and patterns
        crisis_keywords = [
            "suicide", "kill myself", "end it all", "self-harm", "cutting",
            "overdose", "hurt myself", "not worth living", "better off dead"
        ]

        sensitive_keywords = [
            "depression", "anxiety", "panic", "trauma", "abuse", "assault",
            "eating disorder", "addiction", "substance abuse"
        ]

        content_lower = content.lower()

        # Check for crisis indicators
        crisis_flags = [kw for kw in crisis_keywords if kw in content_lower]
        sensitive_flags = [kw for kw in sensitive_keywords if kw in content_lower]

        # Determine safety level
        if crisis_flags:
            safety_level = "crisis"
            safe = False
        elif sensitive_flags:
            safety_level = "sensitive"
            safe = True  # Still safe but needs careful handling
        else:
            safety_level = "safe"
            safe = True

        return {
            "safe": safe,
            "categories": {
                "crisis_indicators": crisis_flags,
                "sensitive_topics": sensitive_flags
            },
            "flagged": not safe,
            "safety_level": safety_level,
            "moderation_type": "keyword_based",
            "recommendations": [
                "Refer to mental health professional" if crisis_flags else None,
                "Handle with care and empathy" if sensitive_flags else None
            ]
        }

# Global LLM service instance
llm_service = LLMService()
