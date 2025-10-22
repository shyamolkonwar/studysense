from typing import List, Dict, Any, Optional, AsyncGenerator
from abc import ABC, abstractmethod
import openai
import logging
import json
import asyncio

from ..types import LLMMessage, LLMResponse, LLMConfig

logger = logging.getLogger(__name__)

class OpenAIProvider:
    """OpenAI API provider implementation"""

    def __init__(self, api_key: str):
        """
        Initialize OpenAI provider

        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        self.default_models = [
            "gpt-4o-mini",
            "gpt-4",
            "gpt-3.5-turbo"
        ]
        logger.info("OpenAI provider initialized")

    def get_default_model(self) -> str:
        """Get default model for this provider"""
        return "gpt-4o-mini"

    async def generate(
        self,
        messages: List[LLMMessage],
        config: LLMConfig
    ) -> LLMResponse:
        """
        Generate response using OpenAI API

        Args:
            messages: List of conversation messages
            config: Configuration for generation

        Returns:
            LLMResponse with generated content
        """
        try:
            # Convert messages to OpenAI format
            openai_messages = self._convert_messages(messages)

            # Prepare request parameters
            request_params = {
                "model": config.model,
                "messages": openai_messages,
                "temperature": config.temperature,
                "stream": False
            }

            # Add optional parameters
            if config.max_tokens:
                request_params["max_tokens"] = config.max_tokens
            if config.top_p:
                request_params["top_p"] = config.top_p
            if config.frequency_penalty:
                request_params["frequency_penalty"] = config.frequency_penalty
            if config.presence_penalty:
                request_params["presence_penalty"] = config.presence_penalty

            # Add tools if provided
            if config.tools:
                request_params["tools"] = config.tools
                if config.tool_choice:
                    request_params["tool_choice"] = config.tool_choice

            # Make API call
            response = await openai.ChatCompletion.acreate(**request_params)

            # Convert response
            return self._convert_response(response, config)

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        config: LLMConfig
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response

        Args:
            messages: List of conversation messages
            config: Configuration for generation

        Yields:
            Content chunks as they are generated
        """
        try:
            # Convert messages
            openai_messages = self._convert_messages(messages)

            # Prepare request parameters
            request_params = {
                "model": config.model,
                "messages": openai_messages,
                "temperature": config.temperature,
                "stream": True
            }

            # Add optional parameters
            if config.max_tokens:
                request_params["max_tokens"] = config.max_tokens
            if config.tools:
                request_params["tools"] = config.tools

            # Make streaming API call
            response = await openai.ChatCompletion.acreate(**request_params)

            async for chunk in response:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming error: {str(e)}")
            raise

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert LLMMessage format to OpenAI format"""
        openai_messages = []

        for message in messages:
            openai_msg = {
                "role": message.role,
                "content": message.content
            }

            # Add metadata as a separate tool result if it exists
            if message.metadata and message.metadata.get("type") == "tool_result":
                openai_msg["content"] = json.dumps(message.metadata, indent=2)

            # Add tool calls if present
            if message.tool_calls:
                openai_msg["tool_calls"] = [
                    self._convert_tool_call(tool_call)
                    for tool_call in message.tool_calls
                ]

            # Add tool call ID for tool responses
            if message.tool_call_id:
                openai_msg["tool_call_id"] = message.tool_call_id

            openai_messages.append(openai_msg)

        return openai_messages

    def _convert_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Convert tool call format to OpenAI format"""
        return {
            "id": tool_call.get("id", f"call_{hash(str(tool_call))}"),
            "type": "function",
            "function": {
                "name": tool_call["function"]["name"],
                "arguments": tool_call["function"]["arguments"]
            }
        }

    def _convert_response(self, response: Any, config: LLMConfig) -> LLMResponse:
        """Convert OpenAI response to LLMResponse format"""
        try:
            choice = response.choices[0]
            message = choice.message

            # Extract content
            content = message.content if hasattr(message, 'content') and message.content else ""

            # Extract tool calls
            tool_calls = None
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]

            # Extract usage information
            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            # Create response
            return LLMResponse(
                content=content,
                role="assistant",
                tool_calls=tool_calls,
                usage=usage,
                model=config.model,
                provider="openai",
                metadata={
                    "finish_reason": choice.finish_reason,
                    "created": response.created
                }
            )

        except Exception as e:
            logger.error(f"Error converting OpenAI response: {e}")
            # Fallback response
            return LLMResponse(
                content="I apologize, but I encountered an error processing my response.",
                role="assistant",
                model=config.model,
                provider="openai",
                metadata={"error": str(e)}
            )

    async def list_models(self) -> List[str]:
        """List available models"""
        try:
            models = await openai.Model.alist()
            return [model.id for model in models.data if model.id.startswith("gpt-")]
        except Exception as e:
            logger.error(f"Error listing OpenAI models: {e}")
            return self.default_models

    async def embed(
        self,
        texts: List[str],
        config: LLMConfig
    ) -> List[List[float]]:
        """
        Generate embeddings using OpenAI API

        Args:
            texts: List of texts to embed
            config: Configuration for embedding

        Returns:
            List of embedding vectors
        """
        try:
            # Use text-embedding-3-small for efficiency
            embedding_model = "text-embedding-3-small"

            # Prepare batches (OpenAI has limits)
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Make API call
                response = await openai.Embedding.acreate(
                    input=batch_texts,
                    model=embedding_model
                )

                # Extract embeddings
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except Exception as e:
            logger.error(f"OpenAI embedding error: {str(e)}")
            raise

    async def moderate(
        self,
        content: str,
        config: LLMConfig
    ) -> Dict[str, Any]:
        """
        Moderate content using OpenAI Moderation API

        Args:
            content: Content to moderate
            config: Configuration (unused for moderation)

        Returns:
            Moderation results
        """
        try:
            # Use OpenAI moderation API
            response = await openai.Moderation.acreate(
                input=content
            )

            result = response.results[0]

            # Map to our format
            moderation_result = {
                "safe": not result.flagged,
                "categories": result.categories.__dict__,
                "category_scores": result.category_scores.__dict__,
                "flagged": result.flagged,
                "moderation_type": "openai_api"
            }

            return moderation_result

        except Exception as e:
            logger.error(f"OpenAI moderation error: {str(e)}")
            raise

    def validate_config(self, config: LLMConfig) -> bool:
        """Validate configuration for this provider"""
        if config.model not in self.default_models:
            logger.warning(f"Model {config.model} may not be supported by OpenAI")

        return True
