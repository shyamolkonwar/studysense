from typing import List, Dict, Any, Optional, AsyncGenerator
import anthropic
import logging
import json
import asyncio

from ..types import LLMMessage, LLMResponse, LLMConfig

logger = logging.getLogger(__name__)

class AnthropicProvider:
    """Anthropic Claude API provider implementation"""

    def __init__(self, api_key: str):
        """
        Initialize Anthropic provider

        Args:
            api_key: Anthropic API key
        """
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.default_models = [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229"
        ]
        logger.info("Anthropic provider initialized")

    def get_default_model(self) -> str:
        """Get default model for this provider"""
        return "claude-3-haiku-20240307"

    async def generate(
        self,
        messages: List[LLMMessage],
        config: LLMConfig
    ) -> LLMResponse:
        """
        Generate response using Anthropic API

        Args:
            messages: List of conversation messages
            config: Configuration for generation

        Returns:
            LLMResponse with generated content
        """
        try:
            # Convert messages to Anthropic format
            anthropic_messages = self._convert_messages(messages)
            system_prompt = self._extract_system_prompt(messages)

            # Prepare request parameters
            request_params = {
                "model": config.model,
                "messages": anthropic_messages,
                "max_tokens": config.max_tokens or 2000,
                "temperature": config.temperature
            }

            # Add system prompt if present
            if system_prompt:
                request_params["system"] = system_prompt

            # Add tools if provided
            if config.tools:
                request_params["tools"] = self._convert_tools(config.tools)
                if config.tool_choice:
                    request_params["tool_choice"] = config.tool_choice

            # Make API call
            response = await self.client.messages.create(**request_params)

            # Convert response
            return self._convert_response(response, config)

        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
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
            anthropic_messages = self._convert_messages(messages)
            system_prompt = self._extract_system_prompt(messages)

            # Prepare request parameters
            request_params = {
                "model": config.model,
                "messages": anthropic_messages,
                "max_tokens": config.max_tokens or 2000,
                "temperature": config.temperature,
                "stream": True
            }

            # Add system prompt if present
            if system_prompt:
                request_params["system"] = system_prompt

            # Add tools if provided
            if config.tools:
                request_params["tools"] = self._convert_tools(config.tools)

            # Make streaming API call
            async with self.client.messages.stream(**request_params) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Anthropic streaming error: {str(e)}")
            raise

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert LLMMessage format to Anthropic format"""
        anthropic_messages = []

        for message in messages:
            if message.role == "system":
                # System messages are handled separately in Anthropic
                continue

            anthropic_msg = {
                "role": message.role,
                "content": message.content
            }

            # Add tool results if present
            if message.tool_call_id:
                anthropic_msg = {
                    "role": "tool",
                    "content": message.content,
                    "tool_use_id": message.tool_call_id
                }

            # Add tool calls if present
            if message.tool_calls:
                anthropic_msg["content"] = [
                    {
                        "type": "text",
                        "text": message.content
                    }
                ]

                # Add tool use blocks
                for tool_call in message.tool_calls:
                    anthropic_msg["content"].append({
                        "type": "tool_use",
                        "id": tool_call.get("id"),
                        "name": tool_call["function"]["name"],
                        "input": json.loads(tool_call["function"]["arguments"])
                    })

            anthropic_messages.append(anthropic_msg)

        return anthropic_messages

    def _extract_system_prompt(self, messages: List[LLMMessage]) -> Optional[str]:
        """Extract system prompt from messages"""
        for message in messages:
            if message.role == "system":
                return message.content
        return None

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools to Anthropic format"""
        anthropic_tools = []

        for tool in tools:
            anthropic_tool = {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"]
            }
            anthropic_tools.append(anthropic_tool)

        return anthropic_tools

    def _convert_response(self, response: Any, config: LLMConfig) -> LLMResponse:
        """Convert Anthropic response to LLMResponse format"""
        try:
            # Extract content
            content = ""
            tool_calls = None

            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "id": block.id,
                        "type": "tool_use",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input)
                        }
                    })

            # Extract usage information
            usage = None
            if hasattr(response, 'usage'):
                usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }

            # Create response
            return LLMResponse(
                content=content,
                role="assistant",
                tool_calls=tool_calls,
                usage=usage,
                model=config.model,
                provider="anthropic",
                metadata={
                    "stop_reason": response.stop_reason,
                    "stop_sequence": response.stop_sequence,
                    "id": response.id
                }
            )

        except Exception as e:
            logger.error(f"Error converting Anthropic response: {e}")
            # Fallback response
            return LLMResponse(
                content="I apologize, but I encountered an error processing my response.",
                role="assistant",
                model=config.model,
                provider="anthropic",
                metadata={"error": str(e)}
            )

    async def list_models(self) -> List[str]:
        """List available models"""
        return self.default_models

    def validate_config(self, config: LLMConfig) -> bool:
        """Validate configuration for this provider"""
        if config.model not in self.default_models:
            logger.warning(f"Model {config.model} may not be supported by Anthropic")

        # Anthropic has different parameter ranges
        if config.temperature is not None and (config.temperature < 0 or config.temperature > 1):
            logger.warning("Anthropic temperature should be between 0 and 1")

        return True
