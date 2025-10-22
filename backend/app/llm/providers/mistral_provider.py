from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx
import logging
import json
import asyncio

from ..types import LLMMessage, LLMResponse, LLMConfig

logger = logging.getLogger(__name__)

class MistralProvider:
    """Mistral AI API provider implementation"""

    def __init__(self, api_key: str):
        """
        Initialize Mistral provider

        Args:
            api_key: Mistral AI API key
        """
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1"
        self.default_models = [
            "mistral-large-latest",
            "mistral-medium",
            "mistral-small",
            "mistral-7b-instruct",
            "mixtral-8x7b-instruct"
        ]

        logger.info("Mistral AI provider initialized")

    def get_default_model(self) -> str:
        """Get default model for this provider"""
        return "mistral-large-latest"

    async def generate(
        self,
        messages: List[LLMMessage],
        config: LLMConfig
    ) -> LLMResponse:
        """
        Generate response using Mistral AI API

        Args:
            messages: List of conversation messages
            config: Configuration for generation

        Returns:
            LLMResponse with generated content
        """
        try:
            # Convert messages to Mistral format
            mistral_messages = self._convert_messages(messages)

            # Prepare request parameters
            request_params = {
                "model": config.model,
                "messages": mistral_messages,
                "temperature": config.temperature,
                "stream": False
            }

            # Add optional parameters
            if config.max_tokens:
                request_params["max_tokens"] = config.max_tokens
            if config.top_p:
                request_params["top_p"] = config.top_p

            # Make API call using httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=request_params,
                    timeout=60.0
                )

                if response.status_code != 200:
                    raise Exception(f"Mistral API error: {response.status_code} - {response.text}")

                response_data = response.json()

            # Convert response
            return self._convert_response(response_data, config)

        except Exception as e:
            logger.error(f"Mistral AI API error: {str(e)}")
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
            mistral_messages = self._convert_messages(messages)

            # Prepare request parameters
            request_params = {
                "model": config.model,
                "messages": mistral_messages,
                "temperature": config.temperature,
                "stream": True
            }

            # Add optional parameters
            if config.max_tokens:
                request_params["max_tokens"] = config.max_tokens

            # Make streaming API call using httpx
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=request_params,
                    timeout=60.0
                ) as response:
                    if response.status_code != 200:
                        raise Exception(f"Mistral API error: {response.status_code}")

                    async for line in response.aiter_lines():
                        if line.strip():
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])  # Remove "data: " prefix
                                    if data.get("choices") and data["choices"][0].get("delta", {}).get("content"):
                                        yield data["choices"][0]["delta"]["content"]
                                except json.JSONDecodeError:
                                    continue

        except Exception as e:
            logger.error(f"Mistral AI streaming error: {str(e)}")
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

    def _convert_response(self, response_data: Dict[str, Any], config: LLMConfig) -> LLMResponse:
        """Convert Mistral AI response to LLMResponse format"""
        try:
            choice = response_data["choices"][0]
            message = choice["message"]

            # Extract content
            content = message.get("content", "")

            # Extract tool calls
            tool_calls = None
            if "tool_calls" in message and message["tool_calls"]:
                tool_calls = [
                    {
                        "id": tc["id"],
                        "type": tc["type"],
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    }
                    for tc in message["tool_calls"]
                ]

            # Extract usage information
            usage = None
            if "usage" in response_data:
                usage_data = response_data["usage"]
                usage = {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0)
                }

            # Create response
            return LLMResponse(
                content=content,
                role="assistant",
                tool_calls=tool_calls,
                usage=usage,
                model=config.model,
                provider="mistral",
                metadata={
                    "finish_reason": choice.get("finish_reason"),
                    "created": response_data.get("created")
                }
            )

        except Exception as e:
            logger.error(f"Error converting Mistral AI response: {e}")
            # Fallback response
            return LLMResponse(
                content="I apologize, but I encountered an error processing my response.",
                role="assistant",
                model=config.model,
                provider="mistral",
                metadata={"error": str(e)}
            )

    async def list_models(self) -> List[str]:
        """List available models"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    data = response.json()
                    return [model["id"] for model in data.get("data", []) if any(mistral_model in model["id"] for mistral_model in ["mistral", "mixtral"])]
                else:
                    logger.warning(f"Failed to fetch models: {response.status_code}")
                    return self.default_models
        except Exception as e:
            logger.error(f"Error listing Mistral AI models: {e}")
            return self.default_models

    def validate_config(self, config: LLMConfig) -> bool:
        """Validate configuration for this provider"""
        if config.model not in self.default_models:
            logger.warning(f"Model {config.model} may not be supported by Mistral AI")

        return True
