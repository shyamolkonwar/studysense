from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
import json
import asyncio
import subprocess
import sys
from pathlib import Path

from ..llm_service import LLMMessage, LLMResponse, LLMConfig

logger = logging.getLogger(__name__)

class LocalProvider:
    """Local model provider using Ollama or similar local inference"""

    def __init__(self, model_endpoint: str = "http://localhost:11434"):
        """
        Initialize local provider

        Args:
            model_endpoint: Endpoint for local model service
        """
        self.model_endpoint = model_endpoint
        self.default_models = [
            "llama2-7b-chat",
            "mistral-7b-instruct",
            "codellama-7b-instruct"
        ]
        logger.info("Local provider initialized")

    def get_default_model(self) -> str:
        """Get default model for this provider"""
        return "llama2-7b-chat"

    async def generate(
        self,
        messages: List[LLMMessage],
        config: LLMConfig
    ) -> LLMResponse:
        """
        Generate response using local model

        Args:
            messages: List of conversation messages
            config: Configuration for generation

        Returns:
            LLMResponse with generated content
        """
        try:
            # Convert messages to format expected by local model
            local_messages = self._convert_messages(messages)
            system_prompt = self._extract_system_prompt(messages)

            # Prepare request payload
            payload = {
                "model": config.model,
                "messages": local_messages,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens or 2000
                }
            }

            # Add system prompt if present
            if system_prompt:
                payload["system"] = system_prompt

            # Make API call to local model
            response = await self._call_local_api(payload)

            # Convert response
            return self._convert_response(response, config)

        except Exception as e:
            logger.error(f"Local model error: {str(e)}")
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
            local_messages = self._convert_messages(messages)
            system_prompt = self._extract_system_prompt(messages)

            # Prepare request payload
            payload = {
                "model": config.model,
                "messages": local_messages,
                "stream": True,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens or 2000
                }
            }

            # Add system prompt if present
            if system_prompt:
                payload["system"] = system_prompt

            # Make streaming API call
            async for chunk in self._call_local_api_stream(payload):
                yield chunk

        except Exception as e:
            logger.error(f"Local model streaming error: {str(e)}")
            raise

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert LLMMessage format to local model format"""
        local_messages = []

        for message in messages:
            if message.role == "system":
                # System messages are handled separately
                continue

            local_msg = {
                "role": message.role,
                "content": message.content
            }

            local_messages.append(local_msg)

        return local_messages

    def _extract_system_prompt(self, messages: List[LLMMessage]) -> Optional[str]:
        """Extract system prompt from messages"""
        for message in messages:
            if message.role == "system":
                return message.content
        return None

    async def _call_local_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call local model API"""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.model_endpoint}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f"API returned status {response.status}: {await response.text()}")

        except ImportError:
            # Fallback to subprocess if aiohttp not available
            return await self._call_local_subprocess(payload)

    async def _call_local_api_stream(self, payload: Dict[str, Any]):
        """Call local model API with streaming"""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.model_endpoint}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line.decode('utf-8').strip())
                                    if 'response' in data:
                                        yield data['response']
                                except json.JSONDecodeError:
                                    continue
                    else:
                        raise Exception(f"API returned status {response.status}")

        except ImportError:
            # Fallback for streaming without aiohttp
            async for chunk in self._call_local_subprocess_stream(payload):
                yield chunk

    async def _call_local_subprocess(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback using subprocess to call local model"""
        try:
            cmd = [
                "curl",
                "-X", "POST",
                f"{self.model_endpoint}/api/generate",
                "-H", "Content-Type: application/json",
                "-d", json.dumps(payload)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return json.loads(stdout.decode('utf-8'))
            else:
                raise Exception(f"Process failed with code {process.returncode}: {stderr.decode('utf-8')}")

        except Exception as e:
            logger.error(f"Subprocess call failed: {e}")
            raise

    async def _call_local_subprocess_stream(self, payload: Dict[str, Any]):
        """Fallback streaming using subprocess"""
        # This is a simplified implementation - in production, you'd want proper async handling
        yield "Local model response (streaming fallback not fully implemented in subprocess mode)"

    def _convert_response(self, response: Dict[str, Any], config: LLMConfig) -> LLMResponse:
        """Convert local model response to LLMResponse format"""
        try:
            content = response.get("response", "")

            # Create usage information (estimated)
            usage = {
                "prompt_tokens": len(str(response.get("prompt", "")) // 4),
                "completion_tokens": len(content // 4),
                "total_tokens": len(str(response.get("prompt", "")) + content) // 4
            }

            # Create response
            return LLMResponse(
                content=content,
                role="assistant",
                usage=usage,
                model=config.model,
                provider="local",
                metadata={
                    "model": response.get("model"),
                    "created_at": response.get("created_at"),
                    "done": response.get("done", True)
                }
            )

        except Exception as e:
            logger.error(f"Error converting local model response: {e}")
            # Fallback response
            return LLMResponse(
                content="I apologize, but I encountered an error processing my response.",
                role="assistant",
                model=config.model,
                provider="local",
                metadata={"error": str(e)}
            )

    async def list_models(self) -> List[str]:
        """List available local models"""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.model_endpoint}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model["name"] for model in data.get("models", [])]
                    else:
                        return self.default_models

        except Exception:
            # Fallback to default models
            return self.default_models

    async def check_connection(self) -> bool:
        """Check if local model service is accessible"""
        try:
            await self.list_models()
            return True
        except:
            return False

    def validate_config(self, config: LLMConfig) -> bool:
        """Validate configuration for this provider"""
        # Local models are more flexible with parameters
        return True