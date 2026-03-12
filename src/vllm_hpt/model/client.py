"""OpenAI API client wrapper for vLLM-served models.

This module provides an LLMClient class that wraps the OpenAI Python client
to call vLLM-served models with configurable sampling parameters, retry logic,
and response caching.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional

import httpx
import openai
from openai import AsyncOpenAI

from vllm_hpt.utils.cache import ResponseCache
from vllm_hpt.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Async OpenAI client wrapper with retry logic and caching.
    
    Wraps AsyncOpenAI to provide:
    - Configurable retry logic with exponential backoff
    - Response caching via ResponseCache
    - Structured logging of API calls
    - Support for vLLM-specific sampling parameters
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        cache: Optional[ResponseCache] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        enable_thinking: Optional[bool] = None,
    ):
        """Initialize the LLM client.
        
        Args:
            base_url: Base URL for the OpenAI-compatible API (e.g., http://localhost:8000/v1)
            api_key: API key for authentication (use "EMPTY" for vLLM without auth)
            model: Model name to use for completions
            cache: Optional ResponseCache instance for caching responses
            timeout: Request timeout in seconds (default: 60.0)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.cache = cache
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        
        http_client = httpx.AsyncClient(
            timeout=timeout,
            event_hooks={"request": [], "response": []},
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            http_client=http_client,
        )
        
        logger.info(
            "llm_client_initialized",
            model=model,
            cache_enabled=cache is not None if cache else False,
            enable_thinking=enable_thinking,
        )

    def _prepare_sampling_params(self, sampling_params: Optional[Dict] = None) -> tuple[Dict, Dict]:
        """Prepare sampling parameters for OpenAI API.
        
        Separates standard OpenAI params from vLLM-specific params that need
        to be passed via extra_body.
        
        Args:
            sampling_params: Dictionary of sampling parameters
            
        Returns:
            Tuple of (standard_params, extra_body_params)
        """
        if sampling_params is None:
            return {}, {}
        
        # Standard OpenAI parameters
        standard_params = {}
        # vLLM-specific parameters that need extra_body
        vllm_params = {}
        
        # Map of parameter names to their destination
        standard_keys = {
            "temperature", "top_p", "max_tokens", "frequency_penalty",
            "presence_penalty", "stop", "n", "stream", "logprobs",
            "top_logprobs", "logit_bias", "seed"
        }
        
        for key, value in sampling_params.items():
            if key in standard_keys:
                standard_params[key] = value
            else:
                # vLLM-specific params like top_k, repetition_penalty, etc.
                vllm_params[key] = value
        
        return standard_params, vllm_params

    def _extract_chat_content(self, response) -> str:
        message = response.choices[0].message
        content = message.content
        if content is not None:
            return content

        reasoning_content = getattr(message, "reasoning_content", None)
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        usage = getattr(response, "usage", None)
        if usage is None:
            usage_dict = None
        elif hasattr(usage, "model_dump"):
            usage_dict = usage.model_dump()
        else:
            usage_dict = str(usage)

        logger.warning(
            "chat_content_missing",
            model=self.model,
            finish_reason=finish_reason,
            has_reasoning_content=reasoning_content is not None,
            reasoning_length=len(reasoning_content) if isinstance(reasoning_content, str) else 0,
            usage=usage_dict,
        )

        if isinstance(reasoning_content, str) and reasoning_content:
            return reasoning_content

        return ""

    async def _call_with_retry(self, api_call_func, *args, **kwargs) -> str:
        """Execute API call with exponential backoff retry logic.
        
        Args:
            api_call_func: Async function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Response text from the API
            
        Raises:
            openai.APIError: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = await api_call_func(*args, **kwargs)
                return response
            except (openai.APIError, openai.APIConnectionError) as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    logger.warning(
                        "api_call_retry",
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        wait_time=wait_time,
                        error=str(e),
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        "api_call_failed",
                        attempts=self.max_retries,
                        error=str(e),
                    )
        
        # All retries exhausted
        raise last_exception or RuntimeError("API call failed without captured exception")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Optional[Dict] = None,
    ) -> str:
        """Send a chat completion request.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            sampling_params: Optional dictionary of sampling parameters
            
        Returns:
            Response text from the model
        """
        # Create cache key from messages
        cache_key = json.dumps(messages, sort_keys=True)
        cache_params = sampling_params or {}
        
        # Check cache first
        if self.cache:
            cached_response = self.cache.get(cache_key, cache_params)
            if cached_response is not None:
                logger.debug(
                    "chat_cache_hit",
                    model=self.model,
                    message_count=len(messages),
                    response_length=len(cached_response),
                )
                return cached_response
        
        # Prepare parameters
        standard_params, vllm_params = self._prepare_sampling_params(sampling_params)
        
        # Build request kwargs
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            **standard_params,
        }
        
        if vllm_params:
            request_kwargs["extra_body"] = vllm_params

        if self.enable_thinking is not None:
            request_kwargs.setdefault("extra_body", {})["enable_thinking"] = self.enable_thinking
        
        # Make API call with retry logic
        async def _api_call():
            response = await self.client.chat.completions.create(**request_kwargs)
            return self._extract_chat_content(response)
        
        response_text = await self._call_with_retry(_api_call)
        
        # Cache the response
        if self.cache:
            self.cache.set(cache_key, cache_params, response_text)
        
        logger.debug(
            "chat_completion",
            model=self.model,
            message_count=len(messages),
            response_length=len(response_text),
            cache_hit=False,
        )
        
        return response_text

    async def complete(
        self,
        prompt: str,
        sampling_params: Optional[Dict] = None,
    ) -> str:
        """Send a completion request.
        
        Args:
            prompt: Prompt text for completion
            sampling_params: Optional dictionary of sampling parameters
            
        Returns:
            Response text from the model
        """
        # Check cache first
        cache_params = sampling_params or {}
        
        if self.cache:
            cached_response = self.cache.get(prompt, cache_params)
            if cached_response is not None:
                logger.debug(
                    "completion_cache_hit",
                    model=self.model,
                    prompt_length=len(prompt),
                    response_length=len(cached_response),
                )
                return cached_response
        
        # Prepare parameters
        standard_params, vllm_params = self._prepare_sampling_params(sampling_params)
        
        # Build request kwargs
        request_kwargs = {
            "model": self.model,
            "prompt": prompt,
            **standard_params,
        }
        
        if vllm_params:
            request_kwargs["extra_body"] = vllm_params

        if self.enable_thinking is not None:
            request_kwargs.setdefault("extra_body", {})["enable_thinking"] = self.enable_thinking
        
        # Make API call with retry logic
        async def _api_call():
            response = await self.client.completions.create(**request_kwargs)
            return response.choices[0].text
        
        response_text = await self._call_with_retry(_api_call)
        
        # Cache the response
        if self.cache:
            self.cache.set(prompt, cache_params, response_text)
        
        logger.debug(
            "completion",
            model=self.model,
            prompt_length=len(prompt),
            response_length=len(response_text),
            cache_hit=False,
        )
        
        return response_text
