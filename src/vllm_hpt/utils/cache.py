"""Cache module for API responses using diskcache.

This module provides a ResponseCache class that wraps diskcache to cache
API responses based on (prompt, params) keys. It supports cache hit/miss
tracking and can be disabled via a flag.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import diskcache


class ResponseCache:
    """Cache for API responses with hit/miss tracking.
    
    Uses diskcache to store responses keyed by (prompt, sampling_params).
    Tracks cache statistics and supports disabling.
    """

    def __init__(self, cache_dir: str = "cache/api_responses", enabled: bool = True):
        """Initialize the response cache.
        
        Args:
            cache_dir: Directory path for cache storage
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.cache_dir = cache_dir
        self._hits = 0
        self._misses = 0
        
        if self.enabled:
            # Create cache directory if it doesn't exist
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self._cache = diskcache.Cache(cache_dir)
        else:
            self._cache = None
            
        # Try to import logger, but don't fail if not available
        try:
            from vllm_hpt.utils.logger import get_logger
            self._logger = get_logger(__name__)
        except (ImportError, Exception):
            self._logger = None

    def _make_key(self, prompt: str, params: dict) -> str:
        """Create deterministic hash key from prompt and params.
        
        Args:
            prompt: The prompt text
            params: Dictionary of sampling parameters
            
        Returns:
            SHA256 hash string as cache key
        """
        # Sort params to ensure deterministic ordering
        sorted_params = json.dumps(params, sort_keys=True)
        key_string = f"{prompt}::{sorted_params}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(self, prompt: str, params: dict) -> Optional[str]:
        """Retrieve cached response if available.
        
        Args:
            prompt: The prompt text
            params: Dictionary of sampling parameters
            
        Returns:
            Cached response string or None if not found
        """
        if not self.enabled:
            return None
            
        key = self._make_key(prompt, params)
        response = self._cache.get(key)
        
        if response is not None:
            self._hits += 1
            if self._logger:
                self._logger.debug("cache_hit", key=key[:16])
        else:
            self._misses += 1
            if self._logger:
                self._logger.debug("cache_miss", key=key[:16])
                
        return response

    def set(self, prompt: str, params: dict, response: str) -> None:
        """Store response in cache.
        
        Args:
            prompt: The prompt text
            params: Dictionary of sampling parameters
            response: Response string to cache
        """
        if not self.enabled:
            return
            
        key = self._make_key(prompt, params)
        self._cache.set(key, response)
        
        if self._logger:
            self._logger.debug("cache_set", key=key[:16])

    def clear(self) -> None:
        """Clear all cached entries."""
        if self.enabled and self._cache:
            self._cache.clear()
            if self._logger:
                self._logger.info("cache_cleared")

    def stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with hit/miss counts and hit rate
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": hit_rate,
            "enabled": self.enabled,
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.enabled and self._cache:
            self._cache.close()
