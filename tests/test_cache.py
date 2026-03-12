"""Tests for vllm_hpt.utils.cache module."""

import pytest

from vllm_hpt.utils.cache import ResponseCache


class TestResponseCacheEnabled:

    def test_set_and_get(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)
        cache.set("prompt1", {"temp": 0.7}, "response1")
        result = cache.get("prompt1", {"temp": 0.7})
        assert result == "response1"

    def test_cache_miss(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)
        result = cache.get("nonexistent", {"temp": 0.7})
        assert result is None

    def test_hit_miss_stats(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)
        cache.set("p", {"t": 1}, "r")
        cache.get("p", {"t": 1})
        cache.get("missing", {"t": 1})

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total"] == 2
        assert stats["hit_rate"] == 0.5
        assert stats["enabled"] is True

    def test_deterministic_key(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)
        k1 = cache._make_key("prompt", {"a": 1, "b": 2})
        k2 = cache._make_key("prompt", {"b": 2, "a": 1})
        assert k1 == k2

    def test_different_params_different_keys(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)
        k1 = cache._make_key("prompt", {"a": 1})
        k2 = cache._make_key("prompt", {"a": 2})
        assert k1 != k2

    def test_clear(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)
        cache.set("p", {"t": 1}, "r")
        cache.clear()
        assert cache.get("p", {"t": 1}) is None

    def test_context_manager(self, tmp_path):
        with ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True) as cache:
            cache.set("p", {}, "r")
            assert cache.get("p", {}) == "r"


class TestResponseCacheDisabled:

    def test_get_returns_none(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=False)
        assert cache.get("p", {}) is None

    def test_set_is_noop(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=False)
        cache.set("p", {}, "r")
        assert cache.get("p", {}) is None

    def test_stats_show_disabled(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=False)
        stats = cache.stats()
        assert stats["enabled"] is False
        assert stats["total"] == 0

    def test_clear_is_safe(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=False)
        cache.clear()


class TestResponseCacheStats:

    def test_initial_stats(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
