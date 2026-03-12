"""Tests for vllm_hpt.tuning.params module."""

import pytest

from vllm_hpt.tuning.params import (
    PARAM_RANGES,
    SamplingParams,
    clamp_params,
)


# ---------------------------------------------------------------------------
# SamplingParams defaults
# ---------------------------------------------------------------------------

class TestSamplingParamsDefaults:
    """Test default parameter values."""

    def test_defaults(self):
        p = SamplingParams()
        assert p.temperature == 0.7
        assert p.top_p == 0.8
        assert p.top_k == 20
        assert p.repetition_penalty == 1.2
        assert p.max_tokens == 10240

    def test_custom_values(self):
        p = SamplingParams(
            temperature=1.0, top_p=0.5, top_k=10,
            repetition_penalty=1.5, max_tokens=256,
        )
        assert p.temperature == 1.0
        assert p.top_p == 0.5
        assert p.top_k == 10
        assert p.repetition_penalty == 1.5
        assert p.max_tokens == 256


# ---------------------------------------------------------------------------
# Clamping (out of range values)
# ---------------------------------------------------------------------------

class TestSamplingParamsClamping:
    """Test that out-of-range values are clamped."""

    def test_temperature_too_high(self):
        p = SamplingParams(temperature=5.0)
        assert p.temperature == 2.0

    def test_temperature_too_low(self):
        p = SamplingParams(temperature=-1.0)
        assert p.temperature == 0.0

    def test_top_p_too_high(self):
        p = SamplingParams(top_p=1.5)
        assert p.top_p == 1.0

    def test_top_p_too_low(self):
        p = SamplingParams(top_p=-0.5)
        assert p.top_p == 0.0

    def test_top_k_too_high(self):
        p = SamplingParams(top_k=200)
        assert p.top_k == 100

    def test_top_k_too_low(self):
        p = SamplingParams(top_k=-5)
        assert p.top_k == -1

    def test_repetition_penalty_too_high(self):
        p = SamplingParams(repetition_penalty=5.0)
        assert p.repetition_penalty == 2.0

    def test_repetition_penalty_too_low(self):
        p = SamplingParams(repetition_penalty=0.5)
        assert p.repetition_penalty == 1.0

    def test_max_tokens_too_high(self):
        p = SamplingParams(max_tokens=20000)
        assert p.max_tokens == 10240

    def test_max_tokens_too_low(self):
        p = SamplingParams(max_tokens=0)
        assert p.max_tokens == 1

    def test_invalid_type_returns_default(self):
        """Non-numeric types should fall back to defaults."""
        p = SamplingParams(temperature="invalid")
        assert p.temperature == 0.7

        p = SamplingParams(top_k="bad")
        assert p.top_k == 20


# ---------------------------------------------------------------------------
# to_api_dict
# ---------------------------------------------------------------------------

class TestToApiDict:
    """Test conversion to API-compatible dictionary."""

    def test_includes_all_params(self):
        p = SamplingParams(top_k=30)
        d = p.to_api_dict()
        assert "temperature" in d
        assert "top_p" in d
        assert "repetition_penalty" in d
        assert "max_tokens" in d
        assert "top_k" in d
        assert d["top_k"] == 30

    def test_excludes_top_k_when_disabled(self):
        """top_k=-1 means disabled, should not appear in API dict."""
        p = SamplingParams(top_k=-1)
        d = p.to_api_dict()
        assert "top_k" not in d

    def test_exclude_max_tokens(self):
        p = SamplingParams()
        d = p.to_api_dict(include_max_tokens=False)
        assert "max_tokens" not in d

    def test_values_match(self):
        p = SamplingParams(
            temperature=1.2, top_p=0.8,
            repetition_penalty=1.3, max_tokens=100,
        )
        d = p.to_api_dict()
        assert d["temperature"] == 1.2
        assert d["top_p"] == 0.8
        assert d["repetition_penalty"] == 1.3
        assert d["max_tokens"] == 100


# ---------------------------------------------------------------------------
# PARAM_RANGES
# ---------------------------------------------------------------------------

class TestParamRanges:
    """Test PARAM_RANGES constant."""

    def test_all_params_have_ranges(self):
        expected_keys = {"temperature", "top_p", "top_k", "repetition_penalty", "max_tokens"}
        assert set(PARAM_RANGES.keys()) == expected_keys

    def test_ranges_are_tuples(self):
        for key, (lo, hi) in PARAM_RANGES.items():
            assert lo <= hi, f"{key}: min ({lo}) > max ({hi})"


# ---------------------------------------------------------------------------
# clamp_params (standalone function)
# ---------------------------------------------------------------------------

class TestClampParams:
    """Test the clamp_params utility function."""

    def test_clamps_out_of_range(self):
        result = clamp_params({"temperature": 5.0, "top_k": 200})
        assert result["temperature"] == 2.0
        assert result["top_k"] == 100

    def test_passes_through_unknown_keys(self):
        result = clamp_params({"temperature": 1.0, "unknown_param": "hello"})
        assert result["unknown_param"] == "hello"

    def test_invalid_type_uses_default(self):
        result = clamp_params({"temperature": "bad"})
        assert result["temperature"] == 0.7  # default
