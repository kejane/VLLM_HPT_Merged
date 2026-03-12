"""Sampling parameters for vLLM with validation and clamping."""

import random
from typing import Any

from pydantic import BaseModel, Field, field_validator


# Parameter ranges: (min, max)
PARAM_RANGES = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "top_k": (-1, 100),
    "repetition_penalty": (1.0, 2.0),
    "max_tokens": (1, 10240),
}


class SamplingParams(BaseModel):
    """vLLM sampling parameters with automatic clamping to valid ranges.

    Default values correspond to the mandatory first-run configuration:
        max_tokens=10240, repetition_penalty=1.2, temperature=0.7,
        top_k=20, top_p=0.8
    """

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=-1, le=100)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)
    max_tokens: int = Field(default=10240, ge=1, le=10240)

    @field_validator("temperature", mode="before")
    @classmethod
    def clamp_temperature(cls, v: Any) -> float:
        """Clamp temperature to valid range."""
        if not isinstance(v, (int, float)):
            return 0.7
        min_val, max_val = PARAM_RANGES["temperature"]
        return max(min_val, min(max_val, float(v)))

    @field_validator("top_p", mode="before")
    @classmethod
    def clamp_top_p(cls, v: Any) -> float:
        """Clamp top_p to valid range."""
        if not isinstance(v, (int, float)):
            return 0.8
        min_val, max_val = PARAM_RANGES["top_p"]
        return max(min_val, min(max_val, float(v)))

    @field_validator("top_k", mode="before")
    @classmethod
    def clamp_top_k(cls, v: Any) -> int:
        """Clamp top_k to valid range."""
        if not isinstance(v, (int, float)):
            return 20
        min_val, max_val = PARAM_RANGES["top_k"]
        return max(min_val, min(max_val, int(v)))

    @field_validator("repetition_penalty", mode="before")
    @classmethod
    def clamp_repetition_penalty(cls, v: Any) -> float:
        """Clamp repetition_penalty to valid range."""
        if not isinstance(v, (int, float)):
            return 1.2
        min_val, max_val = PARAM_RANGES["repetition_penalty"]
        return max(min_val, min(max_val, float(v)))

    @field_validator("max_tokens", mode="before")
    @classmethod
    def clamp_max_tokens(cls, v: Any) -> int:
        """Clamp max_tokens to valid range."""
        if not isinstance(v, (int, float)):
            return 10240
        min_val, max_val = PARAM_RANGES["max_tokens"]
        return max(min_val, min(max_val, int(v)))

    def to_api_dict(self, include_max_tokens: bool = True) -> dict[str, Any]:
        """
        Convert to dict suitable for OpenAI API call.

        Args:
            include_max_tokens: Whether to include max_tokens in the output.
                               Set to False if using a different parameter name.

        Returns:
            Dictionary of parameters for API call.
        """
        result = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }

        # top_k: -1 means disabled, don't include it
        if self.top_k != -1:
            result["top_k"] = self.top_k

        if include_max_tokens:
            result["max_tokens"] = self.max_tokens

        return result


def clamp_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Clamp parameter values to valid ranges.

    Args:
        params: Dictionary of parameter names to values.

    Returns:
        Dictionary with clamped values.
    """
    clamped = {}
    for key, value in params.items():
        if key in PARAM_RANGES:
            min_val, max_val = PARAM_RANGES[key]
            if isinstance(value, (int, float)):
                # Preserve int/float type
                if isinstance(min_val, int) and isinstance(max_val, int):
                    clamped[key] = max(min_val, min(max_val, int(value)))
                else:
                    clamped[key] = max(min_val, min(max_val, float(value)))
            else:
                # Invalid type, use default from SamplingParams
                clamped[key] = getattr(SamplingParams(), key)
        else:
            # Unknown parameter, pass through
            clamped[key] = value
    return clamped


def random_perturbation(params: SamplingParams, scale: float = 0.1) -> SamplingParams:
    """Apply bounded random perturbation to parameters."""
    current = params.model_dump()
    perturbed: dict[str, Any] = {}

    for key, value in current.items():
        if key not in PARAM_RANGES:
            perturbed[key] = value
            continue

        min_val, max_val = PARAM_RANGES[key]
        range_size = max_val - min_val

        if isinstance(value, int):
            delta = int(random.uniform(-scale * range_size, scale * range_size))
            perturbed[key] = value + delta
        else:
            delta = random.uniform(-scale * range_size, scale * range_size)
            perturbed[key] = value + delta

    return SamplingParams(**perturbed)
