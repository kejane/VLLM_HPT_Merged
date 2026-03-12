"""Configuration management for vLLM HPT using Pydantic settings.

Supports both A2A (LLM-based) and Traditional (Optuna-based) tuning modes.
In Traditional mode only EXAM_AGENT_* variables are required.
In A2A mode TUNER_AGENT_* variables must also be set.
"""

from functools import lru_cache
from typing import Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    exam_agent_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for the Exam Agent API",
    )
    exam_agent_api_key: str = Field(
        default="EMPTY",
        description="API key for the Exam Agent",
    )
    exam_agent_model: str = Field(
        ...,
        description="Model name for the Exam Agent (required)",
    )

    tuner_agent_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for the Tuner Agent API (A2A mode only)",
    )
    tuner_agent_api_key: str = Field(
        default="EMPTY",
        description="API key for the Tuner Agent (A2A mode only)",
    )
    tuner_agent_model: Optional[str] = Field(
        default=None,
        description="Model name for the Tuner Agent (required for A2A mode)",
    )

    def get_exam_agent_config(self) -> Dict[str, str]:
        """Get Exam Agent configuration as a dictionary."""
        return {
            "base_url": self.exam_agent_base_url,
            "api_key": self.exam_agent_api_key,
            "model": self.exam_agent_model,
        }

    def get_tuner_agent_config(self) -> Dict[str, str]:
        """Get Tuner Agent configuration (A2A mode only).

        Raises:
            ValueError: If TUNER_AGENT_MODEL is not set.
        """
        if not self.tuner_agent_model:
            raise ValueError(
                "TUNER_AGENT_MODEL is required for A2A tuning mode. "
                "Set it in your .env file or as an environment variable."
            )
        return {
            "base_url": self.tuner_agent_base_url,
            "api_key": self.tuner_agent_api_key,
            "model": self.tuner_agent_model,
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)."""
    return Settings()
