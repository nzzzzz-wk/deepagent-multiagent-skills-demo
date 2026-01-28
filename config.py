"""Configuration settings for the multi-agent system."""

import os
from typing import Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class ModelSettings(BaseModel):
    """Model configuration settings."""
    api_key: str = "dummy"
    base_url: Optional[str] = None
    model: str = "gpt-4o"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    max_retries: int = 2

    def create_model(self) -> ChatOpenAI:
        """Create and return a ChatOpenAI instance."""
        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key if self.api_key else None,
            base_url=self.base_url if self.base_url else None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )


def get_model_settings() -> ModelSettings:
    """Get model settings from environment variables."""
    return ModelSettings(
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        temperature=float(os.environ.get("OPENAI_TEMPERATURE", 0)) if os.environ.get("OPENAI_TEMPERATURE") else None,
        max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS")) if os.environ.get("OPENAI_MAX_TOKENS") else None,
    )
