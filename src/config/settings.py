"""
Application configuration using Pydantic settings.
Loads from environment variables and .env file.
"""
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Secure configuration management using Pydantic.
    - Loads from .env file
    - Validates required keys exist
    - Uses SecretStr for sensitive values (prevents logging exposure)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required API Keys
    openai_api_key: SecretStr = Field(..., validation_alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="ANTHROPIC_API_KEY"
    )
    tavily_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="TAVILY_API_KEY"
    )

    # LangSmith Configuration
    langsmith_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="LANGSMITH_API_KEY"
    )
    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_project: str = Field(
        default="price-compare",
        validation_alias="LANGSMITH_PROJECT"
    )
    langsmith_endpoint: Optional[str] = Field(
        default=None,
        validation_alias="LANGSMITH_ENDPOINT"
    )

    # Database Configuration
    sqlite_path: str = Field(default="./data/db/products.db")
    chroma_path: str = Field(default="./data/db/chroma")

    # Embedding Configuration
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_batch_size: int = Field(default=50)

    # Search Configuration
    default_search_limit: int = Field(default=10)
    confidence_threshold: float = Field(default=0.5)
    enable_live_search: bool = Field(default=True)

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, validation_alias="PORT")
    cors_origins: List[str] = Field(default=["*"])  # Allow all origins for cloud deployment

    # Security Configuration
    encryption_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="ENCRYPTION_KEY"
    )

    # Processing Configuration
    batch_size: int = Field(default=100)
    max_workers: int = Field(default=4)

    @property
    def sqlite_url(self) -> str:
        """Get SQLite connection URL."""
        return f"sqlite:///{self.sqlite_path}"

    @property
    def async_sqlite_url(self) -> str:
        """Get async SQLite connection URL."""
        return f"sqlite+aiosqlite:///{self.sqlite_path}"

    def ensure_directories(self) -> None:
        """Create necessary data directories if they don't exist."""
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
