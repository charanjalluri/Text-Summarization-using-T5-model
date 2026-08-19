from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Summarization Engine Settings
    MODEL_NAME: str = "t5-small"
    DEVICE: str = "auto"  # 'auto', 'cuda', or 'cpu'
    MAX_INPUT_TOKENS: int = 512
    MIN_SUMMARY_TOKENS: int = 30
    MAX_SUMMARY_TOKENS: int = 150
    NUM_BEAMS: int = 4
    LENGTH_PENALTY: float = 2.0

    # Long-Document Handling Settings
    CHUNK_SIZE: int = 500  # Number of words per chunk
    CHUNK_OVERLAP: int = 50  # Words overlap between chunks

    # Google Gemini Settings
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # Translation settings
    DEFAULT_TARGET_LANGUAGE: str = "en"

    @field_validator("MAX_SUMMARY_TOKENS")
    @classmethod
    def validate_lengths(cls, v: int, info) -> int:
        min_len = info.data.get("MIN_SUMMARY_TOKENS", 30)
        if v < min_len:
            raise ValueError(f"MAX_SUMMARY_TOKENS ({v}) cannot be less than MIN_SUMMARY_TOKENS ({min_len})")
        return v

    @field_validator("DEVICE")
    @classmethod
    def validate_device(cls, v: str) -> str:
        val = v.lower()
        if val not in {"auto", "cuda", "cpu"}:
            raise ValueError("DEVICE must be one of 'auto', 'cuda', or 'cpu'")
        return val

# Instantiated settings instance
settings = Settings()
