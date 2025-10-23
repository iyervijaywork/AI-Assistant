from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Settings:
    """Application configuration loaded from environment variables."""

    openai_api_key: str
    model: str = "gpt-4o-mini"
    transcription_model: str = "whisper-1"
    sample_rate: int = 16000
    chunk_duration: float = 5.0

    @classmethod
    def load(cls, env_path: Optional[Path] = None) -> "Settings":
        if env_path:
            load_dotenv(env_path)
        else:
            # Load from default .env if it exists
            load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please set it in your environment or .env file."
            )

        model = os.getenv("OPENAI_MODEL", cls.model)
        transcription_model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", cls.transcription_model)
        sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", str(cls.sample_rate)))
        chunk_duration = float(os.getenv("AUDIO_CHUNK_DURATION", str(cls.chunk_duration)))

        return cls(
            openai_api_key=api_key,
            model=model,
            transcription_model=transcription_model,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
        )


def get_settings() -> Settings:
    return Settings.load()
