from __future__ import annotations

import io
from typing import List

from openai import OpenAI

from .audio_capture import AudioChunk
from .config import Settings


class AIClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        self._conversation: List[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a concise, helpful real-time assistant that listens to a "
                    "conversation and surfaces relevant facts, answers questions, and "
                    "suggests follow-up ideas without overwhelming the user."
                ),
            }
        ]

    def transcribe(self, chunk: AudioChunk) -> str:
        audio_file = io.BytesIO(chunk.to_wav_bytes())
        audio_file.name = "chunk.wav"
        response = self.client.audio.transcriptions.create(
            model=self.settings.transcription_model,
            file=audio_file,
            response_format="text",
        )
        return response

    def chat_completion(self, latest_transcript: str) -> str:
        self._conversation.append({"role": "user", "content": latest_transcript})
        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=self._conversation,
            temperature=0.6,
        )
        message = response.choices[0].message.content or ""
        self._conversation.append({"role": "assistant", "content": message})
        return message
