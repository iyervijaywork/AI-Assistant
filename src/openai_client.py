from __future__ import annotations

import io
from typing import Iterable, List, Sequence

from openai import OpenAI

from .audio_capture import AudioChunk
from .config import Settings


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, helpful real-time assistant that listens to a conversation "
    "and surfaces relevant facts, answers questions, and suggests follow-up ideas "
    "without overwhelming the user. Focus on interview preparation, offering "
    "specific, actionable feedback and examples grounded in the provided knowledge "
    "base."
)


class AIClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)

    @property
    def system_message(self) -> dict[str, str]:
        return {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}

    def transcribe(self, chunk: AudioChunk) -> str:
        audio_file = io.BytesIO(chunk.to_wav_bytes())
        audio_file.name = "chunk.wav"
        response = self.client.audio.transcriptions.create(
            model=self.settings.transcription_model,
            file=audio_file,
            response_format="text",
        )
        return response

    def chat_completion(
        self,
        conversation: Sequence[dict[str, str]],
        knowledge_context: Iterable[str] | None = None,
    ) -> str:
        messages: List[dict[str, str]] = [
            {"role": msg["role"], "content": msg["content"]} for msg in conversation
        ]

        context_snippets: List[str] = []
        if knowledge_context:
            for index, snippet in enumerate(knowledge_context, start=1):
                snippet = snippet.strip()
                if snippet:
                    context_snippets.append(f"Reference {index}:\n{snippet}")

        if context_snippets:
            joined = "\n\n".join(context_snippets)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "In your next reply, prioritize the following reference material. "
                        "Ground your answer in these notes when relevant, mention "
                        "specific examples, and keep the advice practical.\n\n"
                        f"{joined}"
                    ),
                }
            )

        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            temperature=0.3,
        )
        message = response.choices[0].message.content or ""
        return message

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]
