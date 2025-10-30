from __future__ import annotations

import io
import json
from typing import Dict, Iterable, List, Optional, Sequence

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

    def transcribe(self, chunk: AudioChunk, *, prompt: Optional[str] = None) -> str:
        audio_file = io.BytesIO(chunk.to_wav_bytes())
        audio_file.name = "chunk.wav"
        params: Dict[str, object] = {
            "model": self.settings.transcription_model,
            "file": audio_file,
            "response_format": "text",
            "temperature": 0,
        }
        if prompt:
            params["prompt"] = prompt
        response = self.client.audio.transcriptions.create(**params)
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

    def generate_carl_sections(
        self,
        conversation: Sequence[dict[str, str]],
        knowledge_context: Iterable[str] | None = None,
        prep_summary: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        messages: List[dict[str, str]] = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conversation
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
                        "Prioritise the following reference material. "
                        "Ground your answer in these notes when relevant, "
                        "mention specific examples, and keep the advice practical.\n\n"
                        f"{joined}"
                    ),
                }
            )

        if prep_summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Candidate preparation summary for grounding context:\n"
                        f"{prep_summary.strip()}"
                    ),
                }
            )

        messages.append(
            {
                "role": "system",
                "content": (
                    "Respond as the candidate in the first person using the CARL framework. "
                    "Return your entire reply as JSON with keys 'context', 'actions', "
                    "'results', and 'learnings'. Each key must map to an array of concise "
                    "strings (maximum 40 words each). Avoid filler, emphasise metrics, and "
                    "keep the tone confident and direct."
                ),
            }
        )

        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}

        return {
            "context": self._ensure_list(parsed.get("context")),
            "actions": self._ensure_list(parsed.get("actions")),
            "results": self._ensure_list(parsed.get("results")),
            "learnings": self._ensure_list(parsed.get("learnings")),
        }

    @staticmethod
    def _ensure_list(value: object) -> List[str]:
        if isinstance(value, list):
            cleaned: List[str] = []
            for item in value:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        cleaned.append(text)
            return cleaned
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]
