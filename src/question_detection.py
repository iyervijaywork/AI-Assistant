"""Utilities for detecting question boundaries from streaming audio and text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .audio_capture import AudioChunk


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


@dataclass
class QuestionBoundaryDetector:
    """Detect interviewer question boundaries using text and voice cues."""

    min_question_words: int = 4
    speech_threshold: float = 0.012
    min_silence_seconds: float = 0.7
    max_buffer_words: int = 120
    question_prefixes: List[str] = field(
        default_factory=lambda: [
            "what",
            "why",
            "how",
            "when",
            "where",
            "who",
            "which",
            "tell me",
            "could you",
            "would you",
            "do you",
            "can you",
            "walk me",
            "share",
            "describe",
        ]
    )
    question_suffixes: List[str] = field(
        default_factory=lambda: [
            "?",
            "right",
            "correct",
            "okay",
            "ok",
            "yeah",
            "please",
        ]
    )

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._segments: List[str] = []
        self._question_active = False
        self._explicit_question = False
        self._silence_accumulator = 0.0

    def observe(self, chunk: AudioChunk, new_text: str) -> str | None:
        """Observe a new audio chunk and optional text addition.

        Returns the committed question text when the detector is confident the
        interviewer has finished speaking; otherwise returns ``None``.
        """

        addition = new_text.strip()
        if addition:
            self._segments.append(addition)
            self._question_active = True
            if "?" in addition:
                self._explicit_question = True
        current_text = self.current_text

        if chunk.rms >= self.speech_threshold:
            self._silence_accumulator = 0.0
        else:
            if self._question_active:
                self._silence_accumulator += max(chunk.duration, 0.0)

        if not current_text:
            return None

        if not self._question_active and self._looks_like_question_start(current_text):
            self._question_active = True

        if self._silence_accumulator >= self.min_silence_seconds and self._ready_to_commit():
            return self._commit()

        if self._ready_to_commit(force=True) and _word_count(current_text) >= self.max_buffer_words:
            return self._commit()

        return None

    @property
    def current_text(self) -> str:
        return " ".join(self._segments).strip()

    def _commit(self) -> str | None:
        text = self.current_text
        self.reset()
        return text or None

    def _looks_like_question_start(self, text: str) -> bool:
        normalised = text.lower()
        return any(normalised.startswith(prefix) for prefix in self.question_prefixes)

    def _ready_to_commit(self, force: bool = False) -> bool:
        text = self.current_text
        if not text:
            return False

        word_count = _word_count(text)
        if not force and word_count < self.min_question_words and not self._explicit_question:
            return False

        normalised = text.lower().rstrip()
        if normalised.endswith("?"):
            return True
        if any(normalised.endswith(suffix) for suffix in self.question_suffixes):
            return True
        if any(prefix in normalised for prefix in self.question_prefixes):
            return True
        if self._explicit_question:
            return True

        return force and word_count >= self.min_question_words
