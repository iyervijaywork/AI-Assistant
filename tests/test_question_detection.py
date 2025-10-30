from __future__ import annotations

import numpy as np

from src.audio_capture import AudioChunk
from src.question_detection import QuestionBoundaryDetector


def make_chunk(rms: float, duration: float, text: str = "") -> tuple[AudioChunk, str]:
    samples = max(int(16000 * duration), 1)
    data = np.zeros(samples, dtype=np.float32)
    return (
        AudioChunk(
            data=data,
            sample_rate=16000,
            timestamp=0.0,
            rms=rms,
            duration=duration,
        ),
        text,
    )


def test_detector_commits_after_silence():
    detector = QuestionBoundaryDetector(min_silence_seconds=0.5)

    speech_chunk, text = make_chunk(0.05, 0.8, "Can you share your favorite leadership win")
    assert detector.observe(speech_chunk, text) is None

    silence_chunk, empty = make_chunk(0.0, 0.6)
    committed = detector.observe(silence_chunk, empty)
    assert committed is not None
    assert committed.lower().startswith("can you share")


def test_detector_handles_question_without_question_mark():
    detector = QuestionBoundaryDetector(min_silence_seconds=0.5)

    speech_chunk, text = make_chunk(0.04, 0.9, "Tell me about a time you disagreed with a manager")
    assert detector.observe(speech_chunk, text) is None

    silence_chunk, empty = make_chunk(0.0, 0.6)
    committed = detector.observe(silence_chunk, empty)
    assert committed is not None
    assert "disagreed" in committed
