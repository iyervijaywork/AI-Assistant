from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


@dataclass
class AudioChunk:
    data: np.ndarray
    sample_rate: int
    timestamp: float

    def to_wav_bytes(self) -> bytes:
        """Convert the chunk to 16-bit PCM WAV bytes."""
        import io
        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            audio_int16 = np.int16(np.clip(self.data * 32767, -32768, 32767))
            wav_file.writeframes(audio_int16.tobytes())
        return buffer.getvalue()


class MicrophoneListener:
    """Capture audio from the system microphone and emit audio chunks."""

    def __init__(
        self,
        sample_rate: int,
        chunk_duration: float = 5.0,
        callback: Optional[Callable[[AudioChunk], None]] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.callback = callback
        self._audio_queue: "queue.Queue[AudioChunk]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        frames_per_chunk = int(self.sample_rate * self.chunk_duration)
        buffer = np.empty((0,), dtype=np.float32)

        def audio_callback(indata, frames, time_info, status):  # type: ignore[override]
            nonlocal buffer
            if status:
                print(f"Audio callback status: {status}")
            buffer = np.concatenate((buffer, indata[:, 0]))
            while len(buffer) >= frames_per_chunk:
                chunk_data = buffer[:frames_per_chunk]
                buffer = buffer[frames_per_chunk:]
                chunk = AudioChunk(
                    data=chunk_data.copy(),
                    sample_rate=self.sample_rate,
                    timestamp=time.time(),
                )
                self._audio_queue.put(chunk)
                if self.callback:
                    self.callback(chunk)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ) as stream:
            self._stream = stream
            while not self._stop_event.is_set():
                time.sleep(0.1)

    def get_queue(self) -> "queue.Queue[AudioChunk]":
        return self._audio_queue
