from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .openai_client import AIClient


@dataclass
class KnowledgeChunk:
    content: str
    source: str
    embedding: np.ndarray


class KnowledgeBase:
    """In-memory vector store populated from user-provided documents."""

    def __init__(
        self,
        ai_client: AIClient,
        chunk_char_length: int = 900,
        chunk_overlap: int = 200,
    ) -> None:
        self.ai_client = ai_client
        self.chunk_char_length = chunk_char_length
        self.chunk_overlap = chunk_overlap
        self._chunks: List[KnowledgeChunk] = []
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()

    def ingest_files(self, file_paths: Iterable[Path]) -> int:
        """Load text from paths, embed, and store in memory.

        Returns the number of chunks added.
        """

        paths = [Path(path) for path in file_paths]
        texts: List[str] = []
        sources: List[str] = []

        for path in paths:
            if not path.exists():
                continue
            text = self._read_text_from_file(path)
            if not text:
                continue

            for chunk in self._chunk_text(text):
                texts.append(chunk)
                sources.append(str(path))

        if not texts:
            return 0

        embeddings = self.ai_client.embed_texts(texts)

        with self._lock:
            for content, source, embedding in zip(texts, sources, embeddings):
                self._chunks.append(
                    KnowledgeChunk(content=content, source=source, embedding=np.array(embedding))
                )

            return len(texts)

    def is_empty(self) -> bool:
        with self._lock:
            return not self._chunks

    def top_matches(self, query: str, top_k: int = 4) -> List[str]:
        if not query.strip():
            return []

        with self._lock:
            if not self._chunks:
                return []
            chunk_embeddings = np.stack([chunk.embedding for chunk in self._chunks])
            chunk_contents = [chunk.content for chunk in self._chunks]

        query_embedding = np.array(self.ai_client.embed_texts([query])[0])
        scores = self._cosine_similarity(query_embedding, chunk_embeddings)
        top_indices = scores.argsort()[::-1][:top_k]
        return [chunk_contents[idx] for idx in top_indices]

    def listed_sources(self) -> List[str]:
        with self._lock:
            return sorted({chunk.source for chunk in self._chunks})

    def _chunk_text(self, text: str) -> List[str]:
        normalised = re.sub(r"\s+", " ", text.strip())
        if not normalised:
            return []

        chunks: List[str] = []
        start = 0
        length = len(normalised)
        while start < length:
            end = min(start + self.chunk_char_length, length)
            if end < length:
                # try to break on sentence boundary
                sentence_break = normalised.rfind(". ", start, end)
                if sentence_break != -1 and sentence_break > start + self.chunk_char_length // 2:
                    end = sentence_break + 1
            chunk = normalised[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= length:
                break
            overlap_start = max(end - self.chunk_overlap, 0) if self.chunk_overlap else end
            next_start = overlap_start if overlap_start < end else end
            if next_start <= start:
                next_start = end
            start = next_start
        return chunks

    def _read_text_from_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".markdown"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:  # pragma: no cover - optional dependency at runtime
                raise RuntimeError(
                    "pypdf is required to parse PDF documents. Install it via requirements.txt."
                )

            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        return ""

    @staticmethod
    def _cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        vector_norm = np.linalg.norm(vector)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        # Avoid division by zero
        denom = np.clip(vector_norm * matrix_norms, a_min=1e-12, a_max=None)
        return matrix @ vector / denom
