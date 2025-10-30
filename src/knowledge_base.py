from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .openai_client import AIClient


@dataclass
class KnowledgeChunk:
    content: str
    source: str
    embedding: np.ndarray
    session_id: Optional[str]


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
        self._session_sources: dict[Optional[str], set[str]] = {}
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._session_sources.clear()

    def ingest_files(
        self, file_paths: Iterable[Path], session_id: Optional[str] = None
    ) -> int:
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
                    KnowledgeChunk(
                        content=content,
                        source=source,
                        embedding=np.array(embedding),
                        session_id=session_id,
                    )
                )
                self._session_sources.setdefault(session_id, set()).add(source)
            return len(texts)

    def upsert_session_pair(
        self,
        session_id: str,
        turn_index: int,
        question: str,
        answer: str,
    ) -> None:
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            return

        source_identifier = f"session::{session_id}::{turn_index}"
        snippet = f"Question: {question}\nAnswer: {answer}"
        embedding = np.array(self.ai_client.embed_texts([snippet])[0])

        with self._lock:
            self._remove_source_locked(source_identifier)
            self._chunks.append(
                KnowledgeChunk(
                    content=snippet,
                    source=source_identifier,
                    embedding=embedding,
                    session_id=session_id,
                )
            )
            self._session_sources.setdefault(session_id, set()).add(source_identifier)

    def is_empty(self) -> bool:
        with self._lock:
            return not self._chunks

    def is_empty_for_session(self, session_id: Optional[str]) -> bool:
        with self._lock:
            return not any(chunk.session_id == session_id for chunk in self._chunks)

    def top_matches(
        self,
        query: str,
        top_k: int = 4,
        session_id: Optional[str] = None,
        include_global: bool = True,
    ) -> List[str]:
        if not query.strip():
            return []

        with self._lock:
            if not self._chunks:
                return []
            filtered_chunks: List[KnowledgeChunk] = []
            for chunk in self._chunks:
                if chunk.session_id == session_id:
                    filtered_chunks.append(chunk)
                elif include_global and chunk.session_id is None:
                    filtered_chunks.append(chunk)

            if not filtered_chunks:
                return []

            chunk_embeddings = np.stack([chunk.embedding for chunk in filtered_chunks])
            chunk_contents = [chunk.content for chunk in filtered_chunks]

        query_embedding = np.array(self.ai_client.embed_texts([query])[0])
        scores = self._cosine_similarity(query_embedding, chunk_embeddings)
        top_indices = scores.argsort()[::-1][:top_k]
        return [chunk_contents[idx] for idx in top_indices]

    def listed_sources(
        self, session_id: Optional[str] = None, include_global: bool = True
    ) -> List[str]:
        with self._lock:
            sources: set[str] = set()
            if include_global and None in self._session_sources:
                sources.update(self._session_sources[None])
            if session_id in self._session_sources:
                sources.update(self._session_sources[session_id])
        return sorted(self._format_source_label(source) for source in sources)

    def sessions_with_content(self) -> Sequence[Optional[str]]:
        with self._lock:
            return tuple(self._session_sources.keys())

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

    def _remove_source_locked(self, source_identifier: str) -> None:
        if not source_identifier:
            return
        self._chunks = [chunk for chunk in self._chunks if chunk.source != source_identifier]
        for sources in self._session_sources.values():
            sources.discard(source_identifier)

    @staticmethod
    def _format_source_label(source: str) -> str:
        if source.startswith("session::"):
            parts = source.split("::")
            if len(parts) == 3:
                session_fragment = parts[1][:8]
                turn_label = parts[2]
                return f"Session {session_fragment} — turn {turn_label}"
        return source
