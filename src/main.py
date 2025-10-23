from __future__ import annotations

import queue
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .audio_capture import MicrophoneListener
from .config import Settings, get_settings
from .knowledge_base import KnowledgeBase
from .openai_client import AIClient, DEFAULT_SYSTEM_PROMPT


@dataclass
class ChatSession:
    session_id: str
    title: str
    conversation: List[dict[str, str]] = field(default_factory=list)
    transcript_segments: List[str] = field(default_factory=list)
    transcript_display: str = ""
    assistant_segments: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.conversation:
            self.conversation.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    @property
    def transcript_text(self) -> str:
        return self.transcript_display

    @property
    def assistant_text(self) -> str:
        return "\n\n".join(self.assistant_segments).strip()

    @property
    def last_transcript_segment(self) -> str:
        if not self.transcript_segments:
            return ""
        return self.transcript_segments[-1]


class SessionSwitchBridge(QObject):
    requested = pyqtSignal(str, object, str)


class TranscriptionWorker(QObject):
    transcript_ready = pyqtSignal(str, str)
    user_message_committed = pyqtSignal(str, str)
    assistant_ready = pyqtSignal(str, str)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        audio_queue: "queue.Queue",
        ai_client: AIClient,
        knowledge_base: Optional[KnowledgeBase],
        session_id: str,
        conversation: List[dict[str, str]],
        transcript_text: str,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.audio_queue = audio_queue
        self.ai_client = ai_client
        self.knowledge_base = knowledge_base
        self.session_id = session_id
        self._conversation: List[dict[str, str]] = [
            {"role": msg["role"], "content": msg["content"]} for msg in conversation
        ]
        self._stop_event = threading.Event()
        self._previous_transcript = transcript_text
        self._user_buffer = ""

    @pyqtSlot()
    def run(self) -> None:
        self.status_changed.emit("Listening…")
        while not self._stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self.status_changed.emit("Transcribing…")
                transcript = self.ai_client.transcribe(chunk)
                addition = self._extract_new_text(transcript)
                if not addition:
                    self.status_changed.emit("Listening…")
                    continue

                self.transcript_ready.emit(self.session_id, addition)
                self._user_buffer = (self._user_buffer + " " + addition).strip()

                if not self._should_respond(self._user_buffer):
                    self.status_changed.emit("Listening…")
                    continue

                user_message = self._user_buffer
                self._user_buffer = ""
                self._append_conversation({"role": "user", "content": user_message})
                self.user_message_committed.emit(self.session_id, user_message)

                context: List[str] = []
                if self.knowledge_base and not self.knowledge_base.is_empty():
                    context = self.knowledge_base.top_matches(user_message)

                self.status_changed.emit("Thinking…")
                response = self.ai_client.chat_completion(self._conversation, context).strip()
                if response:
                    self._append_conversation({"role": "assistant", "content": response})
                    self.assistant_ready.emit(self.session_id, response)
                self.status_changed.emit("Listening…")
            except Exception as exc:  # pragma: no cover - runtime safeguard
                self.error_occurred.emit(str(exc))
                self.status_changed.emit("Error")

        self.status_changed.emit("Idle")

    def stop(self) -> None:
        self._stop_event.set()

    @pyqtSlot(str, object, str)
    def switch_session(
        self, session_id: str, conversation: object, transcript_text: str
    ) -> None:
        self.session_id = session_id
        if isinstance(conversation, list):
            self._conversation = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in conversation
                if isinstance(msg, dict) and "role" in msg and "content" in msg
            ]
        else:
            self._conversation = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        self._previous_transcript = transcript_text
        self._user_buffer = ""

    def _should_respond(self, buffer: str) -> bool:
        if not buffer:
            return False
        if any(buffer.endswith(punct) for punct in (".", "?", "!")):
            return True
        return len(buffer.split()) >= 12

    def _extract_new_text(self, transcript: str) -> str:
        cleaned = transcript.strip()
        if not cleaned:
            return ""
        previous = self._previous_transcript.strip()
        addition = cleaned
        if previous and cleaned.startswith(previous):
            addition = cleaned[len(previous) :].strip()
        elif previous:
            overlap = self._longest_overlap(previous, cleaned)
            addition = cleaned[overlap:].strip()
        self._previous_transcript = cleaned
        return addition

    @staticmethod
    def _longest_overlap(previous: str, current: str) -> int:
        max_len = min(len(previous), len(current))
        for size in range(max_len, 0, -1):
            if previous[-size:] == current[:size]:
                return size
        return 0

    def _append_conversation(self, message: dict[str, str]) -> None:
        self._conversation.append(message)
        if len(self._conversation) > 50:
            system_message = self._conversation[0]
            recent_messages = self._conversation[-49:]
            self._conversation = [system_message, *recent_messages]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Live Conversation Assistant")
        self.resize(1320, 760)

        self.settings: Optional[Settings] = None
        self.ai_client: Optional[AIClient] = None
        self.knowledge_base: Optional[KnowledgeBase] = None
        self.listener: Optional[MicrophoneListener] = None
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[TranscriptionWorker] = None
        self.session_bridge: Optional[SessionSwitchBridge] = None

        self.sessions: Dict[str, ChatSession] = {}
        self.current_session_id: Optional[str] = None

        self._setup_ui()
        self._create_new_session()

    def _setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        outer_layout = QVBoxLayout(central_widget)

        header_layout = QHBoxLayout()
        title_label = QLabel("AI Conversation Companion")
        title_label.setStyleSheet("font-size: 24px; font-weight: 600;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("color: #555; font-size: 14px;")
        header_layout.addWidget(self.status_label)

        self.toggle_button = QPushButton("Start Listening")
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self._toggle_listening)
        header_layout.addWidget(self.toggle_button)

        outer_layout.addLayout(header_layout)

        content_layout = QHBoxLayout()
        outer_layout.addLayout(content_layout, 1)

        sidebar = QWidget()
        sidebar.setMaximumWidth(260)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)

        chats_label = QLabel("Chats")
        chats_label.setStyleSheet("font-size: 18px; font-weight: 500; padding-bottom: 6px;")
        sidebar_layout.addWidget(chats_label)

        self.session_list = QListWidget()
        self.session_list.setSpacing(4)
        self.session_list.setStyleSheet("QListWidget { background: #ffffff; border-radius: 6px; }")
        self.session_list.currentItemChanged.connect(self._on_session_selected)
        sidebar_layout.addWidget(self.session_list, 1)

        buttons_layout = QVBoxLayout()
        new_chat_button = QPushButton("New chat")
        new_chat_button.clicked.connect(self._on_new_chat)
        buttons_layout.addWidget(new_chat_button)

        load_button = QPushButton("Load reference files…")
        load_button.clicked.connect(self._load_reference_material)
        buttons_layout.addWidget(load_button)

        sidebar_layout.addLayout(buttons_layout)

        self.kb_status_label = QLabel("Knowledge base: empty")
        self.kb_status_label.setStyleSheet("color: #666; font-size: 12px;")
        sidebar_layout.addWidget(self.kb_status_label)

        sidebar_layout.addStretch()

        content_layout.addWidget(sidebar)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.transcript_view = QTextEdit()
        self.transcript_view.setAcceptRichText(False)
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setPlaceholderText("Real-time transcription will appear here…")
        self.transcript_view.setStyleSheet(
            "font-size: 16px; padding: 12px; background: #fcfcfc; border-radius: 8px;"
        )

        self.assistant_view = QTextEdit()
        self.assistant_view.setAcceptRichText(False)
        self.assistant_view.setReadOnly(True)
        self.assistant_view.setPlaceholderText(
            "Assistant insights and responses will show here…"
        )
        self.assistant_view.setStyleSheet(
            "font-size: 16px; padding: 12px; background: #f7fbff; border-radius: 8px;"
        )

        splitter.addWidget(self._wrap_with_label("Live Transcript", self.transcript_view))
        splitter.addWidget(self._wrap_with_label("Assistant Response", self.assistant_view))
        splitter.setSizes([620, 620])

        content_layout.addWidget(splitter, 1)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#f0f2f5"))
        self.setPalette(palette)

    def _wrap_with_label(self, label: str, widget: QWidget) -> QWidget:
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel(label)
        title.setStyleSheet("font-size: 18px; font-weight: 500; padding-bottom: 6px;")
        container_layout.addWidget(title)
        container_layout.addWidget(widget)
        return container

    def _toggle_listening(self, checked: bool) -> None:
        if checked:
            try:
                self._start_listening()
                self.toggle_button.setText("Stop Listening")
            except Exception as exc:
                self.toggle_button.setChecked(False)
                QMessageBox.critical(self, "Error", str(exc))
        else:
            self._stop_listening()
            self.toggle_button.setText("Start Listening")

    def _start_listening(self) -> None:
        if self.worker_thread and self.worker_thread.isRunning():
            return

        ai_client = self._ensure_client()
        if not self.current_session_id:
            session = self._create_new_session()
        else:
            session = self.sessions[self.current_session_id]

        self.listener = MicrophoneListener(
            sample_rate=self.settings.sample_rate,
            chunk_duration=self.settings.chunk_duration,
        )

        audio_queue = self.listener.get_queue()

        self.worker_thread = QThread()
        self.worker = TranscriptionWorker(
            audio_queue,
            ai_client,
            self.knowledge_base,
            session.session_id,
            list(session.conversation),
            session.last_transcript_segment,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.transcript_ready.connect(self._append_transcript)
        self.worker.user_message_committed.connect(self._register_user_message)
        self.worker.assistant_ready.connect(self._append_assistant)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.error_occurred.connect(self._handle_worker_error)

        self.session_bridge = SessionSwitchBridge()
        self.session_bridge.requested.connect(self.worker.switch_session)

        self.worker_thread.start()

        self.session_bridge.requested.emit(
            session.session_id,
            list(session.conversation),
            session.last_transcript_segment,
        )

        self.listener.start()
        self.status_label.setText("Listening…")

    def _stop_listening(self) -> None:
        if self.listener:
            self.listener.stop()
            self.listener = None

        if self.worker:
            self.worker.stop()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None

        if self.session_bridge:
            try:
                self.session_bridge.requested.disconnect()
            except TypeError:
                pass
            self.session_bridge = None

        self.status_label.setText("Idle")

    @pyqtSlot(str, str)
    def _append_transcript(self, session_id: str, text: str) -> None:
        session = self.sessions.get(session_id)
        if not session or not text:
            return

        session.transcript_segments.append(text)
        cleaned = text.strip()
        if cleaned:
            if session.transcript_display and not session.transcript_display.endswith((" ", "\n")):
                session.transcript_display += " "
            session.transcript_display += cleaned
            if cleaned.endswith((".", "?", "!")):
                session.transcript_display += "\n"

        if session_id == self.current_session_id:
            self.transcript_view.setPlainText(session.transcript_text)
            self._auto_scroll(self.transcript_view)

    @pyqtSlot(str, str)
    def _register_user_message(self, session_id: str, message: str) -> None:
        session = self.sessions.get(session_id)
        if not session or not message:
            return

        session.conversation.append({"role": "user", "content": message})
        self._trim_session_history(session)

    @pyqtSlot(str, str)
    def _append_assistant(self, session_id: str, text: str) -> None:
        session = self.sessions.get(session_id)
        if not session or not text:
            return

        session.assistant_segments.append(text)
        session.conversation.append({"role": "assistant", "content": text})
        self._trim_session_history(session)

        if session_id == self.current_session_id:
            self.assistant_view.setPlainText(session.assistant_text)
            self._auto_scroll(self.assistant_view)

    @pyqtSlot(str)
    def _handle_worker_error(self, message: str) -> None:
        QMessageBox.critical(self, "Assistant Error", message)

    def _auto_scroll(self, widget: QTextEdit) -> None:
        widget.verticalScrollBar().setValue(widget.verticalScrollBar().maximum())

    def _ensure_client(self) -> AIClient:
        if not self.ai_client:
            self.settings = get_settings()
            self.ai_client = AIClient(self.settings)
            if not self.knowledge_base:
                self.knowledge_base = KnowledgeBase(self.ai_client)
        return self.ai_client

    def _create_new_session(self) -> ChatSession:
        session_id = str(uuid.uuid4())
        session = ChatSession(session_id=session_id, title=f"Session {len(self.sessions) + 1}")
        self.sessions[session_id] = session

        item = QListWidgetItem(session.title)
        item.setData(Qt.ItemDataRole.UserRole, session_id)
        self.session_list.addItem(item)
        self.session_list.setCurrentItem(item)

        self.current_session_id = session_id
        self._render_session(session)
        return session

    def _on_new_chat(self) -> None:
        self._create_new_session()

    def _on_session_selected(
        self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]
    ) -> None:
        if not current:
            return
        session_id = current.data(Qt.ItemDataRole.UserRole)
        if not session_id:
            return

        session_id = str(session_id)
        self.current_session_id = session_id
        session = self.sessions[session_id]
        self._render_session(session)

        if self.worker and self.session_bridge:
            self.session_bridge.requested.emit(
                session.session_id,
                list(session.conversation),
                session.last_transcript_segment,
            )

    def _render_session(self, session: ChatSession) -> None:
        self.transcript_view.setPlainText(session.transcript_text)
        self._auto_scroll(self.transcript_view)
        self.assistant_view.setPlainText(session.assistant_text)
        self._auto_scroll(self.assistant_view)

    def _trim_session_history(self, session: ChatSession, max_turns: int = 20) -> None:
        if len(session.conversation) <= 1:
            return
        system_message = session.conversation[0]
        remainder = session.conversation[1:]
        if len(remainder) <= max_turns * 2:
            return
        session.conversation = [system_message, *remainder[-max_turns * 2 :]]

    def _load_reference_material(self) -> None:
        try:
            self._ensure_client()
        except RuntimeError as exc:
            QMessageBox.critical(self, "Configuration error", str(exc))
            return

        start_dir = str(Path.home())
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select reference documents",
            start_dir,
            "Documents (*.txt *.md *.markdown *.pdf);;All files (*)",
        )
        if not file_paths or not self.knowledge_base:
            return

        try:
            added_chunks = self.knowledge_base.ingest_files(Path(path) for path in file_paths)
        except Exception as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            return

        if added_chunks == 0:
            QMessageBox.information(
                self,
                "No text detected",
                "We could not extract readable text from the selected files.",
            )
            return

        self._update_kb_status()
        QMessageBox.information(
            self,
            "Knowledge base updated",
            f"Loaded {added_chunks} reference snippets from {len(file_paths)} files.",
        )

    def _update_kb_status(self) -> None:
        if not self.knowledge_base or self.knowledge_base.is_empty():
            self.kb_status_label.setText("Knowledge base: empty")
            self.kb_status_label.setToolTip("")
            return

        sources = self.knowledge_base.listed_sources()
        if not sources:
            self.kb_status_label.setText("Knowledge base: ready")
            self.kb_status_label.setToolTip("")
            return

        self.kb_status_label.setText(f"Knowledge base: {len(sources)} files")
        self.kb_status_label.setToolTip("\n".join(sources))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_listening()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
