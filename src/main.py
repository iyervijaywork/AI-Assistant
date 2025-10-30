from __future__ import annotations

import html
import queue
import re
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .audio_capture import MicrophoneListener
from .chatgpt_sync import ChatGPTSync, ChatGPTSyncError
from .chatgpt_share import ChatGPTShareError, ChatGPTShareImporter, SharedChat
from .config import Settings, get_settings
from .knowledge_base import KnowledgeBase
from .openai_client import AIClient, DEFAULT_SYSTEM_PROMPT
from .question_detection import QuestionBoundaryDetector


@dataclass
class PreparationProfile:
    interview_type: str = ""
    role: str = ""
    company: str = ""
    focus_areas: str = ""
    success_criteria: str = ""

    def summary_lines(self) -> List[str]:
        lines: List[str] = []
        if self.interview_type:
            lines.append(f"Interview type: {self.interview_type.strip()}")
        if self.role:
            lines.append(f"Role: {self.role.strip()}")
        if self.company:
            lines.append(f"Company: {self.company.strip()}")
        if self.focus_areas:
            focus = " ".join(self.focus_areas.split())
            lines.append(f"Focus areas: {focus}")
        if self.success_criteria:
            success = " ".join(self.success_criteria.split())
            lines.append(f"Success criteria: {success}")
        return lines

    def description(self) -> str:
        lines = self.summary_lines()
        return "\n".join(lines)


@dataclass
class QAPair:
    question: str
    answer: str = ""


@dataclass
class ChatSession:
    session_id: str
    title: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    conversation: List[dict[str, str]] = field(default_factory=list)
    transcript_segments: List[str] = field(default_factory=list)
    transcript_questions: List[str] = field(default_factory=list)
    partial_question: str = ""
    assistant_segments: List[str] = field(default_factory=list)
    assistant_display_segments: List[str] = field(default_factory=list)
    assistant_display_html_segments: List[str] = field(default_factory=list)
    qa_pairs: List[QAPair] = field(default_factory=list)
    prep_notes: str = ""
    chatgpt_conversation_id: Optional[str] = None
    chatgpt_share_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.conversation:
            self.conversation.append({"role": "system", "content": self.system_prompt})

    @property
    def transcript_text(self) -> str:
        lines: List[str] = []
        for index, question in enumerate(self.transcript_questions, start=1):
            cleaned = question.strip()
            if cleaned:
                lines.append(f"Q{index}: {cleaned}")
        if self.partial_question:
            lines.append(
                f"Q{len(self.transcript_questions) + 1} (capturing): {self.partial_question.strip()}"
            )
        return "\n\n".join(lines).strip()

    @property
    def assistant_text(self) -> str:
        return "\n\n".join(self.assistant_display_segments).strip()

    @property
    def assistant_html(self) -> str:
        return "".join(self.assistant_display_html_segments).strip()

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
        chunk_duration: float = 1.0,
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
        min_silence = max(chunk_duration * 0.6, 0.5)
        self._question_detector = QuestionBoundaryDetector(min_silence_seconds=min_silence)

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
                question_text = self._question_detector.observe(chunk, addition)

                if addition:
                    self.transcript_ready.emit(self.session_id, addition)

                if not question_text:
                    self.status_changed.emit("Listening…")
                    continue

                user_message = question_text
                self._append_conversation({"role": "user", "content": user_message})
                self.user_message_committed.emit(self.session_id, user_message)

                context: List[str] = []
                if self.knowledge_base and not self.knowledge_base.is_empty():
                    context = self.knowledge_base.top_matches(
                        user_message, session_id=self.session_id
                    )

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
        self._question_detector.reset()

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
        self.chatgpt_sync: Optional[ChatGPTSync] = None
        self.chatgpt_share_importer: Optional[ChatGPTShareImporter] = ChatGPTShareImporter()
        self.listener: Optional[MicrophoneListener] = None
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[TranscriptionWorker] = None
        self.session_bridge: Optional[SessionSwitchBridge] = None

        self.sessions: Dict[str, ChatSession] = {}
        self.chatgpt_session_map: Dict[str, str] = {}
        self.chatgpt_share_map: Dict[str, str] = {}
        self.current_session_id: Optional[str] = None
        self.prep_profile: PreparationProfile = PreparationProfile()

        self._setup_ui()
        QTimer.singleShot(0, self._post_init_setup)

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

        prep_group = QGroupBox("Interview Prep")
        prep_group.setStyleSheet(
            "QGroupBox { font-size: 16px; font-weight: 500; margin-top: 12px; }"
        )
        prep_layout = QFormLayout(prep_group)
        prep_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        prep_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.interview_type_input = QLineEdit()
        self.interview_type_input.setPlaceholderText("e.g. Behavioral, Technical")
        prep_layout.addRow("Type", self.interview_type_input)

        self.role_input = QLineEdit()
        self.role_input.setPlaceholderText("Target role title")
        prep_layout.addRow("Role", self.role_input)

        self.company_input = QLineEdit()
        self.company_input.setPlaceholderText("Company or team")
        prep_layout.addRow("Company", self.company_input)

        self.focus_input = QPlainTextEdit()
        self.focus_input.setPlaceholderText("Key focus areas or anticipated topics…")
        self.focus_input.setMaximumHeight(70)
        prep_layout.addRow("Focus", self.focus_input)

        self.success_input = QPlainTextEdit()
        self.success_input.setPlaceholderText(
            "Any success criteria, must-mention stories, or preparation prompts…"
        )
        self.success_input.setMaximumHeight(70)
        prep_layout.addRow("Success", self.success_input)

        self.apply_prep_button = QPushButton("Apply prep context")
        self.apply_prep_button.clicked.connect(self._apply_prep_context)
        prep_layout.addRow(self.apply_prep_button)

        self.prep_summary_label = QLabel("No prep context applied.")
        self.prep_summary_label.setWordWrap(True)
        self.prep_summary_label.setStyleSheet("color: #555; font-size: 12px;")
        prep_layout.addRow(self.prep_summary_label)

        sidebar_layout.addWidget(prep_group)
        self._update_prep_summary_label()

        buttons_layout = QVBoxLayout()
        new_chat_button = QPushButton("New chat")
        new_chat_button.clicked.connect(self._on_new_chat)
        buttons_layout.addWidget(new_chat_button)

        self.sync_button = QPushButton("Sync ChatGPT")
        self.sync_button.setEnabled(False)
        self.sync_button.clicked.connect(self._on_sync_chatgpt)
        buttons_layout.addWidget(self.sync_button)

        self.import_share_button = QPushButton("Import shared project…")
        self.import_share_button.clicked.connect(self._on_import_chatgpt_share)
        buttons_layout.addWidget(self.import_share_button)

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

        self.transcript_view = QTextBrowser()
        self.transcript_view.setOpenExternalLinks(True)
        self.transcript_view.setStyleSheet(
            "QTextBrowser {"
            "  background: #f7f9fc;"
            "  border: none;"
            "  border-radius: 12px;"
            "  padding: 12px;"
            "  font-size: 15px;"
            "  color: #1f2933;"
            "}"
        )

        self.assistant_view = QTextBrowser()
        self.assistant_view.setOpenExternalLinks(True)
        self.assistant_view.setStyleSheet(
            "QTextBrowser {"
            "  background: #f8fbff;"
            "  border: none;"
            "  border-radius: 12px;"
            "  padding: 12px;"
            "  font-size: 15px;"
            "  color: #1f2933;"
            "}"
        )

        splitter.addWidget(self._wrap_with_label("Live Transcript", self.transcript_view))
        splitter.addWidget(self._wrap_with_label("Assistant Response", self.assistant_view))
        splitter.setSizes([620, 620])

        content_layout.addWidget(splitter, 1)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#f0f2f5"))
        self.setPalette(palette)

    def _post_init_setup(self) -> None:
        error_message: Optional[str] = None
        try:
            self._initialize_clients()
        except Exception as exc:
            error_message = str(exc)

        if self.chatgpt_sync:
            self._refresh_chatgpt_sessions(initial=True)

        if not self.sessions:
            self._create_new_session()

        if error_message:
            QMessageBox.warning(self, "Configuration warning", error_message)

    def _wrap_with_label(self, label: str, widget: QWidget) -> QWidget:
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel(label)
        title.setStyleSheet("font-size: 18px; font-weight: 500; padding-bottom: 6px;")
        container_layout.addWidget(title)
        container_layout.addWidget(widget)
        return container

    def _update_prep_summary_label(self) -> None:
        if not hasattr(self, "prep_summary_label"):
            return
        summary = self.prep_profile.description().strip()
        if summary:
            self.prep_summary_label.setText(summary)
        else:
            self.prep_summary_label.setText("No prep context applied.")

    def _compose_system_prompt(self, profile: Optional[PreparationProfile] = None) -> str:
        profile = profile or self.prep_profile
        base_prompt = DEFAULT_SYSTEM_PROMPT
        structure_prompt = (
            "Respond as the candidate in the first person using the CARL framework:\n"
            "1. Context — describe the situation, stakeholders, and objectives in one tight paragraph.\n"
            "2. Actions — list 2-4 decisive steps you personally drove, highlighting collaboration and reasoning.\n"
            "3. Results — quantify the impact with metrics or concrete outcomes that matter to the interviewer.\n"
            "4. Learnings — close with one or two reflective insights or forward-looking adjustments.\n"
            "Stay focused, avoid filler, and mirror the interviewer's terminology when appropriate."
        )

        context_lines = profile.summary_lines()
        context_block = ""
        if context_lines:
            formatted = "\n".join(f"- {line}" for line in context_lines)
            context_block = f"Interview context:\n{formatted}"

        components = [base_prompt, structure_prompt, context_block]
        return "\n\n".join(part for part in components if part).strip()

    def _initialize_clients(self) -> None:
        if self.settings:
            return

        settings = get_settings()
        self.settings = settings
        self.ai_client = AIClient(settings)
        self.knowledge_base = KnowledgeBase(self.ai_client)
        base_origin = settings.chatgpt_base_url.split("/backend-api")[0]
        self.chatgpt_share_importer = ChatGPTShareImporter(base_url=base_origin)

        if settings.chatgpt_access_token:
            self.chatgpt_sync = ChatGPTSync(
                session_token=settings.chatgpt_access_token,
                base_url=settings.chatgpt_base_url,
                bearer_token=settings.chatgpt_bearer_token,
            )
            self.sync_button.setEnabled(True)
            self.sync_button.setToolTip("Refresh conversation history from ChatGPT")
        else:
            self.sync_button.setToolTip(
                "Set CHATGPT_ACCESS_TOKEN in your environment to enable ChatGPT syncing."
            )

    def _apply_prep_context(self) -> None:
        profile = PreparationProfile(
            interview_type=self.interview_type_input.text().strip(),
            role=self.role_input.text().strip(),
            company=self.company_input.text().strip(),
            focus_areas=self.focus_input.toPlainText().strip(),
            success_criteria=self.success_input.toPlainText().strip(),
        )
        self.prep_profile = profile
        self._update_prep_summary_label()

        if self.current_session_id:
            session = self.sessions[self.current_session_id]
            session.system_prompt = self._compose_system_prompt(profile)
            session.prep_notes = profile.description()
            if session.conversation:
                session.conversation[0] = {"role": "system", "content": session.system_prompt}
            else:
                session.conversation.append({"role": "system", "content": session.system_prompt})

            if self.worker and self.session_bridge:
                self.session_bridge.requested.emit(
                    session.session_id,
                    list(session.conversation),
                    session.last_transcript_segment,
                )

            if session.prep_notes:
                self.prep_summary_label.setText(session.prep_notes)

        self.status_label.setText("Prep context applied")

    def _refresh_chatgpt_sessions(self, initial: bool = False) -> None:
        if not self.chatgpt_sync or not self.settings:
            if not initial:
                QMessageBox.information(
                    self,
                    "ChatGPT sync unavailable",
                    "Provide a CHATGPT_ACCESS_TOKEN in your environment to import chats.",
                )
            return

        try:
            conversations = self.chatgpt_sync.list_conversations(
                limit=self.settings.chatgpt_sync_limit
            )
        except ChatGPTSyncError as exc:
            if not initial:
                QMessageBox.warning(self, "ChatGPT sync failed", str(exc))
            return

        imported_new = False
        for conversation in conversations:
            try:
                messages = self.chatgpt_sync.fetch_messages(conversation.conversation_id)
            except ChatGPTSyncError as exc:
                if not initial:
                    QMessageBox.warning(self, "ChatGPT sync failed", str(exc))
                continue

            existing_session_id = self.chatgpt_session_map.get(conversation.conversation_id)
            if existing_session_id and existing_session_id in self.sessions:
                session = self.sessions[existing_session_id]
                self._apply_messages_to_session(session, messages, conversation.title)
                self._update_session_item_title(existing_session_id, session.title)
                if existing_session_id == self.current_session_id:
                    self._render_session(session)
            else:
                session = self._create_session_from_chatgpt(
                    conversation.conversation_id, conversation.title, messages
                )
                if session:
                    imported_new = True

        if imported_new and not initial:
            QMessageBox.information(
                self,
                "ChatGPT sync",
                "Imported the latest conversations from your ChatGPT account.",
            )

    def _on_import_chatgpt_share(self) -> None:
        url, accepted = QInputDialog.getText(
            self,
            "Import ChatGPT shared project",
            "Paste the ChatGPT share link:",
        )
        if not accepted or not url.strip():
            return

        importer = self.chatgpt_share_importer or ChatGPTShareImporter()
        try:
            shared = importer.fetch(url.strip())
        except ChatGPTShareError as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            return

        self._import_shared_chat(shared)

    def _import_shared_chat(self, shared: SharedChat) -> None:
        existing_session_id = self.chatgpt_share_map.get(shared.share_id)
        if existing_session_id and existing_session_id in self.sessions:
            session = self.sessions[existing_session_id]
            self._apply_messages_to_session(session, shared.messages, shared.title)
            session.chatgpt_share_id = shared.share_id
            self._update_session_item_title(session.session_id, session.title)
            self._select_session_in_list(session.session_id)
            QMessageBox.information(
                self,
                "Shared project refreshed",
                "Updated the imported ChatGPT project with the latest content.",
            )
            return

        system_message = (
            dict(self.ai_client.system_message)
            if self.ai_client
            else {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        )
        session = ChatSession(
            session_id=str(uuid.uuid4()),
            title=shared.title or "Shared ChatGPT project",
            system_prompt=system_message.get("content", DEFAULT_SYSTEM_PROMPT),
            chatgpt_share_id=shared.share_id,
        )
        self._apply_messages_to_session(session, shared.messages, session.title)
        self.chatgpt_share_map[shared.share_id] = session.session_id
        self._register_session(session, make_current=True)
        QMessageBox.information(
            self,
            "Shared project imported",
            "Loaded the shared ChatGPT project into a new interview session.",
        )

    def _select_session_in_list(self, session_id: str) -> None:
        for index in range(self.session_list.count()):
            item = self.session_list.item(index)
            if not item:
                continue
            item_id = item.data(Qt.ItemDataRole.UserRole)
            if item_id and str(item_id) == session_id:
                self.session_list.setCurrentItem(item)
                break

    def _apply_messages_to_session(
        self, session: ChatSession, messages: List[dict[str, str]], title: str
    ) -> None:
        session.title = title or session.title
        system_message = (
            dict(self.ai_client.system_message)
            if self.ai_client
            else {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        )
        session.system_prompt = system_message.get("content", DEFAULT_SYSTEM_PROMPT)
        session.conversation = [system_message]
        session.transcript_segments = []
        session.transcript_questions = []
        session.partial_question = ""
        session.assistant_segments = []
        session.assistant_display_segments = []
        session.qa_pairs = []
        session.prep_notes = ""

        for message in messages:
            role = message.get("role", "user")
            content = (message.get("content") or "").strip()
            if not content:
                continue

            session.conversation.append({"role": role, "content": content})
            if role == "assistant":
                session.assistant_segments.append(content)
                if session.qa_pairs:
                    session.qa_pairs[-1].answer = content
                    formatted_text, formatted_html = self._format_structured_answer_assets(
                        len(session.qa_pairs), session.qa_pairs[-1].question, content
                    )
                    session.assistant_display_segments.append(formatted_text)
                    session.assistant_display_html_segments.append(formatted_html)
                else:
                    session.assistant_display_segments.append(content)
                    fallback_sections = {
                        "context": [content],
                        "actions": [],
                        "results": [],
                        "learnings": [],
                    }
                    session.assistant_display_html_segments.append(
                        self._assistant_card_html(
                            len(session.assistant_display_segments),
                            "Assistant Response",
                            fallback_sections,
                        )
                    )
            else:
                session.transcript_segments.append(content)
                session.transcript_questions.append(content)
                session.qa_pairs.append(QAPair(question=content))

    def _create_session_from_chatgpt(
        self,
        conversation_id: str,
        title: str,
        messages: List[dict[str, str]],
    ) -> Optional[ChatSession]:
        session = ChatSession(
            session_id=str(uuid.uuid4()),
            title=title or "ChatGPT chat",
            system_prompt=self.ai_client.system_message.get("content", DEFAULT_SYSTEM_PROMPT)
            if self.ai_client
            else DEFAULT_SYSTEM_PROMPT,
            chatgpt_conversation_id=conversation_id,
        )
        self._apply_messages_to_session(session, messages, session.title)
        self.chatgpt_session_map[conversation_id] = session.session_id
        self._register_session(session, make_current=False)
        return session

    def _update_session_item_title(self, session_id: str, title: str) -> None:
        for index in range(self.session_list.count()):
            item = self.session_list.item(index)
            item_id = item.data(Qt.ItemDataRole.UserRole)
            if item_id and str(item_id) == session_id:
                if item.text() != title:
                    item.setText(title)
                break

    def _on_sync_chatgpt(self) -> None:
        self._refresh_chatgpt_sessions()

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
            chunk_duration=self.settings.chunk_duration,
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
            if session.partial_question:
                session.partial_question = f"{session.partial_question} {cleaned}".strip()
            else:
                session.partial_question = cleaned

        if session_id == self.current_session_id:
            self._render_transcript_view(session)

    @pyqtSlot(str, str)
    def _register_user_message(self, session_id: str, message: str) -> None:
        session = self.sessions.get(session_id)
        if not session or not message:
            return

        session.conversation.append({"role": "user", "content": message})
        question = message.strip()
        if question:
            session.transcript_questions.append(question)
            session.qa_pairs.append(QAPair(question=question))
        session.partial_question = ""
        self._trim_session_history(session)

        if session_id == self.current_session_id:
            self._render_transcript_view(session)

    @pyqtSlot(str, str)
    def _append_assistant(self, session_id: str, text: str) -> None:
        session = self.sessions.get(session_id)
        if not session or not text:
            return

        session.assistant_segments.append(text)
        session.conversation.append({"role": "assistant", "content": text})

        cleaned = text.strip()
        if session.qa_pairs:
            session.qa_pairs[-1].answer = cleaned
            response_index = len(session.qa_pairs)
            question_text = session.qa_pairs[-1].question
        else:
            response_index = len(session.assistant_display_segments) + 1
            question_text = "Assistant Response"

        formatted_text, formatted_html = self._format_structured_answer_assets(
            response_index, question_text, text
        )
        session.assistant_display_segments.append(formatted_text)
        session.assistant_display_html_segments.append(formatted_html)
        self._trim_session_history(session)

        if session_id == self.current_session_id:
            self._render_assistant_view(session)

    @pyqtSlot(str)
    def _handle_worker_error(self, message: str) -> None:
        QMessageBox.critical(self, "Assistant Error", message)

    def _auto_scroll(self, widget: QTextBrowser) -> None:
        widget.verticalScrollBar().setValue(widget.verticalScrollBar().maximum())

    def _render_transcript_view(self, session: ChatSession) -> None:
        self.transcript_view.setHtml(self._build_transcript_html(session))
        self._auto_scroll(self.transcript_view)

    def _render_assistant_view(self, session: ChatSession) -> None:
        self.assistant_view.setHtml(self._build_assistant_html(session))
        self._auto_scroll(self.assistant_view)

    def _build_transcript_html(self, session: ChatSession) -> str:
        cards: List[str] = [
            self._transcript_card_html(index, question)
            for index, question in enumerate(session.transcript_questions, start=1)
            if question.strip()
        ]
        if session.partial_question.strip():
            cards.append(
                self._transcript_card_html(
                    len(session.transcript_questions) + 1,
                    session.partial_question,
                    capturing=True,
                )
            )
        if not cards:
            cards.append(
                self._placeholder_card_html(
                    title="Waiting for the interviewer…",
                    subtitle="Live questions will stream in as soon as we hear them through the microphone.",
                    column_class="transcript",
                )
            )
        return self._styled_html_document(cards, column_class="transcript")

    def _build_assistant_html(self, session: ChatSession) -> str:
        cards = list(session.assistant_display_html_segments)
        if not cards:
            cards.append(
                self._placeholder_card_html(
                    title="Your interview coach is ready.",
                    subtitle=(
                        "Structured answers, evidence, and follow-ups will appear here as soon as a"
                        " full question has been captured."
                    ),
                    column_class="assistant",
                )
            )
        return self._styled_html_document(cards, column_class="assistant")

    def _transcript_card_html(self, index: int, question: str, capturing: bool = False) -> str:
        clean_question = html.escape(question.strip()) if question.strip() else "Listening…"
        capture_badge = (
            "<span class='badge badge-capturing'>capturing</span>" if capturing else ""
        )
        card_class = "card transcript-card capturing" if capturing else "card transcript-card"
        return (
            f"<div class='{card_class}'>"
            f"  <div class='card-title'>Question {index} {capture_badge}</div>"
            f"  <div class='card-content'>{clean_question}</div>"
            "</div>"
        )

    def _assistant_card_html(
        self, index: int, question: str, sections: Dict[str, List[str]]
    ) -> str:
        safe_question = html.escape(question.strip()) if question.strip() else "Interview question"
        context_html = self._format_section_html(sections["context"], bullet=False)
        actions_html = self._format_section_html(sections["actions"], bullet=True)
        results_html = self._format_section_html(sections["results"], bullet=True)
        learnings_html = self._format_section_html(sections["learnings"], bullet=True)
        return (
            "<div class='card assistant-card'>"
            f"  <div class='card-title'>Q{index} · {safe_question}</div>"
            "  <div class='assistant-section'>"
            "    <h4>Context</h4>"
            f"    {context_html}"
            "  </div>"
            "  <div class='assistant-section'>"
            "    <h4>Actions</h4>"
            f"    {actions_html}"
            "  </div>"
            "  <div class='assistant-section'>"
            "    <h4>Results</h4>"
            f"    {results_html}"
            "  </div>"
            "  <div class='assistant-section'>"
            "    <h4>Learnings</h4>"
            f"    {learnings_html}"
            "  </div>"
            "</div>"
        )

    def _placeholder_card_html(self, title: str, subtitle: str, column_class: str) -> str:
        safe_title = html.escape(title)
        safe_subtitle = html.escape(subtitle)
        return (
            f"<div class='card placeholder-card {column_class}'>"
            f"  <div class='card-title'>{safe_title}</div>"
            f"  <div class='card-content muted'>{safe_subtitle}</div>"
            "</div>"
        )

    def _styled_html_document(self, cards: List[str], column_class: str) -> str:
        style = """
        <style>
            body {
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
                color: #1f2933;
            }
            .column {
                display: flex;
                flex-direction: column;
                gap: 14px;
            }
            .card {
                background: #ffffff;
                border-radius: 14px;
                padding: 16px 18px;
                box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
            }
            .transcript-card {
                border-left: 4px solid #2563eb;
            }
            .transcript-card.capturing {
                border-style: dashed;
                border-color: #f59e0b;
                background: #fff7ed;
            }
            .assistant-card {
                border-left: 4px solid #0f766e;
            }
            .placeholder-card {
                border: 2px dashed #cbd2d9;
                background: #f9fafb;
                color: #52606d;
            }
            .card-title {
                font-size: 14px;
                font-weight: 600;
                letter-spacing: 0.02em;
                margin-bottom: 8px;
                text-transform: uppercase;
                color: #334155;
            }
            .card-content {
                font-size: 16px;
                line-height: 1.5;
            }
            .card-content.muted {
                color: #64748b;
            }
            .assistant-section {
                margin-top: 12px;
            }
            .assistant-section h4 {
                margin: 0 0 6px 0;
                font-size: 15px;
                font-weight: 600;
                color: #0f172a;
            }
            .assistant-section ul {
                margin: 0;
                padding-left: 18px;
            }
            .assistant-section li {
                margin-bottom: 4px;
            }
            .assistant-section p {
                margin: 0;
            }
            .badge {
                display: inline-block;
                padding: 2px 8px;
                margin-left: 8px;
                border-radius: 999px;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                background: #e0f2fe;
                color: #0369a1;
            }
            .badge-capturing {
                background: #fef3c7;
                color: #92400e;
            }
            .section-empty {
                color: #94a3b8;
                font-style: italic;
            }
        </style>
        """
        content = "".join(cards)
        return (
            "<html><head>"
            f"{style}"
            "</head><body>"
            f"  <div class='column column-{column_class}'>"
            f"    {content}"
            "  </div>"
            "</body></html>"
        )

    def _format_structured_answer(self, index: int, question: str, answer: str) -> str:
        formatted_text, _ = self._format_structured_answer_assets(index, question, answer)
        return formatted_text

    def _format_structured_answer_assets(
        self, index: int, question: str, answer: str
    ) -> tuple[str, str]:
        sections = self._parse_structured_sections(answer)
        heading = f"Q{index}: {question.strip()}" if question.strip() else f"Q{index}"
        divider = "-" * min(len(heading), 80)
        lines: List[str] = [heading, divider, "Context:"]
        lines.extend(self._format_section_lines(sections["context"], bullet=False))
        lines.append("")
        lines.append("Actions:")
        lines.extend(self._format_section_lines(sections["actions"], bullet=True))
        lines.append("")
        lines.append("Results:")
        lines.extend(self._format_section_lines(sections["results"], bullet=True))
        lines.append("")
        lines.append("Learnings:")
        lines.extend(self._format_section_lines(sections["learnings"], bullet=True))
        text_answer = "\n".join(lines).strip()
        html_answer = self._assistant_card_html(index, question, sections)
        return text_answer, html_answer

    def _parse_structured_sections(self, answer: str) -> Dict[str, List[str]]:
        keys = {"context": [], "actions": [], "results": [], "learnings": []}
        if not answer.strip():
            return keys

        section_aliases = {
            "context": ["context", "situation", "background"],
            "actions": ["actions", "steps", "what i did", "approach"],
            "results": ["results", "outcome", "impact", "metrics"],
            "learnings": ["learnings", "lessons", "takeaways", "insights"],
        }

        current_key = "context"
        for raw_line in answer.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            normalised = re.sub(r"[^a-z0-9]+", " ", line.lower()).strip()
            matched_key = None
            for key, aliases in section_aliases.items():
                if any(normalised.startswith(alias) for alias in aliases):
                    matched_key = key
                    break
            if matched_key:
                current_key = matched_key
                continue
            keys.setdefault(current_key, []).append(line)

        return keys

    def _format_section_lines(self, lines: List[str], bullet: bool) -> List[str]:
        cleaned = self._normalise_section_lines(lines)
        if not cleaned:
            return ["(not captured)"]
        if bullet:
            return [f"- {line}" for line in cleaned]
        return cleaned

    def _format_section_html(self, lines: List[str], bullet: bool) -> str:
        cleaned = self._normalise_section_lines(lines)
        if not cleaned:
            return "<div class='section-empty'>(not captured)</div>"
        if bullet:
            items = "".join(f"<li>{html.escape(item)}</li>" for item in cleaned)
            return f"<ul>{items}</ul>"
        paragraphs = "".join(f"<p>{html.escape(item)}</p>" for item in cleaned)
        return paragraphs or "<div class='section-empty'>(not captured)</div>"

    def _normalise_section_lines(self, lines: List[str]) -> List[str]:
        normalised: List[str] = []
        for line in lines:
            cleaned = re.sub(r"^[\-•\d\.\)\s]+", "", line).strip()
            if cleaned:
                normalised.append(cleaned)
        return normalised

    def _ensure_client(self) -> AIClient:
        if not self.ai_client:
            self._initialize_clients()
        if not self.ai_client:
            raise RuntimeError("OpenAI client is not configured.")
        return self.ai_client

    def _create_new_session(self, title: Optional[str] = None) -> ChatSession:
        system_prompt = self._compose_system_prompt()
        session = ChatSession(
            session_id=str(uuid.uuid4()),
            title=title or f"Session {len(self.sessions) + 1}",
            system_prompt=system_prompt,
            prep_notes=self.prep_profile.description(),
        )
        self._register_session(session, make_current=True)
        return session

    def _register_session(self, session: ChatSession, make_current: bool = False) -> None:
        self.sessions[session.session_id] = session

        item = QListWidgetItem(session.title)
        item.setData(Qt.ItemDataRole.UserRole, session.session_id)
        self.session_list.addItem(item)

        if make_current or self.session_list.count() == 1:
            self.session_list.setCurrentItem(item)
            self.current_session_id = session.session_id
            self._render_session(session)

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
        self._render_transcript_view(session)
        self._render_assistant_view(session)
        if session.prep_notes:
            self.prep_summary_label.setText(session.prep_notes)
        else:
            self._update_prep_summary_label()
        self._update_kb_status(session.session_id)

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

        session = self.sessions.get(self.current_session_id) if self.current_session_id else None
        session_id = session.session_id if session else None

        try:
            added_chunks = self.knowledge_base.ingest_files(
                (Path(path) for path in file_paths), session_id=session_id
            )
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

        self._update_kb_status(session_id)
        scope_message = (
            "for this interview session."
            if session_id
            else "available to every session."
        )
        QMessageBox.information(
            self,
            "Knowledge base updated",
            (
                "Loaded {chunks} reference snippets from {files} files "
                f"{scope_message}"
            ).format(chunks=added_chunks, files=len(file_paths)),
        )

    def _update_kb_status(self, session_id: Optional[str] = None) -> None:
        if not self.knowledge_base or self.knowledge_base.is_empty():
            self.kb_status_label.setText("Knowledge base: empty")
            self.kb_status_label.setToolTip("")
            return

        sources = self.knowledge_base.listed_sources(session_id=session_id)
        if not sources:
            self.kb_status_label.setText("Knowledge base: ready")
            self.kb_status_label.setToolTip("")
            return

        self.kb_status_label.setText(
            f"Knowledge base: {len(sources)} files for this session"
        )
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
