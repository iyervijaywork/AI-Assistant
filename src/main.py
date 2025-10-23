from __future__ import annotations

import queue
import sys
import threading
from typing import Optional

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
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
from .openai_client import AIClient


class TranscriptionWorker(QObject):
    transcript_ready = pyqtSignal(str)
    assistant_ready = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        audio_queue: "queue.Queue",
        ai_client: AIClient,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.audio_queue = audio_queue
        self.ai_client = ai_client
        self._stop_event = threading.Event()

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
                if transcript:
                    self.transcript_ready.emit(transcript)
                    self.status_changed.emit("Thinking…")
                    response = self.ai_client.chat_completion(transcript)
                    if response:
                        self.assistant_ready.emit(response)
                else:
                    self.status_changed.emit("No speech detected")
            except Exception as exc:  # pragma: no cover - runtime safeguard
                self.error_occurred.emit(str(exc))
                self.status_changed.emit("Error")

        self.status_changed.emit("Idle")

    def stop(self) -> None:
        self._stop_event.set()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Live Conversation Assistant")
        self.resize(1200, 700)

        self.settings: Optional[Settings] = None
        self.listener: Optional[MicrophoneListener] = None
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[TranscriptionWorker] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

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

        layout.addLayout(header_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setPlaceholderText("Real-time transcription will appear here…")
        self.transcript_view.setStyleSheet(
            "font-size: 16px; padding: 12px; background: #fcfcfc; border-radius: 8px;"
        )

        self.assistant_view = QTextEdit()
        self.assistant_view.setReadOnly(True)
        self.assistant_view.setPlaceholderText("Assistant insights and responses will show here…")
        self.assistant_view.setStyleSheet(
            "font-size: 16px; padding: 12px; background: #f7fbff; border-radius: 8px;"
        )

        splitter.addWidget(self._wrap_with_label("Live Transcript", self.transcript_view))
        splitter.addWidget(self._wrap_with_label("Assistant Response", self.assistant_view))
        splitter.setSizes([600, 600])

        layout.addWidget(splitter, 1)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#f0f2f5"))
        self.setPalette(palette)

    def _wrap_with_label(self, label: str, widget: QWidget) -> QWidget:
        container = QWidget()
        container_layout = QVBoxLayout(container)
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

        self.settings = get_settings()
        self.listener = MicrophoneListener(
            sample_rate=self.settings.sample_rate,
            chunk_duration=self.settings.chunk_duration,
        )

        ai_client = AIClient(self.settings)
        audio_queue = self.listener.get_queue()

        self.worker_thread = QThread()
        self.worker = TranscriptionWorker(audio_queue, ai_client)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.transcript_ready.connect(self._append_transcript)
        self.worker.assistant_ready.connect(self._append_assistant)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.error_occurred.connect(self._handle_worker_error)
        self.worker_thread.start()

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

        self.status_label.setText("Idle")

    @pyqtSlot(str)
    def _append_transcript(self, text: str) -> None:
        self.transcript_view.append(text)
        self.transcript_view.verticalScrollBar().setValue(
            self.transcript_view.verticalScrollBar().maximum()
        )

    @pyqtSlot(str)
    def _append_assistant(self, text: str) -> None:
        self.assistant_view.append(text)
        self.assistant_view.verticalScrollBar().setValue(
            self.assistant_view.verticalScrollBar().maximum()
        )

    @pyqtSlot(str)
    def _handle_worker_error(self, message: str) -> None:
        QMessageBox.critical(self, "Assistant Error", message)

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
