import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

try:
    from PyQt6 import QtWidgets
except ImportError as exc:  # pragma: no cover - skip if Qt dependencies missing
    import pytest

    pytest.skip(f"PyQt6 unavailable: {exc}", allow_module_level=True)

QApplication = QtWidgets.QApplication

try:
    from src.main import MainWindow
except Exception as exc:  # pragma: no cover - skip if GUI deps missing
    import pytest

    pytest.skip(f"MainWindow unavailable: {exc}", allow_module_level=True)


def _get_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_transcript_and_response_flow():
    app = _get_app()
    window = MainWindow()

    # Allow any deferred setup to run
    app.processEvents()

    assert window.current_session_id is not None
    session_id = window.current_session_id
    session = window.sessions[session_id]

    transcript_fragment = "Tell me about a time you led a team"
    window._append_transcript(session_id, transcript_fragment)
    assert transcript_fragment.lower() in window.transcript_view.toHtml().lower()

    question = "Tell me about a time you led a team to deliver under pressure."
    window._register_user_message(session_id, question)
    assert session.transcript_questions[-1] == question
    assert session.qa_pairs[-1].question == question

    answer = (
        "Context: I led our data migration under a four-week deadline.\n"
        "Actions:\n"
        "- I created a war room with engineering, QA, and analytics to unblock issues daily.\n"
        "- I negotiated phased rollouts with the business to keep customer impact at zero.\n"
        "Results:\n"
        "- We migrated 42 services with zero severity-one incidents and hit our launch date.\n"
        "Learnings:\n"
        "- I would automate more validation to catch regressions faster next time."
    )
    window._append_assistant(session_id, answer)

    assert session.qa_pairs[-1].answer.startswith("Context")
    assert "Q1" in session.assistant_html
    assert "I led" in session.assistant_segments[-1]
    assert "context" in window.assistant_view.toHtml().lower()

    kb = window.knowledge_base
    assert kb is not None
    matches = kb.top_matches("war room", session_id=session_id)
    assert matches and "war room" in matches[0].lower()

    window.close()
