import os

from src.knowledge_base import KnowledgeBase


class DummyAIClient:
    def embed_texts(self, texts):
        keywords = ["migration", "scal", "require"]
        embeddings = []
        for text in texts:
            lower = text.lower()
            vector = [float(lower.count(word)) for word in keywords]
            embeddings.append(vector)
        return embeddings


def test_session_specific_matches(tmp_path):
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    ai_client = DummyAIClient()
    kb = KnowledgeBase(ai_client, chunk_char_length=512, chunk_overlap=0)

    session_one = "session-one"
    session_two = "session-two"

    file_one = tmp_path / "leadership.txt"
    file_one.write_text("I led a migration that saved the company money.")
    file_two = tmp_path / "scaling.txt"
    file_two.write_text("I scaled a system to handle millions of requests.")
    file_general = tmp_path / "general.txt"
    file_general.write_text("Always clarify requirements before coding.")

    added_session_one = kb.ingest_files([file_one], session_id=session_one)
    added_session_two = kb.ingest_files([file_two], session_id=session_two)
    added_general = kb.ingest_files([file_general])

    assert added_session_one > 0
    assert added_session_two > 0
    assert added_general > 0

    matches_one = kb.top_matches("migration leadership", session_id=session_one)
    assert matches_one, "Expected matches for session one"
    assert "migration" in matches_one[0].lower()

    matches_two = kb.top_matches("scaling requests", session_id=session_two)
    assert matches_two, "Expected matches for session two"
    assert "scaled" in matches_two[0].lower()

    matches_one_without_global = kb.top_matches(
        "requirements", session_id=session_one, include_global=False
    )
    assert all("clarify" not in match.lower() for match in matches_one_without_global)

    sources_one = kb.listed_sources(session_id=session_one)
    assert str(file_one) in sources_one
    assert str(file_general) in sources_one
    assert str(file_two) not in sources_one

    assert not kb.is_empty_for_session(session_one)
    assert not kb.is_empty_for_session(None)


def test_upsert_session_pair_overwrites_previous():
    ai_client = DummyAIClient()
    kb = KnowledgeBase(ai_client)

    session_id = "session-xyz"
    kb.upsert_session_pair(session_id, 1, "Tell me about migration", "I led a migration.")
    first_match = kb.top_matches("migration", session_id=session_id)
    assert first_match, "Expected the session pair to be indexed"
    assert "migration" in first_match[0].lower()

    kb.upsert_session_pair(
        session_id,
        1,
        "Tell me about migration",
        "I led a migration that scaled to millions of users.",
    )

    updated_match = kb.top_matches("scaled", session_id=session_id)
    assert updated_match, "Expected the updated session pair to be retrievable"
    assert "millions" in updated_match[0].lower()

    sources = kb.listed_sources(session_id=session_id)
    assert any("turn 1" in source for source in sources)
