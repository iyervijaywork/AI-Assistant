import pytest
from urllib.parse import urlparse

from src.chatgpt_share import ChatGPTShareError, ChatGPTShareImporter


UUID = "12345678-1234-1234-1234-1234567890ab"


def test_extract_share_id_from_standard_share_link():
    importer = ChatGPTShareImporter()
    parsed = urlparse(f"https://chatgpt.com/share/{UUID}")
    assert importer._extract_share_id(parsed) == UUID


def test_extract_share_id_from_nested_link_with_uuid():
    importer = ChatGPTShareImporter()
    parsed = urlparse(f"https://chat.openai.com/g/project/{UUID}?foo=bar")
    assert importer._extract_share_id(parsed) == UUID


def test_extract_share_id_from_query_parameter():
    importer = ChatGPTShareImporter()
    parsed = urlparse(f"https://chat.openai.com/project?shareId={UUID}")
    assert importer._extract_share_id(parsed) == UUID


def test_fetch_uses_share_domain(monkeypatch):
    importer = ChatGPTShareImporter()
    captured = {}

    def fake_fetch(self, share_id, *, base_url=None):
        captured["share_id"] = share_id
        captured["base_url"] = base_url
        return {"mapping": {}, "title": "Shared"}

    monkeypatch.setattr(ChatGPTShareImporter, "_fetch_share_payload", fake_fetch)

    shared = importer.fetch(f"https://chatgpt.com/share/{UUID}")

    assert captured["share_id"] == UUID
    assert captured["base_url"] == "https://chatgpt.com"
    assert shared.title == "Shared"
    assert shared.messages == []


def test_fetch_rejects_invalid_url():
    importer = ChatGPTShareImporter()
    with pytest.raises(ChatGPTShareError):
        importer.fetch("not-a-valid-share-link")
