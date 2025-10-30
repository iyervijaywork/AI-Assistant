"""Helpers for importing shared ChatGPT projects without direct account access."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests


class ChatGPTShareError(RuntimeError):
    """Raised when importing a shared ChatGPT project fails."""


@dataclass
class SharedChat:
    share_id: str
    title: str
    messages: List[Dict[str, str]]


class ChatGPTShareImporter:
    """Fetch conversations from a ChatGPT share link."""

    def __init__(self, base_url: str = "https://chat.openai.com", timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json, text/html",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
            }
        )

    def fetch(self, share_url: str) -> SharedChat:
        share_id = self._extract_share_id(share_url)
        if not share_id:
            raise ChatGPTShareError("The provided link does not look like a ChatGPT share URL.")

        payload = self._fetch_share_payload(share_id)
        mapping: Dict[str, Dict[str, object]] = payload.get("mapping", {})
        messages = self._parse_mapping(mapping)
        title = payload.get("title") or "Shared ChatGPT project"
        return SharedChat(share_id=share_id, title=title, messages=messages)

    def _fetch_share_payload(self, share_id: str) -> Dict[str, object]:
        endpoints = [
            f"{self.base_url}/backend-api/share/{share_id}",
            f"{self.base_url}/backend-api/share/{share_id}/conversation",
        ]
        last_error: Optional[str] = None
        for endpoint in endpoints:
            response = self.session.get(endpoint, timeout=self.timeout)
            if response.status_code == 404:
                last_error = "The shared project could not be found."
                continue
            if "application/json" not in response.headers.get("Content-Type", ""):
                try:
                    data = self._extract_payload_from_html(response.text)
                except ChatGPTShareError as exc:
                    last_error = str(exc)
                    continue
                else:
                    return data
            try:
                return response.json()
            except ValueError as exc:
                last_error = f"Unexpected response while parsing shared project: {exc}"
                continue
        raise ChatGPTShareError(
            last_error
            or "Unable to import the shared project. Ensure the link is public and try again."
        )

    def _extract_payload_from_html(self, html_text: str) -> Dict[str, object]:
        marker = "__NEXT_DATA__"
        if marker not in html_text:
            raise ChatGPTShareError(
                "The shared page did not include the expected payload. Download the JSON export "
                "from ChatGPT and import that file instead."
            )
        start_token = '<script id="__NEXT_DATA__" type="application/json">'
        start_index = html_text.find(start_token)
        if start_index == -1:
            raise ChatGPTShareError("Unable to locate share metadata within the page.")
        start_index += len(start_token)
        end_index = html_text.find("</script>", start_index)
        if end_index == -1:
            raise ChatGPTShareError("The shared project metadata was truncated.")
        raw = html_text[start_index:end_index]
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ChatGPTShareError("Unable to decode the shared project metadata.") from exc
        data = payload.get("props", {}).get("pageProps", {}).get("serverResponse", {})
        if not data:
            raise ChatGPTShareError("The shared project metadata was empty.")
        return data

    def _parse_mapping(self, mapping: Dict[str, Dict[str, object]]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        for node in mapping.values():
            message = node.get("message") if isinstance(node, dict) else None
            if not isinstance(message, dict):
                continue
            author = message.get("author", {})
            role = "user"
            if isinstance(author, dict) and author.get("role") == "assistant":
                role = "assistant"
            content = message.get("content", {})
            parts: Iterable[str] = []
            if isinstance(content, dict):
                raw_parts = content.get("parts")
                if isinstance(raw_parts, list):
                    parts = [part for part in raw_parts if isinstance(part, str)]
            text = "\n\n".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
            if not text:
                continue
            created = message.get("create_time") or 0
            messages.append({"role": role, "content": text, "create_time": created})
        messages.sort(key=lambda item: item.get("create_time", 0))
        ordered = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        return ordered

    def _extract_share_id(self, share_url: str) -> Optional[str]:
        parsed = urlparse(share_url.strip())
        path = parsed.path.strip("/")
        if not path:
            return None
        parts = [segment for segment in path.split("/") if segment]
        if not parts:
            return None
        if parts[0] != "share":
            return None
        return parts[-1]
