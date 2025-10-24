from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import requests


class ChatGPTSyncError(RuntimeError):
    """Raised when synchronising with the ChatGPT web API fails."""


@dataclass
class ChatGPTConversation:
    conversation_id: str
    title: str


class ChatGPTSync:
    """Thin wrapper around the ChatGPT web API used by chat.openai.com."""

    def __init__(
        self,
        access_token: str,
        base_url: str = "https://chat.openai.com/backend-api",
        timeout: float = 15.0,
    ) -> None:
        if not access_token:
            raise ValueError("A ChatGPT access token is required for synchronisation.")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def list_conversations(self, limit: int = 12) -> List[ChatGPTConversation]:
        """Return the most recent ChatGPT conversations."""

        endpoint = f"{self.base_url}/conversations"
        response = self.session.get(
            endpoint,
            params={"offset": 0, "limit": max(limit, 1)},
            timeout=self.timeout,
        )
        self._raise_for_status(response)
        payload = response.json() if response.content else {}
        items = payload.get("items", [])
        conversations: List[ChatGPTConversation] = []
        for item in items:
            conversation_id = item.get("id")
            if not conversation_id:
                continue
            title = item.get("title") or "Untitled chat"
            conversations.append(ChatGPTConversation(conversation_id=conversation_id, title=title))
        return conversations

    def fetch_messages(self, conversation_id: str) -> List[Dict[str, str]]:
        """Return ordered messages for a specific ChatGPT conversation."""

        endpoint = f"{self.base_url}/conversation/{conversation_id}"
        response = self.session.get(endpoint, timeout=self.timeout)
        self._raise_for_status(response)
        payload = response.json() if response.content else {}
        mapping: Dict[str, Dict[str, object]] = payload.get("mapping", {})
        messages: List[Dict[str, object]] = []

        for node in mapping.values():
            message = node.get("message") if isinstance(node, dict) else None
            if not isinstance(message, dict):
                continue
            author = message.get("author", {})
            role = "user"
            if isinstance(author, dict):
                role = "assistant" if author.get("role") == "assistant" else "user"
            content = message.get("content", {})
            parts: Iterable[str] = []
            if isinstance(content, dict):
                raw_parts = content.get("parts")
                if isinstance(raw_parts, list):
                    parts = [part for part in raw_parts if isinstance(part, str)]
            text = "\n\n".join(part.strip() for part in parts if part.strip())
            if not text:
                continue
            created = message.get("create_time")
            messages.append({"role": role, "content": text, "create_time": created})

        messages.sort(key=lambda msg: msg.get("create_time") or 0)
        ordered = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        return ordered

    def _raise_for_status(self, response: requests.Response) -> None:
        if response.ok:
            return
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if payload and isinstance(payload, dict):
            message = payload.get("detail") or payload.get("message")
        else:
            message = response.text
        raise ChatGPTSyncError(message or f"ChatGPT API call failed with status {response.status_code}")


__all__ = ["ChatGPTSync", "ChatGPTConversation", "ChatGPTSyncError"]
