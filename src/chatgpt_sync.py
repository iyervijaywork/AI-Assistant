from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import time

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
        session_token: str,
        base_url: str = "https://chat.openai.com/backend-api",
        timeout: float = 15.0,
        bearer_token: Optional[str] = None,
    ) -> None:
        if not session_token:
            raise ValueError("A ChatGPT session token is required for synchronisation.")

        self.base_url = base_url.rstrip("/")
        self.origin = "https://chat.openai.com"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Referer": "https://chat.openai.com/",
                "Origin": "https://chat.openai.com",
            }
        )
        # The ChatGPT web backend expects the session cookie when minting a bearer token.
        # Provide the value copied from the browser's `__Secure-next-auth.session-token`.
        self.session.cookies.set(
            "__Secure-next-auth.session-token",
            session_token,
            domain="chat.openai.com",
            secure=True,
        )
        self._access_token: Optional[str] = bearer_token
        self._access_token_expires_at: Optional[float] = None
        self._manual_bearer = bearer_token is not None
        self._token_unavailable = False

    def list_conversations(self, limit: int = 12) -> List[ChatGPTConversation]:
        """Return the most recent ChatGPT conversations."""

        endpoint = f"{self.base_url}/conversations"
        response = self._authorized_get(
            endpoint,
            params={"offset": 0, "limit": max(limit, 1)},
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
        response = self._authorized_get(endpoint)
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
        if response.ok and "application/json" in response.headers.get("Content-Type", ""):
            return
        # ChatGPT sometimes serves an HTML challenge page that asks the user to
        # enable JavaScript and cookies. Surface a clearer hint in that case so
        # users know to refresh their session token from the browser.
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ChatGPTSyncError(
                "Received an HTML challenge page from ChatGPT. "
                "Verify that your session cookie is valid and that you've copied "
                "the `__Secure-next-auth.session-token` value from an active browser session."
            )
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if payload and isinstance(payload, dict):
            message = payload.get("detail") or payload.get("message")
        else:
            message = response.text
        raise ChatGPTSyncError(message or f"ChatGPT API call failed with status {response.status_code}")

    def _authorized_get(self, endpoint: str, **kwargs) -> requests.Response:
        """Issue a GET request with a valid ChatGPT bearer token."""

        self._ensure_access_token()
        headers = kwargs.pop("headers", {}) or {}
        if self._access_token:
            headers.setdefault("Authorization", f"Bearer {self._access_token}")
        else:
            headers.pop("Authorization", None)
        response = self.session.get(
            endpoint,
            headers=headers,
            timeout=self.timeout,
            **kwargs,
        )
        if response.status_code == 401 and self._access_token:
            # The access token likely expired. Refresh and retry once.
            self._access_token = None
            self._access_token_expires_at = None
            self._ensure_access_token()
            if not self._access_token:
                return response
            headers["Authorization"] = f"Bearer {self._access_token}"
            response = self.session.get(
                endpoint,
                headers=headers,
                timeout=self.timeout,
                **kwargs,
            )
        elif response.status_code == 401:
            raise ChatGPTSyncError(
                "ChatGPT rejected the request using only the session cookie. "
                "Provide a valid bearer token via CHATGPT_BEARER_TOKEN or refresh the session."
            )
        return response

    def _ensure_access_token(self) -> None:
        """Retrieve and cache a bearer token for ChatGPT web requests."""

        if self._access_token and self._token_is_fresh():
            return
        if self._manual_bearer or self._token_unavailable:
            return

        session_endpoint = f"{self.origin}/api/auth/session"
        response = self.session.get(session_endpoint, timeout=self.timeout)
        if response.status_code in {401, 403}:
            raise ChatGPTSyncError(
                "ChatGPT rejected the provided session cookie. "
                "Refresh the `__Secure-next-auth.session-token` value from an active browser session."
            )
        self._raise_for_status(response)

        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - defensive parsing guard
            raise ChatGPTSyncError(
                "Unexpected response while retrieving ChatGPT access token."
            ) from exc

        token = None
        if isinstance(payload, dict):
            token = payload.get("accessToken") or payload.get("access_token")
        if not token:
            # Some accounts no longer expose an access token via the session
            # endpoint. Fall back to cookie-only requests and surface clearer
            # guidance if those requests later fail.
            self._token_unavailable = True
            self._access_token = None
            self._access_token_expires_at = None
            return

        expires_at = self._parse_expiry(payload.get("accessTokenExpires") or payload.get("expires"))
        self._access_token = token
        self._access_token_expires_at = expires_at

    def _token_is_fresh(self) -> bool:
        if not self._access_token:
            return False
        if self._access_token_expires_at is None:
            return True
        # Refresh the token a little before it expires to avoid mid-request failures.
        return time.time() < self._access_token_expires_at - 30

    @staticmethod
    def _parse_expiry(expires: Optional[str]) -> Optional[float]:
        if not expires:
            return None
        try:
            if expires.endswith("Z"):
                expires = expires[:-1] + "+00:00"
            dt = datetime.fromisoformat(expires)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:  # pragma: no cover - tolerate unexpected formats
            return None


__all__ = ["ChatGPTSync", "ChatGPTConversation", "ChatGPTSyncError"]
