"""Decoupled LLM client.

A single abstraction handles all communication with the external vLLM server.
The Generator and Semantic Checker just pass their prompts + model name here.
Tests inject a FakeLLMClient implementing the same protocol so the whole
pipeline runs with no GPU/network.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import aiohttp

Message = dict[str, str]


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat-completion interface (OpenAI-compatible)."""

    async def chat(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.7,
    ) -> str:
        """Return the assistant message content for the given prompt."""
        ...


class AiohttpLLMClient:
    """LLMClient backed by a vLLM OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str, api_token: str, *, timeout: float = 120.0):
        # Normalize so both "http://host:8000" and "http://host:8000/v1" work.
        self._base_url = base_url.rstrip("/")
        if not self._base_url.endswith("/v1"):
            self._base_url += "/v1"
        self._api_token = api_token
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "AiohttpLLMClient":
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def chat(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.7,
    ) -> str:
        session = self._ensure_session()
        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_token}"}
        payload = {"model": model, "messages": messages, "temperature": temperature}
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
        return data["choices"][0]["message"]["content"]
