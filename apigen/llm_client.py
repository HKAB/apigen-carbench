"""Decoupled LLM client.

A single abstraction handles all communication with the external vLLM server.
The Generator and Semantic Checker just pass their prompts + model name here.
Tests inject a FakeLLMClient implementing the same protocol so the whole
pipeline runs with no GPU/network.
"""

from __future__ import annotations

import asyncio
import random
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

    def __init__(
        self,
        base_url: str,
        api_token: str,
        *,
        timeout: float = 120.0,
        retries: int = 4,
        retry_base_delay: float = 0.5,
        max_inflight: int = 16,
    ):
        # Normalize so both "http://host:8000" and "http://host:8000/v1" work.
        self._base_url = base_url.rstrip("/")
        if not self._base_url.endswith("/v1"):
            self._base_url += "/v1"
        self._api_token = api_token
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None
        self._retries = max(0, retries)
        self._retry_base_delay = max(0.0, retry_base_delay)
        self._req_sem = asyncio.Semaphore(max(1, max_inflight))

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
        temperature: float = 1.0,
    ) -> str:
        session = self._ensure_session()
        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_token}"}
        payload = {"model": model, "messages": messages, "temperature": temperature}

        last_exc: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                async with self._req_sem:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        # Retry only on known transient statuses.
                        if resp.status == 429 or resp.status >= 500:
                            body = await resp.text()
                            raise aiohttp.ClientResponseError(
                                request_info=resp.request_info,
                                history=resp.history,
                                status=resp.status,
                                message=body[:500],
                                headers=resp.headers,
                            )
                        resp.raise_for_status()
                        data = await resp.json()
                return data["choices"][0]["message"]["content"]
            except (
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientConnectionError,
                aiohttp.ClientOSError,
                aiohttp.ClientPayloadError,
                asyncio.TimeoutError,
                aiohttp.ClientResponseError,
            ) as exc:
                last_exc = exc
                retryable = True
                if isinstance(exc, aiohttp.ClientResponseError):
                    retryable = exc.status == 429 or exc.status >= 500
                if (not retryable) or attempt >= self._retries:
                    raise
                # Exponential backoff + jitter to reduce thundering herd.
                delay = self._retry_base_delay * (2**attempt) + random.uniform(0, 0.25)
                await asyncio.sleep(delay)

        # Defensive fallback (normally unreachable due raise above).
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM request failed without an exception")
