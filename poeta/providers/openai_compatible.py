"""Proveedor para cualquier API compatible con OpenAI (llama.cpp, NVIDIA NIM, OpenAI, ...)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from .base import Chunk, Provider, ProviderError, ProviderUnavailable, ToolCall


class OpenAICompatibleProvider(Provider):
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        cloud: bool = False,
        timeout: float = 120.0,
        connect_timeout: float = 5.0,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.is_cloud = cloud
        self._timeout = httpx.Timeout(timeout, connect=connect_timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> AsyncIterator[Chunk]:
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": True, **params}
        if tools:
            body["tools"] = tools

        pending: dict[int, dict[str, str]] = {}

        def flush() -> list[ToolCall]:
            calls = [ToolCall(s["id"], s["name"], s["arguments"]) for _, s in sorted(pending.items())]
            pending.clear()
            return calls

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=body, headers=self._headers()
                ) as resp:
                    if resp.status_code >= 400:
                        detail = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                        raise ProviderError(
                            f"{self.name}: HTTP {resp.status_code}: {detail}", status=resp.status_code
                        )

                    async for raw in resp.aiter_lines():
                        line = raw.strip()
                        if not line or line.startswith(":") or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            obj = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        if obj.get("usage"):
                            yield Chunk(usage=obj["usage"])

                        for choice in obj.get("choices") or []:
                            delta = choice.get("delta") or {}
                            for tc in delta.get("tool_calls") or []:
                                slot = pending.setdefault(
                                    tc.get("index", 0), {"id": "", "name": "", "arguments": ""}
                                )
                                if tc.get("id"):
                                    slot["id"] = tc["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    slot["name"] += fn["name"]
                                if fn.get("arguments"):
                                    slot["arguments"] += fn["arguments"]

                            chunk = Chunk(
                                text=delta.get("content") or "",
                                reasoning=delta.get("reasoning_content") or delta.get("reasoning") or "",
                                finish_reason=choice.get("finish_reason"),
                            )
                            if chunk.finish_reason and pending:
                                chunk.tool_calls = flush()
                            if chunk.text or chunk.reasoning or chunk.tool_calls or chunk.finish_reason:
                                yield chunk

                    if pending:  # el servidor cerro sin finish_reason
                        yield Chunk(tool_calls=flush(), finish_reason="tool_calls")

        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ProviderUnavailable(f"{self.name}: no se pudo conectar a {self.base_url}") from exc
        except httpx.TimeoutException as exc:
            raise ProviderError(f"{self.name}: tiempo de espera agotado") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name}: error de comunicacion: {exc}") from exc

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0)) as client:
                resp = await client.get(f"{self.base_url}/models", headers=self._headers())
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ProviderUnavailable(f"{self.name}: no se pudo conectar a {self.base_url}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name}: error de comunicacion: {exc}") from exc
        if resp.status_code >= 400:
            raise ProviderError(f"{self.name}: HTTP {resp.status_code}", status=resp.status_code)
        return [m["id"] for m in resp.json().get("data", []) if "id" in m]
