import asyncio

import pytest

from poeta.providers import OpenAICompatibleProvider, ProviderError, ProviderUnavailable


async def collect(gen):
    return [c async for c in gen]


def provider(url, key=None):
    return OpenAICompatibleProvider("test", url, api_key=key)


def test_streams_text(fake_llm):
    chunks = asyncio.run(collect(provider(fake_llm.url).stream_chat([{"role": "user", "content": "hola"}], "m")))
    assert "".join(c.text for c in chunks) == "Hola, mundo!"
    assert chunks[-1].finish_reason == "stop"
    body = fake_llm.posts()[0]["body"]
    assert body["stream"] is True and body["model"] == "m"


def test_reasoning_is_separated_from_text(fake_llm):
    chunks = asyncio.run(collect(provider(fake_llm.url).stream_chat([], "thinking")))
    assert "".join(c.reasoning for c in chunks) == "Pienso..."
    assert "".join(c.text for c in chunks) == "Respuesta."


def test_tool_calls_are_accumulated(fake_llm):
    tools = [{"type": "function", "function": {"name": "buscar"}}]
    chunks = asyncio.run(collect(provider(fake_llm.url).stream_chat([], "tools", tools=tools)))
    calls = [tc for c in chunks for tc in c.tool_calls]
    assert len(calls) == 1
    assert (calls[0].id, calls[0].name, calls[0].arguments) == ("call_1", "buscar", '{"q": "hola"}')
    assert fake_llm.posts()[0]["body"]["tools"] == tools


def test_bearer_header_only_when_key(fake_llm):
    asyncio.run(collect(provider(fake_llm.url, key="secreto").stream_chat([], "m")))
    asyncio.run(collect(provider(fake_llm.url).stream_chat([], "m")))
    first, second = (r["headers"] for r in fake_llm.posts())
    assert first["Authorization"] == "Bearer secreto"
    assert "Authorization" not in second


def test_http_error_is_typed(fake_llm):
    with pytest.raises(ProviderError) as exc:
        asyncio.run(collect(provider(fake_llm.url).stream_chat([], "boom")))
    assert exc.value.status == 500
    assert not isinstance(exc.value, ProviderUnavailable)


def test_connection_refused_is_unavailable(dead_url):
    with pytest.raises(ProviderUnavailable):
        asyncio.run(collect(provider(dead_url).stream_chat([], "m")))


def test_connection_error_message_includes_cause(dead_url):
    with pytest.raises(ProviderUnavailable, match="ConnectError"):
        asyncio.run(collect(provider(dead_url).stream_chat([], "m")))


def test_local_provider_ignores_proxy_env(fake_llm, monkeypatch):
    for var in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:1")  # proxy inexistente
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    local = OpenAICompatibleProvider("local", fake_llm.url, cloud=False)
    chunks = asyncio.run(collect(local.stream_chat([], "m")))
    assert "".join(c.text for c in chunks) == "Hola, mundo!"
    assert asyncio.run(local.list_models()) == ["fake-model"]


def test_list_models_and_health(fake_llm, dead_url):
    assert asyncio.run(provider(fake_llm.url).list_models()) == ["fake-model"]
    assert asyncio.run(provider(fake_llm.url).health()) is True
    assert asyncio.run(provider(dead_url).health()) is False
