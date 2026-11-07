import asyncio

import pytest

from poeta.config import load_config
from poeta.core import AgentCore
from poeta.models import ModelRouter
from poeta.providers import ProviderUnavailable


def make_agent(url, **kw):
    cfg = load_config()
    cfg["providers"]["providers"]["local"]["base_url"] = url
    cfg["models"]["llm"]["default"] = kw.pop("model", "nemotron-bolt-e4b")
    return AgentCore(ModelRouter(cfg), system_prompt="Eres POETA.", **kw)


async def collect(agent, text):
    return [e async for e in agent.respond(text)]


def test_conversation_builds_history_and_sends_system_prompt(fake_llm):
    agent = make_agent(fake_llm.url)
    events = asyncio.run(collect(agent, "hola"))
    assert [e.kind for e in events] == ["text"] * 4 + ["done"]
    assert agent.history == [
        {"role": "user", "content": "hola"},
        {"role": "assistant", "content": "Hola, mundo!"},
    ]
    sent = fake_llm.posts()[0]["body"]["messages"]
    assert sent[0] == {"role": "system", "content": "Eres POETA."}
    assert sent[1] == {"role": "user", "content": "hola"}


def test_second_turn_includes_previous_context(fake_llm):
    agent = make_agent(fake_llm.url)
    asyncio.run(collect(agent, "uno"))
    asyncio.run(collect(agent, "dos"))
    sent = fake_llm.posts()[1]["body"]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]


def test_history_is_trimmed(fake_llm):
    agent = make_agent(fake_llm.url, max_history=2)
    asyncio.run(collect(agent, "uno"))
    asyncio.run(collect(agent, "dos"))
    sent = fake_llm.posts()[1]["body"]["messages"]
    assert len(sent) == 3  # system + 2 ultimos


def test_reasoning_is_not_part_of_the_answer(fake_llm):
    agent = make_agent(fake_llm.url, model="thinking")
    events = asyncio.run(collect(agent, "hola"))
    assert [e.kind for e in events] == ["reasoning", "text", "done"]
    assert agent.history[-1] == {"role": "assistant", "content": "Respuesta."}


def test_interruption_keeps_partial_answer(fake_llm):
    agent = make_agent(fake_llm.url)

    async def interrupted():
        gen = agent.respond("hola")
        async for ev in gen:
            if ev.kind == "text" and ev.text == ", ":
                break
        await gen.aclose()

    asyncio.run(interrupted())
    assert agent.history[-1] == {"role": "assistant", "content": "Hola, "}


def test_failure_without_output_removes_user_message(dead_url):
    agent = make_agent(dead_url)
    with pytest.raises(ProviderUnavailable):
        asyncio.run(collect(agent, "hola"))
    assert agent.history == []


def test_tool_calls_are_emitted_but_not_executed(fake_llm):
    agent = make_agent(fake_llm.url, model="tools")
    events = asyncio.run(collect(agent, "busca"))
    calls = [tc for e in events if e.kind == "tool_calls" for tc in e.tool_calls]
    assert [c.name for c in calls] == ["buscar"]
