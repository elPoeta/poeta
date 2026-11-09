import asyncio
import json
from types import SimpleNamespace

import pytest

from poeta.core import AgentCore
from poeta.memory import MemoryStore
from poeta.providers import Chunk, ToolCall
from poeta.tools import Tool, ToolRegistry
from tests.test_tools import setup


class ScriptRouter:
    """Router falso: cada llamada a stream_chat consume un 'guion' (lista de Chunk)."""

    def __init__(self, scripts):
        self.scripts, self.calls = list(scripts), []
        self.active = SimpleNamespace(is_cloud=False)

    async def stream_chat(self, messages, tools=None, **params):
        self.calls.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        for ch in self.scripts.pop(0):
            await asyncio.sleep(0)
            yield ch


def tc(name, **args):
    return ToolCall("call_" + name, name, json.dumps(args))


def calls_chunk(*calls, text=""):
    return ([Chunk(text=text)] if text else []) + [Chunk(tool_calls=list(calls), finish_reason="tool_calls")]


def texts(*pieces):
    return [Chunk(text=p) for p in pieces] + [Chunk(finish_reason="stop")]


def make(tmp_path, scripts, **kw):
    reg, pm, launched, ws = setup(tmp_path, **kw.pop("toolcfg", {}))
    router = ScriptRouter(scripts)
    agent = AgentCore(router, "Eres POETA.", tools=reg, **kw)
    return agent, router, reg, ws


async def collect(agent, text):
    return [e async for e in agent.respond(text)]


def test_tool_round_then_final_answer(tmp_path):
    agent, router, reg, ws = make(tmp_path, [
        calls_chunk(tc("get_datetime"), text="Voy a mirar. "),
        texts("Son las ", "diez."),
    ])
    ev = asyncio.run(collect(agent, "qué hora es"))
    kinds = [e.kind for e in ev]
    assert kinds == ["text", "tool_start", "tool_result", "text", "text", "done"]
    assert ev[2].result.ok and ev[1].text == "get_datetime"
    # 2a llamada: ve el mensaje assistant con tool_calls y el resultado como mensaje 'tool'
    m = router.calls[1]["messages"]
    assert m[-2]["role"] == "assistant" and m[-2]["content"] == "Voy a mirar. "
    assert m[-2]["tool_calls"][0]["function"]["name"] == "get_datetime"
    assert m[-1]["role"] == "tool" and m[-1]["tool_call_id"] == "call_get_datetime" and "UTC" in m[-1]["content"]
    # las herramientas se ofrecen en ambas rondas
    assert router.calls[0]["tools"] and router.calls[1]["tools"]


def test_history_keeps_only_user_and_final_answer(tmp_path):
    agent, router, *_ = make(tmp_path, [calls_chunk(tc("get_datetime")), texts("Listo.")])
    asyncio.run(collect(agent, "hora?"))
    assert agent.history == [{"role": "user", "content": "hora?"}, {"role": "assistant", "content": "Listo."}]
    # el turno siguiente no arrastra los mensajes de herramientas
    router.scripts.append(texts("Otra."))
    asyncio.run(collect(agent, "y ahora?"))
    assert all(m["role"] in ("system", "user", "assistant") and "tool_calls" not in m for m in router.calls[-1]["messages"])


def test_denied_tool_is_reported_to_the_model_not_raised(tmp_path):
    agent, router, reg, ws = make(tmp_path, [
        calls_chunk(tc("read_file", path="/etc/passwd")), texts("No pude leerlo.")])
    ev = asyncio.run(collect(agent, "lee /etc/passwd"))
    res = [e for e in ev if e.kind == "tool_result"][0].result
    assert res.denied and "root:" not in res.text
    assert "Accion denegada" in router.calls[1]["messages"][-1]["content"]


def test_tool_rounds_are_limited_and_last_round_offers_no_tools(tmp_path):
    scripts = [calls_chunk(tc("get_datetime")) for _ in range(2)] + [texts("Basta.")]
    agent, router, *_ = make(tmp_path, scripts, max_tool_rounds=2)
    ev = asyncio.run(collect(agent, "bucle"))
    assert [c["tools"] is not None for c in router.calls] == [True, True, False]
    assert sum(e.kind == "tool_start" for e in ev) == 2 and "".join(e.text for e in ev if e.kind == "text") == "Basta."


def test_model_insisting_on_tools_in_last_round_does_not_loop_forever(tmp_path):
    scripts = [calls_chunk(tc("get_datetime")) for _ in range(3)]  # tambien en la ultima ronda (ignorado)
    agent, router, *_ = make(tmp_path, scripts, max_tool_rounds=2)
    ev = asyncio.run(collect(agent, "bucle"))
    assert len(router.calls) == 3 and sum(e.kind == "tool_start" for e in ev) == 2 and ev[-1].kind == "done"


def test_several_calls_in_one_round_all_answered_in_order(tmp_path):
    agent, router, *_ = make(tmp_path, [
        calls_chunk(tc("get_datetime"), tc("list_dir")), texts("Hecho.")])
    asyncio.run(collect(agent, "dos cosas"))
    m = router.calls[1]["messages"]
    assert [x["role"] for x in m[-3:]] == ["assistant", "tool", "tool"]
    assert len(m[-3]["tool_calls"]) == 2 and m[-2]["tool_call_id"] != m[-1]["tool_call_id"]


def test_without_registry_tool_calls_are_only_reported():
    router = ScriptRouter([calls_chunk(tc("x"))])
    agent = AgentCore(router, "p")
    ev = asyncio.run(collect(agent, "hola"))
    assert [e.kind for e in ev] == ["tool_calls", "done"] and len(router.calls) == 1


def test_untrusted_result_taints_following_actions_in_same_turn_only(tmp_path):
    agent, router, reg, ws = make(tmp_path, [calls_chunk(tc("list_dir")), texts("ok")])
    asyncio.run(collect(agent, "lista"))
    assert reg.tainted
    router.scripts.append(texts("otra"))
    asyncio.run(collect(agent, "nuevo turno"))
    assert not reg.tainted


def test_cancellation_during_tool_keeps_partial_text_and_leaves_no_tool_messages(tmp_path):
    agent, router, reg, ws = make(tmp_path, [calls_chunk(tc("lenta"), text="Un momento. ")])

    async def slow(a):
        await asyncio.sleep(10)

    reg.register(Tool("lenta", "d", {"type": "object", "properties": {}}, slow, "system_info"))

    async def go():
        async def consume():
            async for _ in agent.respond("hazlo"):
                pass

        t = asyncio.create_task(consume())
        await asyncio.sleep(0.2)
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)

    asyncio.run(go())
    assert agent.history[-1] == {"role": "assistant", "content": "Un momento."} or agent.history[-1]["content"].startswith("Un momento")
    assert all("tool_calls" not in m for m in agent.history)


def test_failed_turn_without_text_removes_user_message(tmp_path):
    agent, router, *_ = make(tmp_path, [])  # sin guiones: stream_chat revienta con IndexError
    with pytest.raises(IndexError):
        asyncio.run(collect(agent, "hola"))
    assert agent.history == []


def test_tool_usage_is_logged_in_memory_with_redaction(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    agent, router, reg, ws = make(tmp_path, [calls_chunk(tc("write_file", path="a.txt", content="token=sk-secret1234567890abcdef")),
                                             texts("Guardado.")],
                                  memory=store, memory_cfg={"save_conversations": True, "context": {"inject": False}})
    asyncio.run(collect(agent, "guarda"))
    rows = store.tool_history()
    assert rows and rows[0]["tool"] == "write_file" and rows[0]["status"] == "ok"
    assert "sk-secret1234567890abcdef" not in json.dumps(rows)
    store.close()


def test_incognito_does_not_log_tools(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    agent, *_ = make(tmp_path, [calls_chunk(tc("get_datetime")), texts("ok")], memory=store,
                     memory_cfg={"save_conversations": True, "context": {"inject": False}})
    agent.incognito = True
    asyncio.run(collect(agent, "hora"))
    assert store.tool_history() == []
    store.close()
