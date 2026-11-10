"""Fase 9: bus de eventos, metricas, gestos en la conversacion y estabilidad del conjunto."""
import asyncio
import json
import threading
import time
import tracemalloc
from types import SimpleNamespace as NS

import pytest

from poeta.config import load_config
from poeta.core import EventBus
from poeta.gestures import GestureEvent, GestureHub, GesturePolicy, make_gesture_confirmer
from poeta.monitoring import MetricsLog, Sampler, gpu_sample, percentile, rss_mb, summarize
from poeta.providers import ToolCall
from poeta.security import ActionRequest, PermissionManager
from poeta.tools import Tool, ToolRegistry
from tests.test_voice_loop import build, drain


def run(coro):
    return asyncio.run(coro)


# -- EventBus ---------------------------------------------------------------------------------------------
def test_bus_filters_replays_and_isolates_failing_subscribers():
    bus, got, other = EventBus(history=3), [], []
    bus.publish("state", state="idle")
    unsub = bus.subscribe(got.append, kinds=("state",), replay=True)
    bus.subscribe(lambda e: 1 / 0)  # suscriptor roto: no afecta a los demas
    bus.subscribe(other.append)
    bus.publish("gesture", name="palm")
    bus.publish("state", state="speaking")
    assert [e.data["state"] for e in got] == ["idle", "speaking"] and len(other) == 2
    unsub()
    bus.publish("state", state="idle")
    assert len(got) == 2 and bus.state == "idle"
    assert len(bus.recent()) == 3  # historial acotado


def test_bus_set_state_ignores_repeats_and_is_thread_safe():
    bus, got = EventBus(), []
    bus.subscribe(got.append, kinds=("state",))
    bus.set_state("thinking")
    bus.set_state("thinking")
    assert len(got) == 1
    ts = [threading.Thread(target=lambda i=i: [bus.publish("gesture", n=i) for _ in range(200)]) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(bus.recent("gesture")) <= 200


# -- metricas ---------------------------------------------------------------------------------------------
def test_percentile_and_summary():
    assert percentile([1, 2, 3, 4, 5], 50) == 3 and percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    with pytest.raises(ValueError):
        percentile([], 50)
    rows = [{"stt_s": 1.0, "total_s": 3.0, "rss_mb": 500}, {"stt_s": 3.0, "total_s": 5.0, "interrupted": True,
                                                              "rss_mb": 520, "gpu_used_mib": 5600}]
    s = summarize(rows)
    assert s["turns"] == 2 and s["interrupted"] == 1 and s["stt_s"]["p50"] == 2.0
    assert s["rss_mb"]["first"] == 500 and s["rss_mb"]["last"] == 520 and s["gpu_used_mib"]["max"] == 5600
    assert "first_audio_s" not in s


def test_metrics_log_private_rotates_and_tolerates_garbage(tmp_path):
    m = MetricsLog(tmp_path / "logs" / "m.jsonl", max_bytes=300)
    for i in range(20):
        m.record(stt_s=1.0 + i, nada=None)
    assert (tmp_path / "logs" / "m.jsonl.1").exists()
    assert ((tmp_path / "logs" / "m.jsonl").stat().st_mode & 0o777) == 0o600
    with open(m.path, "a") as f:
        f.write("{no json\n")
    rows = m.read()
    assert rows and all("nada" not in r for r in rows)
    assert MetricsLog(tmp_path / "x.jsonl").read() == []


def test_gpu_sample_parsing_and_failures():
    ok = lambda *a, **k: NS(stdout="5624, 8151, 37, 61\n")  # noqa: E731
    assert gpu_sample(ok, lambda n: "/usr/bin/nvidia-smi") == {"gpu_used_mib": 5624.0, "gpu_total_mib": 8151.0,
                                                               "gpu_util_pct": 37.0, "gpu_temp_c": 61.0}
    assert gpu_sample(ok, lambda n: None) is None
    assert gpu_sample(lambda *a, **k: NS(stdout="No devices"), lambda n: "x") is None

    def boom(*a, **k):
        raise OSError("sin permiso")

    assert gpu_sample(boom, lambda n: "x") is None
    assert rss_mb() > 1 and rss_mb("/no/existe") is None


def test_sampler_caches_gpu():
    now, calls = [0.0], []
    s = Sampler(ttl=10, gpu=lambda: calls.append(1) or {"gpu_used_mib": 1.0}, clock=lambda: now[0])
    s()
    s()
    assert len(calls) == 1
    now[0] = 11
    assert "gpu_used_mib" in s() and len(calls) == 2


# -- GestureHub: politica de confirmaciones --------------------------------------------------------------------
def req(kind="window_control", tainted=False):
    return ActionRequest(kind, "Cerrar la ventana", {"tool": "window_action"}, tainted=tainted)


def hub_(**kw):
    out = []
    h = GestureHub(GesturePolicy(**kw), EventBus(), out=lambda *a, **k: out.append(" ".join(map(str, a))))
    h.active = True
    h.out_lines = out
    return h


def test_eligibility_rules():
    h = hub_()
    assert h.eligible(req(), False)
    assert not h.eligible(req(), True)                    # nivel 4 / explicita: solo teclado
    assert not h.eligible(req(tainted=True), False)       # tras datos no confiables: solo teclado
    assert not h.eligible(req("cloud_transfer"), False)   # nube: solo teclado
    h.active = False
    assert not h.eligible(req(), False)                   # servicio de gestos detenido
    assert not hub_(confirm_with_gestures=False).eligible(req(), False)


class SlowKeyboard:
    """Teclado que nunca contesta (hasta que se le diga) y registra si lo cancelaron."""

    def __init__(self, answer=None, delay=0.0):
        self.answer, self.delay, self.cancelled, self.calls = answer, delay, False, 0

    async def __call__(self, r, explicit):
        self.calls += 1
        try:
            await asyncio.sleep(self.delay if self.answer is not None else 30)
            return self.answer
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def fire(h, name, after=0.05):
    asyncio.get_running_loop().call_later(after, h.on_event, GestureEvent(name, 0.0))


def test_thumb_up_authorizes_and_cancels_keyboard_and_is_audited():
    audit = []
    h = hub_()
    h.audit = NS(record=lambda ev, **f: audit.append((ev, f)))
    kb = SlowKeyboard()

    async def go():
        c = make_gesture_confirmer(kb, h)
        fire(h, "thumb_up")
        return await asyncio.wait_for(c(req(), False), 2)

    assert run(go()) is True and kb.cancelled and not h.waiting
    assert audit == [("gesture_confirmation", {"kind": "window_control", "tool": "window_action", "allowed": True,
                                               "gesture": "thumb_up"})]
    assert any("con gestos" in l for l in h.out_lines)


def test_thumb_down_denies():
    h = hub_()

    async def go():
        c = make_gesture_confirmer(SlowKeyboard(), h)
        fire(h, "thumb_down")
        return await asyncio.wait_for(c(req(), False), 2)

    assert run(go()) is False


def test_keyboard_answer_wins_and_gesture_is_cancelled():
    h = hub_()

    async def go():
        c = make_gesture_confirmer(SlowKeyboard(answer=True, delay=0.05), h)
        r = await asyncio.wait_for(c(req(), False), 2)
        h.on_event(GestureEvent("thumb_down", 0.0))  # llega tarde: no debe afectar a nada
        return r

    assert run(go()) is True and not h.waiting


def test_ineligible_requests_never_see_gestures():
    h = hub_()
    kb = SlowKeyboard(answer=False)

    async def go():
        c = make_gesture_confirmer(kb, h)
        fire(h, "thumb_up", 0.01)  # un pulgar arriba no puede autorizar una accion explicita
        return await c(req(), True), await c(req(tainted=True), False)

    assert run(go()) == (False, False) and kb.calls == 2 and not h.waiting


def test_stale_gestures_before_a_prompt_are_ignored():
    h = hub_()
    h.on_event(GestureEvent("thumb_up", 0.0))  # nadie preguntaba
    kb = SlowKeyboard(answer=False, delay=0.05)

    async def go():
        return await make_gesture_confirmer(kb, h)(req(), False)

    assert run(go()) is False


def test_palm_interrupts_only_when_nothing_is_pending():
    h, hits = hub_(), []
    h.on_interrupt = lambda: hits.append(1)

    async def go():
        h.on_event(GestureEvent("palm", 0.0))
        assert hits == [1]
        fut = h.begin()
        h.on_event(GestureEvent("palm", 0.0))  # esa palma esta activando la interaccion para los pulgares
        assert hits == [1] and not fut.done()
        h.end()
        assert not hub_(interrupt_with_palm=False).on_interrupt
        off = hub_(interrupt_with_palm=False)
        off.on_interrupt = lambda: hits.append(2)
        off.on_event(GestureEvent("palm", 0.0))
        return hits

    assert run(go()) == [1]


def test_gesture_events_are_published_on_the_bus():
    h, seen = hub_(), []
    h.bus.subscribe(seen.append, kinds=("gesture",))
    h.on_event(GestureEvent("swipe_left", 0.0))
    assert [e.data["name"] for e in seen] == ["swipe_left"]


# -- gestos + permisos reales + herramientas ---------------------------------------------------------------------------
def build_registry(tmp_path, confirmer):
    cfg = load_config()
    pm = PermissionManager(cfg, confirmer=confirmer, root=tmp_path)
    reg = ToolRegistry(pm, 8000, 30)
    ran = []

    async def close_it(a):
        ran.append("cerrada")
        return "ok"

    async def read_web(a):
        return "contenido de una pagina"

    reg.register(Tool("close_win", "cierra", {"type": "object", "properties": {}, "additionalProperties": False},
                      close_it, "window_control"))
    reg.register(Tool("web", "lee", {"type": "object", "properties": {}, "additionalProperties": False},
                      read_web, "desktop_read", untrusted=True))
    return reg, ran


def call(name):
    return ToolCall(id="c", name=name, arguments="{}")


def test_gesture_confirms_a_real_tool_call(tmp_path):
    h = hub_()

    async def go():
        reg, ran = build_registry(tmp_path, make_gesture_confirmer(SlowKeyboard(), h))
        fire(h, "thumb_up")
        res = await asyncio.wait_for(reg.execute(call("close_win")), 3)
        return res, ran

    res, ran = run(go())
    assert res.ok and ran == ["cerrada"]


def test_gesture_denial_blocks_the_tool(tmp_path):
    h = hub_()

    async def go():
        reg, ran = build_registry(tmp_path, make_gesture_confirmer(SlowKeyboard(), h))
        fire(h, "thumb_down")
        return await asyncio.wait_for(reg.execute(call("close_win")), 3), ran

    res, ran = run(go())
    assert res.denied and ran == []


def test_after_untrusted_content_gesture_cannot_authorize(tmp_path):
    # tras leer algo no confiable (una pagina, titulos de ventanas...), cerrar una ventana exige el teclado
    h = hub_()
    kb = SlowKeyboard(answer=False, delay=0.05)

    async def go():
        reg, ran = build_registry(tmp_path, make_gesture_confirmer(kb, h))
        assert (await reg.execute(call("web"))).ok and reg.tainted
        fire(h, "thumb_up", 0.01)
        return await asyncio.wait_for(reg.execute(call("close_win")), 3), ran

    res, ran = run(go())
    assert res.denied and ran == [] and kb.calls >= 1


# -- VoiceLoop: palma, estados, metricas ---------------------------------------------------------------------------
def test_palm_interrupts_a_response_and_is_ignored_when_idle():
    async def go():
        audio, pipe, out, cmds, mk = build(replies=[["Una frase larga para decir. "] * 20], delay=0.05)
        keys, agent, loop = await mk()
        t = asyncio.create_task(loop.run())
        assert loop.interrupt() is False  # sin respuesta en curso: nada que callar
        keys.put("cuentame algo largo")
        await asyncio.sleep(0.4)
        assert loop.interrupt() is True
        await asyncio.sleep(0.3)
        keys.put("/quit")
        await asyncio.wait_for(t, 3)
        return audio, agent, out

    audio, agent, out = run(go())
    assert agent.cancelled and any("interrumpido" in o for o in out) and not audio.real.speaking


def test_palm_interrupts_speech_after_generation_finished():
    async def go():
        audio, pipe, out, cmds, mk = build(replies=[["Respuesta corta lista ya."]])
        keys, agent, loop = await mk()
        t = asyncio.create_task(loop.run())
        keys.put("hola")
        await asyncio.sleep(0.4)
        assert audio.real.speaking
        loop.interrupt()
        await asyncio.sleep(0.3)
        assert not audio.real.speaking
        keys.put("/quit")
        await asyncio.wait_for(t, 3)
        return out

    assert any("interrumpido" in o for o in run(go()))


def test_states_and_metrics_are_published_without_any_text(tmp_path):
    async def go():
        audio, pipe, out, cmds, mk = build(replies=[["Hola, ¿cómo estás? Todo bien."]])
        keys, agent, loop = await mk()
        bus, events = EventBus(), []
        bus.subscribe(events.append)
        loop.bus, loop.metrics = bus, MetricsLog(tmp_path / "m.jsonl")
        loop.sampler = lambda: {"rss_mb": 123.0}
        loop.meta = {"provider": "nvidia", "stt": "large-v3-turbo/cpu"}
        t = asyncio.create_task(loop.run())
        keys.put("mensaje secreto del usuario")
        await asyncio.sleep(0.5)
        drain(audio)
        await asyncio.sleep(0.2)
        keys.put("/quit")
        await asyncio.wait_for(t, 3)
        return events, loop

    events, loop = run(go())
    states = [e.data["state"] for e in events if e.kind == "state"]
    assert states[:2] == ["thinking", "speaking"] and states[-2:] == ["idle", "stopped"]
    rows = MetricsLog(tmp_path / "m.jsonl").read()
    assert rows[0]["provider"] == "nvidia" and rows[0]["stt"] == "large-v3-turbo/cpu"
    assert len(rows) == 1 and rows[0]["rss_mb"] == 123.0 and rows[0]["total_s"] >= 0 and rows[0]["interrupted"] is False
    blob = json.dumps(rows) + json.dumps([e.data for e in events])
    assert "secreto" not in blob and "Hola" not in blob  # solo estados y numeros


def test_metrics_failure_never_breaks_the_conversation(tmp_path):
    async def go():
        audio, pipe, out, cmds, mk = build(replies=[["Listo."], ["Otra vez."]])
        keys, agent, loop = await mk()
        loop.metrics = SimpleNamespace_raising()
        loop.sampler = lambda: 1 / 0
        t = asyncio.create_task(loop.run())
        keys.put("uno")
        await asyncio.sleep(0.4)
        drain(audio)
        keys.put("dos")
        await asyncio.sleep(0.4)
        drain(audio)
        keys.put("/quit")
        await asyncio.wait_for(t, 3)
        return agent

    assert run(go()).heard == ["uno", "dos"]


def SimpleNamespace_raising():
    def boom(**k):
        raise OSError("disco lleno")

    return NS(record=boom)


# -- estabilidad ----------------------------------------------------------------------------------------------------------
def test_many_turns_do_not_leak_threads_tasks_or_memory(tmp_path):
    turns = 40

    async def go():
        audio, pipe, out, cmds, mk = build(replies=[[f"Respuesta numero {i}. "] for i in range(turns)])
        keys, agent, loop = await mk()
        loop.bus, loop.metrics = EventBus(), MetricsLog(tmp_path / "m.jsonl")
        t = asyncio.create_task(loop.run())
        base_threads = threading.active_count()
        tracemalloc.start()
        snap0 = None
        for i in range(turns):
            keys.put(f"pregunta {i}")
            for _ in range(60):
                await asyncio.sleep(0.01)
                if audio.real.speaking:
                    break
            drain(audio)
            await asyncio.sleep(0.03)
            if i == 9:
                snap0 = tracemalloc.take_snapshot()
        snap1 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        keys.put("/quit")
        await asyncio.wait_for(t, 3)
        grown = sum(s.size_diff for s in snap1.compare_to(snap0, "filename"))
        pending = [x for x in asyncio.all_tasks() if x is not asyncio.current_task()]
        return agent, base_threads, grown, pending, audio

    agent, base_threads, grown, pending, audio = run(go())
    assert len(agent.heard) == turns
    assert threading.active_count() <= base_threads + 1
    assert grown < 1_500_000, f"la memoria crecio {grown} bytes en 30 turnos"
    assert not pending
    assert len(MetricsLog(tmp_path / "m.jsonl").read()) == turns


def test_interruption_storm_leaves_audio_and_mic_in_a_clean_state():
    async def go():
        audio, pipe, out, cmds, mk = build(replies=[["Una frase larga para decir. "] * 10 for _ in range(12)], delay=0.01)
        keys, agent, loop = await mk()
        t = asyncio.create_task(loop.run())
        for i in range(12):
            keys.put(f"pregunta {i}")
            await asyncio.sleep(0.03 + 0.01 * (i % 4))
            loop.interrupt() if i % 2 else keys.put("")  # alterna palma y Enter, en momentos distintos
            await asyncio.sleep(0.08)
        keys.put("/quit")
        await asyncio.wait_for(t, 5)
        return audio, loop

    audio, loop = run(go())
    time.sleep(0.1)
    assert not audio.real.speaking and not audio.real.muted and not loop._responding


# -- servicio y modo headless -----------------------------------------------------------------------------------------------
def test_render_unit_contents():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("svc", "scripts/install_service.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    unit = mod.render_unit(Path("/home/u/AI/poeta"), Path("/home/u/AI/poeta/.venv/bin/python"), gestures=True)
    assert "ExecStart=/home/u/AI/poeta/.venv/bin/python /home/u/AI/poeta/main.py voice --wake --headless --gestures" in unit
    assert "Restart=on-failure" in unit and "KillMode=process" in unit
    assert "NoNewPrivileges" not in unit.replace("NoNewPrivileges/", "") and "TasksMax=" not in unit
    assert "WantedBy=graphical-session.target" in unit
    assert "--gestures" not in mod.render_unit(Path("/p"), Path("/p/py"))


def test_headless_requires_wake_or_always_and_denies_without_keyboard(capsys):
    import main as poeta_main
    assert asyncio.run(poeta_main.voice(load_config(), None, None, False, False, False, False, True)) == 2
    assert "--headless necesita" in capsys.readouterr().err
    r = ActionRequest("window_control", "Cerrar la ventana X", {})
    assert asyncio.run(poeta_main.deny_without_keyboard(r, False)) is False
    assert "Cerrar la ventana X" in capsys.readouterr().out
