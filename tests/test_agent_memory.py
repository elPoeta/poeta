import asyncio

import pytest

from poeta.config import load_config
from poeta.core import AgentCore
from poeta.memory import MemoryStore
from poeta.models import ModelRouter
from poeta.providers import ProviderUnavailable
from poeta.security import PermissionManager


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "secret-key")


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    yield s
    s.close()


def make(local_url, store, cloud_url=None, model="nemotron-bolt-e4b", confirmer=None, **memory_overrides):
    cfg = load_config()
    cfg["providers"]["providers"]["local"]["base_url"] = local_url
    if cloud_url:
        cfg["providers"]["providers"]["nvidia"]["base_url"] = cloud_url
        cfg["models"]["llm"]["cloud"]["nvidia"]["model"] = "cloud-model"
    cfg["models"]["llm"]["default"] = model
    cfg["memory"]["context"].update(memory_overrides.pop("context", {}))
    cfg["memory"].update(memory_overrides)
    router = ModelRouter(cfg, permissions=PermissionManager(cfg, confirmer=confirmer))
    return AgentCore(router, system_prompt="Eres POETA.", memory=store, memory_cfg=cfg["memory"]), router


async def collect(agent, text):
    return [e async for e in agent.respond(text)]


def say(agent, text):
    return asyncio.run(collect(agent, text))


def system_of(server, index=-1):
    return server.posts()[index]["body"]["messages"][0]["content"]


# -- guardado -----------------------------------------------------------------------------------

def test_exchange_is_saved_with_provider_and_model(fake_llm, store):
    agent, _ = make(fake_llm.url, store)
    say(agent, "hola")
    cid = store.last_conversation_id()
    msgs = store.get_messages(cid)
    assert [(m["role"], m["content"], m["partial"]) for m in msgs] == [("user", "hola", 0), ("assistant", "Hola, mundo!", 0)]
    assert msgs[0]["provider"] == "local" and msgs[0]["model"] == "nvidia-3.5-Bolt-4B"   # el alias: lo que realmente se envia


def test_turns_accumulate_in_one_conversation(fake_llm, store):
    agent, _ = make(fake_llm.url, store)
    say(agent, "uno")
    say(agent, "dos")
    assert len(store.list_conversations()) == 1 and len(store.get_messages(agent.conversation_id)) == 4


def test_reset_starts_a_new_conversation(fake_llm, store):
    agent, _ = make(fake_llm.url, store)
    say(agent, "uno")
    agent.reset()
    say(agent, "dos")
    assert len(store.list_conversations()) == 2


def test_interrupted_answer_is_saved_as_partial(fake_llm, store):
    agent, _ = make(fake_llm.url, store)

    async def interrupted():
        gen = agent.respond("hola")
        async for ev in gen:
            if ev.kind == "text" and ev.text == ", ":
                break
        await gen.aclose()

    asyncio.run(interrupted())
    msgs = store.get_messages(agent.conversation_id)
    assert msgs[1]["content"] == "Hola, " and msgs[1]["partial"] == 1


def test_failed_turn_saves_nothing(dead_url, store):
    agent, _ = make(dead_url, store)
    with pytest.raises(ProviderUnavailable):
        say(agent, "hola")
    assert store.stats()["counts"]["messages"] == 0 and store.list_conversations() == []


def test_incognito_saves_nothing_and_can_be_turned_off(fake_llm, store):
    agent, _ = make(fake_llm.url, store)
    agent.incognito = True
    say(agent, "secreto")
    assert store.stats()["counts"]["messages"] == 0
    agent.incognito = False
    say(agent, "esto si")
    assert [m["content"] for m in store.get_messages(agent.conversation_id)][0] == "esto si"


def test_save_conversations_false_in_config(fake_llm, store):
    agent, _ = make(fake_llm.url, store, save_conversations=False)
    say(agent, "hola")
    assert store.stats()["counts"]["messages"] == 0


def test_agent_works_without_memory(fake_llm):
    agent, _ = make(fake_llm.url, None)
    assert [e.kind for e in say(agent, "hola")][-1] == "done"
    assert system_of(fake_llm) == "Eres POETA."


# -- continuidad ------------------------------------------------------------------------------------

def test_resume_restores_context_across_restart_and_model_change(fake_llm, fake_cloud, tmp_path):
    path = tmp_path / "m.db"
    first = MemoryStore(path)
    agent, _ = make(fake_llm.url, first)
    say(agent, "mi numero favorito es el 42")
    cid = agent.conversation_id
    first.close()

    second = MemoryStore(path)                                   # "reinicio"
    agent2, router2 = make(fake_llm.url, second, cloud_url=fake_cloud.url, model="thinking")   # y otro modelo
    assert agent2.resume(cid) == 2
    say(agent2, "cual era?")
    sent = fake_llm.posts()[-1]["body"]["messages"]
    assert [m["content"] for m in sent[1:]] == ["mi numero favorito es el 42", "Hola, mundo!", "cual era?"]
    assert fake_llm.posts()[-1]["body"]["model"] == "thinking"
    saved = second.get_messages(cid)                                                # sigue la misma conversacion
    assert agent2.conversation_id == cid and len(second.list_conversations()) == 1 and len(saved) == 4
    assert [m["model"] for m in saved] == ["nvidia-3.5-Bolt-4B"] * 2 + ["thinking"] * 2
    second.close()


def test_resume_respects_limit_and_validates(fake_llm, store):
    agent, _ = make(fake_llm.url, store, resume_last_messages=2)
    for t in ("a1", "a2", "a3"):
        say(agent, t)
    cid = agent.conversation_id
    other, _ = make(fake_llm.url, store, resume_last_messages=2)
    assert other.resume(cid) == 2
    with pytest.raises(LookupError):
        other.resume(9999)


# -- inyeccion de memoria -------------------------------------------------------------------------------

def test_relevant_memory_goes_into_system_prompt_for_local(fake_llm, store):
    store.add_memory("semantic", "Ana cumple años el 4 de marzo")
    store.set_preference("idioma", "español rioplatense")
    agent, _ = make(fake_llm.url, store)
    say(agent, "cuando cumple Ana")
    system = system_of(fake_llm)
    assert system.startswith("Eres POETA.\n\nMemoria guardada")
    assert "4 de marzo" in system and "preferencia idioma" in system
    assert len(fake_llm.posts()[-1]["body"]["messages"]) == 2     # un solo mensaje de sistema + el del usuario


def test_memory_block_is_not_saved_in_history_or_database(fake_llm, store):
    store.add_memory("semantic", "Ana cumple años el 4 de marzo")
    agent, _ = make(fake_llm.url, store)
    say(agent, "cuando cumple Ana")
    assert all("Memoria guardada" not in m["content"] for m in agent.history)
    assert all("Memoria guardada" not in m["content"] for m in store.get_messages(agent.conversation_id))


def test_memory_is_recomputed_each_turn(fake_llm, store):
    store.add_memory("semantic", "Ana cumple años el 4 de marzo")
    store.add_memory("semantic", "El auto es un Fiat Uno")
    agent, _ = make(fake_llm.url, store)
    say(agent, "cuando cumple Ana")
    say(agent, "que auto tengo, un Fiat")
    first, second = system_of(fake_llm, 0), system_of(fake_llm, 1)
    assert "4 de marzo" in first and "Fiat" not in first
    assert "Fiat" in second and "4 de marzo" not in second


def test_inject_false_disables_injection(fake_llm, store):
    store.add_memory("semantic", "Ana cumple años el 4 de marzo")
    agent, _ = make(fake_llm.url, store, context={"inject": False})
    say(agent, "cuando cumple Ana")
    assert system_of(fake_llm) == "Eres POETA."


def test_incognito_still_uses_memory_but_does_not_save(fake_llm, store):
    store.add_memory("semantic", "Ana cumple años el 4 de marzo")
    agent, _ = make(fake_llm.url, store)
    agent.incognito = True
    say(agent, "cuando cumple Ana")
    assert "4 de marzo" in system_of(fake_llm)


# -- la memoria no viaja a la nube por defecto ----------------------------------------------------------------

def test_memory_never_reaches_cloud_by_default(fake_llm, fake_cloud, store):
    store.add_memory("semantic", "Ana cumple años el 4 de marzo")
    store.set_preference("idioma", "español rioplatense")
    agent, router = make(fake_llm.url, store, cloud_url=fake_cloud.url, confirmer=lambda req, explicit: True)
    router.select("nvidia")
    say(agent, "cuando cumple Ana")
    sent = fake_cloud.posts()[0]["body"]["messages"]
    assert sent[0]["content"] == "Eres POETA."
    assert "4 de marzo" not in str(sent) and "idioma" not in str(sent)


def test_cloud_memory_requires_opt_in_and_is_shown_in_the_confirmation(fake_llm, fake_cloud, store):
    store.add_memory("semantic", "Ana cumple años el 4 de marzo")
    shown = []

    def confirmer(req, explicit):
        shown.append(str(req.payload["messages"]))
        return True

    agent, router = make(fake_llm.url, store, cloud_url=fake_cloud.url, confirmer=confirmer, context={"inject_into_cloud": True})
    router.select("nvidia")
    say(agent, "cuando cumple Ana")
    assert "4 de marzo" in shown[0]                                         # el usuario lo ve antes de que salga
    assert "4 de marzo" in fake_cloud.posts()[0]["body"]["messages"][0]["content"]


def test_cloud_conversation_is_still_saved_locally(fake_llm, fake_cloud, store):
    agent, router = make(fake_llm.url, store, cloud_url=fake_cloud.url, confirmer=lambda req, explicit: True)
    router.select("nvidia")
    say(agent, "hola nube")
    assert store.get_messages(agent.conversation_id)[0]["provider"] == "nvidia"


def test_denied_cloud_turn_is_not_saved(fake_llm, fake_cloud, store):
    from poeta.models import CloudNotAuthorized

    agent, router = make(fake_llm.url, store, cloud_url=fake_cloud.url, confirmer=lambda req, explicit: False)
    router.select("nvidia")
    with pytest.raises(CloudNotAuthorized):
        say(agent, "dato que no debe salir")
    assert store.stats()["counts"]["messages"] == 0 and fake_cloud.requests == []


# -- la memoria no rompe la conversacion ---------------------------------------------------------------------------

def test_broken_memory_does_not_break_the_chat(fake_llm, store):
    agent, _ = make(fake_llm.url, store)
    store.close()                                   # la base deja de estar disponible
    events = say(agent, "hola")
    assert "".join(e.text for e in events if e.kind == "text") == "Hola, mundo!"
    assert agent.memory_errors >= 1
    assert agent.history[-1]["content"] == "Hola, mundo!"
