import asyncio

import pytest

from poeta.config import ConfigError, load_config
from poeta.core import AgentCore
from poeta.core.reasoning import ReasoningConfigError, ReasoningPolicy
from poeta.models import ModelRouter

P = ReasoningPolicy()


@pytest.mark.parametrize("text, think", [
    ("hola, ¿cómo estás?", False), ("contame un chiste", False), ("gracias, eso es todo", False),
    ("abrí firefox", True), ("bajá el volumen al 30", True), ("cerrá la ventana de konsole", True),
    ("pausá la música", True), ("buscá noticias sobre llama.cpp", True), ("qué hora es", True),
    ("creá un archivo con mis notas", True), ("¿qué ventanas tengo abiertas?", True),
])
def test_auto_reasons_only_when_tools_are_likely(text, think):
    assert P.wants(text, tools_available=True) is think


def test_auto_without_tools_only_reasons_for_long_questions():
    assert P.wants("abrí firefox", tools_available=False) is False
    assert P.wants(" ".join(["palabra"] * 30), tools_available=False) is True


def test_modes():
    assert ReasoningPolicy("on").wants("hola", False) and not ReasoningPolicy("off").wants("abrí firefox", True)
    assert ReasoningPolicy("server").params_for("abrí firefox", True, False) == ({}, None)


def test_params_shape_and_cloud_gets_nothing():
    assert P.params_for("abrí firefox", True, False) == ({"chat_template_kwargs": {"enable_thinking": True}}, True)
    assert P.params_for("hola", True, False)[0] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert P.params_for("abrí firefox", True, True) == ({}, None)  # nube: no se le manda nada


@pytest.mark.parametrize("patch, msg", [
    ({"mode": "mucho"}, "mode"), ({"long_words": 0}, "long_words"), ({"triggers": "x"}, "triggers"),
    ({"triggers": ["(sin cerrar"]}, "invalida"), ("x", "mapa"),
])
def test_config_validation(patch, msg):
    cfg = load_config()
    cfg["settings"]["agent"]["reasoning"] = patch
    with pytest.raises(ReasoningConfigError, match=msg):
        ReasoningPolicy.from_config(cfg)


def test_custom_triggers_and_bad_config_surfaces_as_configerror(tmp_path):
    cfg = load_config()
    cfg["settings"]["agent"]["reasoning"] = {"mode": "auto", "triggers": [r"\bplanet\w+"]}
    pol = ReasoningPolicy.from_config(cfg)
    assert pol.wants("hablame del planeta", True) and not pol.wants("abrí firefox", True)
    import shutil
    import yaml
    d = tmp_path / "config"
    shutil.copytree("config", d)
    y = yaml.safe_load((d / "settings.yaml").read_text())
    y["agent"]["reasoning"] = {"mode": "tal vez"}
    (d / "settings.yaml").write_text(yaml.safe_dump(y))
    with pytest.raises(ConfigError, match="settings.yaml"):
        load_config(d)


# -- el agente manda el parametro al servidor local y no a la nube ----------------------------------------------
def make_agent(cfg, url, cloud=False, mode="auto"):
    cfg["providers"]["providers"]["local"]["base_url"] = url
    cfg["providers"]["providers"]["nvidia"]["base_url"] = url
    router = ModelRouter(cfg)
    return router, AgentCore(router, "sistema", reasoning=ReasoningPolicy(mode))


async def talk(agent, text, **kw):
    return [e async for e in agent.respond(text, **kw)]


def test_agent_sends_enable_thinking_to_local_server(fake_llm):
    cfg = load_config()
    router, agent = make_agent(cfg, fake_llm.url)
    router.select("local", "fake-model")
    asyncio.run(talk(agent, "hola, ¿cómo estás?"))
    assert fake_llm.posts()[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert agent.last_reasoning is False
    asyncio.run(talk(agent, "abrí firefox", tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}]))
    assert fake_llm.posts()[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert agent.last_reasoning is True


def test_caller_params_win_and_server_mode_sends_nothing(fake_llm):
    cfg = load_config()
    router, agent = make_agent(cfg, fake_llm.url)
    router.select("local", "fake-model")
    asyncio.run(talk(agent, "hola", chat_template_kwargs={"enable_thinking": True}))
    assert fake_llm.posts()[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": True}
    agent.reasoning.mode = "server"
    asyncio.run(talk(agent, "hola"))
    assert "chat_template_kwargs" not in fake_llm.posts()[-1]["body"] and agent.last_reasoning is None
