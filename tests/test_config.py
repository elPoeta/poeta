import shutil
from pathlib import Path

import pytest
import yaml

from poeta.config import CONFIG_FILES, PROJECT_ROOT, ConfigError, load_config


@pytest.fixture
def cfg_dir(tmp_path: Path) -> Path:
    dst = tmp_path / "config"
    shutil.copytree(PROJECT_ROOT / "config", dst)
    return dst


def edit(cfg_dir: Path, name: str, fn) -> None:
    path = cfg_dir / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fn(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def test_default_config_loads():
    cfg = load_config()
    assert set(cfg) == set(CONFIG_FILES)
    assert cfg["providers"]["defaults"]["provider"] == "local"
    assert cfg["voice"]["wake_word"]["phrase"] == "poeta"


def test_env_expansion_and_default(monkeypatch, cfg_dir):
    monkeypatch.setenv("AI_HOME", "/data/ai")
    cfg = load_config(cfg_dir)
    assert cfg["settings"]["paths"]["ai_home"] == "/data/ai"
    assert cfg["settings"]["paths"]["llama_cpp"] == "/data/ai/llama/llama.cpp"

    monkeypatch.delenv("AI_HOME")
    cfg = load_config(cfg_dir)
    assert cfg["settings"]["paths"]["ai_home"] == str(Path("~/AI").expanduser())


def test_missing_file_raises(cfg_dir):
    (cfg_dir / "voice.yaml").unlink()
    with pytest.raises(ConfigError, match="voice.yaml"):
        load_config(cfg_dir)


def test_cloud_fallback_true_is_rejected(cfg_dir):
    edit(cfg_dir, "providers", lambda d: d["defaults"].update(cloud_fallback=True))
    with pytest.raises(ConfigError, match="cloud_fallback"):
        load_config(cfg_dir)


def test_default_provider_must_exist(cfg_dir):
    edit(cfg_dir, "providers", lambda d: d["defaults"].update(provider="nope"))
    with pytest.raises(ConfigError, match="no esta definido"):
        load_config(cfg_dir)


def test_default_provider_cannot_be_cloud(cfg_dir):
    edit(cfg_dir, "providers", lambda d: d["defaults"].update(provider="nvidia"))
    with pytest.raises(ConfigError, match="local"):
        load_config(cfg_dir)


def test_cloud_provider_cannot_embed_api_key(cfg_dir):
    edit(cfg_dir, "providers", lambda d: d["providers"]["nvidia"].update(api_key="nvapi-secreto"))
    with pytest.raises(ConfigError, match="api_key_env"):
        load_config(cfg_dir)


def test_cloud_confirmation_cannot_be_disabled(cfg_dir):
    edit(cfg_dir, "permissions", lambda d: d["cloud"].update(require_confirmation=False))
    with pytest.raises(ConfigError, match="require_confirmation"):
        load_config(cfg_dir)


def test_assistant_name_defaults_to_poeta():
    from poeta.config import assistant_name, system_prompt

    cfg = load_config()
    assert assistant_name(cfg) == "POETA"
    assert system_prompt(cfg).startswith("Eres POETA, un asistente personal")
    assert "{assistant_name}" not in system_prompt(cfg)


def test_assistant_name_is_configurable(cfg_dir):
    from poeta.config import assistant_name, system_prompt

    edit(cfg_dir, "settings", lambda d: d["assistant"].update(name="  Atlas "))
    cfg = load_config(cfg_dir)
    assert assistant_name(cfg) == "Atlas"
    prompt = system_prompt(cfg)
    assert prompt.startswith("Eres Atlas,") and "POETA" not in prompt


def test_custom_prompt_without_placeholder_and_with_other_braces(cfg_dir):
    from poeta.config import system_prompt

    edit(cfg_dir, "settings", lambda d: d["agent"].update(system_prompt='Responde en JSON {"a": 1}.'))
    assert system_prompt(load_config(cfg_dir)) == 'Responde en JSON {"a": 1}.'


@pytest.mark.parametrize("bad", ["", "   ", None, 5])
def test_invalid_assistant_name_is_rejected(cfg_dir, bad):
    edit(cfg_dir, "settings", lambda d: d["assistant"].update(name=bad))
    with pytest.raises(ConfigError, match="assistant.name"):
        load_config(cfg_dir)


def test_llama_server_port_matches_local_provider():
    cfg = load_config()
    port = cfg["models"]["llama_server"]["port"]
    assert cfg["providers"]["providers"]["local"]["base_url"] == f"http://127.0.0.1:{port}/v1"


def test_default_local_model_has_a_file():
    cfg = load_config()
    key = cfg["models"]["llm"]["default"]
    assert cfg["models"]["llm"]["local"][key]["file"].endswith(".gguf")


def test_mcp_servers_disabled_by_default():
    cfg = load_config()
    assert cfg["mcp"]["authorized_servers"] == []
    assert all(not s["enabled"] for s in cfg["mcp"]["servers"].values())
