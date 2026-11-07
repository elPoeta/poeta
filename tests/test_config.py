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


def test_mcp_servers_disabled_by_default():
    cfg = load_config()
    assert cfg["mcp"]["authorized_servers"] == []
    assert all(not s["enabled"] for s in cfg["mcp"]["servers"].values())
