import importlib.util
from pathlib import Path

import pytest

from poeta.config import PROJECT_ROOT, load_config


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location("run_llama_server", PROJECT_ROOT / "scripts" / "run_llama_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cfg(tmp_path: Path):
    cfg = load_config()
    models = tmp_path / "models"
    (models / "llama" / "nvidia").mkdir(parents=True)
    (models / "llama" / "nvidia" / cfg["models"]["llm"]["local"]["nemotron-bolt-e4b"]["file"]).touch()
    llama = tmp_path / "llama.cpp"
    (llama / "build" / "bin").mkdir(parents=True)
    (llama / "build" / "bin" / "llama-server").touch()
    cfg["settings"]["paths"]["models"] = str(models)
    cfg["settings"]["paths"]["llama_cpp"] = str(llama)
    return cfg


def pair(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_command_reproduces_working_script_flags(launcher, cfg):
    cmd = launcher.build_command(cfg)
    assert pair(cmd, "-a") == "nvidia-3.5-Bolt-4B"
    assert pair(cmd, "-c") == "65536"
    assert pair(cmd, "-ngl") == "999"
    assert pair(cmd, "--flash-attn") == "on"
    assert pair(cmd, "-b") == "2048" and pair(cmd, "-ub") == "512"
    assert pair(cmd, "-t") == "8" and pair(cmd, "--parallel") == "2"
    assert pair(cmd, "--reasoning") == "on"
    assert pair(cmd, "--port") == "8088"
    assert pair(cmd, "--cache-reuse") == "256"
    for flag in ("--jinja", "--kv-unified", "--cache-idle-slots", "--cache-prompt", "--metrics"):
        assert flag in cmd


def test_server_binds_to_loopback_by_default(launcher, cfg):
    assert pair(launcher.build_command(cfg), "--host") == "127.0.0.1"


def test_null_values_omit_the_flag(launcher, cfg):
    cfg["models"]["llama_server"]["gpu_layers"] = None
    assert "-ngl" not in launcher.build_command(cfg)


def test_missing_model_file_is_reported(launcher, cfg):
    cfg["models"]["llm"]["local"]["nemotron-bolt-e4b"]["file"] = "no-existe.gguf"
    with pytest.raises(FileNotFoundError):
        launcher.build_command(cfg)
