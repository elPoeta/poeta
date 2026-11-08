import os
import stat

import pytest

from poeta.config import load_config
from poeta.envfile import load_env_file
from poeta.models import ModelRouter


@pytest.fixture(autouse=True)
def isolated_environ(monkeypatch):
    """El cargador escribe en os.environ: se aisla para no contaminar otras pruebas."""
    monkeypatch.setattr(os, "environ", os.environ.copy())


def write_env(tmp_path, content, mode=0o600, encoding="utf-8"):
    p = tmp_path / ".env"
    p.write_bytes(content.encode(encoding))
    p.chmod(mode)
    return p


def test_basic_formats(tmp_path):
    p = write_env(tmp_path, (
        "# comentario\n"
        "\n"
        "A=uno\n"
        "export B=dos\n"
        "C = tres  \n"
        'D="con espacios y # almohadilla"\n'
        "E='simple'\n"
        "F=valor # comentario al final\n"
        "G=nvapi-abc#def\n"          # '#' pegado al valor: parte de la clave
        "H=a=b=c\n"                  # '=' dentro del valor (p. ej. base64)
    ))
    r = load_env_file(p)
    assert r.found and not r.malformed_lines and r.error is None
    assert [os.environ[k] for k in "ABCDEFGH"] == [
        "uno", "dos", "tres", "con espacios y # almohadilla", "simple", "valor", "nvapi-abc#def", "a=b=c",
    ]
    assert r.loaded == list("ABCDEFGH")


def test_crlf_and_bom(tmp_path):
    p = write_env(tmp_path, "﻿NVIDIA_API_KEY=nvapi-xyz\r\nOTRA=1\r\n")
    r = load_env_file(p)
    assert os.environ["NVIDIA_API_KEY"] == "nvapi-xyz" and os.environ["OTRA"] == "1"
    assert r.loaded == ["NVIDIA_API_KEY", "OTRA"]


def test_shell_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "del-shell")
    r = load_env_file(write_env(tmp_path, "NVIDIA_API_KEY=del-archivo\n"))
    assert os.environ["NVIDIA_API_KEY"] == "del-shell"
    assert r.kept_existing == ["NVIDIA_API_KEY"] and r.loaded == []


def test_empty_values_are_ignored_and_do_not_shadow(tmp_path):
    r = load_env_file(write_env(tmp_path, "NVIDIA_API_KEY=\nOPENAI_API_KEY=''\n"))
    assert "NVIDIA_API_KEY" not in os.environ and "OPENAI_API_KEY" not in os.environ
    assert r.loaded == [] and r.malformed_lines == []


def test_empty_shell_variable_is_replaced(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    load_env_file(write_env(tmp_path, "NVIDIA_API_KEY=nvapi-real\n"))
    assert os.environ["NVIDIA_API_KEY"] == "nvapi-real"


def test_malformed_lines_are_reported_by_number_not_content(tmp_path):
    secret = "nvapi-NO-DEBE-APARECER"
    p = write_env(tmp_path, (
        "BUENA=1\n"
        f"sin igual {secret}\n"        # linea 2
        f'ROTA="{secret}\n'            # linea 3: comilla sin cerrar
        f'EXTRA="a" {secret}\n'        # linea 4: texto tras la comilla
        "1MALA=x\n"                    # linea 5: nombre invalido
        "OTRA=2\n"
    ))
    r = load_env_file(p)
    assert r.malformed_lines == [2, 3, 4, 5]
    assert r.loaded == ["BUENA", "OTRA"]
    assert secret not in repr(r) and "ROTA" not in os.environ


def test_missing_file_is_not_an_error(tmp_path):
    r = load_env_file(tmp_path / ".env")
    assert not r.found and r.loaded == [] and r.error is None


def test_unreadable_encoding_is_reported_not_raised(tmp_path):
    p = tmp_path / ".env"
    p.write_bytes(b"A=\xff\xfe\xfa\n")
    r = load_env_file(p)
    assert r.found and r.error == "UnicodeDecodeError" and r.loaded == []


def test_result_never_contains_values(tmp_path):
    r = load_env_file(write_env(tmp_path, "NVIDIA_API_KEY=nvapi-VALOR-SECRETO\n"))
    assert "VALOR-SECRETO" not in repr(r)


@pytest.mark.parametrize("mode, insecure", [(0o600, False), (0o400, False), (0o644, True), (0o640, True)])
def test_permission_warning(tmp_path, mode, insecure):
    r = load_env_file(write_env(tmp_path, "A=1\n", mode=mode))
    assert r.insecure_permissions is insecure, oct(stat.S_IMODE(os.stat(tmp_path / ".env").st_mode))


def test_key_from_env_file_reaches_the_cloud_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    cfg = load_config()
    assert ModelRouter(cfg).status()["providers"] == ["local", "nvidia"]

    # sin .env: el proveedor cloud no tiene clave
    router = ModelRouter(cfg)
    router.select("nvidia")
    assert router.active.api_key is None

    # con .env cargado ANTES de crear el router: la clave llega al proveedor
    load_env_file(write_env(tmp_path, "NVIDIA_API_KEY=nvapi-desde-env\n"))
    router = ModelRouter(cfg)
    router.select("nvidia")
    assert router.active.api_key == "nvapi-desde-env"
