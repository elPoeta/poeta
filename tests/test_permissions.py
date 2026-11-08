import asyncio
import os

import pytest

from poeta.config import ConfigError, load_config
from poeta.security import ActionRequest, AuditLog, PermissionManager, verify_chain


@pytest.fixture
def env(tmp_path):
    cfg = load_config()
    root = tmp_path / "proj"
    for d in ("workspace", "config", "logs"):
        (root / d).mkdir(parents=True)
    docs = tmp_path / "docs"
    docs.mkdir()
    cfg["permissions"]["file_access"]["allowed_dirs"] = ["workspace", str(docs)]
    cfg["permissions"]["file_access"]["auto_create_dirs"] = ["workspace"]
    cfg["permissions"]["commands"]["allowlist"] = ["ls"]
    return cfg, root, docs


def make(env, confirmer=None, audit=None):
    cfg, root, _ = env
    return PermissionManager(cfg, confirmer=confirmer, audit=audit, root=root)


def check(pm, kind, details=None, **kw):
    return asyncio.run(pm.check(ActionRequest(kind, f"prueba {kind}", details or {}, **kw)))


class Asker:
    """Confirmador que registra cada llamada."""

    def __init__(self, answer=True):
        self.answer, self.calls = answer, []

    async def __call__(self, req, explicit):
        self.calls.append((req.kind, explicit))
        return self.answer


# -- clasificacion --------------------------------------------------------------

def test_automatic_actions_need_no_confirmer(env):
    pm = make(env)
    assert check(pm, "open_app").allowed
    assert check(pm, "search_web").allowed and check(pm, "browse_web").allowed


def test_unknown_action_is_denied(env):
    d = check(make(env, Asker()), "inventada")
    assert not d.allowed and "no catalogada" in d.reason


def test_privileged_is_never_allowed_even_if_user_says_yes(env):
    asker = Asker(True)
    d = check(make(env, asker), "privileged")
    assert not d.allowed and asker.calls == []


def test_confirm_level_fails_closed_without_confirmer(env):
    _, _, docs = env
    d = check(make(env), "modify_file", {"path": str(docs / "a.txt")})
    assert not d.allowed and "interfaz de confirmacion" in d.reason


def test_confirm_and_reject(env):
    _, _, docs = env
    target = {"path": str(docs / "a.txt")}
    assert check(make(env, Asker(True)), "modify_file", target).allowed
    assert not check(make(env, Asker(False)), "modify_file", target).allowed


def test_only_a_literal_true_authorizes(env):
    _, _, docs = env
    for answer in (1, "si", object(), None):
        d = check(make(env, lambda req, explicit, a=answer: a), "modify_file", {"path": str(docs / "a.txt")})
        assert not d.allowed, f"{answer!r} no deberia autorizar"


def test_sync_confirmer_is_supported(env):
    _, _, docs = env
    assert check(make(env, lambda req, explicit: True), "modify_file", {"path": str(docs / "a.txt")}).allowed


def test_level_4_requires_explicit_confirmation(env):
    _, _, docs = env
    asker = Asker(True)
    d = check(make(env, asker), "delete_data", {"path": str(docs / "a.txt")})
    assert d.allowed and asker.calls == [("delete_data", True)]


def test_tainted_request_escalates_to_explicit(env):
    _, _, docs = env
    asker = Asker(True)
    check(make(env, asker), "modify_file", {"path": str(docs / "a.txt")}, tainted=True)
    check(make(env, asker), "modify_file", {"path": str(docs / "a.txt")}, tainted=False)
    assert asker.calls == [("modify_file", True), ("modify_file", False)]


def test_every_operation_is_asked_separately(env):
    _, _, docs = env
    asker = Asker(True)
    pm = make(env, asker)
    for _ in range(3):
        check(pm, "modify_file", {"path": str(docs / "a.txt")})
    assert len(asker.calls) == 3


def test_confirmer_exception_denies(env):
    _, _, docs = env

    def boom(req, explicit):
        raise RuntimeError("fallo")

    d = check(make(env, boom), "modify_file", {"path": str(docs / "a.txt")})
    assert not d.allowed and "RuntimeError" in d.reason


def test_confirmation_timeout_denies(env):
    cfg, root, docs = env
    cfg["permissions"]["confirmation_timeout_s"] = 0.05

    async def slow(req, explicit):
        await asyncio.sleep(5)
        return True

    d = check(make(env, slow), "modify_file", {"path": str(docs / "a.txt")})
    assert not d.allowed and "tiempo" in d.reason


def test_policy_is_a_copy_not_a_live_view(env):
    cfg, _, docs = env
    pm = make(env)
    cfg["permissions"]["actions"]["privileged"] = 1   # alguien altera el dict despues de crear el manager
    cfg["permissions"]["actions"]["modify_file"] = 1
    assert not check(pm, "privileged").allowed
    assert not check(pm, "modify_file", {"path": str(docs / "a.txt")}).allowed  # sigue exigiendo confirmacion


# -- nube ------------------------------------------------------------------------

def cloud_payload(media=False):
    content = [{"type": "text", "text": "hola"}, {"type": "image_url", "image_url": {"url": "data:..."}}] if media else "hola"
    return {"messages": [{"role": "user", "content": content}], "tools": None, "params": {}}


def test_cloud_transfer_always_asks_and_media_is_explicit(env):
    asker = Asker(True)
    pm = make(env, asker)
    check(pm, "cloud_transfer", payload=cloud_payload())
    check(pm, "cloud_transfer", payload=cloud_payload(media=True))
    assert asker.calls == [("cloud_transfer", False), ("cloud_transfer", True)]


def test_cloud_media_is_detected_from_payload_not_from_declared_details(env):
    asker = Asker(True)
    check(make(env, asker), "cloud_transfer", {"media_parts": 0}, payload=cloud_payload(media=True))
    assert asker.calls == [("cloud_transfer", True)]


def test_cloud_transfer_without_payload_is_denied(env):
    d = check(make(env, Asker(True)), "cloud_transfer")
    assert not d.allowed and "sin contenido" in d.reason


# -- archivos ----------------------------------------------------------------------

def test_read_inside_allowed_dirs_is_automatic(env):
    _, root, _ = env
    assert check(make(env), "read_authorized_file", {"path": str(root / "workspace" / "x.txt")}).allowed


@pytest.mark.parametrize("rel", ["../../etc/passwd", "../config/x.yaml"])
def test_path_escape_is_denied(env, rel):
    _, root, _ = env
    d = check(make(env, Asker(True)), "read_authorized_file", {"path": str(root / "workspace" / rel)})
    assert not d.allowed


def test_outside_allowed_dirs_is_denied(env, tmp_path):
    d = check(make(env, Asker(True)), "read_authorized_file", {"path": str(tmp_path / "otro" / "x")})
    assert not d.allowed and "fuera de las carpetas" in d.reason


def test_protected_paths_are_denied_even_for_reads_and_with_consent(env):
    cfg, root, _ = env
    cfg["permissions"]["file_access"]["allowed_dirs"].append(str(root))  # la raiz entera permitida...
    asker = Asker(True)
    pm = make(env, asker)
    for p in (root / "config" / "models.yaml", root / "logs" / "audit.jsonl", root / ".env"):
        d = check(pm, "read_authorized_file", {"path": str(p)})
        assert not d.allowed and "protegida" in d.reason
        assert not check(pm, "modify_file", {"path": str(p)}).allowed
    assert asker.calls == []


def test_symlink_cannot_escape_allowed_dir(env, tmp_path):
    _, root, _ = env
    outside = tmp_path / "secreto"
    outside.mkdir()
    os.symlink(outside, root / "workspace" / "atajo")
    d = check(make(env, Asker(True)), "read_authorized_file", {"path": str(root / "workspace" / "atajo" / "k.txt")})
    assert not d.allowed


def test_relative_or_missing_path_is_denied(env):
    pm = make(env, Asker(True))
    assert not check(pm, "read_authorized_file", {"path": "workspace/x.txt"}).allowed
    assert not check(pm, "read_authorized_file", {}).allowed


def test_create_document_rules(env):
    _, root, docs = env
    asker = Asker(True)
    pm = make(env, asker)
    # nuevo en la zona de creacion automatica: sin confirmar
    assert check(pm, "create_document", {"path": str(root / "workspace" / "nota.md")}).allowed
    assert asker.calls == []
    # nuevo en otra carpeta permitida: confirma
    check(pm, "create_document", {"path": str(docs / "nota.md")})
    assert asker.calls == [("create_document", False)]
    # sobrescribir uno existente equivale a modificar: confirma aunque este en la zona automatica
    existing = root / "workspace" / "viejo.md"
    existing.write_text("x")
    check(pm, "create_document", {"path": str(existing)})
    assert asker.calls[-1] == ("create_document", False)
    # fuera de carpetas autorizadas: denegado sin preguntar
    n = len(asker.calls)
    assert not check(pm, "create_document", {"path": str(docs.parent / "afuera.md")}).allowed
    assert len(asker.calls) == n


# -- comandos ----------------------------------------------------------------------------

def test_commands_need_allowlist_and_confirmation(env):
    asker = Asker(True)
    pm = make(env, asker)
    assert not check(pm, "run_command", {"argv": ["rm", "-rf", "/"]}).allowed
    assert not check(pm, "run_command", {"argv": ["/tmp/x/ls"]}).allowed      # ruta en lugar de nombre
    assert not check(pm, "run_command", {"argv": "ls -la; rm x"}).allowed     # texto de shell
    assert not check(pm, "run_command", {"argv": []}).allowed
    assert asker.calls == []
    assert check(pm, "run_command", {"argv": ["ls", "-la"]}).allowed
    assert asker.calls == [("run_command", False)]


# -- auditoria ----------------------------------------------------------------------------

def test_decisions_are_audited_without_content(env, tmp_path):
    path = tmp_path / "audit.jsonl"
    pm = make(env, Asker(True), audit=AuditLog(path))
    payload = {"messages": [{"role": "user", "content": "DATO-PRIVADO-123"}], "tools": None, "params": {}}
    d = check(pm, "cloud_transfer", {"provider": "nvidia"}, payload=payload)
    assert d.allowed and d.digest

    raw = path.read_text(encoding="utf-8")
    assert "DATO-PRIVADO-123" not in raw
    events = [__import__("json").loads(line) for line in raw.splitlines()]
    assert [e["event"] for e in events] == ["permission_requested", "permission_decided"]
    assert events[1]["allowed"] is True and events[1]["digest"] == d.digest
    assert verify_chain(path)[0]


def test_denials_are_audited_too(env, tmp_path):
    path = tmp_path / "audit.jsonl"
    check(make(env, audit=AuditLog(path)), "privileged")
    import json
    last = json.loads(path.read_text().splitlines()[-1])
    assert last["event"] == "permission_decided" and last["allowed"] is False


def test_if_audit_fails_the_action_is_denied(env, tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = AuditLog(path)
    path.mkdir()  # el archivo deja de ser escribible
    pm = make(env, audit=audit)
    d = check(pm, "open_app")   # accion automatica
    assert not d.allowed and "auditoria" in d.reason


# -- invariantes de configuracion ------------------------------------------------------------

def edit_perm(tmp_path, fn):
    import shutil
    import yaml
    from poeta.config import PROJECT_ROOT

    dst = tmp_path / "config"
    shutil.copytree(PROJECT_ROOT / "config", dst)
    p = dst / "permissions.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    fn(data)
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return dst


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda d: d["risk_levels"][3].update(policy="auto"), "risk_levels.3"),
        (lambda d: d["risk_levels"][4].update(policy="confirm"), "risk_levels.4"),
        (lambda d: d["risk_levels"][1].update(policy="quizas"), "risk_levels.1"),
        (lambda d: d["never_automatic"].clear(), "never_automatic"),
        (lambda d: d["actions"].update(privileged=2), "actions.privileged"),
        (lambda d: d["actions"].update(delete_data=3), "actions.delete_data"),
        (lambda d: d["actions"].update(modify_file=1), "actions.modify_file"),
        (lambda d: d["actions"].update(cloud_transfer=2), "actions.cloud_transfer"),
        (lambda d: d["actions"].update(open_app=9), "actions.open_app"),
    ],
)
def test_unsafe_permission_config_is_rejected(tmp_path, mutation, expected):
    with pytest.raises(ConfigError, match=expected):
        load_config(edit_perm(tmp_path, mutation))


def test_stricter_config_is_allowed(tmp_path):
    cfg = load_config(edit_perm(tmp_path, lambda d: d["actions"].update(open_app=3, modify_file=4)))
    assert cfg["permissions"]["actions"]["modify_file"] == 4
