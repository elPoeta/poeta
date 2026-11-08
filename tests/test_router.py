import asyncio
import json

import pytest

from poeta.config import load_config
from poeta.models import (
    CloudNotAuthorized,
    ModelNotSelected,
    ModelRouter,
    ProviderDisabled,
)
from poeta.providers import MissingCredentials, ProviderUnavailable
from poeta.security import AuditLog, PermissionManager, verify_chain

MSGS = [{"role": "user", "content": "texto privado"}]


async def collect(gen):
    return [c async for c in gen]


def run(router, messages=MSGS, **kw):
    return asyncio.run(collect(router.stream_chat(messages, **kw)))


def make_cfg(local_url, cloud_url, cloud_model="cloud-model"):
    cfg = load_config()
    cfg["providers"]["providers"]["local"]["base_url"] = local_url
    cfg["providers"]["providers"]["nvidia"]["base_url"] = cloud_url
    cfg["models"]["llm"]["cloud"]["nvidia"]["model"] = cloud_model
    return cfg


def make_router(cfg, confirmer=None, audit=None):
    return ModelRouter(cfg, permissions=PermissionManager(cfg, confirmer=confirmer, audit=audit))


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "secret-key")


def text(chunks):
    return "".join(c.text for c in chunks)


def test_default_is_local_with_configured_model(fake_llm, fake_cloud):
    router = ModelRouter(make_cfg(fake_llm.url, fake_cloud.url))
    assert router.status()["provider"] == "local"
    assert router.active_model == "nvidia-3.5-Bolt-4B"  # alias de models.yaml, el mismo que -a de llama-server
    assert text(run(router)) == "Hola, mundo!"
    assert fake_llm.posts()[0]["body"]["model"] == "nvidia-3.5-Bolt-4B"
    assert fake_cloud.requests == []


# -- criterio de aceptacion de la Fase 2 --------------------------------------------------

def test_cloud_request_is_blocked_until_authorization_arrives(fake_llm, fake_cloud):
    cfg = make_cfg(fake_llm.url, fake_cloud.url)

    async def scenario():
        gate = asyncio.Event()

        async def confirmer(req, explicit):
            await gate.wait()
            return True

        router = make_router(cfg, confirmer)
        router.select("nvidia")
        task = asyncio.create_task(collect(router.stream_chat(MSGS)))
        await asyncio.sleep(0.3)
        blocked = (fake_cloud.requests == [], not task.done())   # nada salio mientras se esperaba
        gate.set()
        return blocked, await task

    (nothing_sent, still_waiting), chunks = asyncio.run(scenario())
    assert nothing_sent and still_waiting
    assert text(chunks) == "Hola, mundo!" and len(fake_cloud.posts()) == 1


def test_cloud_blocked_without_confirmer(fake_llm, fake_cloud):
    router = ModelRouter(make_cfg(fake_llm.url, fake_cloud.url))   # sin manager explicito: deniega
    router.select("nvidia")
    with pytest.raises(CloudNotAuthorized, match="interfaz de confirmacion"):
        run(router)
    assert fake_cloud.requests == []


def test_cloud_blocked_when_user_denies(fake_llm, fake_cloud):
    router = make_router(make_cfg(fake_llm.url, fake_cloud.url), lambda req, explicit: False)
    router.select("nvidia")
    with pytest.raises(CloudNotAuthorized, match="rechazado"):
        run(router)
    assert fake_cloud.requests == []


def test_cloud_allowed_shows_exact_payload_and_sends_key(fake_llm, fake_cloud):
    seen = []

    async def confirmer(req, explicit):
        seen.append((req, explicit))
        return True

    router = make_router(make_cfg(fake_llm.url, fake_cloud.url), confirmer)
    router.select("nvidia")
    assert text(run(router)) == "Hola, mundo!"

    (req, explicit), = seen
    assert explicit is False
    assert (req.details["provider"], req.details["model"]) == ("nvidia", "cloud-model")
    assert req.payload["messages"] == MSGS and req.details["messages"] == 1
    sent = fake_cloud.posts()[0]
    assert sent["headers"]["Authorization"] == "Bearer secret-key"
    assert sent["body"]["messages"] == MSGS


def test_what_is_shown_is_what_is_sent_even_if_original_changes_meanwhile(fake_llm, fake_cloud):
    messages = [{"role": "user", "content": "mensaje original"}]

    async def confirmer(req, explicit):
        messages[0]["content"] = "mensaje alterado despues de mostrarlo"
        return True

    router = make_router(make_cfg(fake_llm.url, fake_cloud.url), confirmer)
    router.select("nvidia")
    run(router, messages)
    assert fake_cloud.posts()[0]["body"]["messages"][0]["content"] == "mensaje original"


def test_images_or_audio_require_explicit_confirmation(fake_llm, fake_cloud):
    seen = []
    router = make_router(make_cfg(fake_llm.url, fake_cloud.url), lambda req, explicit: seen.append(explicit) or True)
    router.select("nvidia")
    media = [{"role": "user", "content": [{"type": "text", "text": "mira"}, {"type": "image_url", "image_url": {"url": "data:x"}}]}]
    run(router, media)
    assert seen == [True]


def test_authorization_is_per_operation(fake_llm, fake_cloud):
    calls = []
    router = make_router(make_cfg(fake_llm.url, fake_cloud.url), lambda req, explicit: calls.append(1) or True)
    router.select("nvidia")
    run(router)
    run(router)
    assert len(calls) == 2


def test_local_never_asks_for_authorization(fake_llm, fake_cloud):
    router = make_router(make_cfg(fake_llm.url, fake_cloud.url), lambda req, explicit: 1 / 0)
    assert text(run(router)) == "Hola, mundo!"


def test_missing_key_fails_before_asking_user(monkeypatch, fake_llm, fake_cloud):
    monkeypatch.delenv("NVIDIA_API_KEY")
    asked = []
    router = make_router(make_cfg(fake_llm.url, fake_cloud.url), lambda req, explicit: asked.append(1) or True)
    router.select("nvidia")
    with pytest.raises(MissingCredentials):
        run(router)
    assert asked == [] and fake_cloud.requests == []


def test_no_automatic_fallback_to_cloud(dead_url, fake_cloud):
    router = make_router(make_cfg(dead_url, fake_cloud.url), lambda req, explicit: True)
    assert router.status()["provider"] == "local"
    with pytest.raises(ProviderUnavailable):
        run(router)
    assert router.status()["provider"] == "local"
    assert fake_cloud.requests == []


def test_switching_provider_needs_no_code_changes(fake_llm, fake_cloud):
    router = make_router(make_cfg(fake_llm.url, fake_cloud.url), lambda req, explicit: True)
    run(router)
    router.select("nvidia")
    run(router)
    router.select("local")
    run(router)
    assert len(fake_llm.posts()) == 2 and len(fake_cloud.posts()) == 1


def test_cloud_without_model_selected(fake_llm, fake_cloud):
    router = make_router(make_cfg(fake_llm.url, fake_cloud.url, cloud_model=None), lambda r, e: True)
    router.select("nvidia")
    with pytest.raises(ModelNotSelected):
        run(router)
    router.select(model="otro-modelo")
    assert text(run(router)) == "Hola, mundo!"
    assert fake_cloud.posts()[0]["body"]["model"] == "otro-modelo"


def test_disabled_or_unknown_provider_cannot_be_selected(fake_llm, fake_cloud):
    router = ModelRouter(make_cfg(fake_llm.url, fake_cloud.url))
    with pytest.raises(ProviderDisabled):  # openai esta enabled: false
        router.select("openai")
    with pytest.raises(ProviderDisabled):
        router.select("inexistente")


# -- auditoria de las transferencias --------------------------------------------------------

def events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_authorized_transfer_is_audited_without_content_or_secrets(fake_llm, fake_cloud, tmp_path):
    path = tmp_path / "audit.jsonl"
    router = make_router(make_cfg(fake_llm.url, fake_cloud.url), lambda r, e: True, audit=AuditLog(path))
    router.select("nvidia")
    run(router)

    evs = events(path)
    assert [e["event"] for e in evs] == ["permission_requested", "permission_decided", "cloud_transfer_result"]
    assert evs[2]["status"] == "ok" and evs[2]["digest"] == evs[1]["digest"] and evs[2]["request_id"] == evs[1]["request_id"]
    raw = path.read_text(encoding="utf-8")
    assert "texto privado" not in raw and "secret-key" not in raw
    assert verify_chain(path)[0]


def test_denied_transfer_is_audited_and_nothing_is_sent(fake_llm, fake_cloud, tmp_path):
    path = tmp_path / "audit.jsonl"
    router = make_router(make_cfg(fake_llm.url, fake_cloud.url), lambda r, e: False, audit=AuditLog(path))
    router.select("nvidia")
    with pytest.raises(CloudNotAuthorized):
        run(router)
    evs = events(path)
    assert evs[-1]["event"] == "permission_decided" and evs[-1]["allowed"] is False
    assert "cloud_transfer_result" not in [e["event"] for e in evs] and fake_cloud.requests == []


def test_failed_transfer_is_audited_as_error(fake_llm, fake_cloud, tmp_path):
    path = tmp_path / "audit.jsonl"
    router = make_router(make_cfg(fake_llm.url, fake_cloud.url, cloud_model="boom"), lambda r, e: True, audit=AuditLog(path))
    router.select("nvidia")
    with pytest.raises(Exception):
        run(router)
    result = events(path)[-1]
    assert result["event"] == "cloud_transfer_result" and result["status"] == "error" and result["error"] == "ProviderError"
