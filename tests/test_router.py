import asyncio

import pytest

from poeta.config import load_config
from poeta.models import (
    CloudNotAuthorized,
    ModelNotSelected,
    ModelRouter,
    ProviderDisabled,
    TransferRequest,
)
from poeta.providers import MissingCredentials, ProviderUnavailable

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


def test_cloud_blocked_without_authorizer(fake_llm, fake_cloud):
    router = ModelRouter(make_cfg(fake_llm.url, fake_cloud.url))
    router.select("nvidia")
    with pytest.raises(CloudNotAuthorized):
        run(router)
    assert fake_cloud.requests == []


def test_cloud_blocked_when_user_denies(fake_llm, fake_cloud):
    router = ModelRouter(make_cfg(fake_llm.url, fake_cloud.url), authorizer=lambda req: False)
    router.select("nvidia")
    with pytest.raises(CloudNotAuthorized):
        run(router)
    assert fake_cloud.requests == []


def test_cloud_allowed_shows_exact_payload_and_sends_key(fake_llm, fake_cloud):
    seen: list[TransferRequest] = []

    async def authorizer(req):
        seen.append(req)
        return True

    router = ModelRouter(make_cfg(fake_llm.url, fake_cloud.url), authorizer=authorizer)
    router.select("nvidia")
    assert text(run(router)) == "Hola, mundo!"

    assert len(seen) == 1
    assert (seen[0].provider, seen[0].model, seen[0].messages) == ("nvidia", "cloud-model", MSGS)
    req = fake_cloud.posts()[0]
    assert req["headers"]["Authorization"] == "Bearer secret-key"
    assert req["body"]["messages"] == MSGS


def test_authorization_is_per_operation(fake_llm, fake_cloud):
    calls = []
    router = ModelRouter(
        make_cfg(fake_llm.url, fake_cloud.url), authorizer=lambda req: calls.append(1) or True
    )
    router.select("nvidia")
    run(router)
    run(router)
    assert len(calls) == 2


def test_local_never_asks_for_authorization(fake_llm, fake_cloud):
    router = ModelRouter(make_cfg(fake_llm.url, fake_cloud.url), authorizer=lambda req: 1 / 0)
    assert text(run(router)) == "Hola, mundo!"


def test_missing_key_fails_before_asking_user(monkeypatch, fake_llm, fake_cloud):
    monkeypatch.delenv("NVIDIA_API_KEY")
    asked = []
    router = ModelRouter(
        make_cfg(fake_llm.url, fake_cloud.url), authorizer=lambda req: asked.append(1) or True
    )
    router.select("nvidia")
    with pytest.raises(MissingCredentials):
        run(router)
    assert asked == [] and fake_cloud.requests == []


def test_no_automatic_fallback_to_cloud(dead_url, fake_cloud):
    router = ModelRouter(make_cfg(dead_url, fake_cloud.url), authorizer=lambda req: True)
    assert router.status()["provider"] == "local"
    with pytest.raises(ProviderUnavailable):
        run(router)
    assert router.status()["provider"] == "local"
    assert fake_cloud.requests == []


def test_switching_provider_needs_no_code_changes(fake_llm, fake_cloud):
    router = ModelRouter(make_cfg(fake_llm.url, fake_cloud.url), authorizer=lambda req: True)
    run(router)
    router.select("nvidia")
    run(router)
    router.select("local")
    run(router)
    assert len(fake_llm.posts()) == 2 and len(fake_cloud.posts()) == 1


def test_cloud_without_model_selected(fake_llm, fake_cloud):
    router = ModelRouter(make_cfg(fake_llm.url, fake_cloud.url, cloud_model=None), authorizer=lambda r: True)
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
