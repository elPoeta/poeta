import asyncio
import builtins

import pytest

from poeta.interface.cli_confirm import cli_confirmer, render
from poeta.security import ActionRequest


def answers(monkeypatch, *values):
    it = iter(values)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(it))


def req(text="hola"):
    return ActionRequest(
        "cloud_transfer", "Enviar a nvidia", {"provider": "nvidia"},
        payload={"messages": [{"role": "user", "content": text}], "tools": None, "params": {}},
    )


@pytest.mark.parametrize("value, expected", [("s", True), ("SI", True), ("sí", True), ("n", False), ("", False), ("quizas", False)])
def test_simple_confirmation(monkeypatch, value, expected):
    answers(monkeypatch, value)
    assert asyncio.run(cli_confirmer(req(), False)) is expected


@pytest.mark.parametrize("value, expected", [("CONFIRMAR", True), ("s", False), ("confirmar", False), ("", False)])
def test_explicit_requires_the_word(monkeypatch, value, expected):
    answers(monkeypatch, value)
    assert asyncio.run(cli_confirmer(req(), True)) is expected


def test_eof_and_ctrl_c_cancel(monkeypatch):
    for exc in (EOFError, KeyboardInterrupt):
        def raiser(prompt="", exc=exc):
            raise exc
        monkeypatch.setattr(builtins, "input", raiser)
        assert asyncio.run(cli_confirmer(req(), False)) is False


def test_view_full_then_answer(monkeypatch, capsys):
    long_text = "a" * 1000
    answers(monkeypatch, "v", "s")
    assert asyncio.run(cli_confirmer(req(long_text), False)) is True
    out = capsys.readouterr().out
    assert "+700 caracteres" in out          # primera vista: recortada y avisando que se envia completo
    assert out.count("a" * 1000) == 1        # tras 'v': completa


def test_render_shows_what_will_be_sent_and_flags():
    r = req("texto exacto")
    r.tainted = True
    text = render(r, explicit=True)
    assert "EXPLICITA" in text and "texto exacto" in text and "nvidia" in text and "NO CONFIABLE" in text


def test_render_marks_non_text_parts():
    r = ActionRequest(
        "cloud_transfer", "x", payload={"messages": [{"role": "user", "content": [
            {"type": "text", "text": "mira"}, {"type": "image_url", "image_url": {"url": "data:..."}}]}]},
    )
    assert "contenido no textual: image_url" in render(r, True)
