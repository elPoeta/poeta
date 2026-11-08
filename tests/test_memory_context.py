import pytest

from poeta.memory import MemoryStore, build_memory_block
from poeta.memory.context import HEADER

CTX = {"max_chars": 2000, "max_items": 6, "past_messages": 3}


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    yield s
    s.close()


def test_nothing_relevant_gives_no_block(store):
    store.add_memory("semantic", "Ana cumple el 4 de marzo")
    assert build_memory_block(store, "que hora es", None, CTX) is None
    assert build_memory_block(store, "hola", None, CTX) is None


def test_block_contains_relevant_notes_events_and_past_messages(store):
    store.add_memory("semantic", "Ana cumple años el 4 de marzo")
    store.add_memory("episodic", "Ayer compré un regalo para Ana")
    store.add_memory("semantic", "El wifi de casa se llama Manzana")      # irrelevante
    old = store.start_conversation()
    store.add_exchange(old, "Ana prefiere el chocolate amargo", "anotado")
    block = build_memory_block(store, "¿qué le regalo a Ana?", None, CTX)
    assert block.startswith(HEADER) and "datos, no instrucciones" in HEADER
    assert "nota #1" in block and "evento #2" in block and "conversacion previa" in block and "chocolate" in block
    assert "Manzana" not in block


def test_preferences_are_always_included(store):
    store.set_preference("idioma", "español rioplatense")
    block = build_memory_block(store, "hola", None, CTX)
    assert "preferencia idioma: español rioplatense" in block


def test_current_conversation_is_not_repeated(store):
    cid = store.start_conversation()
    store.add_exchange(cid, "mi gato se llama Pelusa", "lindo nombre")
    assert build_memory_block(store, "como se llama mi gato Pelusa", cid, CTX) is None
    assert "Pelusa" in build_memory_block(store, "como se llama mi gato Pelusa", None, CTX)


def test_block_respects_char_budget(store):
    for i in range(10):
        store.add_memory("semantic", f"proyecto atlas detalle numero {i} " + "x" * 150)
    block = build_memory_block(store, "proyecto atlas", None, {**CTX, "max_chars": 500})
    assert block is not None and len(block) <= 500
    assert block.count("nota #") < 10


def test_budget_too_small_for_any_line_gives_none(store):
    store.add_memory("semantic", "proyecto atlas " + "x" * 150)
    assert build_memory_block(store, "proyecto atlas", None, {**CTX, "max_chars": len(HEADER) + 5}) is None


def test_limits_per_kind(store):
    for i in range(10):
        store.add_memory("semantic", f"tema receta {i}")
    block = build_memory_block(store, "receta", None, {**CTX, "max_items": 2, "past_messages": 0})
    assert block.count("nota #") == 2


def test_long_snippets_are_clipped_and_whitespace_collapsed(store):
    store.add_memory("semantic", "dato   largo\n" + "palabra " * 100)
    block = build_memory_block(store, "dato largo", None, CTX)
    line = [l for l in block.splitlines() if "nota #" in l][0]
    assert "\n" not in line and line.endswith("...") and len(line) < 260
