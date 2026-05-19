import pytest
from pydantic import ValidationError

from memgpt.core_memory import CoreMemory, MemoryBlock, default_core_memory


def test_memory_block_label_must_be_snake_case():
    MemoryBlock(label="project_alpha")
    MemoryBlock(label="x")
    for bad in ["Project", "project-alpha", "1block", "a" * 33, "with space", ""]:
        with pytest.raises(ValidationError):
            MemoryBlock(label=bad)


def test_memory_block_value_must_fit_in_limit():
    with pytest.raises(ValidationError):
        MemoryBlock(label="tiny", value="word " * 500, limit=5)


def test_memory_block_limit_must_be_positive():
    with pytest.raises(ValidationError):
        MemoryBlock(label="tiny", limit=0)


def test_default_core_memory_has_assistant_and_human():
    cm = default_core_memory()
    assert set(cm.blocks) == {"assistant", "human"}
    assert cm.get("assistant").value
    assert cm.get("human").value == ""


def test_with_block_is_immutable_and_validates_idempotency():
    cm = default_core_memory()
    cm2 = cm.with_block(MemoryBlock(label="project_alpha", value="initial"))
    assert "project_alpha" in cm2.blocks
    assert "project_alpha" not in cm.blocks
    with pytest.raises(ValueError):
        cm2.with_block(MemoryBlock(label="project_alpha", value="dup"))


def test_with_block_respects_max_blocks():
    cm = CoreMemory(blocks={}, max_blocks=2, total_token_budget=10000)
    cm = cm.with_block(MemoryBlock(label="a"))
    cm = cm.with_block(MemoryBlock(label="b"))
    with pytest.raises(ValueError):
        cm.with_block(MemoryBlock(label="c"))


def test_total_token_budget_is_enforced():
    with pytest.raises(ValidationError):
        CoreMemory(
            blocks={
                "a": MemoryBlock(label="a", limit=3000),
                "b": MemoryBlock(label="b", limit=3000),
            },
            total_token_budget=5000,
        )


def test_dict_key_must_match_block_label():
    with pytest.raises(ValidationError):
        CoreMemory(blocks={"a": MemoryBlock(label="b")})


def test_with_appended_creates_new_value():
    cm = default_core_memory(human="")
    cm2 = cm.with_appended("human", "name=Maximo")
    assert cm2.get("human").value == "name=Maximo"
    cm3 = cm2.with_appended("human", "lang=es")
    assert cm3.get("human").value == "name=Maximo\nlang=es"
    assert cm.get("human").value == ""


def test_with_appended_rejects_overflow():
    cm = CoreMemory(
        blocks={"tiny": MemoryBlock(label="tiny", value="hello", limit=3)},
        total_token_budget=100,
    )
    with pytest.raises(ValidationError):
        cm.with_appended("tiny", " world this is a long sentence")


def test_with_replaced_substitutes_first_occurrence():
    cm = default_core_memory(human="name=Bob\nrole=Bob")
    cm2 = cm.with_replaced("human", "Bob", "Alice")
    assert cm2.get("human").value == "name=Alice\nrole=Bob"


def test_with_replaced_fails_when_text_missing():
    cm = default_core_memory(human="name=Bob")
    with pytest.raises(ValueError):
        cm.with_replaced("human", "Charlie", "Alice")


def test_without_block_removes_block():
    cm = default_core_memory()
    cm2 = cm.without_block("human")
    assert not cm2.has("human")
    assert cm.has("human")
    with pytest.raises(KeyError):
        cm.without_block("nonexistent")


def test_to_prompt_text_lists_all_blocks():
    cm = default_core_memory(human="name=Maximo")
    text = cm.to_prompt_text()
    assert "Core Memory" in text
    assert "[assistant]" in text
    assert "[human]" in text
    assert "name=Maximo" in text


def test_to_prompt_text_empty_when_no_blocks():
    cm = CoreMemory(blocks={})
    assert cm.to_prompt_text() == ""
