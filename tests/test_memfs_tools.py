"""Tests de las 9 tools de MemFS expuestas al LLM."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from memgpt.agent import _default_tools, build_agent
from memgpt.memfs_store import InMemoryMemFS
from memgpt.memfs_tools import make_memfs_tools


MEMFS_TOOL_NAMES = {
    "memfs_create",
    "memfs_read",
    "memfs_write",
    "memfs_list",
    "memfs_move",
    "memfs_delete",
    "memfs_history",
    "memfs_rollback",
    "memfs_grep",
}


def _by_name(tools, name):
    return next(t for t in tools if t.name == name)


# ----- factory -----


def test_factory_returns_all_nine_tools():
    fs = InMemoryMemFS()
    tools = make_memfs_tools(fs)
    assert {t.name for t in tools} == MEMFS_TOOL_NAMES


def test_default_tools_without_memfs_store_excludes_memfs_tools():
    """Sin ``memfs_store``, las tools de MemFS no entran al catálogo."""
    names = {t.name for t in _default_tools()}
    assert names.isdisjoint(MEMFS_TOOL_NAMES)


def test_default_tools_with_memfs_store_includes_all_nine():
    names = {t.name for t in _default_tools(memfs_store=InMemoryMemFS())}
    assert MEMFS_TOOL_NAMES.issubset(names)


# ----- éxito y errores por tool -----


def test_create_then_read_via_tools():
    fs = InMemoryMemFS()
    tools = make_memfs_tools(fs)
    out = _by_name(tools, "memfs_create").invoke(
        {"path": "/notes/todo.md", "content": "buy milk"}
    )
    assert out.startswith("Created /notes/todo.md (commit ")
    read = _by_name(tools, "memfs_read").invoke({"path": "/notes/todo.md"})
    assert read == "buy milk"


def test_create_duplicate_returns_error_string():
    fs = InMemoryMemFS()
    tools = make_memfs_tools(fs)
    create = _by_name(tools, "memfs_create")
    create.invoke({"path": "/a.md", "content": "x"})
    out = create.invoke({"path": "/a.md", "content": "y"})
    assert out.startswith("ERROR:")
    assert "already exists" in out


def test_read_missing_returns_error_string():
    fs = InMemoryMemFS()
    out = _by_name(make_memfs_tools(fs), "memfs_read").invoke({"path": "/nope.md"})
    assert out.startswith("ERROR:")


def test_write_updates_file_and_creates_commit():
    fs = InMemoryMemFS()
    tools = make_memfs_tools(fs)
    _by_name(tools, "memfs_create").invoke({"path": "/a.md", "content": "v1"})
    out = _by_name(tools, "memfs_write").invoke({"path": "/a.md", "content": "v2"})
    assert out.startswith("Updated /a.md (commit ")
    assert fs.read("/a.md") == "v2"


def test_list_formats_dirs_and_files():
    fs = InMemoryMemFS()
    tools = make_memfs_tools(fs)
    _by_name(tools, "memfs_create").invoke({"path": "/readme.md", "content": "r"})
    _by_name(tools, "memfs_create").invoke({"path": "/proj/a.md", "content": "a"})
    out = _by_name(tools, "memfs_list").invoke({})
    assert "file /readme.md" in out
    assert "dir  /proj/" in out


def test_list_empty_fs_returns_marker():
    fs = InMemoryMemFS()
    out = _by_name(make_memfs_tools(fs), "memfs_list").invoke({})
    assert out == "(empty)"


def test_move_and_delete():
    fs = InMemoryMemFS()
    tools = make_memfs_tools(fs)
    _by_name(tools, "memfs_create").invoke({"path": "/a.md", "content": "x"})
    out_move = _by_name(tools, "memfs_move").invoke({"src": "/a.md", "dst": "/b.md"})
    assert out_move.startswith("Moved /a.md -> /b.md (commit ")
    out_del = _by_name(tools, "memfs_delete").invoke({"path": "/b.md"})
    assert out_del.startswith("Deleted /b.md (commit ")
    assert "/b.md" not in fs.current


def test_history_and_rollback_cycle():
    fs = InMemoryMemFS()
    tools = make_memfs_tools(fs)
    _by_name(tools, "memfs_create").invoke({"path": "/a.md", "content": "v1"})
    _by_name(tools, "memfs_write").invoke({"path": "/a.md", "content": "v2"})
    history_out = _by_name(tools, "memfs_history").invoke({"path": "/a.md"})
    # Dos líneas, dos hashes; el primero corresponde a v1
    lines = history_out.splitlines()
    assert len(lines) == 2
    first_hash = lines[0].split()[1]
    rollback_out = _by_name(tools, "memfs_rollback").invoke(
        {"path": "/a.md", "commit_hash": first_hash}
    )
    assert rollback_out.startswith(f"Rolled back /a.md to {first_hash}")
    assert fs.read("/a.md") == "v1"


def test_rollback_unknown_hash_returns_error():
    fs = InMemoryMemFS()
    tools = make_memfs_tools(fs)
    _by_name(tools, "memfs_create").invoke({"path": "/a.md", "content": "x"})
    out = _by_name(tools, "memfs_rollback").invoke(
        {"path": "/a.md", "commit_hash": "00000000"}
    )
    assert out.startswith("ERROR:")


def test_grep_substring_and_regex_and_scoped():
    fs = InMemoryMemFS()
    tools = make_memfs_tools(fs)
    _by_name(tools, "memfs_create").invoke(
        {"path": "/proj/notes.md", "content": "alpha\nbeta\nalphabet"}
    )
    _by_name(tools, "memfs_create").invoke(
        {"path": "/other.md", "content": "alpha-other"}
    )
    out_sub = _by_name(tools, "memfs_grep").invoke({"pattern": "alpha"})
    # 2 hits en /proj/notes.md + 1 en /other.md = 3 líneas
    assert len(out_sub.splitlines()) == 3
    out_scope = _by_name(tools, "memfs_grep").invoke(
        {"pattern": "alpha", "path": "/proj"}
    )
    assert "/other.md" not in out_scope
    out_regex = _by_name(tools, "memfs_grep").invoke(
        {"pattern": r"^alpha$", "regex": True}
    )
    assert "/proj/notes.md:1:alpha" in out_regex
    assert "alphabet" not in out_regex  # ^alpha$ no matchea alphabet


def test_grep_no_matches():
    fs = InMemoryMemFS()
    _by_name(make_memfs_tools(fs), "memfs_create").invoke(
        {"path": "/a.md", "content": "x"}
    )
    out = _by_name(make_memfs_tools(fs), "memfs_grep").invoke({"pattern": "zzz"})
    assert out == "(no matches)"


def test_grep_invalid_regex_returns_error():
    fs = InMemoryMemFS()
    _by_name(make_memfs_tools(fs), "memfs_create").invoke(
        {"path": "/a.md", "content": "x"}
    )
    out = _by_name(make_memfs_tools(fs), "memfs_grep").invoke(
        {"pattern": "(", "regex": True}
    )
    assert out.startswith("ERROR:")


# ----- end-to-end vía agente con ScriptedLLM -----


class _ScriptedLLM:
    """Stub que reproduce una lista de ``AIMessage`` predeterminados."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        if not self.responses:
            return AIMessage(content="done")
        return self.responses.pop(0)


def _tool_call(name: str, args: dict, call_id: str = "c1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_e2e_create_then_history_via_agent():
    """El LLM llama memfs_create y luego memfs_history; el store cambia."""
    llm = _ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "memfs_create",
                        {"path": "/projects/p.md", "content": "plan v1"},
                        call_id="c1",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "memfs_history",
                        {"path": "/projects/p.md"},
                        call_id="c2",
                    )
                ],
            ),
            AIMessage(content="historia recuperada"),
        ]
    )
    fs = InMemoryMemFS()
    agent = build_agent(
        llm=llm,
        checkpointer=MemorySaver(),
        memfs_store=fs,
    )
    cfg = {"configurable": {"thread_id": "t-memfs"}}
    agent.invoke({"messages": [HumanMessage(content="haz cosas")]}, config=cfg)

    assert fs.read("/projects/p.md") == "plan v1"
    msgs = agent.get_state(cfg).values["messages"]
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    # 1 ToolMessage por cada tool call: create + history
    assert len(tool_msgs) == 2
    assert "Created /projects/p.md" in tool_msgs[0].content
    assert "/projects/p.md" in tool_msgs[1].content  # historia formateada
