"""Tests del backend ``InMemoryMemFS`` (versionado por snapshots)."""

from __future__ import annotations

import pytest

from memgpt.memfs_store import InMemoryMemFS, normalize_path


# ----- normalize_path -----


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("/foo", "/foo"),
        ("/foo/bar.md", "/foo/bar.md"),
        ("/foo//bar", "/foo/bar"),
        ("/foo/./bar", "/foo/bar"),
        ("/foo/baz/../bar", "/foo/bar"),
        ("  /foo/bar  ", "/foo/bar"),
    ],
)
def test_normalize_path_valid(raw, expected):
    assert normalize_path(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "relative/path", "/", "/..", "/has\x00nul"],
)
def test_normalize_path_rejects(raw):
    with pytest.raises(ValueError):
        normalize_path(raw)


def test_normalize_path_collapses_dotdot_at_root():
    # posixpath.normpath colapsa segmentos sobrantes al inicio: '/../x' → '/x'.
    # No escapamos del root, así que el path es válido tras normalizar.
    assert normalize_path("/../escape") == "/escape"


# ----- CRUD básico + commits -----


def test_create_read_write_basic():
    fs = InMemoryMemFS()
    h1 = fs.create("/notes/todo.md", "buy milk")
    assert fs.read("/notes/todo.md") == "buy milk"
    h2 = fs.write("/notes/todo.md", "buy milk\nbuy eggs")
    assert h1 != h2
    assert fs.read("/notes/todo.md") == "buy milk\nbuy eggs"
    assert len(fs.commits) == 2


def test_create_rejects_duplicate():
    fs = InMemoryMemFS()
    fs.create("/a.md", "x")
    with pytest.raises(ValueError):
        fs.create("/a.md", "y")


def test_create_rejects_directory_collision():
    fs = InMemoryMemFS()
    fs.create("/proj/notes.md", "x")
    # /proj se ha vuelto un "directorio implícito"; no se puede crear como fichero.
    with pytest.raises(ValueError):
        fs.create("/proj", "anything")


def test_write_requires_existing_file():
    fs = InMemoryMemFS()
    with pytest.raises(KeyError):
        fs.write("/missing.md", "x")


def test_write_noop_returns_last_commit():
    fs = InMemoryMemFS()
    h1 = fs.create("/a.md", "same")
    h2 = fs.write("/a.md", "same")
    assert h1 == h2
    assert len(fs.commits) == 1  # no se añade commit "vacío"


def test_read_rejects_missing():
    fs = InMemoryMemFS()
    with pytest.raises(KeyError):
        fs.read("/nope.md")


def test_path_normalization_applied_in_ops():
    fs = InMemoryMemFS()
    fs.create("/a/../b.md", "hello")
    # /a/../b.md normaliza a /b.md
    assert fs.read("/b.md") == "hello"


# ----- list / move / delete -----


def test_list_root_and_subdir():
    fs = InMemoryMemFS()
    fs.create("/readme.md", "r")
    fs.create("/proj/a.md", "a")
    fs.create("/proj/sub/b.md", "b")
    root = fs.list()
    assert ("file", "/readme.md") in root
    assert ("dir", "/proj") in root
    proj = fs.list("/proj")
    assert ("file", "/proj/a.md") in proj
    assert ("dir", "/proj/sub") in proj


def test_list_rejects_unknown_directory():
    fs = InMemoryMemFS()
    fs.create("/a.md", "x")
    with pytest.raises(KeyError):
        fs.list("/does/not/exist")


def test_move_renames_and_commits():
    fs = InMemoryMemFS()
    fs.create("/a.md", "content")
    h = fs.move("/a.md", "/b.md")
    assert fs.read("/b.md") == "content"
    with pytest.raises(KeyError):
        fs.read("/a.md")
    assert fs.commits[-1].commit_hash == h


def test_move_rejects_same_or_existing_destination():
    fs = InMemoryMemFS()
    fs.create("/a.md", "x")
    fs.create("/b.md", "y")
    with pytest.raises(ValueError):
        fs.move("/a.md", "/a.md")
    with pytest.raises(ValueError):
        fs.move("/a.md", "/b.md")


def test_delete_removes_file_and_keeps_history():
    fs = InMemoryMemFS()
    h_create = fs.create("/a.md", "x")
    h_delete = fs.delete("/a.md")
    assert "/a.md" not in fs.current
    # Tras borrar, history del path debe incluir creación y borrado.
    path_history = fs.history("/a.md")
    hashes = [c.commit_hash for c in path_history]
    assert h_create in hashes
    assert h_delete in hashes


# ----- history -----


def test_history_all_vs_per_path():
    fs = InMemoryMemFS()
    h_a = fs.create("/a.md", "a")
    h_b = fs.create("/b.md", "b")
    h_a2 = fs.write("/a.md", "a2")
    all_hist = [c.commit_hash for c in fs.history()]
    assert all_hist == [h_a, h_b, h_a2]
    a_hist = [c.commit_hash for c in fs.history("/a.md")]
    assert a_hist == [h_a, h_a2]  # h_b no toca /a.md, se filtra
    b_hist = [c.commit_hash for c in fs.history("/b.md")]
    assert b_hist == [h_b]


# ----- rollback -----


def test_rollback_restores_previous_content():
    fs = InMemoryMemFS()
    h1 = fs.create("/a.md", "v1")
    fs.write("/a.md", "v2")
    fs.write("/a.md", "v3")
    assert fs.read("/a.md") == "v3"
    new = fs.rollback("/a.md", h1)
    assert fs.read("/a.md") == "v1"
    assert new != h1
    # Rollback no destruye historial — sigue habiendo 4 commits.
    assert len(fs.history()) == 4


def test_rollback_recreates_deleted_file():
    fs = InMemoryMemFS()
    h1 = fs.create("/a.md", "v1")
    fs.delete("/a.md")
    assert "/a.md" not in fs.current
    fs.rollback("/a.md", h1)
    assert fs.read("/a.md") == "v1"


def test_rollback_rejects_unknown_commit():
    fs = InMemoryMemFS()
    fs.create("/a.md", "x")
    with pytest.raises(KeyError):
        fs.rollback("/a.md", "deadbeef")


def test_rollback_rejects_file_not_in_commit():
    fs = InMemoryMemFS()
    h1 = fs.create("/a.md", "x")
    fs.create("/b.md", "y")
    # /b.md no existía en h1, no se puede rollback a esa versión
    with pytest.raises(KeyError):
        fs.rollback("/b.md", h1)


# ----- grep -----


def test_grep_substring_across_files():
    fs = InMemoryMemFS()
    fs.create("/a.md", "alpha\nbeta\nalphabet")
    fs.create("/b.md", "gamma\ndelta")
    hits = fs.grep("alpha")
    paths = {(h.path, h.line_number) for h in hits}
    assert paths == {("/a.md", 1), ("/a.md", 3)}


def test_grep_regex():
    fs = InMemoryMemFS()
    fs.create("/a.md", "foo123\nbar\nbaz999")
    hits = fs.grep(r"\d{3}", regex=True)
    assert {h.line for h in hits} == {"foo123", "baz999"}


def test_grep_invalid_regex_raises():
    fs = InMemoryMemFS()
    fs.create("/a.md", "x")
    with pytest.raises(ValueError):
        fs.grep("(", regex=True)


def test_grep_scoped_to_subdir():
    fs = InMemoryMemFS()
    fs.create("/proj/x.md", "needle")
    fs.create("/other.md", "needle")
    hits = fs.grep("needle", "/proj")
    assert [h.path for h in hits] == ["/proj/x.md"]


def test_grep_empty_pattern_rejected():
    fs = InMemoryMemFS()
    with pytest.raises(ValueError):
        fs.grep("")


def test_grep_no_matches_returns_empty():
    fs = InMemoryMemFS()
    fs.create("/a.md", "alpha")
    assert fs.grep("zzz") == []


# ----- hashing -----


def test_commit_hashes_unique_even_for_same_content():
    fs = InMemoryMemFS()
    fs.create("/a.md", "x")
    fs.delete("/a.md")
    h_recreate = fs.create("/a.md", "x")
    # Mismo snapshot final que tras el primer create, pero los hashes
    # llevan timestamp y deben divergir.
    all_hashes = [c.commit_hash for c in fs.history()]
    assert len(set(all_hashes)) == len(all_hashes)
    assert h_recreate == fs.history()[-1].commit_hash
