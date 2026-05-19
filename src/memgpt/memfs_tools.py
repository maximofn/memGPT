"""Tools de MemFS expuestas al LLM.

Patrón **factory + closure** idéntico al de Recall/Archival
(`make_recall_archival_tools` en ``recall_archival_tools.py``): el
``MemFSStore`` se inyecta en el closure de cada tool porque no es
serializable y no debe pasar por el ``MemGPTState`` ni el checkpointer.

Tools (9 en total, alineadas con ``memGPT-resumen.md`` y la fase 12.3 del
plan, más ``grep`` decidido con el usuario):

- ``memfs_create(path, content)``
- ``memfs_read(path)``
- ``memfs_write(path, content)``
- ``memfs_list(path?)``
- ``memfs_move(src, dst)``
- ``memfs_delete(path)``
- ``memfs_history(path?)``
- ``memfs_rollback(path, commit_hash)``
- ``memfs_grep(pattern, path?, regex?)``

Las tools devuelven **texto plano** orientado a que el LLM lo lea en el
siguiente turno. Errores (paths inexistentes, regex inválidos, etc.) se
emiten como ``"ERROR: ..."`` en lugar de levantar excepciones, para que el
agente pueda reaccionar sin romper el grafo.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import tool

from .memfs_store import CommitInfo, GrepHit, MemFSStore


def _format_history(commits: list[CommitInfo]) -> str:
    if not commits:
        return "(no history)"
    return "\n".join(
        f"[{c.timestamp.isoformat()}] {c.commit_hash} — {c.summary}" for c in commits
    )


def _format_list(entries: list[tuple[str, str]]) -> str:
    if not entries:
        return "(empty)"
    lines: list[str] = []
    for kind, full_path in entries:
        if kind == "dir":
            lines.append(f"dir  {full_path}/")
        else:
            lines.append(f"file {full_path}")
    return "\n".join(lines)


def _format_grep(hits: list[GrepHit]) -> str:
    if not hits:
        return "(no matches)"
    return "\n".join(f"{h.path}:{h.line_number}:{h.line}" for h in hits)


def make_memfs_tools(store: MemFSStore) -> list[Callable[..., Any]]:
    """Construye las 9 tools de MemFS cerradas sobre ``store``.

    Returns:
        Lista de tools decoradas con ``@tool``, lista para pasar a
        ``build_agent(tools=...)`` o a un ``ToolNode``.
    """

    @tool
    def memfs_create(path: str, content: str) -> str:
        """Create a new file in MemFS at ``path`` with ``content``.

        Fails if the file already exists, if ``path`` collides with an
        existing directory, or if ``path`` is not a well-formed absolute
        path (must start with ``/``).

        Args:
            path: Absolute path of the new file (e.g. ``/notes/todo.md``).
            content: Initial content of the file.

        Returns:
            ``"Created /path (commit a1b2c3d4)"`` on success or
            ``"ERROR: ..."`` describing the failure.
        """
        try:
            commit = store.create(path, content)
        except (KeyError, ValueError) as exc:
            return f"ERROR: {exc}"
        return f"Created {path} (commit {commit})"

    @tool
    def memfs_read(path: str) -> str:
        """Return the current content of the file at ``path``.

        Fails if the file does not exist.

        Args:
            path: Absolute path of the file to read.

        Returns:
            The file content as-is, or ``"ERROR: ..."`` if not found.
        """
        try:
            return store.read(path)
        except (KeyError, ValueError) as exc:
            return f"ERROR: {exc}"

    @tool
    def memfs_write(path: str, content: str) -> str:
        """Overwrite the file at ``path`` with new ``content``.

        Fails if the file does not exist (use ``memfs_create`` to make a
        new file). Writing the same content as before is a no-op and does
        not create a new commit.

        Args:
            path: Absolute path of the existing file.
            content: New content (replaces the previous content entirely).

        Returns:
            ``"Updated /path (commit b5c6d7e8)"`` on success or
            ``"ERROR: ..."`` describing the failure.
        """
        try:
            commit = store.write(path, content)
        except (KeyError, ValueError) as exc:
            return f"ERROR: {exc}"
        return f"Updated {path} (commit {commit})"

    @tool
    def memfs_list(path: str | None = None) -> str:
        """List the direct children of ``path`` (default: filesystem root).

        Returns one entry per line as ``file /path/name`` or
        ``dir /path/name/``. Subdirectories are inferred from file paths —
        MemFS does not store empty directories.

        Args:
            path: Optional absolute path of the directory to list. Leave
                empty to list the root.

        Returns:
            Entries one per line, ``"(empty)"`` if nothing matches, or
            ``"ERROR: ..."`` if the directory does not exist.
        """
        try:
            entries = store.list(path)
        except (KeyError, ValueError) as exc:
            return f"ERROR: {exc}"
        return _format_list(entries)

    @tool
    def memfs_move(src: str, dst: str) -> str:
        """Rename or move the file at ``src`` to ``dst``.

        Fails if ``src`` does not exist, if ``dst`` already exists, or if
        ``src == dst``.

        Args:
            src: Current absolute path.
            dst: Target absolute path.

        Returns:
            ``"Moved /src -> /dst (commit ...)"`` on success or
            ``"ERROR: ..."`` describing the failure.
        """
        try:
            commit = store.move(src, dst)
        except (KeyError, ValueError) as exc:
            return f"ERROR: {exc}"
        return f"Moved {src} -> {dst} (commit {commit})"

    @tool
    def memfs_delete(path: str) -> str:
        """Delete the file at ``path``.

        The deletion is recorded as a commit, so the previous version can
        still be recovered via ``memfs_rollback``.

        Args:
            path: Absolute path of the file to delete.

        Returns:
            ``"Deleted /path (commit ...)"`` on success or
            ``"ERROR: ..."`` if the file does not exist.
        """
        try:
            commit = store.delete(path)
        except (KeyError, ValueError) as exc:
            return f"ERROR: {exc}"
        return f"Deleted {path} (commit {commit})"

    @tool
    def memfs_history(path: str | None = None) -> str:
        """List commits, optionally filtered to those touching ``path``.

        Each line is ``[ISO8601 timestamp] <hash> — <summary>``. Use this
        before ``memfs_rollback`` to find the commit hash to restore.

        Args:
            path: Optional absolute path. If given, only commits where the
                file content changed (including creation/deletion) appear.

        Returns:
            History lines or ``"(no history)"``.
        """
        try:
            commits = store.history(path)
        except (KeyError, ValueError) as exc:
            return f"ERROR: {exc}"
        return _format_history(commits)

    @tool
    def memfs_rollback(path: str, commit_hash: str) -> str:
        """Restore ``path`` to the content it had at ``commit_hash``.

        Creates a **new** commit on top of history — the rollback is itself
        recorded, no previous commit is destroyed. Useful to recover from
        accidental overwrites or deletions.

        Args:
            path: Absolute path to restore.
            commit_hash: Commit hash where the desired version lives.

        Returns:
            ``"Rolled back /path to <hash> (new commit ...)"`` on success
            or ``"ERROR: ..."`` if the commit or file is unknown.
        """
        try:
            new_commit = store.rollback(path, commit_hash)
        except (KeyError, ValueError) as exc:
            return f"ERROR: {exc}"
        return f"Rolled back {path} to {commit_hash} (new commit {new_commit})"

    @tool
    def memfs_grep(
        pattern: str,
        path: str | None = None,
        regex: bool = False,
    ) -> str:
        """Search file contents in MemFS for ``pattern``.

        Args:
            pattern: Substring or regular expression to look for.
            path: Optional absolute path of a directory to restrict the
                search to (recursive). Leave empty to search the whole FS.
            regex: When ``True``, treat ``pattern`` as a regular expression
                (Python ``re`` syntax). Default ``False`` does substring
                matching, which is faster and never raises on user input.

        Returns:
            One match per line as ``/path:LINE:line content`` or
            ``"(no matches)"``.
        """
        try:
            hits = store.grep(pattern, path, regex=regex)
        except (KeyError, ValueError) as exc:
            return f"ERROR: {exc}"
        return _format_grep(hits)

    return [
        memfs_create,
        memfs_read,
        memfs_write,
        memfs_list,
        memfs_move,
        memfs_delete,
        memfs_history,
        memfs_rollback,
        memfs_grep,
    ]
