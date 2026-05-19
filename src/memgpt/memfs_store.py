"""MemFS: memoria del agente organizada como filesystem versionado.

Capa nueva que se suma a Core Memory / Recall / Archival. A diferencia de
Core Memory (slot dentro del prompt) y de Recall/Archival (búsqueda semántica
sobre episodios), MemFS expone una **jerarquía de ficheros con historial
completo de cambios**, pensada para conocimiento estructurado que el agente
quiere organizar por path y poder retroceder en el tiempo.

`MemFSStore` es la abstracción; `InMemoryMemFS` el único backend incluido en
esta iteración. La persistencia entre sesiones (DiskMemFS / GraphitiMemFS)
queda fuera de scope — la abstracción permite añadirlas sin tocar tools.

Las operaciones que mutan generan un **commit** (snapshot completo del
estado + hash truncado SHA-1). Esto evita la dependencia de `dulwich`: para
una capa in-memory, snapshots simples son equivalentes a git y mucho más
fáciles de razonar y testear.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class CommitInfo:
    """Metadatos de un commit visibles al consumidor (no incluye snapshot)."""

    commit_hash: str
    timestamp: datetime
    summary: str


@dataclass(frozen=True)
class GrepHit:
    """Un match de ``grep`` dentro de un fichero."""

    path: str
    line_number: int
    line: str


def normalize_path(path: str) -> str:
    """Devuelve un path absoluto canonicalizado.

    Reglas:

    - Strip de espacios.
    - Debe empezar por ``/``; si no, se rechaza con ``ValueError``.
    - ``posixpath.normpath`` colapsa ``//``, ``./`` y resuelve ``..`` dentro
      del propio path (un ``..`` que se escape de la raíz se rechaza).
    - Sin trailing ``/`` (salvo la raíz, que aquí no es un path válido para
      ficheros).
    - Sin caracteres de control.

    Lanza ``ValueError`` con mensaje claro si el path no es válido.
    """
    if not isinstance(path, str):
        raise ValueError(f"path must be a string, got {type(path).__name__}")
    p = path.strip()
    if not p:
        raise ValueError("path cannot be empty")
    if not p.startswith("/"):
        raise ValueError(f"path must be absolute (start with '/'): {path!r}")
    if any(ord(c) < 32 for c in p):
        raise ValueError(f"path contains control characters: {path!r}")
    norm = posixpath.normpath(p)
    if norm == "/":
        raise ValueError("path '/' is not a valid file path")
    if norm.startswith("../") or norm == "..":
        raise ValueError(f"path escapes root: {path!r}")
    return norm


class MemFSStore(ABC):
    """Interfaz de un backend MemFS.

    Todas las operaciones mutativas crean un commit y devuelven su hash.
    Las lecturas no crean commits.
    """

    @abstractmethod
    def create(self, path: str, content: str) -> str: ...

    @abstractmethod
    def read(self, path: str) -> str: ...

    @abstractmethod
    def write(self, path: str, content: str) -> str: ...

    @abstractmethod
    def list(self, path: str | None = None) -> list[tuple[str, str]]: ...

    @abstractmethod
    def move(self, src: str, dst: str) -> str: ...

    @abstractmethod
    def delete(self, path: str) -> str: ...

    @abstractmethod
    def history(self, path: str | None = None) -> list[CommitInfo]: ...

    @abstractmethod
    def rollback(self, path: str, commit_hash: str) -> str: ...

    @abstractmethod
    def grep(
        self,
        pattern: str,
        path: str | None = None,
        *,
        regex: bool = False,
    ) -> list[GrepHit]: ...


@dataclass(frozen=True)
class _Commit:
    """Un commit interno guarda metadata + snapshot completo del FS.

    El snapshot es ``dict[path, content]`` inmutable — un nuevo commit copia
    el snapshot anterior y aplica el cambio. Para un MemFS in-memory con
    decenas o cientos de ficheros esto es asumible; si crece, sustituir por
    delta compression.
    """

    commit_hash: str
    timestamp: datetime
    summary: str
    snapshot: dict[str, str]


class InMemoryMemFS(MemFSStore):
    """Backend MemFS en memoria con historial vía snapshots.

    No hay persistencia entre procesos: pensado para tests, sesiones
    efímeras y validar la lógica del agente. La interfaz es la misma que
    expondría un ``DiskMemFS`` con dulwich, así que reemplazar el backend
    no toca tools ni agent.
    """

    def __init__(self) -> None:
        self._current: dict[str, str] = {}
        self._commits: list[_Commit] = []

    # ------- helpers internos -------

    def _commit(self, summary: str) -> str:
        """Sella un commit con el snapshot actual y devuelve su hash."""
        ts = datetime.now(timezone.utc)
        payload = json.dumps(self._current, sort_keys=True, ensure_ascii=False)
        digest_input = f"{ts.isoformat()}|{summary}|{payload}".encode("utf-8")
        commit_hash = hashlib.sha1(digest_input).hexdigest()[:8]
        # Colisión 1/2^32: regenera añadiendo nonce de longitud de la lista.
        if any(c.commit_hash == commit_hash for c in self._commits):
            digest_input = (
                f"{ts.isoformat()}|{summary}|{payload}|{len(self._commits)}"
            ).encode("utf-8")
            commit_hash = hashlib.sha1(digest_input).hexdigest()[:8]
        self._commits.append(
            _Commit(
                commit_hash=commit_hash,
                timestamp=ts,
                summary=summary,
                snapshot=dict(self._current),
            )
        )
        return commit_hash

    def _require_exists(self, path: str) -> None:
        if path not in self._current:
            raise KeyError(f"no such file: {path}")

    def _require_absent(self, path: str) -> None:
        if path in self._current:
            raise ValueError(f"path already exists: {path}")
        # Tampoco permitimos crear un fichero que sea prefijo de otro
        # existente (sería un fichero con el mismo nombre que un directorio).
        prefix = path + "/"
        if any(p.startswith(prefix) for p in self._current):
            raise ValueError(
                f"path {path!r} is a directory (has children); cannot create as file"
            )

    # ------- API pública -------

    def create(self, path: str, content: str) -> str:
        p = normalize_path(path)
        self._require_absent(p)
        self._current[p] = content
        return self._commit(f"Created {p}")

    def read(self, path: str) -> str:
        p = normalize_path(path)
        self._require_exists(p)
        return self._current[p]

    def write(self, path: str, content: str) -> str:
        p = normalize_path(path)
        self._require_exists(p)
        if self._current[p] == content:
            # Sin cambios: no creamos un commit "vacío"; devolvemos el último.
            if self._commits:
                return self._commits[-1].commit_hash
        self._current[p] = content
        return self._commit(f"Updated {p}")

    def list(self, path: str | None = None) -> list[tuple[str, str]]:
        """Lista entradas directas bajo ``path`` (default: raíz).

        Devuelve tuplas ``(kind, full_path)`` con ``kind`` ∈ ``{"file","dir"}``.
        Una entrada es directorio si hay al menos un fichero cuyo path
        empieza por ``prefix + "/"``. Ordenado alfabéticamente.
        """
        prefix = "/" if path is None else normalize_path(path) + "/"
        if path is not None:
            # Para listar un directorio, debe existir al menos una entrada
            # debajo. Si no, es un path inválido (no existe el "dir").
            if not any(p.startswith(prefix) for p in self._current):
                raise KeyError(f"no such directory: {path}")
        files: set[str] = set()
        dirs: set[str] = set()
        for p in self._current:
            if not p.startswith(prefix):
                continue
            rest = p[len(prefix) :]
            if "/" in rest:
                first = rest.split("/", 1)[0]
                dirs.add(prefix + first)
            else:
                files.add(p)
        entries: list[tuple[str, str]] = []
        for d in sorted(dirs):
            entries.append(("dir", d))
        for f in sorted(files):
            entries.append(("file", f))
        return entries

    def move(self, src: str, dst: str) -> str:
        s = normalize_path(src)
        d = normalize_path(dst)
        self._require_exists(s)
        if s == d:
            raise ValueError(f"source and destination are the same: {s}")
        self._require_absent(d)
        self._current[d] = self._current.pop(s)
        return self._commit(f"Moved {s} -> {d}")

    def delete(self, path: str) -> str:
        p = normalize_path(path)
        self._require_exists(p)
        del self._current[p]
        return self._commit(f"Deleted {p}")

    def history(self, path: str | None = None) -> list[CommitInfo]:
        """Devuelve los commits relevantes en orden cronológico.

        - Si ``path`` es ``None``: todos los commits.
        - Si ``path`` se da: solo commits que tocan ese fichero (contenido
          cambia entre commit y su predecesor, incluyendo creación/borrado).
        """
        if path is None:
            return [
                CommitInfo(c.commit_hash, c.timestamp, c.summary)
                for c in self._commits
            ]
        p = normalize_path(path)
        out: list[CommitInfo] = []
        prev: str | None = None
        for c in self._commits:
            curr = c.snapshot.get(p)
            if curr != prev:
                out.append(CommitInfo(c.commit_hash, c.timestamp, c.summary))
            prev = curr
        return out

    def rollback(self, path: str, commit_hash: str) -> str:
        p = normalize_path(path)
        target: _Commit | None = next(
            (c for c in self._commits if c.commit_hash == commit_hash), None
        )
        if target is None:
            raise KeyError(f"no such commit: {commit_hash}")
        if p not in target.snapshot:
            raise KeyError(f"file {p} did not exist at commit {commit_hash}")
        self._current[p] = target.snapshot[p]
        return self._commit(f"Rolled back {p} to {commit_hash}")

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        *,
        regex: bool = False,
    ) -> list[GrepHit]:
        if not pattern:
            raise ValueError("pattern cannot be empty")
        prefix = None if path is None else normalize_path(path) + "/"
        if regex:
            try:
                rx = re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex: {exc}") from exc
            matcher = lambda line: rx.search(line) is not None  # noqa: E731
        else:
            needle = pattern
            matcher = lambda line: needle in line  # noqa: E731
        hits: list[GrepHit] = []
        for fpath in sorted(self._current):
            if prefix is not None and not fpath.startswith(prefix):
                continue
            for i, line in enumerate(self._current[fpath].splitlines(), start=1):
                if matcher(line):
                    hits.append(GrepHit(path=fpath, line_number=i, line=line))
        return hits

    # ------- introspección para tests -------

    @property
    def current(self) -> dict[str, str]:
        """Snapshot inmutable del estado actual (sin commits)."""
        return dict(self._current)

    @property
    def commits(self) -> list[CommitInfo]:
        return [CommitInfo(c.commit_hash, c.timestamp, c.summary) for c in self._commits]
