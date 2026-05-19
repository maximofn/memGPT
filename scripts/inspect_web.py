"""Visualizador web de la ventana de contexto del agente MemGPT.

Sirve una página en `http://localhost:8000` con tres paneles —
**System Prompt**, **Working Context** (Core Memory + recursive summary)
y **FIFO Queue** (mensajes vivos) — más un input de chat para empujar
turnos y ver el contexto evolucionar en vivo.

Uso típico:

    uv run scripts/inspect_web.py
    uv run scripts/inspect_web.py --port 8080 --thread debug

Sin dependencias extra: usa `http.server` de stdlib. Solo modo in-memory
(MemorySaver + sin Recall/Archival) — para inspeccionar un thread
persistido habría que cablear `--persistent` igual que en `chat.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from dotenv import load_dotenv

# Mismo ruido benigno que en chat.py.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from memgpt.agent import DEFAULT_SYSTEM_PROMPT, build_agent
from memgpt.queue_manager import QueueManagerConfig, count_state_tokens


_agent = None
_cfg: dict[str, Any] = {}
_system_prompt = DEFAULT_SYSTEM_PROMPT
_queue_config = QueueManagerConfig.from_settings()
_lock = threading.Lock()


def _message_dict(m: BaseMessage) -> dict[str, Any]:
    if isinstance(m, HumanMessage):
        role = "human"
    elif isinstance(m, AIMessage):
        role = "ai"
    elif isinstance(m, ToolMessage):
        role = "tool"
    elif isinstance(m, SystemMessage):
        role = "system"
    else:
        role = type(m).__name__.lower()
    content = m.content if isinstance(m.content, str) else str(m.content)
    out: dict[str, Any] = {"role": role, "content": content}
    if isinstance(m, AIMessage) and m.tool_calls:
        out["tool_calls"] = [
            {"name": tc.get("name"), "args": tc.get("args")}
            for tc in m.tool_calls
        ]
    if isinstance(m, ToolMessage):
        out["tool_name"] = m.name
    return out


def _snapshot_state() -> dict[str, Any]:
    state = _agent.get_state(_cfg)
    values = state.values or {}
    core = values.get("core_memory")
    summary = values.get("recursive_summary") or ""
    messages: list[BaseMessage] = values.get("messages", []) or []

    tokens_used = count_state_tokens(
        system_prompt=_system_prompt,
        core_memory=core,
        recursive_summary=summary or None,
        messages=messages,
    )
    return {
        "system_prompt": _system_prompt,
        "core_memory": core.to_prompt_text() if core is not None else "",
        "recursive_summary": summary,
        "messages": [_message_dict(m) for m in messages],
        "tokens": {
            "used": tokens_used,
            "window": _queue_config.context_window_tokens,
            "warning": _queue_config.warning_tokens,
            "flush": _queue_config.flush_tokens,
        },
    }


def _send_message(text: str) -> dict[str, Any]:
    with _lock:
        _agent.invoke({"messages": [HumanMessage(content=text)]}, config=_cfg)
        return _snapshot_state()


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MemGPT context inspector</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #161922;
    --border: #262a35;
    --accent: #6ee7b7;
    --warn: #fbbf24;
    --danger: #f87171;
    --muted: #6b7280;
    --text: #e5e7eb;
    --human: #60a5fa;
    --ai: #6ee7b7;
    --tool: #c084fc;
    --system: #94a3b8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    background: var(--bg); color: var(--text);
    font-size: 13px; line-height: 1.5;
  }
  header {
    padding: 12px 16px; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }
  header h1 { margin: 0; font-size: 14px; font-weight: 600; letter-spacing: 0.5px; }
  .meter { display: flex; align-items: center; gap: 12px; min-width: 380px; }
  .bar { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .bar > div { height: 100%; background: var(--accent); transition: width .3s, background .3s; }
  .bar.warn > div { background: var(--warn); }
  .bar.danger > div { background: var(--danger); }
  .meter-label { color: var(--muted); white-space: nowrap; }
  main { display: grid; grid-template-columns: 1fr 1fr 1.2fr; gap: 1px; background: var(--border); height: calc(100vh - 53px - 90px); }
  .panel { background: var(--panel); padding: 12px 16px; overflow-y: auto; }
  .panel h2 { margin: 0 0 10px 0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); }
  .split { display: flex; flex-direction: column; padding: 0; overflow: hidden; }
  .split > .subpanel { padding: 12px 16px; overflow-y: auto; }
  .split > .subpanel.top { flex: 0 0 40%; border-bottom: 1px solid var(--border); }
  .split > .subpanel.bottom { flex: 1 1 auto; min-height: 0; }
  pre { margin: 0; white-space: pre-wrap; word-wrap: break-word; font-family: inherit; font-size: 12px; }
  .empty { color: var(--muted); font-style: italic; }
  .msg { padding: 8px 10px; margin-bottom: 8px; border-left: 3px solid var(--muted); background: rgba(255,255,255,0.02); border-radius: 0 4px 4px 0; }
  .msg.human { border-color: var(--human); }
  .msg.ai { border-color: var(--ai); }
  .msg.tool { border-color: var(--tool); }
  .msg.system { border-color: var(--system); }
  .msg-role { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 4px; }
  .msg.human .msg-role { color: var(--human); }
  .msg.ai .msg-role { color: var(--ai); }
  .msg.tool .msg-role { color: var(--tool); }
  .tool-calls { margin-top: 6px; padding: 6px 8px; background: rgba(192,132,252,0.08); border-radius: 3px; font-size: 11px; }
  .tool-calls .tc-name { color: var(--tool); }
  footer { padding: 12px 16px; border-top: 1px solid var(--border); display: flex; gap: 8px; }
  input[type=text] {
    flex: 1; background: var(--bg); border: 1px solid var(--border); color: var(--text);
    padding: 10px 12px; font-family: inherit; font-size: 13px; border-radius: 4px;
  }
  input[type=text]:focus { outline: none; border-color: var(--accent); }
  button {
    background: var(--accent); color: var(--bg); border: 0; padding: 0 18px;
    font-family: inherit; font-weight: 600; cursor: pointer; border-radius: 4px;
  }
  button:disabled { opacity: 0.4; cursor: wait; }
</style>
</head>
<body>
<header>
  <h1>memgpt · context inspector</h1>
  <div class="meter">
    <span class="meter-label" id="tokens-label">…</span>
    <div class="bar" id="bar"><div id="bar-fill" style="width:0"></div></div>
  </div>
</header>
<main>
  <section class="panel">
    <h2>System prompt</h2>
    <pre id="system"></pre>
  </section>
  <section class="panel">
    <h2>Working context</h2>
    <pre id="core"></pre>
  </section>
  <section class="panel split">
    <div class="subpanel top">
      <h2>Slot 0 · Recursive summary</h2>
      <pre id="summary"></pre>
    </div>
    <div class="subpanel bottom" id="fifo-scroll">
      <h2>FIFO queue · slots 1+</h2>
      <div id="fifo"></div>
    </div>
  </section>
</main>
<footer>
  <input id="input" type="text" placeholder="Type a message and press Enter…" autofocus>
  <button id="send">Send</button>
</footer>
<script>
const $ = (id) => document.getElementById(id);

function renderEmpty(el, text) {
  el.innerHTML = `<span class="empty">${text}</span>`;
}

function renderMessages(msgs) {
  const fifo = $("fifo");
  if (!msgs.length) { renderEmpty(fifo, "(empty)"); return; }
  fifo.innerHTML = msgs.map((m) => {
    const tc = m.tool_calls ? m.tool_calls.map(
      (c) => `<div><span class="tc-name">${c.name}</span>(${JSON.stringify(c.args)})</div>`
    ).join("") : "";
    const tcBlock = tc ? `<div class="tool-calls">${tc}</div>` : "";
    const label = m.tool_name ? `tool · ${m.tool_name}` : m.role;
    const content = m.content || (tc ? "" : "(no content)");
    return `<div class="msg ${m.role}">
      <div class="msg-role">${label}</div>
      <pre>${escapeHtml(content)}</pre>
      ${tcBlock}
    </div>`;
  }).join("");
  const scroller = $("fifo-scroll");
  if (scroller) scroller.scrollTop = scroller.scrollHeight;
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
}

function renderTokens(t) {
  const pct = t.window ? (t.used / t.window) * 100 : 0;
  $("bar-fill").style.width = Math.min(100, pct).toFixed(1) + "%";
  const bar = $("bar");
  bar.classList.remove("warn", "danger");
  if (t.used >= t.flush) bar.classList.add("danger");
  else if (t.used >= t.warning) bar.classList.add("warn");
  $("tokens-label").textContent =
    `${t.used.toLocaleString()} / ${t.window.toLocaleString()} tokens (warn ${t.warning.toLocaleString()})`;
}

function render(state) {
  $("system").textContent = state.system_prompt || "";
  state.core_memory ? ($("core").textContent = state.core_memory) : renderEmpty($("core"), "(empty)");
  state.recursive_summary ? ($("summary").textContent = state.recursive_summary) : renderEmpty($("summary"), "(none)");
  renderMessages(state.messages);
  renderTokens(state.tokens);
}

async function refresh() {
  const r = await fetch("/api/state");
  render(await r.json());
}

async function send() {
  const input = $("input");
  const text = input.value.trim();
  if (!text) return;
  const btn = $("send");
  btn.disabled = true; input.disabled = true;
  input.value = "";
  try {
    const r = await fetch("/api/send", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({message: text}),
    });
    if (!r.ok) throw new Error(await r.text());
    render(await r.json());
  } catch (e) {
    alert("error: " + e.message);
  } finally {
    btn.disabled = false; input.disabled = false;
    input.focus();
  }
}

$("send").addEventListener("click", send);
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });

let polling = true;
async function pollLoop() {
  while (true) {
    if (polling) {
      try { await refresh(); } catch (e) { /* ignore transient errors */ }
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
}
// Pausar el polling mientras estás escribiendo o enviando para no pisar
// la respuesta del POST con un GET concurrente.
$("input").addEventListener("focus", () => { polling = false; });
$("input").addEventListener("blur", () => { polling = true; });
refresh().then(pollLoop);
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Silenciar el logging por defecto (cada GET ensucia la terminal).
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path == "/index.html":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/state":
            self._send_json(200, _snapshot_state())
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/send":
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("content-length", "0") or "0")
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = (payload.get("message") or "").strip()
            if not text:
                self._send_json(400, {"error": "empty message"}); return
            self._send_json(200, _send_message(text))
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})


def _serve(args) -> int:
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[inspect] serving on {url}  (thread_id={args.thread})")
    print("[inspect] Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        server.server_close()
    return 0


def main() -> int:
    global _agent, _cfg

    parser = argparse.ArgumentParser(description="MemGPT context window web inspector")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--thread", default="inspector")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--persistent",
        action="store_true",
        help=(
            "Usa PostgresSaver + GraphitiStore (requiere docker compose up). "
            "Necesario si quieres compartir el thread con otro proceso "
            "(p. ej. scripts/chat.py --persistent --thread <mismo>)."
        ),
    )
    args = parser.parse_args()

    load_dotenv()
    _cfg = {"configurable": {"thread_id": args.thread}}

    if args.persistent:
        from graphiti_core import Graphiti  # type: ignore[import-not-found]

        from memgpt.config import get_settings
        from memgpt.memory_store import GraphitiStore
        from memgpt.persistence import build_persistent_agent, postgres_checkpointer

        settings = get_settings()
        client = Graphiti(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
        )
        store = GraphitiStore(client, group_id=f"chat-{args.thread}")
        with postgres_checkpointer(settings.postgres_dsn) as saver:
            agent, _registry = build_persistent_agent(
                checkpointer=saver,
                memory_store=store,
                event_store_dsn=settings.postgres_dsn,
                model_id=args.model,
                queue_config=_queue_config,
            )
            _agent = agent
            return _serve(args)

    _agent = build_agent(model_id=args.model, queue_config=_queue_config)
    return _serve(args)


if __name__ == "__main__":
    sys.exit(main())
