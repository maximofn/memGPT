"""REPL interactivo para charlar con el agente MemGPT desde la terminal.

Uso típico:

    # Modo efímero (in-memory, sin Postgres ni Neo4j).
    uv run scripts/chat.py

    # Persistente: requiere `docker compose up -d` (Postgres + Neo4j).
    uv run scripts/chat.py --persistent

Flags:
- ``--thread``: id del thread LangGraph. Mismo id = misma conversación
  (Core Memory + FIFO se recuperan vía checkpointer).
- ``--persistent``: usa `PostgresSaver` + `GraphitiStore` (Recall/Archival
  reales) en lugar de `MemorySaver` + sin store.
- ``--model``: override del LLM. Por defecto se lee de
  ``PRIMARY_LLM_MODEL`` en `.env`.

Comandos dentro del REPL:
- ``/exit`` o ``/quit`` (también Ctrl-D / Ctrl-C): salir.
- ``/state``: dump rápido del estado actual (Core Memory + nº mensajes).
- Cualquier otra cosa se envía al agente como `HumanMessage`.
"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

# El driver Python de Neo4j escupe un WARNING por cada propiedad/label
# que Graphiti consulta antes de que exista en la BD. Es ruido benigno:
# las primeras escrituras crean el schema. Silenciar a nivel ERROR.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


def _print_response(out: dict) -> None:
    final = out["messages"][-1]
    content = final.content if isinstance(final, AIMessage) else str(final)
    print(f"\n{content}\n")


def _print_state(agent, cfg: dict) -> None:
    snapshot = agent.get_state(cfg)
    values = snapshot.values
    core = values.get("core_memory")
    msgs = values.get("messages", [])
    print(f"\n[state] messages={len(msgs)}")
    if core is not None:
        print(f"[state] core_memory:\n{core.to_prompt_text()}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="MemGPT chat REPL")
    parser.add_argument("--thread", default="repl", help="LangGraph thread_id.")
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Usa PostgresSaver + GraphitiStore (requiere docker compose up).",
    )
    parser.add_argument("--model", default=None, help="Override del primary_llm_model.")
    args = parser.parse_args()

    load_dotenv()
    cfg = {"configurable": {"thread_id": args.thread}}

    if args.persistent:
        from graphiti_core import Graphiti  # type: ignore[import-not-found]

        from memgpt.config import get_settings
        from memgpt.memory_store import GraphitiStore
        from memgpt.persistence import build_persistent_agent, postgres_checkpointer
        from memgpt.queue_manager import QueueManagerConfig

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
                queue_config=QueueManagerConfig.from_settings(),
            )
            return _loop(agent, cfg)

    from memgpt.agent import build_agent
    from memgpt.queue_manager import QueueManagerConfig

    agent = build_agent(
        model_id=args.model,
        queue_config=QueueManagerConfig.from_settings(),
    )
    return _loop(agent, cfg)


def _loop(agent, cfg: dict) -> int:
    print(f"[chat] thread_id={cfg['configurable']['thread_id']} — /exit para salir")
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user in {"/exit", "/quit"}:
            return 0
        if user == "/state":
            _print_state(agent, cfg)
            continue
        try:
            out = agent.invoke(
                {"messages": [HumanMessage(content=user)]}, config=cfg
            )
        except Exception as exc:
            print(f"\n[error] {type(exc).__name__}: {exc}\n")
            continue
        _print_response(out)


if __name__ == "__main__":
    sys.exit(main())
