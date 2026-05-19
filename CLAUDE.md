# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sobre el proyecto

Implementación desde cero de **MemGPT** (arXiv:2310.08560) sobre **LangGraph**. El sistema gestiona un agente con jerarquía de memoria (Core Memory en prompt, FIFO Queue trabajando, Recall + Archival externas, MemFS versionado) y un Queue Manager con doble umbral que decide cuándo alertar al LLM y cuándo expulsar mensajes resumiéndolos.

El roadmap por fases (0 → 11) vive en `posts/papers/memGPT-plan.md` (fuera de este directorio, en `../papers/`). Las fases 0-10 están marcadas como completadas. Consulta ese fichero cuando necesites contexto sobre decisiones de diseño — los docstrings remiten a sus secciones (§3 Core Memory, §4 Queue Manager, §5 Recall/Archival, etc.).

## Comandos

### Entorno

```bash
# El entorno vive en .venv (a nivel raíz del proyecto, no del padre).
conda deactivate && source .venv/bin/activate
```

Todos los scripts y tests se ejecutan con `uv run`. El usuario tiene una preferencia global: desactivar conda antes de activar `.venv` si existe — este repo tiene `.venv` en su raíz, así que aplica.

### Servicios externos

Postgres (`pgvector/pgvector:pg16`, puerto **5433**) + Neo4j (5.26 con APOC; ≥ 5.24 es obligatorio porque Graphiti usa `SET n:$(...)` dynamic labels) corren vía docker-compose:

```bash
docker compose up -d
uv run scripts/check_services.py   # smoke test de conectividad
```

Postgres se usa para el `PostgresSaver` checkpointer y el `PostgresEventStore`. Neo4j respalda `GraphitiStore` para Recall + Archival. Sin docker, todo Tests/benchmarks funcionan con `InMemoryStore` / `MemorySaver`.

### Tests

```bash
uv run pytest                                 # toda la suite
uv run pytest tests/test_queue_manager.py     # un fichero
uv run pytest tests/test_phase6_events.py -k dispatch  # un patrón
uv run pytest -x                              # parar al primer fallo
```

`pyproject.toml` ya tiene `asyncio_mode = "auto"` y `addopts = "-ra -q"`.

### Benchmarks (Fases 8-10)

Los runners viven en `scripts/` y aceptan `--limit` para smoke + `--output runs/<nombre>.json` para dump incremental. Cada uno tiene una sección de "Uso típico" detallada en su docstring.

```bash
uv run scripts/run_nested_kv.py --limit 5
uv run scripts/run_dmr.py --dataset datasets/msc_self_instruct.jsonl --limit 5
uv run scripts/run_document_qa.py --dataset datasets/nq_open.jsonl --limit 5
```

Flags relevantes comunes: `--baseline` (sin memoria), `--model` (override del LLM), `--graphiti` (solo en DMR, usa Neo4j real), `--download` (descarga el dataset de HF Hub si falta). `datasets/` está en `.gitignore`.

### REPL y debug UI

```bash
uv run scripts/chat.py                          # REPL in-memory
uv run scripts/chat.py --persistent --thread X  # comparte estado vía Postgres+Neo4j
uv run scripts/inspect_web.py                   # http://localhost:8000, visualizador del contexto
```

`chat.py` y `inspect_web.py` pueden compartir el mismo `--thread X --persistent`: el inspector hace polling cada 1 s y verás los turnos del REPL aparecer en vivo. El inspector muestra system prompt, Core Memory, recursive summary y FIFO con barra de tokens tricolor (verde/ámbar/rojo según los umbrales del Queue Manager). Ambos silencian `neo4j.notifications` a `ERROR` para no ahogarse con los warnings benignos del driver sobre labels/propiedades aún inexistentes.

### LLMs y embeddings

`.env` (no versionado) provee `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`. Defaults del código: `primary_llm_model=claude-sonnet-4-6`, `summarizer_llm_model=claude-haiku-4-5` (overridables vía `PRIMARY_LLM_MODEL` / `SUMMARIZER_LLM_MODEL` en `.env`). El id del modelo se acepta tanto en formato langchain (`openai:gpt-4o-mini-...`) como pelado (`claude-sonnet-4-6`): `agent.py:_resolve_model_id` antepone `anthropic:` si no hay namespace; `llm.py:to_litellm_model` traduce `:` → `/` antes de pasar a litellm.

Para Document QA con embeddings locales: `uv sync --extra embeddings-local` (instala `sentence-transformers`).

## Arquitectura

### El grafo del agente (`src/memgpt/agent.py`)

`build_agent(...)` compila un `StateGraph` con esta forma:

```
START → turn_init → recall_sync_in → pressure_check → agent
  → [step_tick]? → recall_sync_post → (tools | END)
tools → heartbeat_check → (recall_sync_in | END)
```

- **`turn_init`** detecta un nuevo `HumanMessage` y resetea contadores de heartbeat del turno.
- **`recall_sync_in/post`** persiste mensajes nuevos en el `MemoryStore` antes de que `pressure_check` pueda expulsarlos, y de nuevo tras el `agent` por si el turno acaba sin tool call.
- **`pressure_check`** mide tokens del estado completo (system prompt + Core Memory + recursive_summary + mensajes). Si cruza `warning_threshold` (70 %) inyecta una `Memory Pressure Alert`; si cruza `flush_threshold` (100 %) expulsa bloques atómicos y regenera el resumen recursivo. **Bloque atómico** = un `AIMessage` con `tool_calls` + sus `ToolMessage` correspondientes — nunca se separan (evita los bugs #111/#126 de `langmem.SummarizationNode`).
- **`agent`** monta el prompt: system → Core Memory (`to_prompt_text()`) → recursive_summary (si existe) → mensajes — luego invoca el LLM bound a las tools. El binding es `bind_tools(..., parallel_tool_calls=False)`: forzamos tools secuenciales porque dos tool calls en paralelo que toquen `core_memory` chocan en el mismo step (`InvalidUpdateError` — el campo no tiene reducer). MemGPT está pensado para function chaining vía heartbeat, no fan-out paralelo.
- **`step_tick`** (solo si hay `EventRegistry`) incrementa `step_count` y dispara iteration callbacks (sleep-time agents).
- **`heartbeat_check` + `heartbeat_router`** gestionan el function chaining tras tools: cuentan repeticiones de tool calls, aplican el modo `NATIVE` (continúa siempre, el LLM decide vía nuevos tool_calls) vs `LEGACY` (continúa solo si `request_heartbeat=True` o el tool está en `auto_continue_tools`), respetan hard cap por turno + timeout wall-clock + loop detection.

### Estado (`src/memgpt/state.py`)

`MemGPTState` es un `BaseModel` Pydantic con `messages: Annotated[..., add_messages]` (reducer LangGraph), `core_memory`, `recursive_summary`, `persisted_message_ids` (dedupe para Recall), `step_count`, contadores de heartbeat, y `last_processed_human_id`. **El recursive_summary NO va como slot 0 de `messages`** — vive en su propio campo para no mezclar el reducer append-only con un slot mutable; `agent_node` lo inyecta como `SystemMessage` al construir el prompt.

### Capas de memoria

1. **Core Memory** (`core_memory.py`): `dict[str, MemoryBlock]` con labels snake_case, presupuesto de tokens por bloque, métodos inmutables (devuelven nueva `CoreMemory`). Las tools `core_memory_*` (`tools.py`) son las únicas que la editan desde el LLM.
2. **FIFO Queue + Queue Manager** (`queue_manager.py`): mensajes vivos en el prompt. `group_into_atomic_blocks` agrupa pares tool_call ↔ tool_message; `select_blocks_to_evict` elige los más viejos hasta acumular `target_eviction_tokens`.
3. **Recall + Archival** (`memory_store.py` + `recall_archival_tools.py`): misma instancia (Graphiti / InMemory) con `source_description` diferente (`"conversation:<role>"` vs `"archival"`). Modelo bi-temporal (`occurred_at` vs `learned_at`). Interfaz **síncrona** a propósito: el grafo se invoca con `.invoke()` y mezclar nodos async rompe con `TypeError: No synchronous function provided to "add"`. `GraphitiStore.__init__` dispara `build_indices_and_constraints()` a mano vía su loop dedicado — el auto-build del driver solo arranca si hay un event loop corriendo, y desde código sync no lo hay (sin esto, la primera búsqueda híbrida revienta con `ProcedureCallFailed: edge_name_and_fact`).
4. **MemFS** (`memfs_store.py` + `memfs_tools.py`): extensión Letta (no en el paper). Filesystem versionado in-memory con commits (snapshot + SHA-1 truncado, no usa `dulwich`). 9 tools: `create / read / write / list / move / delete / history / rollback / grep`. **Vive fuera de `MemGPTState`** — no pasa por el checkpointer.

### Eventos (`events.py`)

Dos tipos:
- **Iteration events** ("sleep-time agents"): callbacks Python ejecutadas cada N steps. No se persisten (no son serializables) — el cliente las re-registra al arrancar.
- **Wall-clock events**: jobs APScheduler que disparan un mensaje al agente. Los **specs sí se persisten** vía `EventStore` (`InMemoryEventStore` para tests, `PostgresEventStore` para producción). `EventRegistry.restore()` los recarga al arrancar.

### Persistencia (`persistence.py`)

`build_persistent_agent(...)` cablea checkpointer (típicamente `PostgresSaver`), `memory_store`, y `EventRegistry` con su store opcional. `postgres_checkpointer(dsn)` es el context manager preferido para scripts y tests. `default_persistent_agent(...)` es un atajo que abre el saver sin context manager — solo para la app principal.

## Convenciones del proyecto

- **Idioma**: docstrings y comentarios largos en español; nombres de símbolos en inglés. Los docstrings remiten al plan (`§N`) con frecuencia — mantén esa práctica al añadir módulos nuevos.
- **Modelos inmutables**: las mutaciones de `CoreMemory` devuelven nuevas instancias para ser amistosas con `Command(update=...)` de LangGraph.
- **Tests del Queue Manager** usan `QueueManagerConfig` con `context_window_tokens` pequeños para forzar flushes; no asumas los defaults de 200k al escribir tests.
- **Document QA + Graphiti**: descartado a propósito. `add_episode` extrae entidades por LLM, prohibitivo con 30 docs/sample. Para evaluar Graphiti usa DMR/MSC.
