# MemGPT on LangGraph

A from-scratch implementation of **MemGPT** ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560)) built on top of **LangGraph**. The agent uses a tiered memory hierarchy (Core Memory in-prompt, FIFO working queue, external Recall + Archival stores, versioned MemFS) and a Queue Manager with a dual-threshold policy that decides when to warn the LLM about memory pressure and when to evict messages by recursive summarization.

The full design and rationale lives in `memGPT-plan.md` (phases 0 → 11). Phases 0-10 are complete.

## Features

- **Tiered memory**
  - **Core Memory**: editable blocks rendered into the system prompt (`persona`, `human`, ...) with per-block token budgets.
  - **FIFO Queue**: live conversation messages, grouped into atomic blocks (`AIMessage` with `tool_calls` + matching `ToolMessage`s never split).
  - **Recall + Archival**: same backing store (Graphiti on Neo4j, or in-memory) with bi-temporal model (`occurred_at` vs `learned_at`).
  - **MemFS**: in-memory versioned filesystem with commits, history and rollback (9 tools: create / read / write / list / move / delete / history / rollback / grep).
- **Queue Manager** with dual thresholds (warning 70 %, flush 100 %) and recursive summarization on eviction.
- **Function chaining via heartbeats** (`NATIVE` and `LEGACY` modes) with hard caps, wall-clock timeout and loop detection.
- **Persistence** via `PostgresSaver` checkpointer + `PostgresEventStore` for wall-clock event specs (APScheduler).
- **Sleep-time agents**: iteration callbacks fired every N steps.
- **Inspector web UI**: live view of system prompt, Core Memory, recursive summary and FIFO with tricolor token bar.
- **Benchmarks**: Nested KV, DMR (MSC self-instruct) and Document QA (NQ-Open) runners.

## Requirements

- Python **≥ 3.11**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Docker + Docker Compose (only if you want Postgres / Neo4j persistence; tests and benchmarks also run fully in-memory)
- An LLM API key: `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/maximofn/memGPT
cd memGPT

# 2. Create the virtualenv and install dependencies
uv sync

# Optional: install local embeddings (sentence-transformers) for Document QA
uv sync --extra embeddings-local

# 3. Activate the virtualenv
#    (deactivate conda first if it is auto-activated)
conda deactivate 2>/dev/null
source .venv/bin/activate

# 4. Configure credentials
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY / OPENAI_API_KEY

# 5. (Optional) Start Postgres + Neo4j for persistence
docker compose up -d
uv run scripts/check_services.py   # connectivity smoke test
```

`docker-compose.yml` brings up:

- **Postgres 16** with `pgvector` on port **5433** (used by `PostgresSaver` checkpointer and `PostgresEventStore`).
- **Neo4j 5.26** with APOC on ports **7474 / 7687** (used by `GraphitiStore` for Recall + Archival; ≥ 5.24 is required because Graphiti relies on dynamic labels).

Without Docker everything still works using `InMemoryStore` + `MemorySaver`.

### Environment variables

Set in `.env` (loaded by `pydantic-settings`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | Required for Claude models |
| `OPENAI_API_KEY` | — | Required for OpenAI models / Graphiti |
| `PRIMARY_LLM_MODEL` | `claude-sonnet-4-6` | Main agent LLM |
| `SUMMARIZER_LLM_MODEL` | `claude-haiku-4-5` | Recursive summarizer LLM |
| `POSTGRES_DSN` | `postgresql://memgpt:memgpt@localhost:5433/memgpt` | Checkpointer / event store |
| `NEO4J_URI` | `bolt://localhost:7687` | Graphiti backend |
| `NEO4J_USER` / `NEO4J_PASSWORD` | `neo4j` / `memgptmemgpt` | Neo4j auth |
| `CONTEXT_WINDOW_TOKENS` | `200000` | Queue Manager window |
| `WARNING_THRESHOLD` / `FLUSH_THRESHOLD` | `0.70` / `1.00` | Pressure thresholds |
| `FLUSH_EVICTION_RATIO` | `0.50` | Fraction of queue evicted per flush |

Model IDs accept both the LangChain format (`anthropic:claude-sonnet-4-6`, `openai:gpt-4o-mini`) and the bare form (`claude-sonnet-4-6`).

## Usage

### REPL

```bash
uv run scripts/chat.py                          # in-memory
uv run scripts/chat.py --persistent --thread X  # Postgres + Neo4j backed
```

### Inspector web UI

```bash
uv run scripts/inspect_web.py   # http://localhost:8000
uv run scripts/inspect_web.py --persistent --thread demo # Postgres + Neo4j backend
```

The REPL and the inspector can share the same `--thread X --persistent` — the inspector polls every second so you see turns appear live.

### Tests

```bash
uv run pytest                                          # full suite
uv run pytest tests/test_queue_manager.py              # single file
uv run pytest tests/test_phase6_events.py -k dispatch  # pattern
uv run pytest -x                                       # stop at first failure
```

### Benchmarks

```bash
uv run scripts/run_nested_kv.py --limit 5
uv run scripts/run_dmr.py --dataset datasets/msc_self_instruct.jsonl --limit 5
uv run scripts/run_document_qa.py --dataset datasets/nq_open.jsonl --limit 5
```

Common flags: `--baseline` (run without memory), `--model` (override LLM), `--graphiti` (DMR only — use real Neo4j), `--download` (fetch the dataset from HF Hub), `--output runs/<name>.json` (incremental dump).

## Architecture

The compiled `StateGraph` looks like:

```
START → turn_init → recall_sync_in → pressure_check → agent
  → [step_tick]? → recall_sync_post → (tools | END)
tools → heartbeat_check → (recall_sync_in | END)
```

- `turn_init` resets per-turn heartbeat counters when a new `HumanMessage` arrives.
- `recall_sync_in/post` persist new messages into the `MemoryStore` before they can be evicted.
- `pressure_check` measures full-state tokens, injects a `Memory Pressure Alert` at the warning threshold, and at the flush threshold evicts atomic blocks while regenerating the recursive summary.
- `agent` builds the prompt (system → Core Memory → recursive summary → messages) and invokes the LLM with `parallel_tool_calls=False` (two tool calls touching `core_memory` in the same step would collide).
- `heartbeat_check` handles function chaining (`NATIVE`: LLM always decides via new tool_calls; `LEGACY`: continue only when `request_heartbeat=True` or tool is in `auto_continue_tools`).

### Layout

```
src/memgpt/
├── agent.py                  # build_agent / build_persistent_agent
├── state.py                  # MemGPTState (Pydantic + add_messages reducer)
├── core_memory.py            # Core Memory blocks (immutable)
├── queue_manager.py          # Pressure check + atomic-block eviction
├── summarizer.py             # Recursive summary regeneration
├── memory_store.py           # InMemoryStore + GraphitiStore (Recall/Archival)
├── memfs_store.py            # Versioned in-memory filesystem
├── memfs_tools.py            # 9 MemFS tools
├── tools.py                  # core_memory_* tools
├── recall_archival_tools.py  # conversation_search, archival_*
├── heartbeat.py              # Function chaining policy
├── events.py                 # Iteration + wall-clock events (APScheduler)
├── persistence.py            # Postgres checkpointer wiring
├── llm.py                    # Model ID resolution (langchain ↔ litellm)
├── embedders.py              # Embeddings (OpenAI / local)
├── config.py                 # Pydantic settings
└── tokens.py                 # Token counting

scripts/
├── chat.py                   # REPL
├── inspect_web.py            # Web inspector
├── check_services.py         # Postgres + Neo4j smoke test
├── check_llm.py              # LLM smoke test
├── run_nested_kv.py          # Benchmark
├── run_dmr.py                # Benchmark (DMR / MSC)
└── run_document_qa.py        # Benchmark (NQ-Open)
```

## Conventions

- Docstrings and long comments in Spanish; symbol names in English. Docstrings often reference plan sections (`§3 Core Memory`, `§4 Queue Manager`, ...) — keep the practice when adding modules.
- `CoreMemory` mutations return new instances (LangGraph-friendly via `Command(update=...)`).
- Tests for the Queue Manager use a tiny `context_window_tokens` to force flushes — do not assume the 200k default in tests.
- Document QA + Graphiti is intentionally avoided (`add_episode` extracts entities via LLM, too expensive at 30 docs/sample). Use DMR / MSC for Graphiti evaluation.

## License

Research / educational implementation. See `MemGPT-Towards-LLMs-as-Operating-Systems.md` for the original paper.
