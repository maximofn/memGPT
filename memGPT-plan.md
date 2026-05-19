# Plan de implementación de MemGPT desde cero

Plan operativo para construir una implementación funcional de la arquitectura MemGPT (paper "MemGPT: Towards LLMs as Operating Systems", arXiv:2310.08560) basada en las decisiones del documento `memGPT-resumen.md`.

## Documentos de referencia

Este plan se apoya en dos documentos hermanos en la misma carpeta:

- **Paper original** (traducido al español, con figuras): [`posts/papers/MemGPT-Towards-LLMs-as-Operating-Systems.md`](./MemGPT-Towards-LLMs-as-Operating-Systems.md). Es la fuente canónica de la arquitectura: jerarquía de memoria, Queue Manager, function chaining, benchmarks. Cuando este plan dice "según el paper" o cita una sección concreta (ej. "sección 2.2" o "Figura 6"), se refiere a este documento.

- **Resumen con toma de decisiones**: [`posts/papers/memGPT-resumen.md`](./memGPT-resumen.md). Documento donde están **todas las decisiones de diseño** que justifican este plan:
  - Por qué LangGraph como runtime y no LangChain core o DeepAgents.
  - Por qué Graphiti como backend de Recall/Archival y no mem0 ni pgvector ingenuo.
  - Qué librerías auxiliares cubren cada componente huérfano (litellm, langmem, APScheduler, instructor, dulwich).
  - Bloques etiquetados de Core Memory, eventos por iteraciones (sleep-time agents), MemFS como extensión de Letta.
  - TODOs y limitaciones del paper que se mejoran en la implementación (eliminación selectiva en FIFO, contadores de búsqueda, prompts dinámicos contra terminación prematura, medición previa de inputs masivos).

Si surge una duda durante la implementación sobre **por qué** una decisión se tomó así, la respuesta está en `memGPT-resumen.md`. Si surge una duda sobre **qué** dice el paper exactamente, la respuesta está en `MemGPT-Towards-LLMs-as-Operating-Systems.md`.

---

## 0. Resumen ejecutivo

**Stack tecnológico elegido**:

- **Runtime / agent loop**: LangGraph (StateGraph + Checkpointer + Store).
- **Backend de Recall + Archival**: Graphiti (knowledge graph bi-temporal + búsqueda híbrida).
- **Conteo de tokens**: `litellm.token_counter`.
- **Resumen recursivo**: `langmem.SummarizationNode` (con workarounds para los bugs #118, #111, #126).
- **Scheduling wall-clock**: APScheduler 3.x.
- **Eventos por iteraciones**: LangChain Middleware.
- **Tool tipado para `request_heartbeat`**: `instructor`.
- **Core Memory con bloques etiquetados**: Pydantic custom (~50 líneas).
- **MemFS versionado** (opcional, fase tardía): `dulwich` + abstracción propia.

**Esfuerzo estimado**: 2-4 semanas para un sistema funcional que replique el paper. Multi-agente, MCP y MemFS son extensiones opcionales que extienden el plazo otras 1-2 semanas.

**Estrategia de validación**: replicar los 3 benchmarks del paper (Nested KV → DMR → Document QA) en orden de menor a mayor complejidad.

---

## 1. Fase 0 — Setup del entorno ✅ Completada

### Objetivos

Tener un entorno reproducible con todas las dependencias y servicios necesarios.

### Tareas

1. Crear repo Python con `pyproject.toml` y entorno virtual con `uv` o `poetry`.
2. Instalar dependencias mínimas:
   - `langgraph`, `langchain`, `langchain-core` (>=1.0 para Middleware).
   - `graphiti-core` + driver de Neo4j o FalkorDB.
   - `litellm`, `instructor`, `pydantic`.
   - `apscheduler` (3.x).
   - `langmem` (con conciencia de los bugs activos).
   - `python-dotenv` para gestión de credenciales.
3. Levantar infraestructura local con Docker Compose:
   - PostgreSQL con `pgvector` (para checkpointer + store de LangGraph).
   - Neo4j o FalkorDB (para Graphiti).
4. Configurar acceso a un LLM moderno (Claude Sonnet 4.6 o GPT-4o) y a un LLM más barato para el summarizer (Claude Haiku 4.5 o GPT-4o-mini).
5. Definir variables de entorno para credenciales.

### Definición de hecho

- ✅ `pytest` corre y pasa con un test trivial.
- ✅ Docker Compose levanta los dos servicios sin errores (Postgres en host:5433 para evitar conflicto con otro Postgres local).
- ✅ Una llamada de prueba al LLM principal (`claude-sonnet-4-6`) y al LLM summarizer (`claude-haiku-4-5`) funciona.

### Implementación

Ubicación: [`posts/memGPT/`](../memGPT/). Estructura:

- `pyproject.toml` con todas las dependencias del plan (langgraph, langchain≥1.0, graphiti-core, neo4j, litellm, instructor, pydantic, pydantic-settings, apscheduler 3.x, langmem, python-dotenv, psycopg) + dev (pytest, pytest-asyncio).
- `docker-compose.yml` con `pgvector/pgvector:pg16` (host:5433) y `neo4j:5.26` con healthchecks.
- `src/memgpt/config.py` — `Settings` Pydantic con carga desde `.env`.
- `src/memgpt/llm.py` — wrappers sync/async sobre `litellm` (`call_primary`, `call_summarizer`).
- `scripts/check_services.py` y `scripts/check_llm.py` — smoke tests de infra y LLMs.
- `tests/test_smoke.py` — 3 tests triviales.

#### Decisión clave: Neo4j ≥ 5.24 (no 5.20)

Graphiti emite queries Cypher que usan la sintaxis **dynamic labels** `SET n:$(node.labels)`, introducida en Neo4j 5.24. La primera versión del compose usaba `neo4j:5.20`, lo que rompe la primera escritura con `CypherSyntaxError: Invalid input '$'`. Fijamos `neo4j:5.26` (LTS más reciente). Si actualizas desde un volume creado por 5.20 hay que `docker compose down -v` antes (no es upgrade compatible para nuestro caso, BD vacía o de prueba).

#### Decisión clave: id del modelo único en `.env`

`agent.py` usa `init_chat_model(...)` de langchain, que espera el formato `provider:model` (p. ej. `openai:gpt-4o-mini-2024-07-18`). `llm.py` usa `litellm.completion(...)`, que espera `provider/model`. Para que el `.env` tenga **una sola** variable (`PRIMARY_LLM_MODEL`) válida para ambos mundos, `llm.py` expone `to_litellm_model(id)`: si el id trae `:` lo convierte a `/`; si no, se devuelve intacto (litellm autodetecta provider por prefijo `claude-`/`gpt-`). El `_resolve_model_id` del agente sigue su lógica original (antepone `anthropic:` si no hay namespace). Resultado: pones `PRIMARY_LLM_MODEL=openai:gpt-4o-mini-2024-07-18` y funcionan agente, summarizer, check_llm y benchmarks sin tocar más.

---

## 2. Fase 1 — Esqueleto del agent loop con LangGraph ✅ Completada

### Objetivos

Tener un `StateGraph` con el ciclo `agent → tools → agent` funcionando end-to-end, sin gestión de memoria todavía.

### Tareas

1. Definir `MemGPTState` como **`BaseModel` de Pydantic** con campos: `messages`, `recursive_summary`, `core_memory`, `step_count`, `memory_pressure_alerted`, `evicted_count`. Se elige `BaseModel` sobre `TypedDict` por:
   - Validación en runtime de invariantes (límites de tokens por bloque, formato de labels, monotonicidad de `step_count`).
   - Serialización nativa a JSON (`model_dump_json` / `model_validate_json`) para la persistencia en PostgreSQL.
   - Coherencia con `MemoryBlock` y `CoreMemory`, que ya son Pydantic (Fase 2).
   - Soporte de validators (`@field_validator`, `@model_validator`) y computed fields (`@computed_field` para totales de tokens).
   - El overhead por update es del orden de microsegundos: irrelevante frente al coste de la inferencia LLM.
   - LangGraph soporta `BaseModel` de primera clase con el patrón `Annotated[list, add_messages]` vía `Field(...)`.
2. **Decisión clave sobre el slot 0 (resumen recursivo)**: separar el resumen recursivo de la FIFO en un campo aparte (`recursive_summary: str | None`) en lugar de mantenerlo como primer elemento de `messages`. Razones:
   - El reducer `add_messages` está diseñado para conversaciones inmutables, no para mutar un slot fijo. Mezclar ambas semánticas obliga a lógica defensiva frágil (IDs fijos, comprobaciones de posición).
   - El resumen es conceptualmente **metadato** sobre mensajes expulsados, no un mensaje real de la conversación.
   - Modificar el resumen pasa a ser `state.recursive_summary = nuevo_resumen` vía `Command(update={...})`, sin tocar `messages`.
   - En el `agent_node`, al construir el prompt enviado al LLM, se prepende el `recursive_summary` como un `SystemMessage` justo después del system prompt y antes de los mensajes reales — el modelo lo sigue viendo como "el primer slot", la divergencia con el paper es solo de implementación, no de semántica.
   - Tests más simples y separación de responsabilidades clara entre la lógica de flush y la del resumen.
2. Construir el grafo:
   - Nodo `agent_node`: invoca el LLM con el estado actual.
   - Nodo `tools_node`: ejecuta tools usando `ToolNode` de LangGraph.
   - Edge condicional desde `agent_node`: si la salida tiene tool calls → `tools_node`; si no → `END`.
   - Edge desde `tools_node` → `agent_node`.
3. Definir un tool de prueba (ej. `get_current_time`) para validar el ciclo.
4. Compilar el grafo con `MemorySaver` (checkpointer en memoria, temporal).

### Definición de hecho

- ✅ El agente responde a una pregunta básica (test E2E `test_agent_responds_to_basic_question`).
- ✅ El agente puede invocar un tool y usar su resultado para responder (test E2E `test_agent_uses_tool_and_consumes_result`).
- ✅ El estado se persiste entre invocaciones del mismo `thread_id` (tests `test_state_persists_across_invocations_with_same_thread_id` y E2E `test_state_persists_across_invocations_same_thread`).

### Implementación

- `src/memgpt/state.py` — `MemGPTState` Pydantic con `messages` (reducer `add_messages`), `recursive_summary`, `core_memory`, `step_count`, `memory_pressure_alerted`, `evicted_count`.
- `src/memgpt/tools.py` — tool `get_current_time` (ISO 8601 UTC).
- `src/memgpt/agent.py` — `build_agent()`: `StateGraph` con nodos `agent` y `tools`, edge condicional `tools_condition`, `MemorySaver` por defecto. Acepta `llm` inyectable (para tests con stub) y `model_id` configurable. Usa `init_chat_model("anthropic:<model>")`. Inyecta `recursive_summary` como `SystemMessage` tras el system prompt.
- Dependencia añadida: `langchain-anthropic>=0.3` (provider para `init_chat_model`).
- Tests (15/15 passing): `tests/test_state.py`, `tests/test_agent_structure.py` (con LLM stub), `tests/test_agent_e2e.py` (skip si no hay API key, ejecutados contra Anthropic real).

---

## 3. Fase 2 — Core Memory (bloques etiquetados) ✅ Completada

### Objetivos

Implementar los bloques editables de Core Memory con la semántica de Letta (`assistant`, `human`, custom).

### Tareas

1. Definir `MemoryBlock` con Pydantic:
   ```python
   class MemoryBlock(BaseModel):
       label: str
       value: str
       limit: int  # presupuesto de tokens
   ```
2. Implementar `CoreMemory` como contenedor de **`dict[str, MemoryBlock]`** (no como una clase con campos fijos). Esta decisión es importante: permite añadir bloques arbitrarios al construir el estado inicial **y** durante el runtime sin tocar el schema. Métodos clave:
   - `to_prompt_text()` que itera sobre todos los bloques y los formatea para el system prompt.
   - `add_block(label, value, limit)`, `delete_block(label)`, `with_new_block(...)` (versión inmutable que devuelve nueva instancia).
   - Validación de límites por bloque y de presupuesto total.
3. Implementar las tools básicas (modificación de bloques existentes):
   - `core_memory_append(label, content)` que devuelve `Command(update={"core_memory": ...})`.
   - `core_memory_replace(label, old, new)` con la misma estrategia.
4. **Tools opcionales para creación/borrado dinámico de bloques en runtime** (decidir si exponerlas según el caso de uso):
   - `core_memory_create_block(label, initial_content, limit)`: para que el agente cree bloques nuevos al detectar dimensiones nuevas en la conversación (ej. "project_alpha", "team_members").
   - `core_memory_delete_block(label)`: para liberar espacio cuando un bloque deja de ser relevante.
5. Inyectar el contenido del Core Memory en el system prompt en cada inferencia (función helper en `agent_node`).
6. Inicializar bloques por defecto: `assistant` (identidad del agente) y `human` (vacío al inicio). Cualquier bloque adicional específico de la aplicación (`project_context`, `team_members`, etc.) se añade en la inicialización del estado, no requiere cambios en el código del agent loop.

### Caveats a tener en cuenta al implementar

Si se exponen las tools de creación/borrado dinámico (paso 4), considerar:

1. **Presupuesto total de tokens**: la suma de `limit` de todos los bloques no debe exceder el espacio reservado al Working Context dentro de la ventana de contexto (si no, se come la FIFO). Validar en `add_block` y rechazar la creación si el nuevo bloque haría desbordar el presupuesto global.
2. **Proliferación de bloques**: si el agente crea bloques sin control, el system prompt crece indefinidamente. Establecer un **límite máximo de bloques** (p. ej. 10) y devolver un error explícito al LLM cuando se alcance, sugiriendo borrar o consolidar bloques antes.
3. **Coherencia de labels**: dos bloques con labels muy parecidos (`project_a` y `project-a`) generan duplicación y caos. Validar **formato estricto** del label:
   - Solo `snake_case` (alfanumérico + guiones bajos).
   - Longitud máxima (p. ej. 32 caracteres).
   - Rechazar si ya existe un bloque con label normalizado equivalente.
4. **Persistencia automática**: como `CoreMemory` vive dentro de `MemGPTState` y se serializa con Pydantic, los bloques nuevos creados en runtime **persisten entre sesiones automáticamente** vía el checkpointer de LangGraph. Sin trabajo extra.
5. **Borrado vs. vaciado**: distinguir entre `core_memory_delete_block` (elimina el bloque del dict) y `core_memory_replace(label, old=block.value, new="")` (vacía el contenido pero mantiene el bloque). Documentar bien para que el LLM use el correcto.
6. **Idempotencia**: `core_memory_create_block` con un label ya existente debe fallar con un error claro, no sobrescribir silenciosamente.

### Definición de hecho

- ✅ El agente ve los bloques en su system prompt (`agent_node` inyecta `to_prompt_text()` como `SystemMessage`; verificado en `test_core_memory_text_is_injected_into_prompt`).
- ✅ Llamar a `core_memory_append` modifica el bloque y persiste vía checkpointer (`test_core_memory_append_updates_state_via_command`).
- ✅ El límite de tokens por bloque se respeta (`test_memory_block_value_must_fit_in_limit`, `test_with_appended_rejects_overflow`).
- ✅ E2E con LLM real: Claude actualiza `human` block al recibir un hecho durable (`test_agent_writes_to_core_memory_on_durable_fact`).

### Implementación

- `src/memgpt/tokens.py` — `count_tokens(text, model)` con `litellm.token_counter` y fallback word-count.
- `src/memgpt/core_memory.py` — `MemoryBlock` (validación `LABEL_RE` snake_case ≤32 chars + presupuesto por bloque) y `CoreMemory` (validación de coherencia clave↔label, `max_blocks=10`, `total_token_budget=8000`). Métodos inmutables: `with_block`, `without_block`, `with_appended`, `with_replaced`, `to_prompt_text`. Helper `default_core_memory(assistant, human, ...)`.
- `src/memgpt/state.py` — `core_memory: CoreMemory` con `default_factory=default_core_memory` (assistant+human por defecto).
- `src/memgpt/tools.py` — 4 tools con `InjectedState`+`InjectedToolCallId` que devuelven `Command(update={core_memory, messages})`: `core_memory_append`, `core_memory_replace`, `core_memory_create_block`, `core_memory_delete_block`. Errores devuelven `ToolMessage(status="error")`.
- `src/memgpt/agent.py` — `_default_tools()` incluye `get_current_time` + las 4 core_memory tools; `agent_node` inyecta el `to_prompt_text()` como `SystemMessage` antes del resumen recursivo.
- Tests Fase 2 (22 nuevos): `tests/test_core_memory.py` (15), `tests/test_core_memory_tools.py` (6), 1 E2E adicional. Total suite: 37/37.

---

## 4. Fase 3 — FIFO Queue + Queue Manager con doble umbral ✅ Completada

### Objetivos

Implementar la lógica de gestión de la ventana de contexto: warning al 70% y flush al 100% con resumen recursivo en llamada LLM separada.

### Umbrales como variables configurables

Los tres porcentajes que aparecen en esta fase **son configurables**, no constantes hardcodeadas. El paper los menciona explícitamente con "e.g." (sección 2.2). Modelarlos como una clase de configuración inyectable:

```python
class QueueManagerConfig(BaseModel):
    warning_threshold: float = 0.70   # default 70% — dispara Memory Pressure Alert
    flush_threshold: float = 1.00     # default 100% — dispara expulsión + resumen
    flush_eviction_ratio: float = 0.50  # default 50% — proporción de FIFO a expulsar
    context_window_tokens: int        # tokens totales del LLM principal (ej. 8192, 128000)
    
    @field_validator("warning_threshold", "flush_threshold", "flush_eviction_ratio")
    @classmethod
    def validate_ratio(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("threshold must be in (0, 1]")
        return v
    
    @model_validator(mode="after")
    def validate_order(self) -> "QueueManagerConfig":
        if self.warning_threshold >= self.flush_threshold:
            raise ValueError("warning_threshold must be < flush_threshold")
        return self
```

Esto permite ajustarlos por aplicación (un agente con LLM grande podría usar 80%/100%; uno con LLM pequeño podría usar 60%/90%) sin tocar el código del Queue Manager.

### Tareas

1. Definir `QueueManagerConfig` con los umbrales **configurables** (defaults: 70%/100%/50%) y pasarlo a la inicialización del agente.
2. Implementar contador de tokens del estado completo en cada paso usando `litellm.token_counter` (cuenta system + core memory + FIFO + recursive_summary).
3. Crear un **LangChain Middleware** con hook `before_model`:
   - Si `tokens > warning_threshold * context_window_tokens` → inyectar mensaje de sistema "Memory Pressure Alert" en la FIFO **una sola vez por episodio** (flag `memory_pressure_alerted` en estado para evitar repetición).
   - Si `tokens > flush_threshold * context_window_tokens` → disparar el flush.
4. Implementar la **llamada al summarizer separada del agent loop**:
   - Función `regenerate_recursive_summary(old_summary, evicted_messages, working_context) -> str`.
   - Usa `langmem.SummarizationNode` o invocación directa al LLM barato con un prompt dedicado.
   - El prompt incluye el Working Context para que el summarizer no sugiera información ya guardada (diseño que mejora al paper).
5. Implementar el flush:
   - Identificar el `flush_eviction_ratio * context_window_tokens` más antiguo de la FIFO (campo `messages`). Default: 50%, configurable.
   - Llamar al summarizer.
   - Actualizar `state.recursive_summary` con el nuevo texto (campo separado, no slot 0 dentro de `messages`).
   - Eliminar los mensajes expulsados de `messages` enviando `RemoveMessage` por sus IDs (el reducer `add_messages` los gestiona limpiamente al estar separados del resumen).
   - Incrementar `state.evicted_count`.
   - Resetear `state.memory_pressure_alerted` a `False` (tras el flush, el contexto baja por debajo del 70% y el ciclo de alertas puede empezar de nuevo).
   - Persistir mensajes expulsados en Recall Storage (Fase 4) — por ahora solo guardarlos en una lista del estado.
6. **Workarounds para los tres bugs activos de `langmem.SummarizationNode`** (cubre el ~80% del componente, deja un ~20% que tienes que parchear):
   - **Issue #118 — no fusiona correctamente con el resumen previo**: el `RunningSummary` no se incorpora al nuevo resumen cuando hay tool calls de por medio, perdiendo lo ya consolidado. Workaround: pasar explícitamente `state.recursive_summary` al prompt del summarizer y validar tras cada flush que el resumen nuevo contiene información del previo (test de regresión). Si se detecta pérdida, llamar al LLM una segunda vez con instrucción explícita de fusionar.
   - **Issue #111 — recorta `HumanMessage` con tool calls a la mitad**: el algoritmo de truncado deja tool calls huérfanas (sin su `ToolMessage` correspondiente o viceversa), provocando errores de schema validation en el siguiente turno. Workaround: pre-procesar los mensajes en "bloques atómicos" `(HumanMessage + sus AssistantMessages + sus ToolMessages)` y hacer que el recorte opere solo a nivel de bloque, nunca dentro de un bloque.
   - **Issue #126 — parallel tool calls se procesan mal**: cuando un `AssistantMessage` lleva varias tool calls paralelas, el recorte puede separarlas de sus respuestas. Workaround: extender la lógica anterior para tratar `(AssistantMessage con N tool calls + sus N ToolMessages)` como unidad atómica. Alternativa peor: configurar `parallel_tool_calls=False` en el LLM (sacrifica velocidad).
   - **Plan B si los workarounds resultan frágiles**: reemplazar `SummarizationNode` por una función propia (~80 líneas) que llama directamente al LLM con un prompt dedicado, sin dependencias. Está contemplado en la tarea 4 como camino alternativo.

### Definición de hecho

- En una conversación que supere el `warning_threshold` (default 70%) se inyecta la alerta una sola vez y el agente puede consolidar en Core Memory.
- En una conversación que supere el `flush_threshold` (default 100%) se ejecuta el flush, se regenera el resumen y la FIFO baja al `(1 - flush_eviction_ratio)` (default 50%).
- Modificar los umbrales en `QueueManagerConfig` cambia el comportamiento sin tocar código de runtime.
- El test de "100 mensajes seguidos" no pierde información clave gracias a la consolidación.
- Tras un flush con tool calls en los mensajes expulsados, el siguiente turno del LLM **no falla** con errores de schema validation (workarounds de los bugs #111 y #126 funcionan).
- Tras dos flushes consecutivos, el segundo resumen recursivo contiene información del primero (workaround del bug #118 funciona).

### Implementación

Decisión tomada: **Plan B** (summarizer custom, sin `langmem`). Razones — los tres bugs (#118, #111, #126) tocan exactamente nuestro caso de uso (fusión con resumen previo, tool calls a mitad, parallel tool calls), los workarounds suman más código frágil que la solución directa, y un summarizer propio nos permite implementar la mejora sobre el paper (Working Context como anti-hint en el prompt). `langmem` eliminado de `pyproject.toml`.

Archivos:

- `src/memgpt/queue_manager.py` — `QueueManagerConfig` (umbrales 70 / 100 / 50 % configurables, validators de orden y rango), helpers `count_messages_tokens`, `count_state_tokens`, `group_into_atomic_blocks` (agrupa `AIMessage` con `tool_calls` + sus `ToolMessages` por `tool_call_id`, soporta parallel tool calls), `select_blocks_to_evict` (granularidad de bloque, nunca parte un par tool_call ↔ tool_message).
- `src/memgpt/summarizer.py` — `regenerate_recursive_summary(...)` con LLM callable inyectable (default `call_summarizer` de litellm). El prompt fusiona resumen previo + Working Context (anti-hint) + mensajes expulsados serializados con tool calls visibles.
- `src/memgpt/agent.py` — nuevo nodo `pressure_check` antes de `agent_node` (equivalente al hook `before_model`). El grafo pasa a `START → pressure_check → agent → (tools | END)`, `tools → pressure_check`. La lógica:
  1. Cuenta tokens del estado completo (system prompt + Core Memory + recursive_summary + FIFO).
  2. Si `total ≥ flush_threshold`: selecciona bloques atómicos a expulsar, llama al summarizer con el contexto actual, emite `RemoveMessage` por cada id evictado, actualiza `recursive_summary`, incrementa `evicted_count`, resetea `memory_pressure_alerted`.
  3. Si `total ≥ warning_threshold` y aún no se alertó: inyecta un `SystemMessage` con el aviso y marca `memory_pressure_alerted=True` (una sola vez por episodio; tras flush se resetea).
- `build_agent(...)` acepta `queue_config`, `summarizer_callable`, `token_count_model` (defaults sensatos: 200 000 tokens de ventana, callable de litellm, modelo primario).

Tests Fase 3 (23 nuevos, suite total 60/60):

- `tests/test_queue_manager.py` (10) — config validators, atomic blocks (incluye parallel tool calls y orphan ToolMessage), eviction sin partir pares, monotonicidad del conteo.
- `tests/test_summarizer.py` (7) — fusión con resumen previo, marcador "first flush" sin previo, anti-hint con Core Memory, serialización de tool calls + resultados, fallback a old_summary cuando el LLM devuelve vacío, no llama al LLM si no hay mensajes evictados.
- `tests/test_phase3_flush.py` (6) — alerta inyectada exactamente una vez por episodio (no se re-inyecta en invocaciones posteriores hasta el siguiente flush), flush trim FIFO + escribe summary + resetea flag, en dos flushes consecutivos el prompt del segundo recibe el primero (preservación), pares tool_call ↔ tool_message no se quedan huérfanos, no flush bajo umbral.

---

## 5. Fase 4 — Recall y Archival con Graphiti ✅ Completada

### Objetivos

Sustituir el almacenamiento ingenuo por Graphiti como backend de búsqueda con knowledge graph bi-temporal.

### Tareas

1. Inicializar cliente Graphiti contra Neo4j/FalkorDB.
2. Modificar la lógica de la FIFO para que **cada mensaje** que entra se persista también en Graphiti (vía `add_episode` o equivalente).
3. Implementar las tools:
   - `conversation_search(query, limit, start_date, end_date)` → consulta Graphiti con búsqueda híbrida.
   - `archival_memory_insert(content)` → inserta texto arbitrario en Graphiti.
   - `archival_memory_search(query, limit)` → búsqueda semántica + BM25 + graph traversal.
4. **Modelo de namespacing: un grafo por agente** (decisión tomada). Como solo se va a implementar un agente único, esta opción es la más simple y limpia:
   - Aislamiento total: imposible que datos de otro agente se mezclen (no hay otros).
   - Queries directas sin filtros adicionales (`group_id` fijo o ausente).
   - Borrar el agente = borrar el grafo entero (operación atómica simple).
   - Si en el futuro se añadieran más agentes, la migración a un grafo compartido con `group_id` por agente está soportada nativamente por Graphiti.
   - Implementación: inicializar un único cliente Graphiti apuntando a una base de datos dedicada (ej. `neo4j://localhost:7687/agent_db`) y no pasar `group_id` en las llamadas (o usar uno constante como `"default"`).
5. Modelar timestamps de los mensajes para aprovechar el modelo bi-temporal (cuándo ocurrió + cuándo se aprendió).

### Definición de hecho

- ✅ El agente puede recuperar mensajes antiguos vía `conversation_search` (test `test_evicted_messages_are_searchable_in_recall`).
- ✅ El agente puede insertar y buscar en Archival Memory (test `test_archival_insert_and_search_via_tools`).
- ✅ Mensajes expulsados por el flush de la Fase 3 siguen siendo accesibles en Recall después de salir de la FIFO.

### Implementación

#### Decisión clave: `MemoryStore` con interfaz síncrona

Al diseñar `MemoryStore` había dos opciones obvias:

- **Async**: parecía la opción "correcta" porque Graphiti **solo expone API async** (`await client.add_episode(...)`), Neo4j habla por red, y el principio "async all the way down" es la regla de oro en Python moderno.
- **Sync**: métodos `def` normales, el backend async (Graphiti) se adapta internamente.

**Por qué descarté la opción async**: LangGraph **no es agnóstico al sync/async de los nodos**. Si un nodo se define como `async def`, el grafo entero solo se puede invocar con `agent.ainvoke()`, no con `agent.invoke()`. Verificado con un mini-test antes de empezar:

```python
async def add(state): return {"n": state["n"] + 1}
graph.invoke({"n": 1})
# TypeError: No synchronous function provided to "add".
# Either initialize with a synchronous function or invoke via the
# async API (ainvoke, astream, etc.)
```

Si `MemoryStore.persist_message` fuera async, el nodo `recall_sync` también tendría que ser async (`await store.persist_message(...)`). Eso obligaría a:

1. Reescribir **toda** la suite Fase 0–3 (60 tests) para usar `await agent.ainvoke(...)` en vez de `agent.invoke(...)`.
2. Convertir cualquier código cliente (scripts, ejemplos) a async.
3. Aceptar que el agente "completo" solo funciona en contexto async, aunque el resto de nodos (queue manager, summarizer, core memory) son CPU-bound y no ganan nada con async.

A cambio gano cero rendimiento real: las llamadas a Graphiti se hacen una a la vez en el flujo del agente, no se paralelizan. La concurrencia que aporta async sería puro andamiaje.

**Cómo se resuelve el desajuste con Graphiti**: `GraphitiStore` necesita llamar a una API async desde un método sync. La solución estándar y mala es `asyncio.run(coro)` por llamada, pero eso (a) crea y destruye un event loop por cada `add_episode` (caro) y (b) rompe el driver de Neo4j, que mantiene conexiones ligadas a un loop concreto. La solución usada: **un event loop dedicado vivo en un hilo de fondo**, y enviar cada coroutine con `run_coroutine_threadsafe`:

```python
class GraphitiStore(MemoryStore):
    def __init__(self, client, *, group_id="default"):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._thread.start()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def persist_message(self, ...):
        self._run(self._client.add_episode(...))   # bloquea hasta terminar
```

Propiedades del diseño:

- El driver de Neo4j vive todo el ciclo en un único loop (no se reabre por llamada).
- El método sync bloquea exactamente el tiempo de la llamada de red — overhead nulo respecto a una API async pura.
- El hilo principal del agente (donde corre LangGraph) sigue siendo síncrono y compatible con todos los tests existentes.
- Si en el futuro queremos paralelismo real, ese mismo loop puede lanzar varias tareas a la vez con `asyncio.gather` envuelto en `_run`.

**El trade-off**: pago una pequeña dosis de complejidad en `GraphitiStore` (≈15 líneas de gestión de hilo + loop) a cambio de mantener intactos los 60 tests previos, una API limpia (`store.persist_message(...)` se lee como código normal en el resto del proyecto) y aislamiento (la complejidad async vive solo dentro de la clase que realmente lo necesita).

Es una aplicación del principio "async all the way down" **invertida**: cuando solo una hoja de la arquitectura es async-first y el resto es sync-first, es la hoja la que debe adaptarse, no el árbol entero.

#### Decisión clave: disparar `build_indices_and_constraints()` a mano

El driver Neo4j de Graphiti (`graphiti_core/driver/neo4j_driver.py:96`) intenta auto-construir índices y constraints fulltext/vector al instanciarse, pero **solo si encuentra un event loop corriendo** (`asyncio.get_running_loop()`); si no, cae en un `except RuntimeError: pass` silencioso. Como `GraphitiStore` se construye desde código síncrono (no hay loop corriendo en el hilo principal cuando lo construyes desde `chat.py`, `inspect_web.py`, `default_persistent_agent`, etc.), los índices **nunca se crean**, y la primera búsqueda híbrida revienta con `ProcedureCallFailed: There is no such fulltext schema index: edge_name_and_fact`.

Solución: en el `__init__` del store, tras arrancar el loop dedicado en hilo de fondo, ejecutar `self._run(self._client.build_indices_and_constraints())` envuelto en un `try/except` permisivo (idempotente — Graphiti usa `CREATE IF NOT EXISTS`; si la BD aún no responde, el primer `add_episode` reintentará y dará un error más informativo). El coste son ~200ms al arrancar el primer agente contra una BD vacía; runs posteriores ven los índices ya creados.

#### Decisión clave: silenciar los warnings benignos del driver

El driver Python de Neo4j escupe un `WARNING` por cada label/propiedad que Graphiti consulta antes de que exista en la BD (`(the missing property name is: valid_at)`, `(the missing label name is: Episodic)`, etc.). En una BD vacía la primera invocación llena la terminal con docenas de líneas que asustan pero son inofensivas (las primeras escrituras crean el schema). Subir el nivel del logger en los entry points de UX que toquen Graphiti:

```python
import logging
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
```

Aplicado en `chat.py` y `inspect_web.py`. Los benchmarks (`run_dmr.py`, etc.) pueden hacer lo mismo si la salida estorba; los tests no porque corren con `InMemoryStore`.

#### Persistencia integrada en el grafo

Dos nodos `recall_sync` con la misma lógica:

- `recall_sync_in` antes de `pressure_check`: garantiza que cualquier mensaje (input del usuario, ToolMessages tras un tool call) entra en Recall **antes** de que un posible flush lo evicte.
- `recall_sync_post` justo después de `agent_node`: captura la nueva `AIMessage` aunque el turno acabe en `END` sin volver al loop.

Grafo final: `START → recall_sync_in → pressure_check → agent → recall_sync_post → (tools | END)`, `tools → recall_sync_in → pressure_check`.

Dedup por `message_id`: el set de IDs persistidos vive en `state.persisted_message_ids` (sobrevive a reinicios vía checkpointer) y también en el propio store (defensa en profundidad). Saltamos `SystemMessage` (memory pressure alert + scaffolding) para no contaminar Recall con infraestructura.

Archivos:

- `src/memgpt/memory_store.py` — `MemoryStore` ABC + modelos `RecallEpisode` / `ArchivalEpisode` con campos bi-temporales (`occurred_at` + `learned_at`). `InMemoryStore` (búsqueda por substring, suficiente para tests deterministas, sin Neo4j ni embeddings). `GraphitiStore` con event loop en hilo de fondo, `add_episode` con `source_description` distinto para Recall (`"conversation:<role>"`) vs Archival (`"archival"`), búsqueda vía `search_` con `COMBINED_HYBRID_SEARCH_RRF` filtrando por `source_description`, namespacing por `group_id` (default `"default"`).
- `src/memgpt/recall_archival_tools.py` — factory `make_recall_archival_tools(store)` (closure pattern, store no es serializable así que no puede ir en state) que devuelve `conversation_search(query, limit, start_date?, end_date?)`, `archival_memory_insert(content)`, `archival_memory_search(query, limit)`. Salida formateada como texto plano para que el LLM la consuma en el siguiente turno (`[timestamp] [role] content` o `(no matches)`).
- `src/memgpt/state.py` — campo nuevo `persisted_message_ids: list[str]`.
- `src/memgpt/agent.py` — parámetro `memory_store` en `build_agent`. Cuando se pasa, las 3 tools se añaden automáticamente al toolset y los nodos `recall_sync_in` / `recall_sync_post` se conectan al grafo. Cuando es `None` el comportamiento es idéntico al de Fase 3 (cero overhead).

Tests Fase 4 (18 nuevos, suite total 78/78):

- `tests/test_memory_store.py` (7) — round-trip persist/search, idempotencia por id, filtro por rango de fechas, límite, independencia Recall ↔ Archival.
- `tests/test_recall_archival_tools.py` (5) — factory expone los 3 nombres correctos, insert+search archival vía tools, mensaje `(no matches)`, filtro de fechas, error legible si la fecha ISO es inválida.
- `tests/test_phase4_recall_sync.py` (6) — ambos roles (`user`, `assistant`) persistidos por turno, sin duplicados al re-invocar, `memory_store=None` sigue funcionando, mensajes evictados siguen recuperables vía `conversation_search`, `SystemMessage` no entra a Recall, `ToolMessage` y `AIMessage` con tool_calls se persisten con role correcto.

---

## 6. Fase 5 — `request_heartbeat` y function chaining ✅ Completada

### Objetivos

Permitir al LLM encadenar varias acciones en un mismo turno sin esperar input del usuario. Implementar **ambos modos** (nativo y legacy) y hacerlos seleccionables por configuración, para que el sistema funcione con LLMs modernos y con LLMs antiguos/pequeños sin tocar el código del runtime.

### Diseño: ambos modos coexistiendo

```python
class HeartbeatMode(str, Enum):
    NATIVE = "native"   # LLMs modernos: confiar en reasoning multi-step nativo
    LEGACY = "legacy"   # LLMs antiguos/pequeños: requieren flag explícito

class HeartbeatConfig(BaseModel):
    mode: HeartbeatMode = HeartbeatMode.NATIVE
    max_chained_heartbeats: int = 50      # red de seguridad (ambos modos)
    turn_timeout_seconds: int = 300       # red de seguridad (ambos modos)
    loop_detection_threshold: int = 3     # repeticiones idénticas que cortan (ambos modos)
```

**Cuándo usar cada modo**:

| Modo | Cuándo elegirlo | LLMs típicos |
|---|---|---|
| `NATIVE` (default) | Reasoning multi-step nativo, function calling robusto | Claude Sonnet 4.6, Claude Haiku 4.5, GPT-4o, GPT-5, Gemini 2.x |
| `LEGACY` | El LLM no encadena fiable sin flag explícito; modelos pequeños/locales | Llama 3, Mistral 7B, modelos open-source <14B, GPT-3.5 |

### Tareas

1. Definir `HeartbeatConfig` como clase Pydantic con los parámetros de arriba, inyectable al inicializar el agente.

2. **Implementar ambos modos en el grafo**, seleccionables vía `config.mode`:

   #### Modo `NATIVE` (default)
   - El edge condicional desde `tools_node` vuelve a `agent_node` siempre que el LLM haya emitido tool calls.
   - El LLM "decide" implícitamente si seguir trabajando emitiendo más tool calls o respondiendo en texto al usuario (lo que se interpreta como yield).
   - No se requiere flag explícito ni tipado especial — se aprovecha el reasoning multi-step nativo del LLM moderno.
   - Ventaja: cero ceremonia, prompts más limpios, mejor rendimiento en LLMs grandes.

   #### Modo `LEGACY`
   - Definir un Pydantic model `ToolCallWithHeartbeat` usando `instructor` para tipar el flag `request_heartbeat: bool` en cada tool call.
   - Modificar el system prompt para instruir al LLM cómo usar el flag (`"set request_heartbeat=true if you need another inference cycle"`).
   - Edge condicional desde `tools_node`: si la última tool call tiene `request_heartbeat=True` → volver a `agent_node`; si no (o ausente) → `END` (yield).
   - Ventaja: comportamiento determinista, funciona con LLMs que no encadenan razonamiento de forma fiable.

3. Implementar la **red de seguridad** (común a ambos modos), gestionada vía middleware o nodo dedicado:
   - **Límite máximo de heartbeats encadenados por turno** (default 50): contador en el estado que se incrementa cada vez que el grafo vuelve a `agent_node` desde `tools_node` sin haber yieldado al usuario; al alcanzar `max_chained_heartbeats` se fuerza `END`.
   - **Timeout total por turno** (default 300s = 5 min): si la duración acumulada del turno supera el límite → forzar `END`.
   - **Detección de loops idénticos** (`loop_detection_threshold`, default 3): mantener un buffer de las últimas tool calls (nombre + args hashados); si la misma tool call con los mismos args se repite el número configurado de veces sin progreso → inyectar un mensaje de sistema avisando del loop ("You have repeated the same tool call N times. Either change strategy or respond to the user.") y forzar `END` si el LLM insiste una vez más. Es más fina que las otras dos redes: detecta loops triviales mucho antes de que se agote `max_chained_heartbeats` o el timeout.
   - Estos contadores se resetean al empezar un nuevo turno (recibir un mensaje del usuario o un evento).

4. **Test del switch entre modos**: validar que el mismo flujo funciona en ambos modos cambiando solo `config.mode`, sin tocar el resto del código.

### Definición de hecho

- ✅ En modo `NATIVE`: el agente encadena tool calls hasta que el LLM emite texto (test `test_native_chains_until_yield`). Validar contra Figura 6/8 del paper queda pendiente para las fases de validación (8-10) que sí ejecutan benchmarks reales.
- ✅ En modo `LEGACY`: sin flag o sin `auto_continue_tool`, el agente cede tras una tool call (`test_legacy_ends_without_flag_or_auto_tool`). Con `request_heartbeat=True` continúa (`test_legacy_continues_with_request_heartbeat_true`). Con tool en `auto_continue_tools`, también (`test_legacy_continues_via_auto_continue_tools`).
- ✅ Cambiar `mode` no toca código del agent loop, solo `HeartbeatConfig` (mismos tests cubren NATIVE y LEGACY con la misma definición de tools).
- ✅ `max_chained_heartbeats` corta loops infinitos exactamente al alcanzarse (`test_native_max_chained_heartbeats_caps_loop`).
- ✅ Detección de loops idénticos: se inyecta el warning al alcanzar `loop_detection_threshold` y se fuerza END en la siguiente repetición (`test_loop_detection_injects_warning_then_ends`).
- ✅ Timeout por turno corta correctamente cuando `turn_started_at` está vencido (`test_native_turn_timeout_forces_end`).
- ✅ Contadores se resetean al recibir un `HumanMessage` con id distinto al `last_processed_human_id` (`test_new_human_message_resets_per_turn_counters`).

### Implementación

#### Decisión clave: `parallel_tool_calls=False` en el binding del LLM

LLMs modernos (OpenAI gpt-4o, Anthropic claude-sonnet-4.x) emiten **tool calls en paralelo** por defecto cuando la petición lo sugiere. Caso típico observado en `chat.py`: el usuario dice "*me llamo Máximo, llámame memcito*" y el LLM dispara `core_memory_append(label='human', ...)` y `core_memory_append(label='assistant', ...)` en el mismo step. Cada tool devuelve `Command(update={"core_memory": ...})`. Como el campo `core_memory` de `MemGPTState` no tiene reducer (es un `CoreMemory` Pydantic plano, no `Annotated[..., reducer]`), LangGraph rechaza con:

```
InvalidUpdateError: At key 'core_memory': Can receive only one value per step.
```

Dos vías obvias y por qué descartamos una:

- **Añadir un reducer de merge a `core_memory`**: rompe el `default_factory=default_core_memory` (LangGraph reinicia el canal a `CoreMemory()` vacío sin los bloques `assistant`/`human` que la fase 2 garantiza por defecto). Habría que reescribir cómo se siembra el estado inicial, lo cual filtra fontanería de LangGraph a `core_memory.py`.
- **Forzar tool calls secuenciales** (elegida): `llm.bind_tools(tools, parallel_tool_calls=False)` en `_build_llm`. El LLM emite un tool a la vez, el grafo lo encadena por heartbeat (que ya existe desde esta misma fase), y la fase siguiente del razonamiento ve el estado ya actualizado por el tool anterior. Es **lo natural en MemGPT**: el agente está diseñado en torno a function chaining encadenado, no a fan-out paralelo.

Detalle de implementación: envolver el `bind_tools` en `try/except TypeError` para tolerar providers/versiones que no acepten el kwarg (caen al binding por defecto, asumiendo que ese provider no emita parallel tool calls por su cuenta — verdad hoy para Anthropic con `langchain-anthropic<0.3.x`).

Coste en latencia: un round-trip extra al LLM por cada tool adicional que el modelo hubiera emitido en paralelo. En la práctica son 1-3 tools por turno y el coste dominante es la inferencia, no el round-trip.

#### Decisión clave 1: `auto_continue_tools` en lugar de envolver tools con un flag

El plan original (`ToolCallWithHeartbeat` vía `instructor`) propone añadir el parámetro `request_heartbeat: bool` al schema de **cada** tool en modo LEGACY. El problema con nuestro toolset actual:

- Los `core_memory_*` (Fase 2) usan `Annotated[MemGPTState, InjectedState]` y `Annotated[str, InjectedToolCallId]`. La inyección de estado de LangGraph se basa en metadata Pydantic v2 que se pierde al reconstruir un schema con `create_model(...)`. Envolverlos rompe la inyección.
- Las tools de Recall/Archival (Fase 4) viven dentro de un closure sobre el `MemoryStore`. Wrappear preservando la closure es factible pero añade complejidad sin valor.
- Tools simples (`get_current_time`) no tienen estado pero el LLM no necesita el flag para ellas; siempre llevan a más razonamiento.

Solución elegida: **mantener el schema de las tools intacto** y declarar por configuración qué tools señalan continuación implícita en LEGACY mode (`auto_continue_tools: frozenset[str]`). La heurística por defecto incluye los 7 tools "preparatorios" del runtime (4 `core_memory_*` + 3 Recall/Archival): editar Core Memory o consultar Recall siempre llevan a un siguiente paso de razonamiento. Para tools custom donde el LLM debe poder elegir, el usuario añade `request_heartbeat: bool = False` en la firma de su función `@tool` y el LLM emite el flag. Soporta los dos modelos de la realidad sin adaptadores frágiles.

El flujo de decisión LEGACY queda:

1. ¿Alguna tool call tiene `request_heartbeat=True` en sus args? → continuar.
2. ¿El nombre de alguna tool call está en `auto_continue_tools`? → continuar.
3. En cualquier otro caso → END.

Este diseño es **conservador hacia el yield**: ante la duda, ceder al usuario antes que loopear.

#### Decisión clave 2: `heartbeat_check` (side effects) + `heartbeat_router` (decisión pura)

Se podría haber resuelto el routing con un único nodo que devuelve `Command(goto=...)`. Lo descarté porque complica los tests: el side effect (incrementar contador, inyectar warning) y el routing condicional acaban entremezclados, y la lógica del router se vuelve difícil de testear de forma aislada. Patrón usado:

- `heartbeat_check`: nodo normal que actualiza state (incrementa `chained_heartbeats`, añade keys al buffer FIFO de `recent_tool_call_keys`, inyecta `SystemMessage` de warning si una key alcanza el threshold).
- `heartbeat_router`: función pura `state → "continue" | "end"` registrada como `add_conditional_edges`. Lee el state ya actualizado y decide. Sin side effects, fácilmente testeable.

#### Detección de nuevo turno

Un turno empieza cuando llega un `HumanMessage` cuyo id no coincide con `state.last_processed_human_id`. El nodo `turn_init` (que sustituye a `START → recall_sync_in`) se encarga del reset:

- Si el último `HumanMessage` ya fue procesado (mismo id) → return `{}` (nada que hacer; estamos a mitad de loop o en una invocación sin novedades).
- Si es nuevo → resetear `chained_heartbeats=0`, `recent_tool_call_keys=[]`, `turn_started_at=now`, y guardar el id.

Esto evita que el contador del turno anterior contamine el nuevo turno y que la detección de loops dispare por repeticiones cruzadas entre turnos.

#### Loop detection: warning + 1 antes de cortar

El plan describe "inyectar warning… y forzar END si el LLM insiste una vez más". Implementación literal:

- En `heartbeat_check`, si una key recién añadida tiene exactamente `threshold` ocurrencias en el buffer reciente → inyectar `SystemMessage` con `LOOP_DETECTION_WARNING_TEMPLATE`.
- En `heartbeat_router`, si la key más reciente tiene **más** de `threshold` ocurrencias → END.

Resultado con threshold=3: 3 llamadas idénticas → warning. La 4ª → END. El test `test_loop_detection_injects_warning_then_ends` valida exactamente este patrón: 4 invocaciones LLM, 1 warning, fin del turno.

#### Grafo final

```
START → turn_init → recall_sync_in → pressure_check → agent → recall_sync_post → tools_condition(tools|END)
tools → heartbeat_check → heartbeat_router(continue|end)
continue → recall_sync_in
end → END
```

`heartbeat_check` solo se ejecuta tras `tools` (es ahí donde tiene sentido: el agente decidió encadenar, el tool corrió, ahora la red de seguridad evalúa si seguir).

#### Archivos

- `src/memgpt/heartbeat.py` — `HeartbeatMode` enum, `HeartbeatConfig` Pydantic (mode, max_chained_heartbeats, turn_timeout_seconds, loop_detection_threshold, auto_continue_tools, recent_keys_buffer), `tool_call_key()` (hash determinista nombre+args), `extract_tool_call_keys()`, `loop_repetition_count()`, `LOOP_DETECTION_WARNING_TEMPLATE`. `DEFAULT_AUTO_CONTINUE_TOOLS` cubre los 7 tools internos.
- `src/memgpt/state.py` — campos nuevos: `chained_heartbeats`, `turn_started_at`, `recent_tool_call_keys`, `last_processed_human_id`.
- `src/memgpt/agent.py` — parámetro `heartbeat_config` en `build_agent`. Nodos nuevos `turn_init` y `heartbeat_check`. Función `heartbeat_router`. Grafo reescrito con `START → turn_init → …` y la rama `tools → heartbeat_check → router`.

#### Tests Fase 5 (18 nuevos, suite total 96/96)

- `tests/test_heartbeat.py` (10) — defaults, validators de campo, coerción de `auto_continue_tools` desde set/list/frozenset, rechazo de garbage, estabilidad de `tool_call_key` ante args desordenados, distinción de args distintos, fallback con args no-JSON, extracción de claves para tool calls paralelas, conteo en buffer, consistencia con `DEFAULT_AUTO_CONTINUE_TOOLS`.
- `tests/test_phase5_chaining.py` (8) — NATIVE encadena hasta yield del LLM; `max_chained_heartbeats` corta exactamente al límite; `turn_timeout` corta cuando `turn_started_at` está vencido; LEGACY sin flag ni auto-tool yield tras 1 tool call; LEGACY con `request_heartbeat=True` continúa; LEGACY con `auto_continue_tools` continúa; loop detection inyecta exactamente 1 warning y termina en la siguiente; nuevo `HumanMessage` con id distinto resetea `chained_heartbeats` y `last_processed_human_id`.

---

## 7. Fase 6 — Eventos automáticos ✅ Completada

### Objetivos

Soportar tanto eventos wall-clock como eventos por iteraciones.

### Tareas

1. **Eventos wall-clock (APScheduler 3.x)**:
   - Crear un `AsyncIOScheduler` global.
   - API para registrar eventos: `schedule_event(agent_id, cron_expression, event_payload)`.
   - Cuando vence un evento: construir un `SystemMessage` con el contenido y disparar `graph.ainvoke()` con un nuevo evento en la FIFO.
2. **Eventos por iteraciones (LangChain Middleware)**:
   - Hook `after_model` que incrementa `step_count` en el estado.
   - Lista de "sleep-time agents" registrados con `(every_n_steps, callback)`.
   - En cada `after_model`, comprobar si `step_count % N == 0` y disparar el callback.
3. Definir API unificada de registro de eventos en una clase `EventRegistry`.
4. Persistir el calendario de eventos en BD para que sobreviva a reinicios.

### Definición de hecho

- ✅ Un evento programado a una hora concreta dispara un mensaje al agente (`test_wallclock_dispatcher_drives_agent_via_systemmessage` valida la cadena dispatcher → SystemMessage → invocación del grafo; el cableado APScheduler → dispatcher se valida en `test_registered_wallclock_job_invokes_dispatcher_with_correct_args` disparando manualmente el `Job` registrado, sin esperar al cron real).
- ✅ Un sleep-time agent "cada N pasos" funciona (`test_iteration_callback_fires_every_n_steps`: 5 LLM calls + `every_n_steps=2` ⇒ callback se ejecuta exactamente en steps 2 y 4).
- ✅ Tras reiniciar el proceso, los eventos programados se recuperan (`test_restore_reloads_persisted_events_into_fresh_registry`: registry-1 persiste en `EventStore`, registry-2 fresco invoca `restore()` y recupera los specs).

### Implementación

#### Decisión clave 1: `step_tick` como nodo del grafo, no middleware

LangGraph 1.x soporta middlewares, pero la integración con un nodo Pydantic-tipado y con el resto del grafo (que ya tiene `pressure_check`, `recall_sync_*`, `heartbeat_check` como nodos) es más limpia y testeable como **nodo dedicado**. `step_tick` se inserta entre `agent` y `recall_sync_post`:

```
agent → step_tick → recall_sync_post → tools_condition(...)
```

El nodo:
1. Incrementa `state.step_count`.
2. Llama a `event_registry.dispatch_iteration(state_with_new_count)` para fusionar todas las callbacks aplicables.
3. Devuelve el update combinado (step_count + cualquier `messages`/otros campos producidos por las callbacks).

Si `event_registry is None`, **el nodo no se añade al grafo en absoluto** (no es un no-op condicional, sino ausencia total). Esto preserva el contrato de Fase 5: los 96 tests previos siguen pasando sin tocar nada.

Las `messages` emitidas por una callback fluyen por el reducer `add_messages` y son persistidas por `recall_sync_post` en el siguiente paso, sin código adicional.

#### Decisión clave 2: dispatcher inyectable separado del registry

El plan habla de "disparar `graph.ainvoke()`", pero el `EventRegistry` no debería **conocer** al agente — eso crea un ciclo `agent → registry → agent` que complica la inicialización (necesitas el agente para crear el registry, pero el agente recibe el registry por constructor).

Solución: el registry recibe un `WallClockDispatcher = Callable[[str, str], None]` inyectable. El módulo provee una factoría `default_wallclock_dispatcher(agent)` que cierra sobre el agente compilado y construye un `SystemMessage` con el payload. El orden de inicialización queda:

```python
registry = EventRegistry(scheduler=..., store=...)  # sin dispatcher todavía
agent = build_agent(..., event_registry=registry)
registry.set_wallclock_dispatcher(default_wallclock_dispatcher(agent))
```

Bonus: `payload_to_message` permite cambiar el wrapping (p. ej. enviar como `HumanMessage` para que entre a Recall, en vez de `SystemMessage` que el `recall_sync` filtra como scaffolding).

#### Decisión clave 3: persistencia solo de wall-clock (no de iteration)

Las callbacks de iteration son `Callable` Python — no serializables. Persistirlas requeriría un mecanismo de registro por nombre (factory pattern) que añade ceremonia para poco beneficio: las iteration están conceptualmente ligadas al código del agente y se re-registran en el módulo de bootstrap en cada arranque.

Los wall-clock events sí se persisten (un `cron_expression` o `interval` es texto puro). El backend `EventStore` es ABC con dos implementaciones:
- `InMemoryEventStore` — tests y agentes single-process sin necesidad de supervivencia.
- `PostgresEventStore` — tabla única `memgpt_wallclock_events(name PK, spec JSONB)`. `psycopg` se importa perezosamente para no obligar a tenerlo instalado si solo se usa el InMemoryStore.

`EventRegistry.restore()` reabre todos los specs desde el store y los reinscribe contra el scheduler vivo. Es idempotente: si un spec ya está registrado en el registry actual, lo salta.

#### Decisión clave 4: `BackgroundScheduler` por defecto, no `AsyncIOScheduler`

El plan menciona `AsyncIOScheduler`, pero el resto del proyecto es **sync-first** (decisión de Fase 4: `MemoryStore` síncrono para no contagiar `async` a los nodos LangGraph). `BackgroundScheduler` corre los jobs en un thread pool propio sin requerir un event loop activo, mantiene la coherencia con el resto del código y no obliga a los tests a `asyncio.run(...)`. Es inyectable (`scheduler=AsyncIOScheduler()`) si en el futuro se monta el agente dentro de un servicio async.

#### Archivos

- `src/memgpt/events.py` — módulo nuevo: `IterationEvent` (Pydantic con `Callable` arbitrary), `WallClockEvent` (spec serializable + `build_trigger()` para los 3 tipos `cron`/`interval`/`date`), `EventStore` Protocol + `InMemoryEventStore` + `PostgresEventStore` (tabla JSONB con `psycopg` perezoso), `EventRegistry` (orquestador unificado: `register_iteration`/`register_wallclock`/`dispatch_iteration`/`restore`/`start`/`shutdown`), `default_wallclock_dispatcher(agent, payload_to_message=...)` factory.
- `src/memgpt/agent.py` — parámetro `event_registry: EventRegistry | None` en `build_agent`. Cuando se pasa, se añade el nodo `step_tick` entre `agent` y `recall_sync_post`; cuando es `None` el grafo es idéntico al de Fase 5 (zero overhead). El nodo `step_tick` incrementa `step_count` y llama a `event_registry.dispatch_iteration(...)`.

#### Tests Fase 6 (27 nuevos, suite total 123/123)

- `tests/test_events.py` (21) — `WallClockEvent` construye los 3 tipos de trigger / rechaza tipos desconocidos / round-trips por JSON; `IterationEvent` exige `every_n_steps > 0`; `InMemoryEventStore` save/list/delete e idempotencia de overwrite por nombre; `EventRegistry` salta dispatch cuando `step_count == 0`, dispara solo en múltiplos de N, fusiona `messages` de varias callbacks y deja "last write wins" para otras claves, rechaza duplicados, `unregister` idempotente, exige dispatcher antes de registrar wall-clock, registra job APScheduler con args correctos, `restore` recarga en registry fresco y salta lo ya registrado, `set_wallclock_dispatcher` post-construcción funciona, `start`/`shutdown` idempotentes; `default_wallclock_dispatcher` envía `SystemMessage` con `thread_id=agent_id` y acepta `payload_to_message` custom.
- `tests/test_phase6_events.py` (6) — E2E con `build_agent`: `step_count` se incrementa exactamente una vez por LLM call y persiste vía checkpointer; sin `event_registry` el `step_count` queda en su default (compat Fase 5); callback "every 2 steps" se ejecuta exactamente en steps 2 y 4 sobre 5 invocaciones; callback que devuelve `messages` los appendea al estado vía el reducer; dispatcher por defecto procesa el `SystemMessage` como nuevo turno y produce respuesta del agente; el `Job` APScheduler registrado dispara al dispatcher con `(agent_id, payload)` correctos al llamar `job.func(*job.args)` manualmente.

---

## 8. Fase 7 — Persistencia robusta entre sesiones ✅ Completada

### Objetivos

Asegurar que todo el estado (Core Memory + FIFO + Graphiti + eventos) sobrevive a cierres de proceso.

### Tareas

1. Migrar checkpointer de `MemorySaver` a `PostgresSaver`.
2. Verificar que Graphiti persiste todo lo escrito (Neo4j/FalkorDB).
3. Implementar test de "kill 9 + restart": al rearrancar, el agente debe ver:
   - Core Memory tal y como quedó.
   - Resumen recursivo de la FIFO en slot 0.
   - Conversaciones pasadas accesibles vía `conversation_search`.
   - Eventos programados pendientes.
4. Definir estrategia de write-through: cada modificación de Core Memory se persiste inmediatamente.
5. Añadir transacciones atómicas en operaciones que tocan varios bloques (ej. `core_memory_replace` que hace borrar + insertar).

### Definición de hecho

- ✅ Test E2E "kill + restart" valida los 3 pilares con backends compartidos (`test_kill_restart_preserves_core_memory_and_summary`, `test_kill_restart_preserves_recall_after_flush`, `test_kill_restart_restores_wallclock_events`).
- ✅ `core_memory_replace` ejecuta una única escritura atómica (`test_core_memory_replace_is_one_atomic_step` recorre `get_state_history` y verifica que ningún snapshot intermedio contiene un estado parcial).
- ✅ Round-trip Pydantic completo de `MemGPTState` con todos los campos de Fases 0-6 (`test_memgpt_state_round_trips_through_pydantic`).
- ✅ La factory `build_persistent_agent` instala el dispatcher por defecto y llama a `restore()` automáticamente (`test_build_persistent_agent_wires_dispatcher_and_restore`).

### Implementación

#### Decisión clave 1: write-through es propiedad emergente, no código

El plan pide "write-through: cada modificación de Core Memory se persiste inmediatamente". En la arquitectura actual, esto **ya ocurre** sin código adicional:

- `MemGPTState.core_memory` es un campo del estado de LangGraph.
- LangGraph llama al checkpointer al final de cada step de nodo.
- `PostgresSaver.put(...)` escribe el snapshot en una transacción.

Por tanto, el momento en que el tool `core_memory_*` devuelve `Command(update={"core_memory": ...})`, el siguiente checkpoint es exactamente la nueva Core Memory. No hay ventana en la que el estado live diverja del estado persistido más allá de la duración de un step. Confirmado en `test_core_memory_replace_is_one_atomic_step`: el `get_state_history` solo muestra valores legales (el original o el final), nunca un híbrido.

Lo mismo aplica a Recall/Archival (Graphiti hace commit por episodio) y a wall-clock events (`PostgresEventStore.save_wallclock` hace commit por insert). El stack persiste por construcción.

#### Decisión clave 2: atomicidad por agrupación en `Command(update=...)`

El plan menciona "transacciones atómicas en operaciones que tocan varios bloques (ej. `core_memory_replace` que hace borrar + insertar)". El diseño actual lo resuelve sin transacciones explícitas:

- `core_memory_replace` calcula la nueva Core Memory (con el bloque modificado) **fuera** del estado, en `CoreMemory.with_replaced(label, old, new)` — un método inmutable que devuelve una instancia nueva.
- Devuelve **un único** `Command(update={"core_memory": <new>, "messages": [<ToolMessage>]})`.
- LangGraph aplica ese update como un único checkpoint write.

No hay borrar + insertar separados que puedan dejar estado intermedio. La granularidad atómica es el step, no la operación de bajo nivel. Esta decisión se tomó implícitamente en Fase 2 al modelar `CoreMemory` como inmutable; Fase 7 simplemente la audita.

#### Decisión clave 3: `PostgresSaver` se expone como context manager

`langgraph.checkpoint.postgres.PostgresSaver.from_conn_string(...)` devuelve un context manager (gestiona la pool de conexiones). Forzar un constructor plano filtraría el ciclo de vida de la pool. El módulo `persistence` provee `postgres_checkpointer(dsn)` como context manager:

```python
with postgres_checkpointer(dsn) as saver:
    agent = build_agent(checkpointer=saver, ...)
    agent.invoke(...)
```

`PostgresSaver.setup()` se llama dentro del with — es idempotente, crea las tablas `checkpoints*` solo si faltan. Apto para arranque de servicio.

#### Decisión clave 4: `build_persistent_agent` cierra el ciclo registry ↔ agent

El `EventRegistry` (Fase 6) necesita un dispatcher que invoca al agente; el agente recibe el registry por constructor. Es un ciclo. La factory `build_persistent_agent`:

1. Crea/recibe el `EventRegistry` (sin dispatcher todavía).
2. Compila el agente con `event_registry=registry`.
3. Instala `default_wallclock_dispatcher(agent)` en el registry.
4. Llama a `registry.restore()` para reinscribir wall-clock events persistidos antes de que el cliente arranque el scheduler.

El cliente recibe `(agent, registry)` y solo tiene que hacer `registry.start()`. Una llamada en lugar de cuatro pasos manuales con orden frágil.

#### Decisión clave 5: kill+restart se simula con backends compartidos

El test E2E del plan pide "conversación → kill → restart → continuar". Reproducirlo contra Postgres real exige levantar contenedores en CI; reproducirlo contra `MemorySaver` + `InMemoryStore` + `InMemoryEventStore` **compartidos entre dos instancias del agente** valida exactamente el mismo contrato:

```python
saver = MemorySaver()
store = InMemoryStore()
agent1 = build_agent(checkpointer=saver, memory_store=store, ...)
agent1.invoke(...)
del agent1                              # "kill -9"
agent2 = build_agent(checkpointer=saver, memory_store=store, ...)  # "restart"
assert agent2.get_state(cfg).values == ...
```

Si el contrato del checkpointer (put/get sobre `thread_id`) se cumple para `MemorySaver`, también se cumple para `PostgresSaver` (ambos heredan de `BaseCheckpointSaver`). Lo mismo para `MemoryStore` y `EventStore`. Este patrón evita infra en tests sin perder cobertura del invariante: **ninguna referencia Python del agente original sobrevive a la "muerte"**.

#### Archivos

- `pyproject.toml` — añadida `langgraph-checkpoint-postgres>=2.0`.
- `src/memgpt/persistence.py` — módulo nuevo: `postgres_checkpointer(dsn)` context manager (llama a `setup()` + cede el `PostgresSaver`); `build_persistent_agent(...)` factory que construye agente + registry + dispatcher + restore en una llamada y devuelve `(agent, registry)`; `default_persistent_agent(...)` atajo que usa los DSN de `Settings`.
- (Sin cambios en `agent.py`, `state.py`, `core_memory.py`, `tools.py` — la persistencia es propiedad emergente del diseño previo.)

#### Tests Fase 7 (7 nuevos, suite total 130/130)

- `tests/test_phase7_persistence.py` (7) — round-trip Pydantic con todos los campos de Fases 0-6 (Core Memory + recursive_summary + step_count + heartbeat counters + persisted_message_ids + turn_started_at); kill+restart con `MemorySaver` + `InMemoryStore` compartidos preserva Core Memory y recursive_summary; tras un flush en sesión 1, los mensajes evictados siguen recuperables por `search_conversation` en sesión 2 y `evicted_count` se mantiene; wall-clock events sobreviven al restart vía `EventRegistry.restore()`; `core_memory_replace` no deja estados intermedios en `get_state_history`; `build_persistent_agent` instala el dispatcher y rechaza la combinación ambigua `event_registry` + `event_store_dsn`.

---

## 9. Fase 8 — Validación con Nested KV (benchmark más simple) ✅ Completada

### Objetivos

Replicar el primer benchmark del paper para validar function chaining + paginación.

### Tareas

1. Descargar el dataset Nested KV de https://huggingface.co/MemGPT/datasets.
2. Cargar 140 pares clave-valor en Archival Memory.
3. Definir el asistente del agente con el prompt del apéndice 6.1.6 ("DO NOT STOP SEARCHING UNTIL...").
4. Ejecutar las 30 configuraciones × 5 niveles de anidamiento = 150 queries.
5. Medir accuracy y comparar con el resultado del paper:
   - GPT-4 baseline: cae a 0% en nivel 3.
   - MemGPT con GPT-4: 100% en todos los niveles.
6. Iterar sobre los fallos hasta superar el umbral del paper.

### Definición de hecho

- ✅ Generación reproducible del dataset (140 pares × 30 configs × 5 niveles = 150 queries) verificada por `tests/test_nested_kv.py` (19 tests).
- ✅ Asistente del agente reproduce literalmente el apéndice 6.1.6 ("DO NOT STOP SEARCHING UNTIL...").
- ✅ Runner CLI (`scripts/run_nested_kv.py`) ejecuta el benchmark contra cualquier modelo (`--model`) y con InMemoryStore o GraphitiStore (`--graphiti`). Sale con código 0 si la accuracy global ≥ 95 % — el umbral del paper.
- ⏳ Accuracy ≥ 95 % y tiempo medio < 30 s queda pendiente de la ejecución contra el LLM real (requiere créditos / decisión del usuario sobre el modelo).

### Cómo funciona el benchmark, paso a paso

**Construcción de cada configuración**. 140 pares clave-valor (UUIDs v4) en los que 5 forman una **cadena guía** y los otros 135 son distractores:

```
k0 → k1 → k2 → k3 → k4 → terminal_no_clave
```

El valor de cada `kᵢ` es el `kᵢ₊₁` siguiente, excepto el último (`k4`), cuyo valor es un UUID `terminal` que **no aparece como clave** en ningún par del dataset. Los distractores tienen claves frescas y valores que no chocan con la cadena (verificado en `test_generate_config_distractor_values_are_not_chain_keys`).

**Las 5 queries por configuración** (una por nivel de anidamiento) parten de un punto distinto de la misma cadena:

| Nivel | Start key | Cadena de búsquedas | Saltos |
|------:|-----------|---------------------|-------:|
| 0 | k4 | k4 → terminal | 1 |
| 1 | k3 | k3 → k4 → terminal | 2 |
| 2 | k2 | k2 → k3 → k4 → terminal | 3 |
| 3 | k1 | k1 → k2 → k3 → k4 → terminal | 4 |
| 4 | k0 | k0 → k1 → k2 → k3 → k4 → terminal | 5 |

Las 5 queries son **independientes**: cada una usa su propio `thread_id` (`nested-kv-cfg{N}-lvl{L}`), arranca con un `MemGPTState` vacío y no comparte mensajes con las demás. Lo que sí comparten es la archival memory: los 140 pares idénticos. Por eso todas las queries OK de una misma configuración predicen el **mismo terminal** — es el final único de la cadena guía.

**El "+1" de la verificación**. El contador de `archival_search_calls` registra `nivel + 2` búsquedas, no `nivel + 1`. La búsqueda extra es la que comprueba que el terminal no es a su vez una clave: el agente lo busca en archival y obtiene un solo match (el par donde aparece como **valor**, no como clave), confirmando que la cadena se acaba ahí. Es la heurística que el system prompt del apéndice 6.1.6 dicta literalmente:

> "DO NOT STOP SEARCHING UNTIL YOU VERIFY THAT THE VALUE IS NOT A KEY."

Sin esa verificación final el agente no sabría distinguir un UUID intermedio (que sí es clave de un par siguiente) de un UUID terminal.

### Implementación

#### Decisión clave 1: regenerar el dataset, no descargarlo

El plan apuntaba a `https://huggingface.co/MemGPT/datasets`, pero la inspección del perfil (`MemGPT-DPO-Dataset`, `MSC-Self-Instruct`, `qa_data`, etc.) confirma que **el dataset Nested KV nunca se publicó allí**. El paper (§3.2.2) describe la construcción con suficiente detalle para reproducirla determinísticamente, así que `generate_dataset(seed=42, n_configs=30)` produce 30 configuraciones idénticas en cualquier máquina sin red ni dependencias adicionales.

Estructura por configuración:
- 140 pares clave-valor de UUIDs v4 derivados del RNG (`Random(seed + cfg_id)` ⇒ determinista).
- `chain_length=5` pares forman la cadena guía: `k0 → k1 → k2 → k3 → k4 → terminal_no_clave`.
- 135 pares distractores con claves frescas y valores que NO colisionan con la cadena (evita anidamiento espurio).
- 5 queries por config: nivel `L` parte de `k_{4-L}` y exige `L+1` lookups hasta llegar al `terminal`.

#### Decisión clave 2: una archival aislada por configuración

Reutilizar la misma `MemoryStore` entre configs contaminaría las búsquedas (la cadena del config N aparecería al consultar la del config M y los distractores explotarían). El runner construye un `MemoryStore` y un `agent` nuevos por config; al terminar las 5 queries cierra el store y descarta el agente. Coste: 30 init de Graphiti — aceptable comparado con el coste de las 150 inferencias.

#### Decisión clave 3: scoring por extracción de UUID, no comparación literal

El LLM puede preceder/seguir el UUID final con explicaciones aunque el asistente pida solo el UUID. `extract_uuid` aplica una regex y se queda con el **último** match (que coincide con la conclusión del razonamiento). La comparación es case-insensitive porque algunos modelos mayúsculan los UUIDs.

#### Decisión clave 4: tests sin LLM con un `_StubAgent`

Los 19 tests en `tests/test_nested_kv.py` no consumen tokens. `_StubAgent` resuelve la cadena leyendo directamente el `InMemoryStore` y devuelve un `AIMessage` con el terminal — equivalente a un LLM perfecto. Eso permite validar el cableado dataset → store → agent → scoring sin depender de la red ni de credenciales. La validación contra LLM real se delega al runner CLI on-demand.

#### Archivos

- `src/memgpt/benchmarks/__init__.py` — paquete nuevo.
- `src/memgpt/benchmarks/nested_kv.py` — generador de dataset, runner programático (`run_benchmark`), asistente del apéndice 6.1.6, plantilla de query, extracción de UUID y scoring.
- `scripts/run_nested_kv.py` — CLI con flags `--configs`, `--levels`, `--seed`, `--model`, `--graphiti`, `--output`.
- `tests/test_nested_kv.py` — 19 tests de generación, scoring, archival, asistente, runner con stub.

#### Cómo ejecutar

```bash
# Smoke run rápido sobre un solo config (5 queries, ~mínimo coste).
uv run scripts/run_nested_kv.py --configs 1

# Benchmark completo (paper) con InMemoryStore.
uv run scripts/run_nested_kv.py --configs 30 --output runs/nested_kv.json

# Contra Graphiti real (requiere docker compose up postgres neo4j).
uv run scripts/run_nested_kv.py --configs 30 --graphiti
```

---

## 10. Fase 9 — Validación con DMR (Multi-Session Chat) ✅ Completada

### Objetivos

Replicar el segundo benchmark del paper para validar Working Context + Recall + persistencia entre sesiones.

### Tareas

1. Descargar el MSC aumentado (sesión 6 incluida) desde Hugging Face.
2. Para cada par de usuarios:
   - Ejecutar las sesiones 1-5 secuencialmente, con cierre y reapertura simulada entre sesiones.
   - Después de la sesión 5, abrir la sesión 6 y formular la pregunta del DMR.
   - Evaluar la respuesta con LLM-as-a-judge (prompt del apéndice 6.1.2).
3. Métricas: ROUGE-L (recall) + accuracy del judge.
4. Comparar con paper:
   - Baseline GPT-4: 32.1% accuracy.
   - MemGPT GPT-4: 92.5%.
   - Objetivo con Graphiti: superar 92.5% (Graphiti reporta 94.8% en este benchmark).

### Definición de hecho

- ✅ Loader del dataset `MemGPT/MSC-Self-Instruct` (500 pares) con descarga lazy desde HF Hub vía `huggingface_hub.hf_hub_download`. Verificado por `tests/test_dmr.py` (parsing + asignación de speakers + orden cronológico de sesiones).
- ✅ Ingesta de las 5 sesiones en Recall **sin gastar LLM**: cada turno se persiste con `persist_message`, anclando cada sesión en su propio `occurred_at` derivado de `time_back`. Esto reproduce la condición del paper ("MemGPT … has access to the full conversation history but must access it via paginated search queries to recall memory") sin pagar 70 inferencias por par.
- ✅ Agente DMR con persona en Core Memory (`persona`/`human` blocks) + system prompt literal del apéndice 6.1.1 ("completely immerse myself in this role").
- ✅ Sesión 6 ejecutada en un thread separado: el agente usa `conversation_search` para recuperar el dato.
- ✅ ROUGE-L recall implementado nativo (LCS DP) — sin dependencia adicional. Ignora puntuación y mayúsculas, alineado con la métrica del paper.
- ✅ LLM-judge con el prompt del apéndice 6.1.2 (litellm `completion`, temperatura 0.0); parser de veredicto tolerante a "CORRECT/WRONG" en cualquier capitalización y que abstiene si aparecen ambos términos.
- ✅ Baseline (control sin memoria) con el preprompt del apéndice 6.1.1 + el resumen lossy `summary_speaker_*` incrustado en el system prompt; ningún tool de memoria.
- ✅ Runner CLI (`scripts/run_dmr.py`) con InMemoryStore o GraphitiStore (`--graphiti`), modo baseline (`--baseline`), límite de muestras (`--limit`), modelo override (`--model`, `--judge-model`), descarga lazy (`--download`), dump incremental, y exit code distinto para `accuracy ≥ 0.92 ∧ ROUGE-L ≥ 0.80` vs fail vs abort.
- ⏳ Accuracy ≥ 92% y ROUGE-L recall ≥ 0.80 quedan pendientes de la ejecución contra LLM real (requiere créditos / decisión del usuario sobre el modelo).

### Cómo funciona el benchmark, paso a paso

**El dataset (`MemGPT/MSC-Self-Instruct`)**. 500 registros JSONL. Cada registro es un par de personas (`Speaker 1` y `Speaker 2`) con todo lo necesario para reconstruir su historia compartida:

- `personas`: dos listas de hechos (uno por speaker), p. ej. *"I like Taylor Swift"*, *"I have two dogs"*. Es el "personaje" que cada uno mantiene durante toda la conversación.
- `previous_dialogs`: 4 sesiones anteriores (sesiones 1-4). Cada sesión es una secuencia de turnos `{"text": ...}` que **alterna estrictamente** Speaker 1 → Speaker 2 (sin ID explícito). Cada sesión incluye `time_num` + `time_unit` (e.g. `5 days`) que indica cuánto tiempo pasó hasta la siguiente.
- `dialog`: la sesión 5, esta sí con `id: "Speaker 1"`/`"Speaker 2"` explícito en cada turno.
- `self_instruct`: `{"B": pregunta, "A": respuesta gold}` — la pregunta + respuesta sintética de la **sesión 6**, generada por un LLM con la condición de que solo se pueda contestar habiendo participado en las sesiones 1-5 (no leyendo solo la persona).
- `summary_speaker_1` / `summary_speaker_2`: resúmenes lossy progresivos de la conversación, usados solo por el baseline para imitar "extended recursive summarization".

**Por par evaluado, el pipeline hace cinco cosas**:

1. **Aislar memoria**. Construye un `MemoryStore` y un agente nuevos. Las conversaciones de un par no deben contaminar las búsquedas de otro.

2. **Volcar las 5 sesiones en Recall directamente**. Recorre `previous_dialogs[0..3]` + `dialog` y por cada turno llama `store.persist_message(content, role, occurred_at, message_id)` con:
   - `role = "assistant"` si el speaker coincide con el agente (default `Speaker 1`), `"user"` si es el otro.
   - `occurred_at` derivado de los `time_back` acumulados desde la sesión 5: la sesión más antigua queda al fondo del eje temporal, la 5 al borde del presente.
   - `message_id` estable `dmr-s{sample_id}-sess{i}-t{j}` para que la operación sea idempotente.
   
   Aquí no se gasta ni un token de LLM. Reproduce la condición del paper ("MemGPT … has access to the full conversation history but must access it via paginated search queries to recall memory") sin pagar las ~120 inferencias que costaría replay-ear las 5 sesiones turn por turn.

3. **Cargar las personas en Core Memory**. Persona del agente en el bloque `assistant`, persona del otro en el bloque `human`. El system prompt es el literal del apéndice 6.1.1 ("The following is information about myself … reply with a best guess using the information in core memory and conversation_search").

4. **Lanzar la sesión 6**. La pregunta `self_instruct['B']` entra como `HumanMessage` en un `thread_id` nuevo (`dmr-s{N}-session6`). El agente — que NO tiene en su FIFO ninguno de los turnos previos — debe llamar `conversation_search` para recuperar el hecho relevante de Recall y responder al estilo de su persona.

5. **Puntuar la respuesta**:
   - **ROUGE-L recall** = |LCS(gold, generada)| / |tokens(gold)|. Ignora puntuación y case. Recall (no F1) porque las respuestas del agente suelen ser más verbosas que el gold corto.
   - **LLM-judge** (apéndice 6.1.2). Una sola llamada con `(question, gold, generated)` y few-shot guiando al juez a ser generoso (responder verbosamente con el tópico correcto = CORRECT). El parser extrae el último CORRECT/WRONG; si aparecen ambos o ninguno, el sample queda como **abstención** (no cuenta en accuracy).

**El baseline (control sin memoria)** comparte la pregunta + el juez, pero cambia los pasos 2-3-4: no hay `MemoryStore` ni tools, y el system prompt es el preprompt del apéndice 6.1.1 con `{conversation_summary}` sustituido por `summary_speaker_*` (el resumen lossy del dataset). El agente solo ve el resumen, no la conversación completa — es justo el modo de fallo que el paper quiere mostrar (GPT-4 baseline cae al 32.1 %).

**Aritmética del run completo**: 500 pares × 1 inferencia del agente (más reintentos por tool-call en el modo MemGPT) + 500 inferencias del juez = ~1500 calls al LLM. Con `gpt-4o-mini` el coste está en el rango de unos pocos dólares; con `gpt-4o` o `claude-sonnet`, ~10× más.

### Implementación

#### Decisión clave 1: ingesta directa en Recall, no replay del agente

El plan dice "ejecutar las sesiones 1-5 secuencialmente". Una lectura literal implica `agent.invoke()` en cada turno — 5 sesiones × ~12 turnos × 2 LLM calls medias = ~120 inferencias por par × 500 pares = ~60 000 inferencias antes de evaluar siquiera la pregunta DMR. Económicamente inviable y, además, **innecesario**: lo que el agente necesita en la sesión 6 es que los hechos estén en Recall, no que él mismo los haya generado. La implementación oficial de Letta hace exactamente esto: bulk-ingesta los turnos como episodios.

`populate_recall(store, sample)` recorre las 5 sesiones y llama `persist_message` para cada turno asignando rol `assistant` (turnos del agente) o `user` (turnos del otro), con `message_id` estable derivado de `(sample_id, session_idx, turn_idx)` para hacer la operación idempotente.

#### Decisión clave 2: `time_back` → `occurred_at` para soportar filtros temporales

Aunque las búsquedas DMR no suelen filtrar por tiempo, el `MemoryStore` ya lo soporta y queremos preservar la cronología. `_time_back_to_days` interpreta `time_num`/`time_unit` (e.g. `5 days`, `2 hours`, `1 month`) y acumulamos hacia atrás desde la sesión 5 (anclada en `now`). La sesión más antigua acaba al fondo; la sesión 5 al borde temporal del tiempo presente.

#### Decisión clave 3: alternancia estricta en `previous_dialogs`

El dataset solo guarda `{"text": ...}` para las sesiones 1-4 — los IDs de speaker no aparecen. El dataset MSC asume estricta alternancia desde Speaker 1, así que `_alternating_turns` asigna `Speaker 1, Speaker 2, Speaker 1, ...` y respeta el `id` cuando existe (sesión 5 lo trae explícito).

#### Decisión clave 4: ROUGE-L recall nativo

El paper especifica recall ("to account for the verbosity of the generated agent replies"). En lugar de añadir `rouge_score` a las dependencias, `rouge_l_recall` implementa LCS-DP en O(n·m) tiempo y O(min(n,m)) memoria. Tokeniza con un regex de puntuación + lower-case, equivalente al modo `rouge1`+LCS de la librería estándar para textos cortos.

#### Decisión clave 5: juez con `litellm.completion` directo

El módulo `llm.py` ya tiene `call_primary` pero hardcodea `model=settings.primary_llm_model`. Como queremos elegir el juez por separado (`--judge-model`), el `default_judge` invoca `litellm.completion` directamente con temperatura 0.0. Eso hace al juez configurable independientemente del agente sin tocar `llm.py`.

#### Decisión clave 6: tests sin LLM con stubs duales

`_RecallReadingStub` resuelve la pregunta leyendo Recall directamente (agente perfecto). `_SummaryReadingStub` lee el `CONVERSATION_SUMMARY` del baseline. Combinados con `_stub_judge_correct`, los 31 tests cubren parsing, métricas, ingesta, runner y baseline sin gastar tokens.

#### Archivos

- `src/memgpt/benchmarks/dmr.py` — dataset loader (`load_dataset`, `parse_record`, `download_dataset`), `populate_recall`, builder de Core Memory + agente DMR (`build_dmr_agent`), métricas (`rouge_l_recall`, `parse_judge_verdict`, `default_judge`), runner programático (`run_benchmark`, `run_baseline_benchmark`).
- `scripts/run_dmr.py` — CLI con flags `--dataset`, `--limit`, `--model`, `--judge-model`, `--graphiti`, `--baseline`, `--download`, `--output`, `--sleep-between`, dump incremental.
- `tests/test_dmr.py` — 31 tests (parsing, métricas, ingesta, runner con stubs, baseline).

#### Cómo ejecutar

```bash
# Descargar el dataset (~8.6 MB) la primera vez.
uv run scripts/run_dmr.py --download --dataset datasets/msc_self_instruct.jsonl --limit 0

# Smoke run sobre 5 pares.
uv run scripts/run_dmr.py --dataset datasets/msc_self_instruct.jsonl --limit 5

# Benchmark completo con InMemoryStore.
uv run scripts/run_dmr.py --dataset datasets/msc_self_instruct.jsonl \
    --output runs/dmr.json

# Baseline (sin memoria).
uv run scripts/run_dmr.py --dataset datasets/msc_self_instruct.jsonl \
    --baseline --output runs/dmr_baseline.json

# Contra Graphiti real (requiere docker compose up postgres neo4j).
uv run scripts/run_dmr.py --dataset datasets/msc_self_instruct.jsonl --graphiti
```

---

## 11. Fase 10 — Validación con Document QA ✅ Completada

### Objetivos

Replicar el tercer benchmark del paper (el más pesado) para validar Archival Storage a escala.

### Tareas

1. Descargar embeddings precalculados de los 20M artículos de Wikipedia desde Hugging Face.
2. Cargar todo en Graphiti (puede llevar horas).
3. Subset de 50 preguntas de NaturalQuestions-Open.
4. Ejecutar el agente con el asistente del apéndice 6.1.4.
5. Evaluar con el LLM-judge del apéndice 6.1.5.
6. Comparar con el paper (Figura 5).

### Definición de hecho

- ✅ Loader del dataset NQ-Open / lost-in-the-middle (formato DPR estándar: `question`, `answers`, `ctxs[{title, text, hasanswer}]`) con descarga lazy desde HF Hub vía `huggingface_hub.hf_hub_download` (`--hf-repo`, `--hf-filename` para apuntar a otra fuente). Verificado por `tests/test_document_qa.py` (parsing + `documents` alias + rechazo de question/answers vacíos).
- ✅ Ingesta a Archival **sin gastar LLM**: `populate_archival(store, sample)` recorre los `ctxs` del sample y llama `insert_archival(content="[title] text")`. Modo "lost-in-the-middle" (cada sample tiene su propia archival aislada) por defecto, hook `shared_store=` para el modo "corpus global" (cargar 20M docs una sola vez fuera del runner — la lógica del agente y del judge es idéntica).
- ✅ Agente DOC-QA con system prompt literal del apéndice 6.1.4 (`MemGPT DOC-QA bot`, "the year is 2018", "keep searching if you can't find the answer") y prompt-de-query exigiendo `ANSWER: [YOUR ANSWER], DOCUMENT: [ARCHIVAL MEMORY TEXT]`.
- ✅ Parser de respuestas (`parse_response`): regex laxa para extraer los campos ANSWER/DOCUMENT y un flag `insufficient` para la abstención `INSUFFICIENT INFORMATION`. Tolerante a separadores extra (coma + salto de línea, etc.) que los LLMs introducen sin querer.
- ✅ Métricas duales:
  - **Exact-match laxo** (sanity sin coste): cualquier `gold_answer` aparece como substring case-insensitive del campo ANSWER, ignorando puntuación final. Equivale a la métrica EM de NQ-Open. Corre offline en cada sample, sirve para detectar regresiones del agente sin gastar tokens.
  - **LLM-judge** (apéndice 6.1.5): `litellm.completion` con temperatura 0.0, prompt que exige un único token CORRECT/INCORRECT. El parser usa word-boundary (`\bCORRECT\b`/`\bINCORRECT\b`) y `set(matches)` para distinguir bien — sin esto, "INCORRECT" matchearía CORRECT por substring. Si aparecen ambos términos, abstiene (el sample no cuenta en accuracy).
- ✅ Baseline (control sin archival) con el preprompt + el query template del apéndice 6.1.4 (modo "references"): top-K documentos numerados como `Document [i]` se incrustan en el HumanMessage; agente sin tools de memoria. `--baseline-top-k 10/20/30` reproduce los puntos de la Figura 5.
- ✅ Runner CLI (`scripts/run_document_qa.py`) con InMemoryStore (único backend soportado — ver Decisión clave 7), modo baseline (`--baseline --baseline-top-k`), límite de muestras (`--limit`), modelo override (`--model`, `--judge-model`), descarga lazy (`--download`, `--hf-repo`, `--hf-filename`), dump incremental, retry con backoff exponencial ante 429, y exit code distinto para `accuracy ≥ 0.40` vs fail vs abort.
- ⏳ Confirmación de que el agente pagina correctamente y la accuracy supera al baseline GPT-4 con K=10 quedan pendientes de la ejecución contra LLM real (requiere créditos / decisión del usuario sobre el modelo).

### Cómo funciona el benchmark, paso a paso

**El dataset (NaturalQuestions-Open + lost-in-the-middle)**. Formato JSONL estándar de DPR / Liu et al. 2023a. Cada record es:

- `question`: pregunta de NQ (extraída de queries reales de Google Search).
- `answers`: lista de respuestas gold aceptables (NQ admite múltiples paráfrasis).
- `ctxs`: lista de pasajes recuperados por el retriever sobre el dump de Wikipedia 2018, cada uno con `title`, `text` y `hasanswer` (flag de oráculo: el pasaje contiene la respuesta gold). Típicamente 10-30 pasajes por pregunta — el gold está mezclado con distractores.

El paper original pre-carga embeddings de 20M artículos en pgvector con HNSW; nosotros admitimos ese modo vía `shared_store=` pero por defecto usamos el modo "lost-in-the-middle" (un Archival aislado por sample con sus 10-30 pasajes), que es testable y reproduce el efecto que el paper quiere mostrar: cuando hay distractores el LLM se atasca eligiendo el documento correcto, MemGPT no porque los pagina.

**Por sample, el pipeline hace cinco cosas**:

1. **Aislar archival**. Construye un `MemoryStore` nuevo. Las archivals de samples distintos no deben mezclarse (ese cruce sí es admisible si pasas un `shared_store` con un corpus global pre-cargado).

2. **Volcar los `ctxs` en Archival**. `populate_archival(store, sample)` itera y llama `insert_archival(content="[title] text")`. No gasta LLM.

3. **Construir el agente**. System prompt = literal del apéndice 6.1.4 ("`MemGPT DOC-QA bot`… `Answer the questions as if though the year is 2018`"). Tools heredadas de Fase 4: `archival_memory_search/insert` (más conversation_search, core_memory_*). Heartbeat NATIVE (Fase 5) para que el agente pueda encadenar varios `archival_memory_search` antes de cerrar el turno.

4. **Lanzar la pregunta**. La pregunta entra envuelta en el query template del apéndice 6.1.4 ("`Search your archival memory…`") como `HumanMessage` en un thread fresco. El agente — que NO tiene los docs en su FIFO — debe llamar `archival_memory_search` para recuperar el pasaje relevante y responder con el formato `ANSWER: X, DOCUMENT: Y`.

5. **Puntuar la respuesta**:
   - **Exact-match**: `gold ⊂ ANSWER (lower, sin puntuación)`. 0 tokens de LLM.
   - **LLM-judge** (apéndice 6.1.5): una llamada con `(question, gold_answers, generated)`. Devuelve un único token CORRECT/INCORRECT. Si la respuesta del agente es `INSUFFICIENT INFORMATION` o le falta el campo `DOCUMENT`, el judge devuelve INCORRECT.

**El baseline (control sin archival)** comparte la pregunta + el judge, pero cambia los pasos 2-4: no hay `MemoryStore` ni tools, y el query template del apéndice 6.1.4 ("references") incluye los `Document [i]` numerados como bloque de texto en el `HumanMessage`. El LLM tiene que extraer la respuesta leyendo el prompt entero — justo el modo de fallo "lost in the middle" que el paper documenta (la accuracy cae con K alto). Con `--baseline-top-k 10/20/30` reproduces los puntos de la Figura 5.

**Aritmética del run completo**: 50 preguntas × 1 inferencia del agente (más reintentos por tool-call en el modo MemGPT, ~3-5 calls a archival por sample) + 50 inferencias del juez = ~150-300 calls al LLM. Mucho más barato que DMR; el coste real está en cargar el corpus si vas a modo global.

### Implementación

#### Decisión clave 1: dos modos de uso (lost-in-the-middle vs corpus global)

La lectura literal del plan ("cargar 20M artículos en Graphiti") es inviable para tests automatizados. La implementación admite dos topologías:

- **Default — lost-in-the-middle**: cada sample tiene sus propios `ctxs` (10-30 pasajes), y el runner les construye un Archival aislado. Reproduce la condición *"el documento gold está mezclado con distractores y puede caer fuera del top-K del retriever"* sin necesitar montar pgvector ni 20M embeddings. Es lo que validan los tests.
- **Hook `shared_store=`** del runner: el caller pre-carga un `MemoryStore` con el corpus completo y se lo pasa al runner. Este NO ingiere por sample (asumimos que el caller lo hizo una vez fuera del bucle). Es el modo "20M Wikipedia" del paper; la lógica del agente, judge y exact-match es idéntica.

#### Decisión clave 2: parsing del formato `ANSWER/DOCUMENT` con regex tolerante

Los LLMs no respetan el formato al pie de la letra: a veces ponen un salto de línea entre `ANSWER:` y `DOCUMENT:`, otras veces se saltan la coma. `parse_response` usa dos regex con `re.DOTALL` y separadores laxos (coma o `DOCUMENT:` mismo) que recuperan los campos en la mayoría de variaciones realistas. La regex `INSUFFICIENT INFORMATION` es independiente y tolera capitalización mixta.

#### Decisión clave 3: word-boundary en el parser del judge para distinguir CORRECT de INCORRECT

`INCORRECT` contiene `CORRECT` como substring. Un parser ingenuo (`"CORRECT" in text`) clasificaría todo como correcto. Usamos `\b(CORRECT|INCORRECT)\b` y comparamos `set(matches)` con `{"CORRECT"}` o `{"INCORRECT"}` — si aparecen ambos términos (juez ambiguo) abstenemos en lugar de elegir uno.

#### Decisión clave 4: judge con `litellm.completion` directo

Mismo patrón que en DMR: el módulo `llm.py` hardcodea `model=settings.primary_llm_model` y queremos el judge configurable por separado (`--judge-model`). `default_judge` usa `litellm.completion` con temperatura 0.0 directamente.

#### Decisión clave 5: tests sin LLM con stubs

`_PerfectArchivalStub` busca en archival y emite el formato `ANSWER:.../DOCUMENT:...`, registrando un tool_call de `archival_memory_search` para que `_count_archival_calls` lo cuente. `_InsufficientStub` siempre responde `INSUFFICIENT INFORMATION`. `_stub_judge_correct` evalúa contra cualquier gold answer como substring. Combinados, los 38 tests cubren parsing, métricas, ingesta, runner, baseline y modo `shared_store` sin gastar tokens.

#### Decisión clave 6: exact-match como métrica secundaria

Aunque el paper solo reporta accuracy del judge, calculamos `exact_match` (gold ⊂ ANSWER) en cada sample sin coste extra. Sirve como sanity-check rápido — si exact-match cae a 0 sabemos que el agente no está formateando bien la respuesta antes de gastar tokens del juez. También permite comparar la implementación contra benchmarks NQ-Open clásicos que reportan EM como métrica principal.

#### Decisión clave 7: solo `InMemoryStore`, sin `GraphitiStore`

Inicialmente el runner exponía `--graphiti` para usar `GraphitiStore` como backend. Lo descartamos: `GraphitiStore.insert_archival` invoca `add_episode`, que internamente extrae entidades y relaciones por LLM antes de embeber e insertar. Con 30 documentos por sample (Figura 5, K=30) eso son ~30 llamadas a LLM **solo de ingestión** antes de que el agente empiece a buscar — varios minutos por sample y un coste prohibitivo. Más importante: arquitecturalmente Graphiti es memoria episódica con grafo de entidades, no vector store de documentos Wikipedia, así que la analogía falla aunque el coste fuera asumible. El paper original tampoco usa knowledge graphs aquí — usa pgvector. Para evaluar Graphiti contra el paper están DMR (Fase 9) y MSC.

#### Archivos

- `src/memgpt/benchmarks/document_qa.py` — dataset loader (`load_dataset`, `parse_record`, `download_dataset`), `populate_archival` + `populate_archival_from_corpus`, builder del agente DOC-QA (`build_doc_qa_agent`), métricas (`parse_response`, `exact_match`, `parse_judge_verdict`, `default_judge`), runner programático (`run_benchmark` con hook `shared_store=`, `run_baseline_benchmark` con `top_k=`).
- `scripts/run_document_qa.py` — CLI con flags `--dataset`, `--limit`, `--model`, `--judge-model`, `--baseline`, `--baseline-top-k`, `--download`, `--hf-repo`, `--hf-filename`, `--output`, `--sleep-between`, dump incremental.
- `tests/test_document_qa.py` — 38 tests (parsing del dataset, parser del formato ANSWER/DOCUMENT, métricas, ingesta, baseline, runner con stubs, modo `shared_store`).

#### Cómo ejecutar

```bash
# Descargar el dataset la primera vez (HF repo configurable).
uv run scripts/run_document_qa.py --download --dataset datasets/nq_open.jsonl --limit 0

# Smoke run sobre 5 preguntas.
uv run scripts/run_document_qa.py --dataset datasets/nq_open.jsonl --limit 5

# Benchmark completo (50 preguntas) con InMemoryStore.
uv run scripts/run_document_qa.py --dataset datasets/nq_open.jsonl \
    --output runs/doc_qa.json

# Baseline con K=10 docs (reproduce un punto de Figura 5).
uv run scripts/run_document_qa.py --dataset datasets/nq_open.jsonl \
    --baseline --baseline-top-k 10 --output runs/doc_qa_baseline_k10.json
```

---

## 12. Fase 11 — Extensiones (opcionales)

### 12.1 Multi-agente

- Definir API para que un agente pueda crear subagentes.
- Cada subagente tiene su propio thread, Core Memory y Graphiti namespace.
- Tools de comunicación: `send_message_to_agent(agent_id, message)`.

### 12.2 MCP tools

- Integrar `langchain-mcp` para exponer servidores MCP como tools del agente.
- Probar con un servidor MCP existente (filesystem, GitHub, etc.).

### 12.3 MemFS versionado

- Construir abstracción `VersionedMemFS` sobre `dulwich`:
  - `memfs.create(path, content)` → commit en repo git in-memory.
  - `memfs.read(path)` → lectura del HEAD.
  - `memfs.history(path)` → log de versiones.
  - `memfs.rollback(path, commit_hash)` → restaurar versión.
- Exponer como tools del agente.

### 12.4 Mejoras propuestas en `memGPT-resumen.md` (TODOs del paper)

- Eliminación selectiva en la FIFO: tool `fifo.delete_messages(message_ids)` invocable durante la Memory Pressure Alert.
- Contadores de búsqueda por página + límite por página.
- Prompts dinámicos para evitar terminación prematura del agente.
- Medición previa de inputs grandes + flush proactivo + chunking.

### 12.5 REPL interactivo (`scripts/chat.py`)

Punto de entrada de UX para charlar con el agente turno a turno desde la terminal sin tener que escribir un script Python cada vez. No es un componente del paper — es tooling que cierra el gap entre los tests E2E (que invocan el grafo en código) y un humano queriendo probar el sistema.

**Tareas**:

1. CLI con `argparse`: `--thread` (id de conversación, mismo id = mismo estado vía checkpointer), `--persistent` (PostgresSaver + GraphitiStore vs MemorySaver + sin store), `--model` (override del `primary_llm_model` del `.env`).
2. Bucle `input() → agent.invoke({"messages": [HumanMessage(...)]}, config=cfg) → print(out["messages"][-1].content)`.
3. Comandos meta: `/exit`, `/quit` (también Ctrl-D / Ctrl-C salen limpio), `/state` (dump rápido de `core_memory.to_prompt_text()` + nº de mensajes vivos vía `agent.get_state(cfg)`).
4. Modo persistente cablea `postgres_checkpointer(...)` como context manager + `GraphitiStore` leyendo DSN/Neo4j de `Settings` — reusa `build_persistent_agent(...)` para no duplicar la receta de Fase 7.

**Definición de hecho**:

- `uv run scripts/chat.py` arranca el REPL contra un agente in-memory, mantiene Core Memory + FIFO entre turnos del mismo proceso.
- `uv run scripts/chat.py --persistent --thread foo` sobrevive a reinicios: cerrar el REPL y volver a abrirlo con el mismo `--thread` recupera la conversación.
- `/state` muestra los bloques de Core Memory actuales sin gastar LLM.

### 12.6 Inspector web de la ventana de contexto (`scripts/inspect_web.py`)

Servidor HTTP de stdlib (`http.server.ThreadingHTTPServer`, sin deps extra) que sirve una página en `http://localhost:8000` con tres paneles — **System Prompt**, **Working Context** (Core Memory + recursive summary) y **FIFO Queue** — más una barra de tokens en vivo y un input de chat embebido. Pensado para depurar: ves cómo cada turno modifica los bloques de Core Memory, qué tools se llamaron (con sus args), y cuándo la ocupación de tokens cruza los umbrales del Queue Manager (warn → ámbar, flush → rojo).

**Tareas**:

1. CLI análogo a `chat.py`: `--port`, `--host`, `--thread`, `--model`, `--persistent`. Reutiliza `build_agent` / `build_persistent_agent`.
2. Tres endpoints stdlib:
   - `GET /` → HTML embebido (single-page, sin build step).
   - `GET /api/state` → JSON con `{system_prompt, core_memory, recursive_summary, messages[], tokens{used, window, warning, flush}}`. Calcula `used` con `count_state_tokens` (misma fórmula que `pressure_check`).
   - `POST /api/send` → invoca el agente con un nuevo `HumanMessage` y devuelve el snapshot actualizado.
3. Polling cliente cada 1 s con `setInterval`, pausado mientras el input está enfocado (para no pisar el POST de envío). Esto permite tener `chat.py` y el inspector compartiendo el mismo `--thread --persistent` en dos terminales y ver los turnos del REPL aparecer en la web en tiempo real.
4. Barra de tokens con tres colores según los umbrales del `QueueManagerConfig` (verde < warn, ámbar warn-flush, rojo ≥ flush).

**Definición de hecho**:

- `uv run scripts/inspect_web.py` sirve la página y los endpoints `/api/state` y `/api/send` responden JSON válido.
- Tras enviar un mensaje, los tres paneles se actualizan; los `tool_calls` aparecen desplegados con `name(args)` bajo el `AIMessage` que los emitió.
- Con `--persistent --thread X` compartido con `chat.py --persistent --thread X`, los turnos enviados desde el REPL aparecen en la web en ≤ 1 s.

---

## 13. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Bugs de `langmem` con tool calls bloquean el flush | Alta | Alto | Implementar el summarizer custom desde el principio si los workarounds no son suficientes |
| Graphiti tiene latencia mayor de la esperada en Document QA | Media | Medio | Tener pgvector como fallback; benchmarks intermedios |
| El LLM moderno se "salta" el flag `request_heartbeat` | Media | Bajo | Modo nativo (sin flag) por defecto, legacy solo si es necesario |
| Persistencia se corrompe en escritura concurrente | Baja | Alto | Cola de eventos secuencial + transacciones atómicas |
| El benchmark DMR no llega al 92.5% | Media | Medio | Iterar sobre prompts del agente y del summarizer; revisar si Graphiti está aprovechando el modelo bi-temporal |
| Dataset de Wikipedia es demasiado pesado | Media | Bajo | Hacer subset de 1M artículos para iterar más rápido |

---

## 14. Decisiones pendientes

Estas decisiones se cierran durante la implementación según resultados:

1. **`request_heartbeat` modo nativo o legacy**: probar primero el modo nativo; si el LLM falla en encadenar acciones complejas, activar legacy.
2. **Graphiti backend**: empezar con Neo4j (más maduro); evaluar FalkorDB si la latencia no es buena.
3. **LLM para el summarizer**: empezar con un modelo barato (Haiku 4.5 / GPT-4o-mini); subir a Sonnet 4.6 / GPT-4o si la calidad del resumen recursivo no es suficiente.
4. **Estrategia de chunking** para inputs masivos (libro entero pegado): empezar con rechazo controlado; evolucionar a chunking automático si el caso de uso lo demanda.
5. **Eliminación selectiva en la FIFO**: se implementará en la Fase 11 como extensión (decisión tomada). En Fase 3 se asume eliminación FIFO clásica por antigüedad.

---

## 15. Cronograma orientativo

Asumiendo dedicación a tiempo completo:

| Semana | Fases | Resultado esperado |
|---|---|---|
| 1 | Fase 0 + Fase 1 + Fase 2 | Esqueleto funcional con Core Memory |
| 2 | Fase 3 + Fase 4 | Queue Manager + Recall/Archival con Graphiti |
| 3 | Fase 5 + Fase 6 + Fase 7 | Function chaining + eventos + persistencia |
| 4 | Fase 8 + Fase 9 | Nested KV + DMR validados |
| (5-6) | Fase 10 + Fase 11 | Document QA + extensiones según prioridad |

**Mínimo viable** (core MemGPT del paper sin extensiones): semanas 1-4.
**Producto completo** (con multi-agente, MCP, MemFS): semanas 1-6.

---

## 16. Criterios de éxito globales

El proyecto se considera exitoso si:

1. ✅ Reproducimos los 3 benchmarks del paper con accuracy igual o superior.
2. ✅ El agente mantiene memoria coherente a través de cierres y reaperturas de sesión.
3. ✅ La latencia media por turno (incluyendo búsquedas) está por debajo de 5s.
4. ✅ El sistema soporta al menos 10 agentes concurrentes sin degradación de rendimiento.
5. ✅ Una persona externa puede leer el código y entender la arquitectura en menos de 1 día.
