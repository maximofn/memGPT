from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import get_settings
from .events import EventRegistry
from .heartbeat import (
    LOOP_DETECTION_WARNING_TEMPLATE,
    HeartbeatConfig,
    HeartbeatMode,
    extract_tool_call_keys,
    loop_repetition_count,
)
from .memory_store import MemoryStore
from .queue_manager import (
    MEMORY_PRESSURE_ALERT_TEXT,
    QueueManagerConfig,
    count_state_tokens,
    select_blocks_to_evict,
)
from .memfs_store import MemFSStore
from .memfs_tools import make_memfs_tools
from .recall_archival_tools import make_recall_archival_tools
from .state import MemGPTState
from .summarizer import LLMCallable, regenerate_recursive_summary
from .tools import CORE_MEMORY_TOOLS, get_current_time

DEFAULT_SYSTEM_PROMPT = (
    "You are MemGPT, a helpful assistant with persistent memory. "
    "Use the available tools when they help you answer the user. "
    "Your Core Memory is shown below and is always visible: it contains "
    "your assistant identity and what you know about the human. Update it with the "
    "core_memory_* tools whenever you learn durable facts that should "
    "outlive the current conversation."
)


def _default_tools(
    memory_store: MemoryStore | None = None,
    memfs_store: MemFSStore | None = None,
) -> list:
    base = [get_current_time, *CORE_MEMORY_TOOLS]
    if memory_store is not None:
        base.extend(make_recall_archival_tools(memory_store))
    if memfs_store is not None:
        base.extend(make_memfs_tools(memfs_store))
    return base


def _role_of(m: BaseMessage) -> str:
    if isinstance(m, HumanMessage):
        return "user"
    if isinstance(m, AIMessage):
        return "assistant"
    if isinstance(m, ToolMessage):
        return "tool"
    return "system"


def _persistable_content(m: BaseMessage) -> str:
    """Best-effort string view of a message for Recall Memory.

    Includes tool-call descriptors so an AIMessage with empty content but
    tool calls is still searchable by tool name / args.
    """
    raw = m.content
    text = raw if isinstance(raw, str) else (str(raw) if raw else "")
    if isinstance(m, AIMessage) and m.tool_calls:
        calls = "; ".join(
            f"{tc.get('name')}({tc.get('args', {})})" for tc in m.tool_calls
        )
        suffix = f" [tool_calls: {calls}]"
        text = (text + suffix).strip()
    return text.strip()


def _resolve_model_id(model_id: str) -> str:
    if ":" in model_id:
        return model_id
    return f"anthropic:{model_id}"


def _build_llm(tools: Sequence[BaseTool], model_id: str | None = None) -> Any:
    settings = get_settings()
    resolved = _resolve_model_id(model_id or settings.primary_llm_model)
    # base_url + api_key se pasan solo si están definidos en Settings: así no
    # pisamos los defaults del provider (Anthropic, OpenAI canónico…). Sirven
    # para apuntar a un endpoint OpenAI-compatible local sin contaminar las
    # env vars globales que también usa Graphiti.
    init_kwargs: dict[str, Any] = {}
    if resolved.startswith("openai:"):
        if settings.primary_llm_base_url:
            init_kwargs["base_url"] = settings.primary_llm_base_url
        if settings.primary_llm_api_key:
            init_kwargs["api_key"] = settings.primary_llm_api_key
    llm: BaseChatModel = init_chat_model(resolved, **init_kwargs)
    # parallel_tool_calls=False: MemGPT está diseñado en torno a tool calls
    # secuenciales encadenadas por heartbeat. Si el LLM emite dos tools en
    # paralelo que actualicen Core Memory (p. ej. `core_memory_append` a
    # `human` y a `assistant` a la vez), ambos `Command(update={"core_memory":
    # ...})` chocan en el mismo step y LangGraph rechaza con
    # `InvalidUpdateError`. Algunos providers (Anthropic en versiones viejas
    # de langchain-anthropic) no aceptan el kwarg — silenciamos el TypeError
    # y caemos al binding por defecto, asumiendo que ese provider no haga
    # parallel tool calls por su cuenta.
    try:
        return llm.bind_tools(list(tools), parallel_tool_calls=False)
    except TypeError:
        return llm.bind_tools(list(tools))


def build_agent(
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tools: Sequence[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    llm: Any | None = None,
    model_id: str | None = None,
    queue_config: QueueManagerConfig | None = None,
    summarizer_callable: LLMCallable | None = None,
    token_count_model: str | None = None,
    memory_store: MemoryStore | None = None,
    memfs_store: MemFSStore | None = None,
    heartbeat_config: HeartbeatConfig | None = None,
    event_registry: EventRegistry | None = None,
) -> CompiledStateGraph:
    """Build the MemGPT agent loop with Phase-3 Queue Manager.

    Graph: ``START → pressure_check → agent → (tools | END)``,
    ``tools → pressure_check``. The `pressure_check` node runs before every
    LLM call (the "before_model" hook of the plan), inspects state size
    against `queue_config`, and:

    - Once per episode, when tokens cross `warning_threshold`, injects a
      Memory Pressure Alert SystemMessage so the LLM can consolidate facts
      into Core Memory before the flush.
    - When tokens cross `flush_threshold`, evicts the oldest atomic blocks
      (~`flush_eviction_ratio` of the window), regenerates the recursive
      summary via the custom summariser (Plan B), and resets the alert flag.

    Parameters
    ----------
    queue_config:
        Thresholds + context window. Defaults to 70 % / 100 % / 50 % over a
        200 000-token window — override for tests or smaller models.
    summarizer_callable:
        LLM callable for the summariser. Defaults to litellm
        ``call_summarizer``. Tests inject a stub.
    token_count_model:
        Model id used by ``count_tokens``. Defaults to the primary LLM.
    memory_store:
        Backend for Recall + Archival memory (Phase 4). When provided,
        ``recall_sync`` nodes ingest every new conversation message into
        the store and the agent gets the ``conversation_search``,
        ``archival_memory_insert`` and ``archival_memory_search`` tools.
        ``None`` disables both (Phases 0-3 behaviour, useful for tests
        that focus on the queue manager).
    memfs_store:
        Backend for the MemFS extension (versioned filesystem-style
        memory; Letta extension, not in the paper). When provided, the
        agent gets the 9 ``memfs_*`` tools (``create``, ``read``,
        ``write``, ``list``, ``move``, ``delete``, ``history``,
        ``rollback``, ``grep``) closed over the store. The store does
        not pass through the LangGraph checkpointer — it lives entirely
        outside ``MemGPTState``. ``None`` keeps the agent free of
        MemFS tools.
    heartbeat_config:
        Function-chaining mode + safety net (Phase 5). Defaults to
        ``HeartbeatMode.NATIVE`` with the safety net active. The
        ``turn_init`` node detects new user turns and resets the
        per-turn counters; ``heartbeat_check`` runs after every tool
        execution and decides whether to continue chaining or yield.
    event_registry:
        Registry of automatic events (Phase 6). When provided, a
        ``step_tick`` node is added between ``agent`` and
        ``recall_sync_post``: it increments ``state.step_count`` and
        dispatches every iteration callback whose ``every_n_steps``
        divides the new count. Wall-clock events are scheduled outside
        the graph (via APScheduler) and trigger normal ``invoke`` calls.
        ``None`` keeps the Phase-5 graph unchanged (zero overhead).
    """
    if tools is None:
        tool_list = _default_tools(memory_store=memory_store, memfs_store=memfs_store)
    else:
        tool_list = list(tools)
    bound_llm = llm if llm is not None else _build_llm(tool_list, model_id=model_id)
    qcfg = queue_config or QueueManagerConfig()
    hbcfg = heartbeat_config or HeartbeatConfig()

    def pressure_check(state: MemGPTState) -> dict[str, Any]:
        total = count_state_tokens(
            system_prompt=system_prompt,
            core_memory=state.core_memory,
            recursive_summary=state.recursive_summary,
            messages=state.messages,
            model=token_count_model,
        )

        update: dict[str, Any] = {}

        if total >= qcfg.flush_tokens and state.messages:
            evicted, _kept = select_blocks_to_evict(
                state.messages,
                target_tokens=qcfg.target_eviction_tokens,
                model=token_count_model,
            )
            if evicted:
                core_text = state.core_memory.to_prompt_text()
                new_summary = regenerate_recursive_summary(
                    old_summary=state.recursive_summary,
                    evicted_messages=evicted,
                    core_memory_text=core_text,
                    llm_callable=summarizer_callable,
                )
                evicted_ids = [m.id for m in evicted if getattr(m, "id", None)]
                update["messages"] = [RemoveMessage(id=mid) for mid in evicted_ids]
                update["recursive_summary"] = new_summary
                update["evicted_count"] = state.evicted_count + len(evicted)
                update["memory_pressure_alerted"] = False
                return update

        if total >= qcfg.warning_tokens and not state.memory_pressure_alerted:
            update["messages"] = [SystemMessage(content=MEMORY_PRESSURE_ALERT_TEXT)]
            update["memory_pressure_alerted"] = True
            return update

        return update

    def agent_node(state: MemGPTState) -> dict[str, Any]:
        prompt: list[Any] = [SystemMessage(content=system_prompt)]
        core_text = state.core_memory.to_prompt_text()
        if core_text:
            prompt.append(SystemMessage(content=core_text))
        if state.recursive_summary:
            prompt.append(
                SystemMessage(
                    content=(
                        "Recursive summary of previously evicted messages "
                        "(treat as authoritative context):\n"
                        f"{state.recursive_summary}"
                    )
                )
            )
        prompt.extend(state.messages)
        response = bound_llm.invoke(prompt)
        return {"messages": [response]}

    def recall_sync(state: MemGPTState) -> dict[str, Any]:
        """Persist any unpersisted message into the Recall backend.

        Runs (a) before ``pressure_check`` so messages reach Recall **before**
        a flush could evict them, and (b) right after ``agent_node`` so the
        new ``AIMessage`` is captured even when the turn ends without a tool
        call. Skips ``SystemMessage`` (memory pressure alert, prompt scaffolding).
        """
        if memory_store is None:
            return {}

        already = set(state.persisted_message_ids)
        new_ids: list[str] = []
        now = datetime.now(timezone.utc)

        for m in state.messages:
            mid = getattr(m, "id", None)
            if not mid or mid in already:
                continue
            if isinstance(m, SystemMessage):
                continue
            content = _persistable_content(m)
            if not content:
                continue
            memory_store.persist_message(
                content=content,
                role=_role_of(m),
                occurred_at=now,
                message_id=mid,
            )
            new_ids.append(mid)

        if not new_ids:
            return {}
        return {"persisted_message_ids": list(state.persisted_message_ids) + new_ids}

    def turn_init(state: MemGPTState) -> dict[str, Any]:
        """Detect a new turn and reset per-turn heartbeat counters.

        A new turn begins when a fresh ``HumanMessage`` (id different from
        ``last_processed_human_id``) reaches the agent. We reset
        ``chained_heartbeats``, ``recent_tool_call_keys`` and arm a new
        ``turn_started_at`` so the safety net measures *this* turn.
        """
        last_human = next(
            (m for m in reversed(state.messages) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            return {}
        last_id = getattr(last_human, "id", None) or ""
        if last_id and last_id == state.last_processed_human_id:
            return {}
        return {
            "last_processed_human_id": last_id,
            "chained_heartbeats": 0,
            "turn_started_at": datetime.now(timezone.utc),
            "recent_tool_call_keys": [],
        }

    def heartbeat_check(state: MemGPTState) -> dict[str, Any]:
        """Update chained_heartbeats + loop-detection buffer; inject warning if needed.

        The actual continue/end routing is delegated to ``heartbeat_router``
        which reads the updated state. This split keeps the bookkeeping in
        one place and the routing logic side-effect-free.
        """
        last_ai = next(
            (m for m in reversed(state.messages) if isinstance(m, AIMessage) and m.tool_calls),
            None,
        )

        new_count = state.chained_heartbeats + 1
        update: dict[str, Any] = {"chained_heartbeats": new_count}

        if last_ai is None:
            return update

        keys_added = extract_tool_call_keys(last_ai.tool_calls)
        recent = (state.recent_tool_call_keys + keys_added)[-hbcfg.recent_keys_buffer :]
        update["recent_tool_call_keys"] = recent

        threshold = hbcfg.loop_detection_threshold
        for k in keys_added:
            if loop_repetition_count(recent, k) == threshold:
                tool_name = k.split("::", 1)[0]
                warning = SystemMessage(
                    content=LOOP_DETECTION_WARNING_TEMPLATE.format(
                        tool_name=tool_name, count=threshold
                    )
                )
                update["messages"] = [warning]
                break

        return update

    def heartbeat_router(state: MemGPTState) -> str:
        """Decide whether to continue chaining or yield to the user (END).

        Order of checks:
        1. Hard cap on chained heartbeats per turn.
        2. Wall-clock timeout per turn.
        3. Loop detection: a tool call repeated past the threshold ends the turn
           (one warning was already injected at threshold by heartbeat_check).
        4. Mode-specific:
           - ``NATIVE``: continue (the LLM will decide via tool_calls).
           - ``LEGACY``: continue only if any tool call had ``request_heartbeat=True``
             or its name is in ``auto_continue_tools``; otherwise END.
        """
        if state.chained_heartbeats >= hbcfg.max_chained_heartbeats:
            return "end"

        if state.turn_started_at is not None:
            elapsed = (
                datetime.now(timezone.utc) - state.turn_started_at
            ).total_seconds()
            if elapsed >= hbcfg.turn_timeout_seconds:
                return "end"

        threshold = hbcfg.loop_detection_threshold
        recent = state.recent_tool_call_keys
        if recent:
            last_key = recent[-1]
            if loop_repetition_count(recent, last_key) > threshold:
                return "end"

        if hbcfg.mode == HeartbeatMode.LEGACY:
            last_ai = next(
                (
                    m
                    for m in reversed(state.messages)
                    if isinstance(m, AIMessage) and m.tool_calls
                ),
                None,
            )
            if last_ai is None:
                return "end"
            for tc in last_ai.tool_calls:
                args = tc.get("args") or {}
                if args.get("request_heartbeat") is True:
                    return "continue"
                if tc.get("name") in hbcfg.auto_continue_tools:
                    return "continue"
            return "end"

        return "continue"

    def step_tick(state: MemGPTState) -> dict[str, Any]:
        """Increment ``step_count`` and dispatch iteration events.

        Runs once per LLM call (between ``agent`` and ``recall_sync_post``).
        Iteration callbacks merge their updates with the new ``step_count``;
        if a callback emits messages they go through the regular
        ``add_messages`` reducer and are persisted by ``recall_sync_post``.
        """
        new_count = state.step_count + 1
        update: dict[str, Any] = {"step_count": new_count}
        if event_registry is None:
            return update
        # Build a transient state-shaped object for the dispatcher: we don't
        # want to mutate `state` (it's frozen-ish from LangGraph's view).
        probe = state.model_copy(update={"step_count": new_count})
        callback_update = event_registry.dispatch_iteration(probe)
        for k, v in callback_update.items():
            update[k] = v
        return update

    graph = StateGraph(MemGPTState)
    graph.add_node("turn_init", turn_init)
    graph.add_node("recall_sync_in", recall_sync)
    graph.add_node("recall_sync_post", recall_sync)
    graph.add_node("pressure_check", pressure_check)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tool_list))
    graph.add_node("heartbeat_check", heartbeat_check)
    if event_registry is not None:
        graph.add_node("step_tick", step_tick)

    graph.add_edge(START, "turn_init")
    graph.add_edge("turn_init", "recall_sync_in")
    graph.add_edge("recall_sync_in", "pressure_check")
    graph.add_edge("pressure_check", "agent")
    if event_registry is not None:
        graph.add_edge("agent", "step_tick")
        graph.add_edge("step_tick", "recall_sync_post")
    else:
        graph.add_edge("agent", "recall_sync_post")
    graph.add_conditional_edges(
        "recall_sync_post", tools_condition, {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "heartbeat_check")
    graph.add_conditional_edges(
        "heartbeat_check",
        heartbeat_router,
        {"continue": "recall_sync_in", "end": END},
    )

    return graph.compile(checkpointer=checkpointer or MemorySaver())
