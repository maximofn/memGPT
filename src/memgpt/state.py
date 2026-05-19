from datetime import datetime
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, NonNegativeInt

from .core_memory import CoreMemory, default_core_memory


class MemGPTState(BaseModel):
    """Estado del agente MemGPT.

    El `recursive_summary` vive en un campo separado, no como slot 0 de
    `messages`, para no mezclar la semántica del reducer `add_messages`
    (conversación append-only) con un slot mutable. El `agent_node` lo
    inyecta como `SystemMessage` al construir el prompt.
    """

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    recursive_summary: str | None = None
    core_memory: CoreMemory = Field(default_factory=default_core_memory)
    step_count: NonNegativeInt = 0
    memory_pressure_alerted: bool = False
    evicted_count: NonNegativeInt = 0
    persisted_message_ids: list[str] = Field(default_factory=list)

    # --- Fase 5: heartbeat + red de seguridad por turno ---
    chained_heartbeats: NonNegativeInt = 0
    """Número de iteraciones tools→agent dentro del turno actual."""

    turn_started_at: datetime | None = None
    """Marca temporal de inicio del turno actual; se compara contra el timeout."""

    recent_tool_call_keys: list[str] = Field(default_factory=list)
    """Buffer FIFO de claves (nombre+args) de tool calls recientes para detectar loops."""

    last_processed_human_id: str | None = None
    """Id del último HumanMessage procesado; cuando llega uno nuevo se resetean los contadores del turno."""
