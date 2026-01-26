from enum import Enum
from pydantic import BaseModel
from typing import Any, Optional

class StreamEvent(str, Enum):
    """Type for stream chunk event of inner agent."""

    START = "start"
    TOOL_REQUEST = "toolcall_requested"
    TOOL_RESULT = "toolcall_result"
    FINAL = "final_answer"
    ABORTED = "aborted"

class StreamLevel(str, Enum):
    """Hierarchy level of agentic chunk.""" 
    OUTER = "outer_agent"
    INNER = "inner_agent"

class StreamChunk(BaseModel):
    """Type for stream chunk of inner agent."""

    level: StreamLevel
    event: StreamEvent
    agent_name: str
    
    info: Optional[str] = None
    query: Optional[str] = None
    toolcall_id: Optional[str] = None
    tool_name: Optional[str] = None
    data: Optional[Any] = None
    aborted: Optional[bool] = None
    abortion_reason: Optional[str] = None
    final_answer: Optional[str] = None



    
