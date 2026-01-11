from enum import Enum
from pydantic import BaseModel
from typing import Literal, Any, Optional

class InnerStreamEvent(str, Enum):
    """Type for stream chunk event of inner agent."""

    SUBAGENT = "subagent"
    NESTED_AGENT = "nested_agent"
    START = "start"
    TOOL_REQUEST = "toolcall_requested"
    TOOL_RESULT = "toolcall_result"
    FINAL = "final_answer"
    ABORTED = "aborted"
    CUSTOM = "custom"
    UNDEFINED = "not yet defined"

class InnerStreamChunk(BaseModel):
    """Type for stream chunk of inner agent."""

    type: Literal["subagent", "nested_agent"]
    event: InnerStreamEvent
    subagent: str

    query: Optional[str] = None
    data: Optional[Any] = None
    toolcall_id: Optional[str] = None
    tool_name: Optional[str] = None
    aborted: Optional[bool] = None
    abortion_reason: Optional[str] = None
    final_answer: Optional[str] = None




    
