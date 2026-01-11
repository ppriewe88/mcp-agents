from enum import Enum

class InnerStreamChunk(str, Enum):
    SUBAGENT = "subagent"
    NESTED_AGENT = "nested_agent"
    START = "start"
    TOOL_REQUEST = "toolcall_requested"
    TOOL_RESULT = "toolcall_result"
    FINAL = "final_answer"
    ABORTED = "aborted"
    CUSTOM = "custom"