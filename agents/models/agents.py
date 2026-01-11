import logging
from enum import Enum
from typing import Any, List, Optional

from langchain.agents.middleware import AgentMiddleware, AgentState
from pydantic import BaseModel, ConfigDict

from agents.models.tools import ToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from typing import TypeAlias

# from langchain_core.tools import BaseTool
MiddlewareT: TypeAlias = AgentMiddleware[AgentState[Any], None]


class AgentBehaviourConfig(BaseModel):
    """Configuration for an agent."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    system_prompt: str
    directanswer_validation_sysprompt: str
    direct_answer_prompt: Optional[str] = None
    toolbased_answer_prompt: Optional[str] = None
    max_toolcalls: Optional[int] = None
    only_one_model_call: bool = False


class CompleteAgentConfig(BaseModel):
    """Type for entries of agent registry."""

    description: str
    behaviour_config: AgentBehaviourConfig
    tool_schemas: list[ToolSchema]
    agents_as_tools: List[Any] = []

class AbortionCodes(str, Enum):
    """Enum of agent response abortion reasons."""

    NO_TOOLMATCH = "NO_MATCHING_TOOL_FOUND"
    TOOL_ERROR = "MCP_TOOL_ERROR"
    DIRECT_ANSWER_UNUSABLE = "DIRECT_AGENT_RESPONSE_UNUSABLE"
    HALLUCINATION = "HALLUCINATION_DETECTED"
    UNKNOWN = "UNKNOWN_ABORTION"


class LoopStatus(str, Enum):
    PENDING = "first model call pending"
    TOOLCALL_REQUEST = "toolcall requested"
    DIRECT_ANSWER = "direct answer (no toolcalls made)"
    TOOLCALL_CONTENTS = "direct return of raw toolcall results"
    TOOL_BASED_ANSWER = "tool based answer"
    ABORTED = "aborted"


class ValidatedAgentResponse(BaseModel):
    response: Optional[str] = None
    valid: bool
    abortion_code: AbortionCodes | None = None
    type: LoopStatus | None = None


class PromptMarkers(str, Enum):
    INITIAL = "initial_prompt"
    DIRECT_ANSWER = "direct_answer_prompt"
    TOOLBASED_ANSWER = "toolbased_answer_prompt"

class MiscMarkers(str, Enum):
    POSTPROCESSING_ERRORMARKER = "TOOL_ERROR"
