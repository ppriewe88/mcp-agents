import logging
from enum import Enum
from typing import Any, List, Optional

from langchain.agents.middleware import AgentMiddleware, AgentState
from pydantic import BaseModel, ConfigDict, Field

from agents.models.tools import ToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from typing import TypeAlias

MiddlewareT: TypeAlias = AgentMiddleware[AgentState[Any], None]


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: Optional[str] = ""
    system_prompt: str
    middleware_loopcontrol: List[MiddlewareT] = Field(default_factory=list)
    directanswer_validation_sysprompt: str
    directanswer_allowed: bool = True


class AgentRegistryEntry(BaseModel):
    """Type for entries of agent registry."""

    description: str
    config: AgentConfig
    tool_schemas: list[ToolSchema]

class AbortionCodes(str, Enum):
    """Enum of agent response abortion reasons."""

    NO_TOOLMATCH = "NO_MATCHING_TOOL_FOUND"
    TOOL_ERROR = "MCP_TOOL_ERROR"
    DIRECT_ANSWERS_FORBIDDEN = "NO_DIRECT_ANSWERS_ALLOWED"
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
    NO_EXTRACTIONS_FOUND = "no extractions found in text"
    POSTPROCESSING_ERRORMARKER = "TOOL_ERROR"
