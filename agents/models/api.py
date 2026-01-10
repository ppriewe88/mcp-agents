from typing import List, Literal

from pydantic import BaseModel

from agents.models.agents import AgentBehaviourConfig
from agents.models.tools import ToolSchema
from enum import Enum

class ChatRole(str, Enum):
    user = "user"
    ai = "ai"

class ChatMessage(BaseModel):
    id: str
    role: ChatRole
    content: str

class GetToolsRequest(BaseModel):
    server_url: str

class StreamAgentRequest(BaseModel):
    messages: List[ChatMessage]
    agent_config: AgentBehaviourConfig
    tool_schemas: List[ToolSchema]
