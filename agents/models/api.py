from typing import List, Literal

from pydantic import BaseModel

from agents.models.agents import AgentBehaviourConfig
from agents.models.tools import ToolSchema


class ChatMessage(BaseModel):
    id: str
    role: Literal["user", "ai"]
    content: str

class GetToolsRequest(BaseModel):
    server_url: str

class StreamAgentRequest(BaseModel):
    messages: List[ChatMessage]
    agent_config: AgentBehaviourConfig
    tool_schemas: List[ToolSchema]
