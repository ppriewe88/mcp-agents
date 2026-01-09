from typing import List

from pydantic import BaseModel

from agents.models.agents import AgentBehaviourConfig
from agents.models.tools import ToolSchema


class GetToolsRequest(BaseModel):
    server_url: str

class StreamAgentRequest(BaseModel):
    message: str
    agent_config: AgentBehaviourConfig
    tool_schemas: List[ToolSchema]
