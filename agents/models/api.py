from pydantic import BaseModel
from typing import List
from agents.models.agents import AgentConfig
from agents.models.tools import ToolSchema

class GetToolsRequest(BaseModel):
    server_url: str

class StreamAgentRequest(BaseModel):
    message: str
    agent_config: AgentConfig
    tool_schemas: List[ToolSchema]
