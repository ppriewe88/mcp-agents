from agents.models.api import (
    StreamAgentRequest
    )
from agents.models.agents import AgentConfig, AgentRegistryEntry
from agents.models.tools import ToolSchema
from agents.factory.factory import AgentFactory, ConfiguredAgent
from typing import List

def assemble_agent(payload: StreamAgentRequest) -> ConfiguredAgent:
    """Assemlbes factory agent from frontend payload."""
    agent_config: AgentConfig = payload.agent_config
    tool_schemas: List[ToolSchema] = payload.tool_schemas
    agent_reg_entry = AgentRegistryEntry(
        description=agent_config.description,
        config=agent_config,
        tool_schemas=tool_schemas
    )
    factory = AgentFactory()
    agent = factory._charge_agent(
        name="Test",
        entry=agent_reg_entry
    )
    return agent

