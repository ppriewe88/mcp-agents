from enum import Enum

from agents.configured_agents.number_one.config import (
    numberone_entry,
)
from agents.configured_agents.number_two.config import (
    numbertwo_entry,
)
from agents.models.agents import AgentRegistryEntry


class AgentName(str, Enum):
    """Enum of agent names."""

    NUMBER_ONE = "NUMBER_ONE"
    SANTA_EXPERT = "SANTA_EXPERT"


AGENT_REGISTRY: dict[AgentName, AgentRegistryEntry] = {
    AgentName.NUMBER_ONE: numberone_entry,
    AgentName.SANTA_EXPERT: numbertwo_entry
}
