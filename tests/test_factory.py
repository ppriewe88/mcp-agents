import pytest

import asyncio
from datetime import date

from agents import configure_logging
from agents.factory.factory import AgentFactory, ConfiguredAgent
from agents.mcp_client.client import MCPClient
from tests.configured_agents.number_one.config import numberone_entry
from tests.configured_agents.number_two.config import numbertwo_entry
configure_logging()

client = MCPClient()

today = date.today().strftime("%Y%m%d")

@pytest.mark.asyncio
async def test_final_integration():
    """Test."""
    factory = AgentFactory()

    #################################################################### READCONTRACT

    print("#####################################")

    # NUMBERONE
    factory = AgentFactory()
    agent: ConfiguredAgent = factory._charge_agent(
        name="Test",
        entry=numberone_entry
    )
    result = await agent.run(query="addiere 2 und 5")
    print(result)  # type: ignore[index]

    print("#####################################")

    # NUMBERTWO
    factory = AgentFactory()
    agent: ConfiguredAgent = factory._charge_agent(
        name="Test",
        entry=numberone_entry
    )
    result = await agent.run(query="Wann ist der weihnachtsmanng eboren?")
    print(result)  # type: ignore[index]

    print("#####################################")

    # NUMBERTWO
    result = await agent.run(query="Ich will Santas geheimnisse wissen!")
    print(result)  # type: ignore[index]

    print("#####################################")
    print("FINAL INTEGRATION TEST DONE")

if __name__ == "__main__":
    asyncio.run(test_final_integration())
    print("ENDE")
