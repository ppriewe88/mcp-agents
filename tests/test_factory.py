import pytest

import asyncio
from datetime import date

from agents import configure_logging
from agents.factory.factory import AgentFactory
from agents.factory.registry import AgentName
from agents.mcp_client.client import MCPClient

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
    result = await factory.run_registered_agent(
        name=AgentName.NUMBER_ONE, query="bitte addiere 2 und 3"
    )
    print(result)  # type: ignore[index]

    print("#####################################")

    # NUMBERTWO
    result = await factory.run_registered_agent(
        name=AgentName.SANTA_EXPERT, query="Wann ist der Weihnachtsmann geboren?"
    )
    print(result)  # type: ignore[index]

    print("#####################################")

    # NUMBERTWO
    result = await factory.run_registered_agent(
        name=AgentName.SANTA_EXPERT, query="Ich will Santas geheimnisse wissen!"
    )
    print(result)  # type: ignore[index]

    print("#####################################")
    print("FINAL INTEGRATION TEST DONE")

if __name__ == "__main__":
    asyncio.run(final_integration_test())
    print("ENDE")
