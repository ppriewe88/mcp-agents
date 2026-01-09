from datetime import date

from agents import configure_logging
from agents.factory.factory import AgentFactory, RunnableAgent
from agents.mcp_client.client import MCPClient
from tests.configured_agents.number_one.config import numberone_entry
from tests.configured_agents.number_two.config import numbertwo_entry

configure_logging()

client = MCPClient()

today = date.today().strftime("%Y%m%d")

async def test_final_integration() -> None:
    """Test."""
    factory = AgentFactory()

    #################################################################### READCONTRACT

    print("#####################################")

    # NUMBERONE
    factory = AgentFactory()
    agent1: RunnableAgent = factory._charge_runnable_agent(
        name="Test", complete_config=numberone_entry
    )  # type: ignore[annotation-unchecked]
    result = await agent1.run(query="addiere 2 und 5")
    print(result)  # type: ignore[index]

    print("#####################################")

    # NUMBERTWO
    factory = AgentFactory()
    agent2: RunnableAgent = factory._charge_runnable_agent(
        name="Test", complete_config=numbertwo_entry
    )  # type: ignore[annotation-unchecked]
    result = await agent2.run(query="Wann ist der weihnachtsmann geboren?")
    print(result)  # type: ignore[index]

    print("#####################################")

if __name__=="__main__":
    import asyncio
    asyncio.run(test_final_integration())