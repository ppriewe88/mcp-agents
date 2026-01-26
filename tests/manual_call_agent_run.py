from datetime import date

from agents import configure_logging
from agents.factory.factory import AgentFactory, RunnableAgent
from agents.mcp_client.client import MCPClient
from tests.configured_agents.config import numberone_entry
from agents.models.api import ChatMessage

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
    message = ChatMessage(
        id = "1",
        role = "user",
        content = """rufe das tool "structured_pydantic" auf."""
    )
    result = await agent1.run(messages=[message])
    print(result)  # type: ignore[index]

    print("#####################################")

    message = ChatMessage(
        id = "1",
        role = "user",
        content = """rufe das tool "structured_dict" auf."""
    )
    result = await agent1.run(messages=[message])
    print(result)  # type: ignore[index]

    print("#####################################")

if __name__=="__main__":
    import asyncio
    asyncio.run(test_final_integration())