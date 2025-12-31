import pytest

from agents.mcp_client.client import MCPClient
from agents.models.client import MCPToolDecision


@pytest.mark.asyncio
async def test_mcp_client_tools_roundtrip():
    """
    Integration test:
    - connect to MCP server
    - retrieve tools
    - call multiple tools
    - ensure results are returned without errors
    """

    client = MCPClient()

    connected = await client.connect()
    assert connected is True

    tools = await client.get_tools()
    assert tools is not None
    assert len(tools) > 0

    add_tool_call = MCPToolDecision.model_validate(
        {
            "name": "add",
            "args": {"a": 2, "b": 5},
            "id": "add-1",
        }
    )

    summarize_tool_call = MCPToolDecision.model_validate(
        {
            "name": "summarize",
            "args": {"birth_year": "1950"},
            "id": "summarize-1",
        }
    )

    birthday_tool_call = MCPToolDecision.model_validate(
        {
            "name": "get_birthday_santaclaus",
            "args": {"query": "Bitte Santas Geburtstag"},
            "id": "birthday-1",
        }
    )

    shopping_tool_call = MCPToolDecision.model_validate(
        {
            "name": "shopping_list",
            "args": {"name": "Patrick"},
            "id": "shopping-1",
        }
    )

    results = await client.call_tools(
        [
            add_tool_call,
            summarize_tool_call,
            birthday_tool_call,
            shopping_tool_call,
        ]
    )

    assert results is not None
    assert len(results) == 4

    for result in results:
        assert result.isError is False

    closed = await client.close()
    assert closed is True
