import asyncio
from agents.mcp_client.client import MCPClient
from agents.models.client import MCPToolDecision

client = MCPClient()

async def main():
    toolcall = MCPToolDecision.model_validate(
        {
            "name": "structured_pydantic",
            "args": {},
            "id": "whatever",
        }
    )
    result = await client.call_tools([toolcall])
    print(result)
    print("done")

    toolcall = MCPToolDecision.model_validate(
        {
            "name": "structured_dict",
            "args": {},
            "id": "whatever",
        }
    )
    result = await client.call_tools([toolcall])
    print(result)
    print("done")
asyncio.run(main())