import asyncio
from agents.containers.mcp_tools import MCPToolContainer

from tests.schemas import schema_add, schema_structured_pydantic, schema_structured_dict

async def main():
    """Test."""
    
    container = MCPToolContainer(
        schemas=[
            schema_add, schema_structured_dict, schema_structured_pydantic
        ],
    )

    result = await container.tools_raw[schema_add.name_for_llm](
        a = "5",
        b = "7",
    )
    print("\n ############################# RAW RESULT:\n", result)

    result = await container.tools_raw[schema_structured_pydantic.name_for_llm]()
    print("\n ############################# RAW RESULT:\n", result)
    
    result = await container.tools_raw[schema_structured_dict.name_for_llm]()
    print("\n ############################# RAW RESULT:\n", result)
    print("Done")

##########################################################################
asyncio.run(main())
##########################################################################
