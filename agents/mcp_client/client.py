import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import List, Optional, cast

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent

from agents.mcp_client.abstract import BaseMCPClient
from agents.models.client import (
    MCPError,
    MCPErrorCode,
    MCPTool,
    MCPToolDecision,
    OpenAITool,
    OpenAIToolParameters,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

MCP_SERVER_ENDPOINT: str = os.getenv("MCP_SERVER_ENDPOINT", "http://127.0.0.1:8000/sse")

class MCPClient(BaseMCPClient):
    """MCP client implementation.
    """

    ################################################################ constructor
    def __init__(
        self,
        mcp_server_endpoint: Optional[str] = None,
    ):
        self.mcp_endpoint = mcp_server_endpoint or MCP_SERVER_ENDPOINT
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None

    ################################################################ connection methods

    async def _setup_client(self) -> None:
        """Setup client, initialize connection to mcp server."""
        try:
            # setup client, connect to server (sse stream), initialize session
            logger.info(
                f"[CLIENT] Setting up client for MCP server at {self.mcp_endpoint}"
            )
            client_ctx = sse_client(url=self.mcp_endpoint)
            read_stream, write_stream = await self.exit_stack.enter_async_context(client_ctx)
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self.session.initialize()
            logger.info("[CLIENT] Successfully initialized connection to mcp server.")

        # error handling. No return
        except Exception as error:
            logger.error(
                f"[CLIENT] {MCPErrorCode.CLIENT}. Failure during client setup and initialization of connection."
            )
            logger.error(
                f"[CLIENT] Error during client setup and initialization of connection: {error}"
            )
            self.session = None
            raise MCPError(
                "[CLIENT] Failure during client setup and initialization of connection.",
                code=MCPErrorCode.CLIENT,
            ) from error

    async def connect(self) -> bool:
        """Establishes connection to mcp server, by getting token, setting up client, and initializing."""
        try:
            # check that no session already exists.
            assert self.session is None

            # setup client, connect to server (sse stream), initialize session
            await self._setup_client()

            # log success, return
            logger.info("[CLIENT] MCP client connection successfully established.")
            return True

        # error handling. No return
        except MCPError:
            # raise MCPErrors coming from lower levels
            raise
        except Exception as error:
            logger.error(
                f"[CLIENT] {MCPErrorCode.UNKNOWN}. MCP connection failed due to unknown reason.",
                exc_info=False,
            )
            raise MCPError(
                "MCP connection failed due to unknown reason.",
                code=MCPErrorCode.UNKNOWN,
            ) from error

    async def close(self) -> bool:
        """Closes connection to mcp server."""
        # check session

        if self.session is None:
            print("No open connection")
            return False

        try:
            await self.exit_stack.aclose()
            self.session = None
            logger.info("[CLIENT] Closure of connection to mcp server")
            return True

        # error handling. No raise, no return
        except Exception:
            logger.info("[CLIENT] Enforced closure of connection to mcp server")
            # destroy connection in any possible error case
            self.session = None
            return True

    async def _check_for_reconnect(self) -> None:
        """Check connection. If connection dead, reconnect."""
        try:
            if self.session:
                # if session exists, ping and reconnect, if ping fails
                try:
                    await self.session.send_ping()
                    logger.info("[CLIENT] Ping successfull, session still active")
                except Exception:
                    logger.error(
                        "[CLIENT] Ping failed, try to reconnect.", exc_info=False
                    )  # suppress logging of stack trace
                    self.session = None
                    await self.connect()
            else:
                await self.connect()

        # error handling: no return
        except MCPError:
            # raise MCPErrors coming from lower levels
            raise
        except Exception as error:
            raise MCPError(
                "MCP connection failed due to unknown reason.",
                code=MCPErrorCode.UNKNOWN,
            ) from error

    ################################################################ tooling methods

    async def get_tools(self) -> List[OpenAITool]:
        """Get list of tools from mcp server. Converts them into OpenAI-suitable format."""
        available_tools: List[OpenAITool] = []

        # retrieve tools from server. If empty return, log, and return empty
        try:
            await self.connect()
            assert isinstance(self.session, ClientSession)
            tools_result = await self.session.list_tools()
            if tools_result.tools:
                for tool in tools_result.tools:
                    logger.debug(
                        f"[CLIENT] {self.mcp_endpoint}, RETRIEVED TOOL: {tool.name}"
                    )
                    logger.debug(f"[CLIENT] {tool.inputSchema}")
                    logger.debug(f"[CLIENT] {type(tool.inputSchema)}")
                mcp_tools = cast(List[MCPTool], tools_result.tools)
                available_tools = self._convert_tools_to_openai_format(mcp_tools)
            else:
                logger.error(
                    f"[CLIENT] {MCPErrorCode.LIST_TOOLS}. Server returned empty tool list.",
                    exc_info=False,
                )
                raise MCPError(
                    "Server returned empty tool list.",
                    code=MCPErrorCode.UNKNOWN,
                )

            # log success, return
            logger.debug("[CLIENT] Successfully fetched tools from mcp server")
            await self.close()
            return available_tools

        # error handling. No return
        except MCPError:
            # raise MCPErrors coming from lower levels
            raise
        except Exception as error:
            logger.error(
                f"[CLIENT] {MCPErrorCode.UNKNOWN}. Getting tools from MCP server failed due to unknown reason.",
                exc_info=False,
            )
            raise MCPError(
                "Getting tools from MCP server failed due to unknown reason.",
                code=MCPErrorCode.UNKNOWN,
            ) from error

    def _convert_tools_to_openai_format(self, mcp_tools: List[MCPTool]) -> List[OpenAITool]:
        """Convert MCP-formatted tools to openai format."""
        openai_tools = []
        for tool in mcp_tools:
            try:
                params = OpenAIToolParameters.model_validate(
                    {**tool.inputSchema, "additionalProperties": False}
                )
                openai_tool = OpenAITool.model_validate(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": params,
                        "strict": True,
                    }
                )
                openai_tools.append(openai_tool)

            except Exception as error:
                logger.error(
                    f"[CLIENT] {MCPErrorCode.CONVERSION}. Converting tool to openai format failed. Tool name: {tool.name}"
                )
                raise MCPError(
                    f"Converting tool to openai format failed. Tool name: {tool.name}",
                    code=MCPErrorCode.CONVERSION,
                ) from error

        logger.debug("[CLIENT] Tools successfully retrieved and converted.")
        return openai_tools

    async def call_tools(self, tooling_decision: List[MCPToolDecision]) -> List[CallToolResult]:
        """Call all tools decided in the last tooling decision step."""
        try:
            if self.session is None:
                await self.connect()
            else:
                await self._check_for_reconnect()

            toolcall_results: List[CallToolResult] = []

            for tool_call in tooling_decision:
                tool_call_dumped = tool_call.model_dump()
                tool_name, tool_args, tool_call_id = (
                    tool_call_dumped["name"],
                    tool_call_dumped["args"],
                    tool_call_dumped["id"],
                )
                assert self.session is not None
                tool_response: CallToolResult = await self.session.call_tool(tool_name, tool_args)
                logger.info(f"[CLIENT] Called tool {tool_name}. Error: {tool_response.isError}")

                ##### first check types. Currently, content allows text only
                assert isinstance(tool_response.content[0], TextContent)
                assert tool_response.content[0].type == "text"
                assert tool_response.content[0].text
                assert tool_response.structuredContent is None or isinstance(
                    tool_response.structuredContent, dict
                )

                ##### append
                toolcall_results.append(tool_response)

            # log success, return
            logger.info("[CLIENT] Successfully called tools on mcp server")
            await self.close()

            return toolcall_results

        # error handling. No return
        except MCPError:
            # raise MCPErrors coming from lower levels
            raise
        except Exception as error:
            error_message = (
                f"[CLIENT] {MCPErrorCode.TOOLING} Calling tools failed. "
                f"Tried toolcall: {tool_call_id}, {tool_name} with args {tool_args}. "
            )
            if isinstance(error, McpError):
                error_message += f"[CLIENT] Internal reason (server): {error}"
                logger.error(error_message, exc_info=False)
            else:
                logger.error(error_message, exc_info=False)
            raise MCPError(error_message, code=MCPErrorCode.TOOLING) from error


if __name__ == "__main__":

    async def main():
        """Main function to run the client."""
        client = MCPClient()
        tools = await client.get_tools()
        print(f"Retrieved tools: {tools}")

        add_tool_call = MCPToolDecision.model_validate(
            {
                "name": "add",
                "args": {"a": 2, "b": 5},
                "id": "add-1",
            }
        )

        # --- tool 2: say_hello ---
        hello_tool_call = MCPToolDecision.model_validate(
            {
                "name": "say_hello_dict",
                "args": {"name": "Patrick"},
                "id": "hello-1",
            }
        )

        # --- tool 3: say_hello ---
        hello_tool_call_string = MCPToolDecision.model_validate(
            {
                "name": "say_hello_string",
                "args": {"name": "Patrick"},
                "id": "hello-1",
            }
        )

        # --- tool 4:  ---
        shopping = MCPToolDecision.model_validate(
            {
                "name": "shopping_list",
                "args": {"name": "Patrick"},
                "id": "hello-1",
            }
        )
        results = await client.call_tools([add_tool_call, hello_tool_call, hello_tool_call_string, shopping])
        print("\n=== TOOL RESULTS ===\n")
        for result in results:
            print(result)
        await client.close()

    asyncio.run(main())


