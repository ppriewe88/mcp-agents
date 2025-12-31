import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from langchain_core.tools.structured import StructuredTool
from mcp.types import CallToolResult, TextContent

from agents.mcp_client.abstract import BaseMCPClient
from agents.mcp_client.client import MCPClient
from agents.models.agents import MiscMarkers
from agents.models.client import MCPToolDecision
from agents.models.tools import (
    DROP_EMPTY_DEFAULTS_MARKER,
    ToolArg,
    ToolSchema,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MCPToolContainer:
    """Container class that builds executable langchain StructuredTools for langchain agent.

    Builds such tools from given tool schema:
      - builds core function (mcp-caller) with constructed signature according to schema
      - wraps constructed core function in sync process (to secure after-agent debugging with breaking points)
      - stores readymade StructuredTool objects in class state, along with raw tools for manual calls (tests)
    """

    def __init__(
        self,
        schemas: List[ToolSchema],
        mcp_client: Optional[BaseMCPClient] = None,
    ):
        # state for tools and execution
        self.mcp_client = mcp_client or MCPClient()
        self.tools_agent = {}
        self.tools_raw = {}

        # build tools
        for schema in schemas:
            # build tool-specific mcp-caller with signature according to schema
            core = self._build_mcp_executable(schema)

            # bind to instance
            bound_core = core.__get__(self)

            # store in state
            self.tools_raw[schema.name_for_llm] = bound_core

            # build sync wrapper
            sync_wrapper = self._make_sync_wrapper(bound_core)

            # build args schema for llm
            args_schema = schema.get_args_schema_for_llm()

            # build StructuredTool for langchain agent
            tool = StructuredTool.from_function(
                name=schema.name_for_llm,
                description=schema.description_for_llm,
                func=sync_wrapper,
                args_schema=args_schema,
            )

            # store in state
            self.tools_agent[schema.name_for_llm] = tool

    def _make_sync_wrapper(self, async_func: Callable[..., Awaitable[Any]]) -> Callable[..., Any]:
        """Create a synchronous wrapper around an asynchronous MCP tool function.

        This wrapper enables synchronous execution of dynamically generated async MCP executables,
        ensuring they can be used seamlessly in environments (such as LangChain tools) that do not
        support async functions. Internally it runs the async function via asyncio.run().

        Args:
            async_func (Callable[..., Awaitable[Any]]): The asynchronous MCP execution function
                that should be exposed as a synchronous callable.

        Returns:
            Callable[..., Any]: A synchronous wrapper function that executes the async function
                and returns its result.
        """

        def wrapper(*args, **kwargs):
            import asyncio

            return asyncio.run(async_func(*args, **kwargs))  # type: ignore[arg-type]

        return wrapper

    def _build_signature(self, schema: ToolSchema) -> inspect.Signature:
        """Construct the Python signature for the LLM-facing tool function.

        Only arguments that are not drop_and_inject (i.e., visible to the LLM)
        appear in the signature. VSNR remains the only forced injection and does
        not appear here.
        """
        parameters = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]

        # get LLM-visible args = all except drop_and_inject
        llm_args = schema.get_args()

        for arg in llm_args:
            default = inspect._empty if arg.required else arg.default
            parameters.append(
                inspect.Parameter(
                    name=arg.name_for_llm,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=str,
                )
            )

        return inspect.Signature(parameters)

    def _construct_complete_server_args(
        self, schema: ToolSchema, llm_kwargs: Dict[str, str]
    ) -> Dict[str, str]:
        """Construct the full dictionary of server-side arguments required for the MCP call.

        This method maps LLM-provided arguments to their server-side equivalents and injects
        all required drop_and_inject fields using available InjectibleArg instances.
        It guarantees that every server argument marked for injection receives a valid value.

        Args:
            schema (ToolSchema): The schema defining both LLM-facing and server-facing argument metadata.
            llm_args (List[ToolArg]): Active non-dropped argument definitions provided by the LLM.
            llm_kwargs (Dict[str, str]): Actual runtime values passed from the LLM during invocation.

        Returns:
            Dict[str, str]: A dictionary mapping server-side argument names to their final values.
        """
        # get active args (i.e. args that are provided by llm, i.e. that are not injected)
        llm_args: List[ToolArg] = schema.get_args()

        # build dict that maps llm args on server args -> empty vessel first
        constructed_server_args: Dict[str, str] = {}

        #################################### fill in args that are provided by llm
        logger.debug("[TOOL ARG CONSTRUCTION] Insert args provided by llm")
        for arg in llm_args:
            # Case 1: LLM provided value -> always forward to server
            if arg.name_for_llm in llm_kwargs:
                constructed_server_args[arg.name_on_server] = llm_kwargs[arg.name_for_llm]
                continue

            # Case 2: LLM did not provide required value -> error
            if arg.required:
                raise ValueError(
                    f"[MCP EXECUTABLE] Missing required LLM argument '{arg.name_for_llm}'."
                )

            # Case 3.1: LLM did not provide optional arg, and server fills it -> dont include
            if arg.default == DROP_EMPTY_DEFAULTS_MARKER:
                logger.debug(
                    f"[TOOL DEFAULT DROP] Omitting server arg '{arg.name_on_server}' because "
                    f"default is DROP marker ({DROP_EMPTY_DEFAULTS_MARKER!r})."
                )
                continue

            # Case 3.2: LLM did not provide optional arg, and schema fills it -> include
            if arg.default is None:
                raise ValueError(
                    f"[MCP EXECUTABLE] Optional argument '{arg.name_for_llm}' has no default "
                    "despite required=False. Schema inconsistent."
                )
            constructed_server_args[arg.name_on_server] = arg.default

        return constructed_server_args

    def _validate_final_server_args(
        self, schema: ToolSchema, constructed_server_args: Dict[str, Any]
    ) -> None:
        """Verify that the constructed server-side argument dictionary exactly matches the schema requirements.

        This method ensures that the MCP call will not fail due to missing required fields or
        unexpected extras. It compares the final argument dictionary against the expected server
        parameter list defined in the ToolSchema.

        Args:
            schema (ToolSchema): The schema specifying required server-side argument names.
            constructed_server_args (Dict[str, Any]): The final argument mapping produced after injection.

        Returns:
            None: Raises ValueError when arguments are missing or unexpected fields are present.
        """
        all_server_names = {inp.name_on_server for inp in schema.args_schema.properties}
        required_server_names = {
            inp.name_on_server for inp in schema.args_schema.properties if inp.required
        }
        constructed = set(constructed_server_args.keys())

        missing_required = required_server_names - constructed
        extra = constructed - all_server_names

        if missing_required:
            raise ValueError(
                f"[MCP EXECUTABLE ERROR] Required server args not fully satisfied.\n"
                f"Required server args: {required_server_names}\n"
                f"Present server args:  {constructed}\n"
                f"Missing required:     {missing_required}\n"
                f"Unexpected extras:    {extra if extra else 'None'}"
            )

        if extra:
            logger.warning(
                f"[TOOL MAP CHECK] Unexpected extra server args detected: {extra} "
                f"for tool {schema.name_for_llm}."
            )

        logger.info("[TOOL MAP CHECK] All required server arguments successfully provided.")
        return None

    def _build_mcp_executable(
        self, schema: ToolSchema
    ) -> Callable[..., Awaitable[str | CallToolResult]]:
        """Create the asynchronous MCP execution function for a specific tool schema.

        This factory method generates the complete dynamic caller, including signature mapping,
        argument injection, schema validation, MCP request handling, and postprocessing.
        The returned async function becomes the core logic used by agents and sync wrappers.

        Args:
            schema (ToolSchema): The schema describing how the tool should be exposed to LLM and server.

        Returns:
            Callable[..., Awaitable[str]]: An async function implementing the full MCP tool call.
        """
        # use active args (i.e. args that are provided by llm, i.e. that are not injected)
        llm_args = schema.get_args()

        ############################### create signature
        signature: inspect.Signature = self._build_signature(schema=schema)

        ############################### create mcp-executable
        # create mcp-caller with signature built from schema and postprocessor from schema
        async def mcp_executable(*args, **kwargs) -> str | CallToolResult:
            """Core mcp caller with signature built from ToolSchema.

            During construction, signature is built from llm-facing tool argument names.
            Drop_and_inject- inputs are dropped and not included in signature.

            args and kwargs: To be provided by llm!

            Returns: string
            """
            self = args[0]
            mcp_client = self.mcp_client

            ###################### start with info logging

            logger.info(
                f"[TOOL START] {schema.name_for_llm} | "
                f"Signature={mcp_executable.__signature__} | "  # type: ignore[attr-defined]
                f"Server args={schema.get_all_server_arg_names()} | "
                f"llm args={kwargs}"
            )

            ###################### construct server-args dict with injections
            constructed_server_args = self._construct_complete_server_args(
                schema=schema,
                llm_kwargs=kwargs,
            )

            ###################### check, if constructed args comply to server tool signature
            self._validate_final_server_args(
                schema=schema, constructed_server_args=constructed_server_args
            )

            logger.info(f"[TOOL MAP] {schema.name_for_llm} LLM→SERVER = {constructed_server_args}")

            ###################### call mcp tool
            await mcp_client.connect()

            toolcall = MCPToolDecision(
                name=schema.name_on_server,
                args=constructed_server_args,
                id="auto",
            )

            result_list: List[CallToolResult] = await mcp_client.call_tools([toolcall])
            await mcp_client.close()
            tool_result: CallToolResult = result_list[0]
            logger.info(f"[TOOL RESULT] {schema.name_for_llm} received raw MCP response")
            assert isinstance(tool_result.content[0], TextContent)

            if tool_result.isError:
                logger.error(
                    f"[TOOL ERROR] {schema.name_for_llm} MCP toolcall resulted in error: "
                    f"{tool_result.content[0].text}"
                )
                return MiscMarkers.POSTPROCESSING_ERRORMARKER.value

            return tool_result.content[0].text

        ############################### set documentation and signature
        mcp_executable.__name__ = f"core_{schema.name_for_llm}"
        mcp_executable.__qualname__ = mcp_executable.__name__
        mcp_executable.__doc__ = (
            f"Dynamisch generierte Core-Funktion für MCP Tool '{schema.name_for_llm}'.\n"
            f"Server-Toolname: {schema.name_on_server}\n"
            f"Inputs: {[inp.name_for_llm for inp in llm_args]}"
        )
        mcp_executable.__signature__ = signature  # type: ignore[attr-defined]

        ############################### log for transparency
        logger.info(
            f"[BUILD TOOL] Constructed raw tool\n"
            f"Tool: {schema.name_for_llm}\n"
            f"Signature: {signature}\n"
            f"Server args: {schema.get_all_server_arg_names()}\n"
        )

        return mcp_executable


##############################################################################

if __name__ == "__main__":
    # 000301450224 000301918893
    from agents.mcp_adaption.schemas import schema_add
    from agents.mcp_client.client import MCPClient

    client = MCPClient()

    async def debug():
        """Test."""
        ################################################################################# only vsnr injection
        container = MCPToolContainer(
            schemas=[
                schema_add
            ],
            mcp_client=client,
        )

        ############################################################################## SFK
        result = await container.tools_raw[schema_add.name_for_llm](
            a = "5",
            b = "7",
        )
        print("\n ############################# RAW RESULT:\n", result)
        print("Done")

    ##########################################################################
    asyncio.run(debug())
    ##########################################################################

    async def raw_test():
        """Test."""
        ################################################################################# only vsnr injection
        container = MCPToolContainer(
            schemas=[
                schema_add
            ],
            mcp_client=client,
        )

        ############################################################################## SFK
        result = await container.tools_raw[schema_add.name_for_llm](
            a = "5",
            b = "7",
        )
        print("\n ############################# RAW RESULT:\n", result)
        print("Done")

    ##########################################################################
    asyncio.run(raw_test())
    ##########################################################################
