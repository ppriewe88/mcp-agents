import logging
from typing import Any, List, Optional, Sequence, cast

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import HumanMessage
from langchain_core.tools.structured import StructuredTool
from langgraph.graph.state import CompiledStateGraph, StateT

from agents.factory.registry import AGENT_REGISTRY, AgentName
from agents.llm.client import model
from agents.mcp_adaption.container import MCPToolContainer
from agents.middleware.middleware import (
    AbortOnToolErrors,
    LoggingMiddlewareSync,
    ModelCallCounterMiddlewareSync,
    OnlyOneModelCallMiddlewareSync,
    configured_validator_async,
    global_toolcall_limit_sync,
    override_final_agentprompt_async,
)
from agents.models.agents import (
    AgentConfig,
    AgentRegistryEntry,
    PromptMarkers,
)
from typing import AsyncGenerator, Any
from langchain.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    ToolMessage,
    HumanMessage,
)
from agents.models.extended_state import CustomStateShared
from agents.models.tools import ToolSchema
from agents.factory.utils import artificial_stream

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
load_dotenv()


class ConfiguredAgent:
    """Provides a high-level wrapper around a fully configured agent.

    The class bundles an agent instance with its configuration.
    This includes system prompts and middleware hooks that shape or restrict behavior inside the underlying ReAct loop.
    It also exposes a unified interface to run the agent with ticket-like inputs.

    Args:
        langchain_agent: CompiledStateGraph - the compiled ReAct agent graph responsible for execution.
        config: AgentConfig - contains system prompt, middleware, and behavioral constraints.
        name: str - optional agent name for identification.
        description: str - optional extended description of the agent's purpose.

    Returns:
        ConfiguredAgent: an initialized wrapper that exposes a uniform run-interface over the compiled agent.
    """

    def __init__(
        self,
        langchain_agent: CompiledStateGraph,
        config: AgentConfig,
        name: Optional[str] = None,
        description: str = "",
    ):
        self.agent: CompiledStateGraph[StateT] = langchain_agent # type: ignore[valid-type]
        self.config: AgentConfig = config
        self.name: str = name or ""
        self.description = description

    async def run(
        self, 
        query: str
    ) -> str | dict[str, Any]:
        """Executes the configured agent using a message."""
        extended_state = CustomStateShared(
            messages=[HumanMessage(query)],
            query=query,
            agent_name=self.name,
            toolcall_error=False,
            error_toolname=None,
            model_call_count=0,
            model_call_limit_reached=False,
            final_agentprompt_switched=False,
            final_agentprompt_used=PromptMarkers.INITIAL.value,
            agent_output_aborted=False,
            agent_output_abortion_reason=None,
            agent_output_description=None,
            validated_agent_output=None
        )

        extended_state["messages"] = [HumanMessage(query)]

        result = await self.agent.ainvoke(extended_state)

        return result

    async def astream(
        self, 
        query: str
    ) -> AsyncGenerator[bytes, None]:
        """Executes the configured agent using a message."""
        extended_state = CustomStateShared(
            messages=[HumanMessage(query)],
            query=query,
            agent_name=self.name,
            toolcall_error=False,
            error_toolname=None,
            model_call_count=0,
            model_call_limit_reached=False,
            final_agentprompt_switched=False,
            final_agentprompt_used=PromptMarkers.INITIAL.value,
            agent_output_aborted=False,
            agent_output_abortion_reason=None,
            agent_output_description=None,
            validated_agent_output=None
        )

        extended_state["messages"] = [HumanMessage(query)]

        emitted_toolcall_ids: set[str] = set()
        emitted_final = False

        async for stream_mode, data in self.agent.astream(
            extended_state,
            stream_mode=["messages", "updates"],
        ):
            if stream_mode != "updates":
                continue
            
            # data: dict[source, update_dict]
            for _source, update in data.items():
                if not isinstance(update, dict):
                    continue
            
                # CASE VALIDATOR ABORT
                if update.get("agent_output_aborted") is True:
                    reason = update.get("agent_output_abortion_reason") or "validation rejected"
                    yield f"[ABORTED:{reason}]".encode("utf-8")
                    return

                # CASE NEW MESSAGE
                msgs = update.get("messages")
                if msgs:
                    last: AnyMessage = msgs[-1]

                    # CASE TOOLCALL REQUESTED
                    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                        for tc in last.tool_calls:
                            tc_id = tc.get("id") or f"{tc.get('name')}::{hash(str(tc.get('args')))}"
                            if tc_id in emitted_toolcall_ids:
                                continue
                            emitted_toolcall_ids.add(tc_id)
                            tool_name = tc.get("name", "unknown_tool")
                            marker = f"[CALLING TOOL:{tool_name}....]"
                            async for chunk in artificial_stream(marker, pause=0.04):
                                yield chunk
                            yield b"\n"

                    # CASE TOOLCALL MADE
                    elif isinstance(last, ToolMessage):
                        marker = f"[TOOLCALL DONE: {last.name}....]"
                        async for chunk in artificial_stream(marker, pause=0.04):
                            yield chunk
                        yield b"\n"

                # CASE NOT FINAL ANSWER YET
                validated: Optional[Any] = update.get("validated_agent_output")
                if validated is None or emitted_final:
                    continue

                # CASE FINAL ANSWER
                text: Optional[str] = None
                if isinstance(validated, AIMessage):
                    if isinstance(validated.content, str):
                        text = validated.content
                    else:
                        text = str(validated.content)
                elif isinstance(validated, str):
                    text = validated
                else:
                    text = str(validated)

                if text:
                    emitted_final = True                    
                    async for chunk in artificial_stream(text, pause=0.04):
                        yield chunk
                    return

class AgentFactory:
    """Provides a unified mechanism for constructing fully configured agents from registry definitions.

    The factory orchestrates:
     - lookup,
     - configuration loading,
     - tool instantiation,
     - and the assembly of compiled agent graphs.

    It ensures that each agent is consistently created based on its registered metadata and tools.
    This enables a predictable and reproducible process for building agents across the system.

    Args:
        registry: dict[str, AgentRegistryEntry] | None - optional custom registry; defaults to AGENT_REGISTRY.

    Returns:
        AgentFactory: a factory instance capable of resolving and creating ConfiguredAgent objects.
    """

    def __init__(
        self,
        registry: Optional[dict[str, AgentRegistryEntry]] = None,
    ):
        self.registry = registry or AGENT_REGISTRY
        self.llm = model
    
    async def run_registered_agent(
        self,
        name: AgentName,
        query: str
    ) -> str | dict[str, Any]:
        """Convenience method to load, build, and run a registered agent.

        This method performs the full lifecycle:
        - load registry entry
        - extract data
        - build tools and agent
        - execute the agent with ticket input

        Args:
            name: AgentName - identifier of the registered agent.
            vsnr_mcp: str - VSNR injected into MCP tools.
            subject: str - ticket subject.
            body: str - ticket body.
            division: str - organizational division.
            ticket_id: str - unique ticket identifier.
            debug: bool - return full agent state if True.

        Returns:
            str | dict[str, Any]: agent result or full state if debug is enabled.
        """
        ################# load registry entry
        entry: AgentRegistryEntry = self._load_registered_agent(name)

        ################# build configured agent (extract, inject, build)
        agent: ConfiguredAgent = self._charge_agent(name=name.value, entry=entry)

        ################# run agent
        return await agent.run(query = query)

    def _load_registered_agent(self, name: AgentName) -> AgentRegistryEntry:
        """Loads agent from registry. Raises if not found.

        Args: name: AgentName

        Returns: entry (AgentRegistryEntry)
        """
        if name not in self.registry:
            raise ValueError(f"Agent is not registered: {name}")

        entry: AgentRegistryEntry = self.registry[name]
        return entry

    def _charge_agent(
        self, name: str, entry: AgentRegistryEntry
    ) -> ConfiguredAgent:
        """Constructs a fully configured agent by resolving its definition from the registry.

        The method looks up the specified agent name, loads its configuration and tool schemas,
        and constructs a tool container with MCP context injected.
        It assembles the final agent using
         - the resolved configuration
         - and dynamically built tool instances,
         - enriching it with metadata and descriptions.

        This produces a ready-to-run ConfiguredAgent that reflects the complete registry specification.

        Args:
            name: AgentName - identifier for selecting the agent's registry entry.
            vsnr_mcp: str - VSNR value injected into the tool container for contextualized tool behavior.
            mcp_client: BaseMCPClient | None - optional MCP client used to execute the agent's tools.

        Returns:
            ConfiguredAgent: the constructed agent instance built from registry config and tool definitions.
        """
        config: AgentConfig = entry.config

        ################################################################### charge tools (inject)
        tools: List[StructuredTool] = self._charge_tools(tool_schemas=entry.tool_schemas)

        ################################################################### construct agent (build)
        description = entry.description + f"\nconfig: {config.name}:" + f"{config.description}"
        agent: ConfiguredAgent = self._create_configured_agent(
            config=config,
            tools=tools,
            description=description,
            name=name
        )

        logger.debug(f"[AGENT CREATION] Successfully created agent {name}")
        return agent

    def _charge_tools(
        self,
        tool_schemas: List[ToolSchema],
    ) -> List[StructuredTool]:
        """Creates tools with container class methodology. Injects mandatory and optional data."""
        ############################################################## build tool container
        tool_container = MCPToolContainer(schemas=tool_schemas)
        tools: List[StructuredTool] = list(tool_container.tools_agent.values())
        return tools

    def _create_configured_agent(
        self,
        config: AgentConfig,
        tools: list[Any],
        description: str = "",
        name: str = "",
    ) -> ConfiguredAgent:
        """Builds a ConfiguredAgent from a flat serializable config + factory-wired middleware."""

        ################################################# assemble middleware
        
        #################### basic middleware
        basic_middleware: list[Any] = [
            LoggingMiddlewareSync(),
            ModelCallCounterMiddlewareSync(),
            AbortOnToolErrors()
        ]

        #################### loop control middleware
        loopcontrol_middleware: list[Any] = []

        if config.only_one_model_call:
            loopcontrol_middleware.append(OnlyOneModelCallMiddlewareSync())

        if config.max_toolcalls is not None:
            if config.max_toolcalls < 0:
                raise ValueError("max_toolcalls must be >= 0 or None")
            loopcontrol_middleware.append(global_toolcall_limit_sync(config.max_toolcalls))

        if (
            config.toolbased_answer_prompt is not None
            or config.direct_answer_prompt is not None
        ):
            effective_toolbased_prompt = (
                config.toolbased_answer_prompt
                if config.toolbased_answer_prompt is not None
                else config.system_prompt
            )

            effective_direct_prompt = (
                config.direct_answer_prompt
                if config.direct_answer_prompt is not None
                else config.system_prompt
            )

            loopcontrol_middleware.extend(
                override_final_agentprompt_async(
                    toolbased_answer_prompt=effective_toolbased_prompt,
                    direct_answer_prompt=effective_direct_prompt,
                )
            )

        #################### validation middleware
        validation_middleware: list[Any] = [
            configured_validator_async(
                directanswer_validation_prompt=config.directanswer_validation_sysprompt,
                allow_direct_answers=config.directanswer_allowed,
            )
        ]

        #################### complete middleware
        complete_middleware = (
            basic_middleware
            + loopcontrol_middleware
            + validation_middleware
        )
        ################################################# build langchain agent (core asset)
        agent: CompiledStateGraph = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=config.system_prompt,
            middleware=cast(Sequence[AgentMiddleware[AgentState[Any], None]], complete_middleware),
        )

        return ConfiguredAgent(
            langchain_agent=agent,
            config=config,
            description=description,
            name=name,
        )

if __name__ == "__main__":
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
    import asyncio
    asyncio.run(test_final_integration())