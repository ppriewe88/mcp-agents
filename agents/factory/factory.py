import logging
from typing import Any, AsyncGenerator, List, Optional, Sequence, cast

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools.structured import StructuredTool
from langgraph.graph.state import CompiledStateGraph, StateT

from agents.containers.mcp_tools import MCPToolContainer
from agents.factory.utils import artificial_stream
from agents.llm.client import model
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
    AgentBehaviourConfig,
    CompleteAgentConfig,
    PromptMarkers,
)
from agents.models.extended_state import CustomStateShared
from agents.models.tools import ToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
load_dotenv()


class RunnableAgent:
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
        config: AgentBehaviourConfig,
        name: Optional[str] = None,
        description: str = "",
    ):
        self.agent: CompiledStateGraph[StateT] = langchain_agent  # type: ignore[valid-type]
        self.config: AgentBehaviourConfig = config
        self.name: str = name or ""
        self.description = description
        self.initial_state = CustomStateShared(
            messages=[HumanMessage("EMPTY_PLACEHOLDER")],
            query="EMPTY_PLACEHOLDER",
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
            validated_agent_output=None,
        )

    async def run(self, query: str) -> str | dict[str, Any]:
        """Executes the configured agent using a message."""
        extended_state = self.initial_state
        extended_state["messages"] = [HumanMessage(query)]
        extended_state["query"] = query

        result = await self.agent.ainvoke(extended_state)

        return result

    async def outer_astream(self, query: str) -> AsyncGenerator[bytes, None]:
        """Executes the configured agent using a message."""
        extended_state = self.initial_state
        extended_state["messages"] = [HumanMessage(query)]
        extended_state["query"] = query

        emitted_toolcall_ids: set[str] = set()
        emitted_final = False

        async for stream_mode, data in self.agent.astream(
            extended_state,
            stream_mode=["messages", "updates", "custom"],
        ):
            if stream_mode == "custom":
                # data is dict sent by inner agent's stream writer
                print("[DEBUG][CUSTOM EVENT]:", data)

                yield f"-------------- [INNER CUSTOM]: {data}".encode("utf-8")
                yield b"\n\n"

                continue  # ganz wichtig: nicht in updates-Logik fallen

            if stream_mode == "messages":
                continue

            assert stream_mode == "updates"
            for _source, update in data.items():  # type: ignore[union-attr]
                if not isinstance(update, dict):
                    continue

                # CASE VALIDATOR ABORT
                if update.get("agent_output_aborted") is True:
                    reason = (
                        update.get("agent_output_abortion_reason")
                        or "validation rejected"
                    )
                    yield f"[ABORTED:{reason}]".encode("utf-8")
                    return

                # CASE NEW MESSAGE
                msgs = update.get("messages")
                if msgs:
                    last: AnyMessage = msgs[-1]

                    # CASE TOOLCALL REQUESTED
                    if isinstance(last, AIMessage) and getattr(
                        last, "tool_calls", None
                    ):
                        for tc in last.tool_calls:
                            tc_id = (
                                tc.get("id")
                                or f"{tc.get('name')}::{hash(str(tc.get('args')))}"
                            )
                            if tc_id in emitted_toolcall_ids:
                                continue
                            emitted_toolcall_ids.add(tc_id)
                            tool_name = tc.get("name", "unknown_tool")
                            marker = f"[+++ CALLING TOOL:{tool_name}....]"
                            async for chunk in artificial_stream(marker, pause=0.04):
                                yield chunk
                            yield b"\n\n"

                    # CASE TOOLCALL MADE
                    elif isinstance(last, ToolMessage):
                        marker = f"[+++ TOOLCALL DONE: {last.name}....]"
                        async for chunk in artificial_stream(marker, pause=0.04):
                            yield chunk
                        yield b"\n\n"

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
    """Provides a unified mechanism for constructing fully configured agents.

    The factory orchestrates:
     - configuration loading,
     - tool instantiation,
     - and the assembly of compiled agent graphs.

    It ensures that each agent is consistently created based on its configuration and tools.
    This enables a predictable and reproducible process for building agents across the system.

    Args:
        registry: dict[str, AgentRegistryEntry] | None - optional custom registry; defaults to AGENT_REGISTRY.

    Returns:
        AgentFactory: a factory instance capable of resolving and creating ConfiguredAgent objects.
    """

    def __init__(
        self,
    ):
        self.llm = model

    def _charge_runnable_agent(
        self, name: str, complete_config: CompleteAgentConfig
    ) -> RunnableAgent:
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
        behaviour_conf: AgentBehaviourConfig = complete_config.behaviour_config

        ################################################################### charge tools (inject)
        tools: List[StructuredTool] = self._charge_tools(
            tool_schemas=complete_config.tool_schemas,
            agents_as_tools=complete_config.agents_as_tools,
        )

        ################################################################### construct agent (build)
        agent: RunnableAgent = self._create_runnable_agent(
            behaviour_config=behaviour_conf,
            tools=tools,
            description=complete_config.description,
            name=name,
        )

        logger.debug(f"[AGENT CREATION] Successfully created agent {name}")
        return agent

    def _charge_tools(
        self, tool_schemas: List[ToolSchema], agents_as_tools: List[Any]
    ) -> List[StructuredTool]:
        """Creates tools with container class methodology. Injects mandatory and optional data."""
        ############################################################## build tool container
        tool_container = MCPToolContainer(schemas=tool_schemas)
        mcp_tools: List[StructuredTool] = list(tool_container.tools_agent.values())
        all_tools = mcp_tools + agents_as_tools
        return all_tools

    def _create_runnable_agent(
        self,
        behaviour_config: AgentBehaviourConfig,
        tools: list[Any],
        description: str = "",
        name: str = "",
    ) -> RunnableAgent:
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

        if behaviour_config.only_one_model_call:
            loopcontrol_middleware.append(OnlyOneModelCallMiddlewareSync())

        if behaviour_config.max_toolcalls is not None:
            if behaviour_config.max_toolcalls < 0:
                raise ValueError("max_toolcalls must be >= 0 or None")
            loopcontrol_middleware.append(
                global_toolcall_limit_sync(behaviour_config.max_toolcalls)
            )

        if (
            behaviour_config.toolbased_answer_prompt is not None
            or behaviour_config.direct_answer_prompt is not None
        ):
            effective_toolbased_prompt = (
                behaviour_config.toolbased_answer_prompt
                if behaviour_config.toolbased_answer_prompt is not None
                else behaviour_config.system_prompt
            )

            effective_direct_prompt = (
                behaviour_config.direct_answer_prompt
                if behaviour_config.direct_answer_prompt is not None
                else behaviour_config.system_prompt
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
                directanswer_validation_prompt=behaviour_config.directanswer_validation_sysprompt,
                allow_direct_answers=behaviour_config.directanswer_allowed,
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
            system_prompt=behaviour_config.system_prompt,
            middleware=cast(
                Sequence[AgentMiddleware[AgentState[Any], None]], complete_middleware
            ),
        )

        return RunnableAgent(
            langchain_agent=agent,
            config=behaviour_config,
            description=description,
            name=name,
        )
