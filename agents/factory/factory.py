import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, cast

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
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
from agents.models.api import ChatMessage, ChatRole
from agents.models.extended_state import CustomStateShared
from agents.models.stream import StreamChunk, StreamEvent, StreamLevel
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
        behaviour_config: AgentBehaviourConfig,
        name: Optional[str] = None,
        description: str = "",
    ):
        self.agent: CompiledStateGraph[StateT] = langchain_agent  # type: ignore[valid-type]
        self.behaviour_config: AgentBehaviourConfig = behaviour_config
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

    async def run(self, messages: List[ChatMessage]) -> str | dict[str, Any]:
        """Executes the configured agent using a message."""
        extended_state = self.initial_state
        extended_state["messages"] = self._construct_thread(messages)  # type: ignore[typeddict-item]
        extended_state["query"] = messages[-1].content

        result = await self.agent.ainvoke(extended_state)

        return result

    async def outer_astream(
            self, 
            messages: List[ChatMessage]
            ) -> AsyncGenerator[bytes, None]:
        """Executes the configured agent using a message."""
        extended_state = self.initial_state
        extended_state["messages"] = self._construct_thread(messages)  # type: ignore[typeddict-item]
        extended_state["query"] = messages[-1].content

        emitted_toolcall_ids: set[str] = set()

        async for stream_mode, data in self.agent.astream(
            extended_state,
            stream_mode=["messages", "updates", "custom"],
        ):
            ########################################### CUSTOM EVENTS FROM SUBAGENTS (SUBTHREAD)
            if stream_mode == "custom":
        
                async for part in self._handle_subagent_stream(data):
                    yield part
                continue
            
            ########################################### OUTER AGENT MESSAGE CHUNKS (SUPPRESS)
            if stream_mode == "messages":
                continue

            ########################################### OUTER AGENT MESSAGE UPDATES (HIGHEST THREAD)
            assert stream_mode == "updates"
            assert isinstance(data, dict)
            async for b in self._handle_agent_stream(
                    data,
                    emitted_toolcall_ids,
                ):
                    yield b

    def _construct_thread(
            self, 
            messages: List[ChatMessage]
            ) -> List[AIMessage | SystemMessage | HumanMessage]:
        """Construct langchain message list from frontend input."""
        thread: list[SystemMessage | HumanMessage | AIMessage] = [
            SystemMessage(self.behaviour_config.system_prompt)
        ]
        for message in messages:
            match message.role:
                case ChatRole.user:
                    thread.append(HumanMessage(message.content))
                case ChatRole.ai:
                    thread.append(AIMessage(message.content))
                case _:
                    raise ValueError(f"Unsupported role: {message.role}")
        return thread

    async def _handle_subagent_stream(
        self,
        data,
    ) -> AsyncGenerator[bytes, None]:
        """
        Handle 'custom' stream_mode events (inner/subagent stream forwarded to outer).
        Emits using the central chunk emitter.
        """
        chunk: Optional[StreamChunk] = None
        try:
            chunk = StreamChunk.model_validate(data)
        except Exception:
            chunk = StreamChunk(
                level=StreamLevel.OUTER.value,  # type: ignore[arg-type]
                event=StreamEvent.ABORTED.value,  # type: ignore[arg-type]
                agent_name=self.name,
                info="[STREAM] Received custom chunk with invalid model!",
                aborted=True,
                abortion_reason="invalid custom chunk model",
                data=data,
            )

        if chunk is None:
            yield self._emit_chunk_ndjson(chunk)
            return
        
        async for b in self._emit_chunk_ndjson(chunk):
            yield b

    async def _handle_agent_stream(
        self,
        data: Dict,
        emitted_toolcall_ids,
    ):
        for _source, update in data.items():  # type: ignore[union-attr]
            if not isinstance(update, dict):
                continue
            
            chunks = self._extract_agent_chunks(
                update, 
                emitted_toolcall_ids=emitted_toolcall_ids
                )

            for chunk in chunks:

                async for b in self._emit_chunk_ndjson(chunk):
                    yield b

                # IMPORTANT: keep current control flow
                if chunk.event in (StreamEvent.ABORTED.value, StreamEvent.FINAL.value):
                    return
            
    def _extract_agent_chunks(
            self,
            update: dict[str, Any],
            emitted_toolcall_ids:set[str],
    ) -> List[StreamChunk]:
        """
        Translate a single outer-agent update dict into 0..n StreamChunk objects.
        Keeps the same semantics as the current inline logic.
        """
        chunks: List[StreamChunk] = []

         ########### CASE VALIDATOR ABORT
        if update.get("agent_output_aborted") is True:
            reason = update.get("agent_output_abortion_reason") or "validation rejected"
            chunks.append(
                StreamChunk(
                    level=StreamLevel.OUTER.value,  # type: ignore[arg-type]
                    event=StreamEvent.ABORTED.value,  # type: ignore[arg-type]
                    agent_name=self.name,
                    aborted=True,
                    abortion_reason=reason,
                )
            )
            return chunks

        ########### CASE NEW MESSAGE: TOOLCALL REQUEST & TOOLCALL RESULTS
        msgs = update.get("messages")
        if msgs:
            last: AnyMessage = msgs[-1]

            ##### TOOL_REQUEST (possibly multiple)
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                for tc in last.tool_calls:
                    tc_id = tc.get("id") or f"{tc.get('name')}::{hash(str(tc.get('args')))}"
                    if tc_id in emitted_toolcall_ids:
                        continue
                    emitted_toolcall_ids.add(tc_id)

                    chunks.append(
                        StreamChunk(
                            level=StreamLevel.OUTER.value,  # type: ignore[arg-type]
                            event=StreamEvent.TOOL_REQUEST.value,  # type: ignore[arg-type]
                            agent_name=self.name,
                            toolcall_id=tc_id,
                            tool_name=tc.get("name", "unknown_tool"),
                            data=tc.get("args"),  # optional; keep for later
                        )
                    )

            ##### TOOL_RESULT (single)
            elif isinstance(last, ToolMessage):
                chunks.append(
                    StreamChunk(
                        level=StreamLevel.OUTER.value,  # type: ignore[arg-type]
                        event=StreamEvent.TOOL_RESULT.value,  # type: ignore[arg-type]
                        agent_name=self.name,
                        tool_name=last.name,
                        data=last.content,
                    )
                )

        ########### CASE FINAL ANSWER
        validated = update.get("validated_agent_output")
        if validated is not None:
            if isinstance(validated, AIMessage):
                text = validated.content if isinstance(validated.content, str) else str(validated.content)
            else:
                text = str(validated)

            if text:
                chunks.append(
                    StreamChunk(
                        level=StreamLevel.OUTER.value,  # type: ignore[arg-type]
                        event=StreamEvent.FINAL.value,  # type: ignore[arg-type]
                        agent_name=self.name,
                        final_answer=text,
                    )
                )

        return chunks

    async def _emit_chunk_ndjson(self, chunk: StreamChunk) -> AsyncGenerator[bytes, None]:
        """Emit one StreamChunk as NDJSON records."""
    
        if chunk.event == StreamEvent.TOOL_RESULT.value:
            if chunk.level == StreamLevel.OUTER.value:
                yield (json.dumps({"level": chunk.level, "type":"tool_results", "data": chunk.data}, ensure_ascii=False) + "\n").encode("utf-8")
                return
            if chunk.level == StreamLevel.INNER.value:
                yield (json.dumps({"level": chunk.level, "type":"tool_results", "data": chunk.data}, ensure_ascii=False) + "\n").encode("utf-8")
                return

        if chunk.event == StreamEvent.FINAL.value:
            text = chunk.final_answer or ""
            if not text:
                return
            
            if chunk.level == StreamLevel.OUTER.value:
                async for part in artificial_stream(text, pause=0.04):
                    yield (json.dumps({"level": chunk.level, "type":"text_final", "data": part}, ensure_ascii=False) + "\n").encode("utf-8")
                return
            
            if chunk.level == StreamLevel.INNER.value:
                yield (json.dumps({"level": chunk.level, "type":"text_final", "data": text}, ensure_ascii=False) + "\n").encode("utf-8")
                return

        marker: str
        if chunk.event == StreamEvent.START.value:
            marker = f"[{chunk.level}] START: {chunk.agent_name}...."

        elif chunk.event == StreamEvent.TOOL_REQUEST.value:
            tool = chunk.tool_name or "unknown_tool"
            tcid = f" (id={chunk.toolcall_id})" if chunk.toolcall_id else ""
            marker = f"[{chunk.level}] CALLING TOOL: {chunk.agent_name}::{tool}{tcid}...."

        elif chunk.event == StreamEvent.ABORTED.value:
            reason = chunk.abortion_reason or "aborted!"
            marker = f"[{chunk.level}] ABORTED: {chunk.agent_name}: {reason}"

        else:
            raise ValueError("[STREAM] Uncovered event!")

        yield (json.dumps({"level": chunk.level, "type":"text_step", "data": marker}, ensure_ascii=False) + "\n").encode("utf-8")

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
            subagents=complete_config.subagents,
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
        self, tool_schemas: List[ToolSchema], subagents: List[Any]
    ) -> List[StructuredTool]:
        """Creates tools with container class methodology. Injects mandatory and optional data."""
        ############################################################## build tool container
        tool_container = MCPToolContainer(schemas=tool_schemas)
        mcp_tools: List[StructuredTool] = list(tool_container.tools_agent.values())
        all_tools = mcp_tools + subagents
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

            loopcontrol_middleware.extend(
                override_final_agentprompt_async(
                    toolbased_answer_prompt=effective_toolbased_prompt,
                    direct_answer_prompt=behaviour_config.direct_answer_prompt,
                )
            )

        #################### validation middleware
        validation_middleware: list[Any] = [
            configured_validator_async(
                directanswer_validation_prompt=behaviour_config.directanswer_validation_sysprompt or None,
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
            behaviour_config=behaviour_config,
            description=description,
            name=name,
        )
