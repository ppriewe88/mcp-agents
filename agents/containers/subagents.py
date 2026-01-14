import asyncio
import logging
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Set, Union

from langchain.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from langchain_core.tools.structured import StructuredTool
from langgraph.config import get_stream_writer
from typing import Optional
from agents.factory.factory import RunnableAgent
from agents.models.stream import InnerStreamEvent, InnerStreamChunk

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AgentAsToolContainer:
    """..."""

    def __init__(self, agents: List[RunnableAgent]) -> None:
        self.subagents: Dict[str, StructuredTool] = {}
        self.subagents_raw: Dict[str, Callable[[str], Awaitable[str]]] = {}

        for agent in agents:
            subagent_name = f"run_{agent.name}"

            # 1) build async core
            core: Callable[[str], Coroutine[Any, Any, str]] = self._build_subagent_as_tool(subagent=agent, subagent_name=subagent_name)

            # 2) store raw (async)
            self.subagents_raw[subagent_name] = core

            # 4) sync wrap for StructuredTool
            sync_wrapper: Callable[[str], str] = self._make_sync_wrapper(core)

            # 5) create StructuredTool + store
            agent_as_tool = StructuredTool.from_function(
                name=subagent_name,
                description=agent.description,
                func=sync_wrapper,
            )
            self.subagents[subagent_name] = agent_as_tool

    def _make_sync_wrapper(
            self, 
            async_func: Callable[[str], Coroutine[Any, Any, str]]
            ) -> Callable[[str], str]:
        def wrapper(user_query: str) -> str:
            return asyncio.run(async_func(user_query))
        return wrapper

    def _build_subagent_as_tool(
        self,
        subagent: Any,
        subagent_name: str,
    ) -> Callable[[str], Coroutine[Any, Any, str]]:
        """Return inner agent as callable async function."""
        async def run_subagent(user_query: str) -> str:
            """
            Runs the inner agent with streaming and forwards inner progress into the
            OUTER agent stream via custom events.
            Returns final inner result as a normal tool return (string).
            """
            writer = get_stream_writer()

            emitted_toolcall_ids: Set[str] = set()
            validated_output: Optional[str] = None

            event = InnerStreamEvent.START.value           
            writer(
                InnerStreamChunk(
                    type="subagent",
                    event = event,
                    subagent= subagent_name,
                    user_query=user_query
                ).model_dump(mode="json")
                )

            extended_state = subagent.initial_state
            extended_state["messages"] = [HumanMessage(user_query)]
            extended_state["query"] = user_query

            async for mode, data in subagent.agent.astream(
                extended_state,
                stream_mode=["updates", "messages", "custom"],
            ):
                ########################################### NESTED SUBAGENTS
                if mode == "custom":
                    event = InnerStreamEvent.UNDEFINED.value
                    writer(
                        InnerStreamChunk(
                            type="nested_agent",
                            event = event,
                            subagent= subagent_name,
                            data=data
                        ).model_dump(mode="json")
                    )
                    continue
                
                ########################################### MESSAGE CHUNKS
                if mode == "messages":
                    continue

                ########################################### UPDATES IN NODES AND MIDDLEWARE
                assert mode == "updates"
                
                ############################### EMPTY UPDATES (middleware returns "None") 
                if not isinstance(data, dict):
                    continue
                
                ############################### DICT UPDATES (middleware updates state) 
                assert isinstance(data, dict)
                for _source, update in data.items():
                    if not isinstance(update, dict):
                        continue

                    ###### UPDATE OF MESSAGES
                    msgs = update.get("messages")
                    if msgs:
                        last: Union[AIMessage, HumanMessage, ToolMessage] = msgs[-1]

                        ####### CASE TOOLCALL REQUESTED
                        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                            for tc in last.tool_calls:
                                tc_id = tc.get("id") or f"{tc.get('name')}::{hash(str(tc.get('args')))}"
                                if tc_id in emitted_toolcall_ids:
                                    continue
                                emitted_toolcall_ids.add(tc_id)

                                event = InnerStreamEvent.TOOL_REQUEST.value
                                writer(
                                    InnerStreamChunk(
                                        type="subagent",
                                        event = event,
                                        subagent= subagent_name,
                                        toolcall_id=tc_id,
                                        tool_name=tc.get("name", "unknown_tool")
                                    ).model_dump(mode="json")
                                )

                        ####### CASE TOOLCALL RESULT
                        elif isinstance(last, ToolMessage):
                            event = InnerStreamEvent.TOOL_RESULT.value
                            writer(
                                InnerStreamChunk(
                                    type="subagent",
                                    event = event,
                                    subagent= subagent_name,
                                    tool_name=last.name
                                ).model_dump(mode="json")
                            )

                    ####### CASE FINAL ANSWER / ABORT (final update made at end)
                    
                    #### UPDATES OF FINAL ATTRIBUTES
                    output_aborted = update.get("agent_output_aborted")
                    validated_output = update.get("validated_agent_output")

                    ## ABORT wins immediately
                    if output_aborted:
                        output_abortion_reason = update.get("agent_output_abortion_reason") or "aborted!"
                        event = InnerStreamEvent.ABORTED.value
                        writer(
                            InnerStreamChunk(
                                type="subagent",
                                event = event,
                                subagent= subagent_name,
                                aborted=True,
                                abortion_reason=output_abortion_reason
                            ).model_dump(mode="json")
                        )
                        return f"[ABORTED: {output_abortion_reason}]"
                    
                    if validated_output is None:
                        continue

                    assert isinstance(validated_output, str) and validated_output
                    event = InnerStreamEvent.FINAL.value
                    writer(
                        InnerStreamChunk(
                            type="subagent",
                            event = event,
                            subagent= subagent_name,
                            final_answer=validated_output
                        ).model_dump(mode="json")
                    )
                    return validated_output

            #### FALLBACK 
            # should not be reached, as loop either aborts or returns validated_output
            return "SUBAGENT DID NOT CONVERGE! LET USER KNOW!"

        # keep it readable in stack traces
        run_subagent.__name__ = subagent_name
        run_subagent.__qualname__ = subagent_name
        return run_subagent
    