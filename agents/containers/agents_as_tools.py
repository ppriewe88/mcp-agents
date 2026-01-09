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

from agents.factory.factory import RunnableAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AgentAsToolContainer:
    """..."""

    def __init__(self, agents: List[RunnableAgent]) -> None:
        self.tools_for_agent: Dict[str, StructuredTool] = {}
        self.tools_raw: Dict[str, Callable[[str], Awaitable[str]]] = {}

        for agent in agents:
            tool_name = f"run_{agent.name}"

            # 1) build async core
            core: Callable[[str], Coroutine[Any, Any, str]] = self._build_agent_as_tool(inner_agent=agent, tool_name=tool_name)

            # 2) store raw (async)
            self.tools_raw[tool_name] = core

            # 4) sync wrap for StructuredTool
            sync_wrapper: Callable[[str], str] = self._make_sync_wrapper(core)

            # 5) create StructuredTool + store
            agent_as_tool = StructuredTool.from_function(
                name=tool_name,
                description=agent.description,
                func=sync_wrapper,
            )
            self.tools_for_agent[tool_name] = agent_as_tool

    def _make_sync_wrapper(
            self, 
            # async_func: Callable[[str], Awaitable[str]]
            async_func: Callable[[str], Coroutine[Any, Any, str]]
            ) -> Callable[[str], str]:
        def wrapper(user_query: str) -> str:
            return asyncio.run(async_func(user_query))
        return wrapper

    def _build_agent_as_tool(
        self,
        inner_agent: Any,
        tool_name: str,
    ) -> Callable[[str], Coroutine[Any, Any, str]]:
        async def run_inner_agent(user_query: str) -> str:
            """
            Runs the inner agent with streaming and forwards inner progress into the
            OUTER agent stream via custom events.
            Returns final inner result as a normal tool return (string).
            """
            writer = get_stream_writer()

            emitted_toolcall_ids: Set[str] = set()
            emitted_final = False

            writer({"type": "inner_agent", "event": "start", "tool": tool_name, "query": user_query})

            extended_state = inner_agent.initial_state
            extended_state["messages"] = [HumanMessage(user_query)]
            extended_state["query"] = user_query

            async for mode, data in inner_agent.agent.astream(
                extended_state,
                stream_mode=["updates", "messages", "custom"],
            ):
                ########################################### Relevant if inner agent itself emits custom (e.g., inner-inner agents)
                if mode == "custom":
                    writer({"type": "inner_agent", "event": "custom", "tool": tool_name, "data": data})
                    continue
                
                ########################################### no streaming of any textual messages (from model)
                if mode == "messages":
                    continue

                ########################################### now for mode == updates
                
                ############################### subcase 
                # update DOES NOT update state (i.e., data is no dict)
                
                if not isinstance(data, dict):
                    continue
                
                ############################### subcase 
                # update DOES update state (i.e., data is dict)
                assert mode == "updates"
                assert isinstance(data, dict)
                for _source, update in data.items():
                    if not isinstance(update, dict):
                        continue

                    # CASE: update contains NEW MESSAGE (i.e.: update comes from model node or tool node)
                    msgs = update.get("messages")
                    if msgs:
                        last: Union[AIMessage, HumanMessage, ToolMessage] = msgs[-1]

                        # CASE TOOLCALL REQUESTED
                        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                            for tc in last.tool_calls:
                                tc_id = tc.get("id") or f"{tc.get('name')}::{hash(str(tc.get('args')))}"
                                if tc_id in emitted_toolcall_ids:
                                    continue
                                emitted_toolcall_ids.add(tc_id)

                                writer(
                                    {
                                        "type": "inner_agent",
                                        "event": "toolcall_requested",
                                        "tool": tool_name,
                                        "toolcall_id": tc_id,
                                        "tool_name": tc.get("name", "unknown_tool"),
                                    }
                                )

                        # CASE TOOLCALL RESULT
                        elif isinstance(last, ToolMessage):
                            writer(
                                {
                                    "type": "inner_agent",
                                    "event": "toolcall_result",
                                    "tool": tool_name,
                                    "tool_name": last.name,
                                }
                            )

                    # CASE FINAL ANSWER: get validated_agent_output from agent state (i.e. final answer)
                    validated_output = update.get("validated_agent_output")
                    if emitted_final or validated_output is None:
                        continue

                    assert isinstance(validated_output, str) and validated_output

                    emitted_final = True
                    writer({"type": "inner_agent", "event": "final_answer", "tool": tool_name, "text": validated_output})

            return validated_output or "(inner agent produced no final text)"

        # keep it readable in stack traces
        run_inner_agent.__name__ = tool_name
        run_inner_agent.__qualname__ = tool_name
        return run_inner_agent
    