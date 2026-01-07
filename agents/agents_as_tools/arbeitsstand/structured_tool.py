import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from langchain_core.tools.structured import StructuredTool
from langgraph.config import get_stream_writer

# Message types (keep compatibility with your current imports)
from langchain.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage  # type: ignore
from agents.factory.factory import AgentFactory, ConfiguredAgent
from agents.models.agents import AgentConfig, CompleteAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AgentAsToolContainer:
    """..."""

    def __init__(self, agents: List[ConfiguredAgent]) -> None:
        self.tools_for_agent: Dict[str, StructuredTool] = {}
        self.tools_raw: Dict[str, Callable[[str], Awaitable[str]]] = {}

        for agent in agents:
            tool_name = f"run_{agent.name}"

            # 1) build async core
            core = self._build_agent_as_tool(inner_agent=agent, tool_name=tool_name)

            # 2) bind core to instance (so we can call self._make_sync_wrapper etc. if needed later)
            bound_core = core.__get__(self)

            # 3) store raw (async)
            self.tools_raw[tool_name] = core

            # 4) sync wrap for StructuredTool
            sync_wrapper = self._make_sync_wrapper(core)

            # 5) create StructuredTool + store
            agent_as_tool = StructuredTool.from_function(
                name=tool_name,
                description=f"Runs inner agent '{agent.name}' with streaming; forwards inner progress as custom events.",
                func=sync_wrapper,
            )
            self.tools_for_agent[tool_name] = agent_as_tool

    def _make_sync_wrapper(self, async_func: Callable[[str], Awaitable[str]]) -> Callable[[str], str]:
        def wrapper(user_query: str) -> str:
            return asyncio.run(async_func(user_query))
        return wrapper

    def _build_agent_as_tool(
        self,
        inner_agent: Any,
        tool_name: str,
    ) -> Callable[..., Awaitable[str]]:
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

            final_text: Optional[str] = None

            # Build extended state (minimal; same pattern as your plain approach)
            extended_state = inner_agent.initial_state
            extended_state["messages"] = [HumanMessage(user_query)]
            extended_state["query"] = user_query

            async for mode, data in inner_agent.agent.astream(
                extended_state,
                stream_mode=["updates", "messages", "custom"],
            ):
                if mode == "custom":
                    # Relevant if inner agent itself emits custom (e.g., inner-inner agents)
                    writer({"type": "inner_agent", "event": "custom", "tool": tool_name, "data": data})
                    continue

                if mode == "messages":
                    continue

                # mode == "updates"
                if not isinstance(data, dict):
                    continue

                for _source, update in data.items():
                    if not isinstance(update, dict):
                        continue

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

                    # CASE FINAL ANSWER – align with your outer naming: validated_agent_output
                    validated: Optional[Any] = update.get("validated_agent_output")
                    if validated is None or emitted_final:
                        continue

                    if isinstance(validated, AIMessage):
                        text = validated.content if isinstance(validated.content, str) else str(validated.content)
                    elif isinstance(validated, str):
                        text = validated
                    else:
                        text = str(validated)

                    if text:
                        emitted_final = True
                        final_text = text
                        writer({"type": "inner_agent", "event": "final_answer", "tool": tool_name, "text": text})

            writer({"type": "inner_agent", "event": "end", "tool": tool_name})

            return final_text or "(inner agent produced no final text)"

        # keep it readable in stack traces
        run_inner_agent.__name__ = tool_name
        run_inner_agent.__qualname__ = tool_name
        return run_inner_agent
    



if __name__ == "__main__":
    # import asyncio
    from typing import Any, Optional

    from langchain.messages import (
        AIMessage,
        AnyMessage,
        HumanMessage,
        ToolMessage,
    )
    from langchain.tools import tool
    from langgraph.config import get_stream_writer

    from agents.factory.factory import AgentFactory, ConfiguredAgent
    from agents.models.agents import AgentConfig, CompleteAgent
    from tests.schemas import schema_add

    ###################################################### setup inner agent (CONFIGURATION! FROM THIS, ACTUAL AGENT OBJECT WILL BE BUILT!)
    inner_agent_configuration = CompleteAgent(
            description="""inner agent.
            It accesses tools for querying contract data.""",
            config=AgentConfig(
                name="one_shot_tooling_with_retrieval",
                description="""Inner agent. Useful for arithmetic operations like adding numbers.""",
                system_prompt="You are a math agent. you can call tools like 'add' do answer the user query",
                only_one_model_call=True,
                directanswer_validation_sysprompt="direct answer is always usable",
                directanswer_allowed = False
            ),
            tool_schemas=[schema_add],
        )

    ###################################################### get ConfiguredAgent
    factory = AgentFactory()
    inner_agent: ConfiguredAgent = factory._charge_agent(
        name="Test",
        entry=inner_agent_configuration
    )

    agents_as_tools = AgentAsToolContainer(
        agents = [inner_agent]
    )

    outer_agent_configuration = CompleteAgent(
        description="""Outer agent.
        Can call inner agent.""",
        config=AgentConfig(
            name="one_shot_tooling_with_retrieval",
            description="""Inner agent. Useful for arithmetic operations like adding numbers.""",
            system_prompt="You are a math agent. you can call tools like 'add' do answer the user query",
            only_one_model_call=True,
            directanswer_validation_sysprompt="direct answer is always usable",
            directanswer_allowed = False
        ),
        tool_schemas=[],
        agents_as_tools = list(agents_as_tools.tools_for_agent.values())
    )

    outer_agent: ConfiguredAgent = factory._charge_agent(
        name="Test",
        entry=outer_agent_configuration
    )

    async def _test_raw() -> None:
        tool_name = f"run_{inner_agent.name}"
        raw_tool = agents_as_tools.tools_raw[tool_name]

        result = await raw_tool("Add 2 and 3 using the add tool.")
        print("\n[RAW TOOL RESULT]")
        print(result)

    async def _test_complete() -> None:
        query = "Call the inner agent to add 2 and 3. Use its answer."
        async for chunk in outer_agent.outer_astream(query):
            print(chunk.decode("utf-8"), end="", flush=True)

    asyncio.run(_test_complete())