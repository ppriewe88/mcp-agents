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

###################################################### setup agent as tool
@tool
async def run_inner_agent(user_query: str) -> str:
    """
    Runs the inner agent with streaming and forwards inner progress into the
    OUTER agent stream via custom events.
    Returns final inner result as a normal tool return (string).
    """
    writer = get_stream_writer()

    emitted_toolcall_ids: set[str] = set()
    emitted_final = False
    
    # Optional: "Start" Marker
    writer({"type": "inner_agent", "event": "start", "query": user_query})

    final_text: str | None = None

    # -------------------------
    # 2) Innerer Agent (hat get_weather)
    # -------------------------
    # Wichtig:
    # - Wir streamen den inneren Agenten hier selbst.
    # - Alles, was "live" nach außen soll, senden wir über writer(...) als custom.
    #
    # stream_mode kann kombiniert werden; dann liefert astream (mode, chunk). :contentReference[oaicite:3]{index=3}
    extended_state = inner_agent.initial_state
    extended_state["messages"] = [HumanMessage(user_query)]
    extended_state["query"] = user_query

    async for mode, data in inner_agent.agent.astream(
        extended_state,
        stream_mode=["updates", "messages", "custom"],
    ):
        if mode == "custom":
            # Relevant, if inner agent has inner agents 
            writer({
                "type": "inner_agent",
                "event": "custom",
                "data": data,
            })

        if mode == "messages": 
            continue

        assert mode == "updates"

        # data: dict[source, update_dict]
        if not isinstance(data, dict):
            continue
        
        for _source, update in data.items():  # type: ignore[union-attr]
            if not isinstance(update, dict):
                continue
            
        # CASE NEW MESSAGE
            msgs = update.get("messages")
            if msgs:
                last: AnyMessage = msgs[-1]

                # CASE TOOLCALL REQUESTED (inner)
                if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                    for tc in last.tool_calls:
                        tc_id = tc.get("id") or f"{tc.get('name')}::{hash(str(tc.get('args')))}"
                        if tc_id in emitted_toolcall_ids:
                            continue
                        emitted_toolcall_ids.add(tc_id)

                        writer({
                            "type": "inner_agent",
                            "event": "toolcall_requested",
                            "toolcall_id": tc_id,
                            "tool_name": tc.get("name", "unknown_tool"),
                            # optional:
                            # "args": tc.get("args"),
                        })

                # CASE TOOLCALL RESULT (inner)
                elif isinstance(last, ToolMessage):
                    writer({
                        "type": "inner_agent",
                        "event": "toolcall_result",
                        "tool_name": last.name,
                        # optional:
                        # "content": last.content,
                    })

            # CASE FINAL ANSWER (inner) – analog zu outer: validated_agent_output
            validated: Optional[Any] = update.get("validated_agent_output")
            if validated is None or emitted_final:
                continue

            text: Optional[str] = None
            if isinstance(validated, AIMessage):
                text = validated.content if isinstance(validated.content, str) else str(validated.content)
            elif isinstance(validated, str):
                text = validated
            else:
                text = str(validated)

            if text:
                emitted_final = True
                final_text = text
                writer({
                    "type": "inner_agent",
                    "event": "final_answer",
                    "text": text,
                })

    writer({"type": "inner_agent", "event": "end"})

    # Tool MUSS normal zurückgeben (kein Generator), damit outer Agent eine ToolMessage bekommt.
    return final_text or "(inner agent produced no final text)"

###################################################### setup outer agent (CONFIGURATION! FROM THIS, ACTUAL AGENT OBJECT WILL BE BUILT!)
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
        agents_as_tools = [run_inner_agent]
    )

outer_agent: ConfiguredAgent = factory._charge_agent(
    name="Test",
    entry=outer_agent_configuration
)


async def main() -> None:
    query = "Call the inner agent to add 2 and 3. Use its answer."
    async for chunk in outer_agent.outer_astream(query):
        print(chunk.decode("utf-8"), end="", flush=True)

if __name__=="__main__":
    import asyncio
    asyncio.run(main())


############### HINWEISE
"""
ALS NÄCHSTES:
- agents_as_tools strukturiert einbauen, siehe approach.py
-- dafür: unbedingt StructuredTool, und dann Funktion, die das zurückgibt (ähnlich wie container)
-- Datenmodell Agent as tool: CompleteAgent mit: toolbeschreibung; args (immer: query als Text)
- streaming: chunks typisieren
- frontend: agent-as-a-tool zusammenstellbar
"""