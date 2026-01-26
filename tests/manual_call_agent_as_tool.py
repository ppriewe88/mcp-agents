import asyncio

from agents.containers.subagents import AgentAsToolContainer
from agents.factory.factory import AgentFactory, RunnableAgent
from agents.models.agents import AgentBehaviourConfig, CompleteAgentConfig
from tests.schemas import schema_add, schema_structured_dict, schema_structured_pydantic
from agents.models.api import ChatMessage

###################################################### setup inner agent (CONFIGURATION! FROM THIS, ACTUAL AGENT OBJECT WILL BE BUILT!)
inner_agent_configuration = CompleteAgentConfig(
        description="""inner agent.
        It accesses tools for querying contract data.""",
        behaviour_config=AgentBehaviourConfig(
            name="one_shot_tooling_with_retrieval",
            description="""Inner agent. Useful for arithmetic operations like adding numbers.""",
            system_prompt="You are a tooling agent. you can call tools to answer the user query",
            only_one_model_call=False,
            directanswer_validation_sysprompt="direct answer is always usable",
            toolbased_answer_prompt="Summarize your toolcall results in a nice and fancy catch phrase!"
        ),
        tool_schemas=[schema_add, schema_structured_dict, schema_structured_pydantic],
    )

###################################################### get ConfiguredAgent
factory = AgentFactory()
inner_agent: RunnableAgent = factory._charge_runnable_agent(
    name="INNER",
    complete_config=inner_agent_configuration
)

agents_as_tools = AgentAsToolContainer(
    agents = [inner_agent]
)

outer_agent_configuration = CompleteAgentConfig(
    description="""Outer agent.
    Can call inner agent.""",
    behaviour_config=AgentBehaviourConfig(
        name="one_shot_tooling_with_retrieval",
        description="""Inner agent. Useful for arithmetic operations like adding numbers.""",
        system_prompt="You are a math agent. you can call tools like 'add' do answer the user query",
        only_one_model_call=False,
        directanswer_validation_sysprompt="direct answer is always usable",
        toolbased_answer_prompt="""Summarize your tooling responses. 
        If you have received infos from a sub agent, cite him and make clear what he  told you!"""
    ),
    tool_schemas=[],
    subagents = list(agents_as_tools.subagents.values())
)

outer_agent: RunnableAgent = factory._charge_runnable_agent(
    name="OUTER",
    complete_config=outer_agent_configuration
)

async def _test_complete() -> None:
    query = "Call the inner agent to retrieve data from structured_dict and structured_pydantic. Use its answer."
    message = ChatMessage(
        id = "1",
        role = "user",
        content = query
    )
    async for chunk in outer_agent.outer_astream([message]):
        print(chunk.decode("utf-8"), end="", flush=True)

asyncio.run(_test_complete())