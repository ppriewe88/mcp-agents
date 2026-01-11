import asyncio

from agents.containers.subagents import AgentAsToolContainer
from agents.factory.factory import AgentFactory, RunnableAgent
from agents.models.agents import AgentBehaviourConfig, CompleteAgentConfig
from tests.schemas import schema_add

###################################################### setup inner agent (CONFIGURATION! FROM THIS, ACTUAL AGENT OBJECT WILL BE BUILT!)
inner_agent_configuration = CompleteAgentConfig(
        description="""inner agent.
        It accesses tools for querying contract data.""",
        behaviour_config=AgentBehaviourConfig(
            name="one_shot_tooling_with_retrieval",
            description="""Inner agent. Useful for arithmetic operations like adding numbers.""",
            system_prompt="You are a math agent. you can call tools like 'add' do answer the user query",
            only_one_model_call=False,
            directanswer_validation_sysprompt="direct answer is always usable",
            toolbased_answer_prompt="Summarize your toolcall results in a nice and fancy catch phrase!"
        ),
        tool_schemas=[schema_add],
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
    query = "Call the inner agent to add 2 and 3. Use its answer."
    async for chunk in outer_agent.outer_astream(query):
        print(chunk.decode("utf-8"), end="", flush=True)

asyncio.run(_test_complete())