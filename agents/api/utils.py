from typing import List

from agents.containers.agents_as_tools import AgentAsToolContainer
from agents.factory.factory import AgentFactory, RunnableAgent
from agents.models.agents import AgentBehaviourConfig, CompleteAgentConfig
from agents.models.api import StreamAgentRequest
from agents.models.tools import ToolSchema
from tests.schemas import schema_add, schema_birthday, schema_shopping_list


def assemble_agent(payload: StreamAgentRequest) -> RunnableAgent:
    """Assemlbes factory agent from frontend payload."""
    agent_config: AgentBehaviourConfig = payload.agent_config
    tool_schemas: List[ToolSchema] = payload.tool_schemas
    agent_reg_entry = CompleteAgentConfig(
        description=agent_config.description,
        behaviour_config=agent_config,
        tool_schemas=tool_schemas
    )
    factory = AgentFactory()
    agent = factory._charge_runnable_agent(
        name="Test",
        complete_config=agent_reg_entry
    )
    return agent


def use_test_agent() -> RunnableAgent:
    ###################################################### setup inner agent (CONFIGURATION! FROM THIS, ACTUAL AGENT OBJECT WILL BE BUILT!)
    inner_agent_configuration = CompleteAgentConfig(
            description="""Inner agent.
            
            CAPABILITIES:
            Has a tool to add numbers.
            Has a tool to give Santas birth year.

            RECEIVES QUERY AS INPUT:
            Feed this inner agent a query that reflects the user's questions.
            You can use one query with all the subquestions of the user. The inner agent is able to use his capabilities to make work on his own subtasks.
            You can also call him multiple times, each time with a dedicated subquery to answer parts of the user question.
            
            USE WHEN:
            Use when user explicitly asks for info from inner agent
            Use when the capabilities might help to answer the user questions.

            RETURNS:
            Returns an answer on the specific query you asked.
            """,
            behaviour_config=AgentBehaviourConfig(
                name="one_shot_tooling_with_retrieval",
                description="""Inner agent. Useful for arithmetic operations like adding numbers.""",
                system_prompt="Use your tools to answer the user query",
                only_one_model_call=False,
                toolbased_answer_prompt="Summarize your toolcall results in a nice and fancy catch phrase!",
                direct_answer_prompt="""If no tools are suitable to help answering the user query:
                - Politely tell the user, that you cannot answer this questions based on your capabilities (tools)."""
            ),
            tool_schemas=[schema_add, schema_birthday],
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
            description="""Outer agent.""",
            system_prompt="""Use your tools to answer the user query:
            - you might call the inner agent.
            - you can use the other tools as well.
            - If no tool mathes, let the user know. Also let him know, if you need inputs that are not given.""",
            only_one_model_call=False,
            toolbased_answer_prompt="""Summarize your tooling responses. 
            If you have received infos from a sub agent, cite him and make clear what he  told you!""",
            direct_answer_prompt="""If no tools are suitable to help answering the user query:
            - Politely tell the user, that you cannot answer this questions based on your capabilities (tools)."""
        ),
        tool_schemas=[schema_shopping_list],
        agents_as_tools = list(agents_as_tools.tools_for_agent.values())
    )

    outer_agent: RunnableAgent = factory._charge_runnable_agent(
        name="OUTER",
        complete_config=outer_agent_configuration
    )
    return outer_agent