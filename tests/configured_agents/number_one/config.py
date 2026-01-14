from agents.models.agents import AgentBehaviourConfig, CompleteAgentConfig
from tests.configured_agents.number_one.prompts_productive import (
    AGENTPROMPT_INITIAL,
)
from tests.schemas import schema_add

###################################################### setup agent
numberone_entry = CompleteAgentConfig(
    description="""First agent.
        It accesses tools for querying contract data.""",
    behaviour_config=AgentBehaviourConfig(
        name="one_shot_tooling_with_retrieval",
        description="""This configuration specifies the following react agent behaviour:
                1. agent logs
                2. agent makes only one model call:
                - agent response is either direct answer, or ToolMessage (results of toolcalls)
                3. postprocess after agentic loop with separate llm task:
                - get retrieval (sabio), and generate answer from toolcalls and retrieval.
                """,
        system_prompt=AGENTPROMPT_INITIAL,
        only_one_model_call=True,
        directanswer_validation_sysprompt=AGENTPROMPT_INITIAL,
    ),
    tool_schemas=[schema_add],
)
