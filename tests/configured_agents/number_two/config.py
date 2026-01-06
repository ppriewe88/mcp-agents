from tests.configured_agents.number_two.prompts_productive import (
    AGENTPROMPT_INITIAL,
    AGENTPROMPT_TOOLBASED_ANSWER,
)
from tests.schemas import schema_birthday, schema_more_info_on_santa
from agents.models.agents import AgentConfig, AgentRegistryEntry

###################################################### setup agent
numbertwo_entry = AgentRegistryEntry(
        description="""Second agent.
        It accesses tools for querying contract data.""",
        config=AgentConfig(
            name="blabla",
            description="""blabla
                """,
            system_prompt=AGENTPROMPT_INITIAL,
            max_toolcalls=5,
            toolbased_answer_prompt=AGENTPROMPT_TOOLBASED_ANSWER,
            direct_answer_prompt=AGENTPROMPT_INITIAL,
            directanswer_validation_sysprompt=AGENTPROMPT_INITIAL,
            directanswer_allowed = False
        ),
        tool_schemas=[schema_birthday, schema_more_info_on_santa],
    )
