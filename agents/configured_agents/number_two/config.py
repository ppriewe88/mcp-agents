from agents.configured_agents.number_two.prompts_productive import (
    AGENTPROMPT_INITIAL,
    AGENTPROMPT_TOOLBASED_ANSWER,
)
from agents.mcp_adaption.schemas import schema_birthday, schema_more_info_on_santa
from agents.middleware.middleware import (
    global_toolcall_limit_sync,
    override_final_agentprompt_async,
)
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
            middleware_loopcontrol=[
                global_toolcall_limit_sync(max_toolcalls=5),  # type: ignore[list-item]
                *override_final_agentprompt_async(
                    toolbased_answer_prompt=AGENTPROMPT_TOOLBASED_ANSWER)
            ],
            directanswer_validation_sysprompt=AGENTPROMPT_INITIAL,
            directanswer_allowed = False
        ),
        tool_schemas=[schema_birthday, schema_more_info_on_santa],
    )
