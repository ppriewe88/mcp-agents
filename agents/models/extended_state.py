
from langchain.agents.middleware import AgentState


############################################################### Custom State
class CustomStateShared(AgentState):
    """Custom State for agents. Vessel for additional information, that is carried along by the agent.

    Can be read and changed by middleware (that inherits from AgentMiddleware with Customstate).
    """

    query: str
    agent_name: str

    toolcall_error: bool
    error_toolname: str | None
    model_call_count: int
    model_call_limit_reached: bool
    final_agentprompt_switched: bool
    final_agentprompt_used: str
    agent_output_aborted: bool
    agent_output_abortion_reason: str | None
    agent_output_description: str | None
    validated_agent_output: str | None
