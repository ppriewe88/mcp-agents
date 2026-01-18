from agents.models.agents import AgentBehaviourConfig, CompleteAgentConfig
from tests.configured_agents.prompts_productive import (
    AGENTPROMPT_INITIAL,
)
from tests.schemas import schema_add, schema_structured_dict, schema_structured_pydantic

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
        direct_answer_prompt=AGENTPROMPT_INITIAL,
        directanswer_validation_sysprompt="""
        Direkte Antworten sind IMMER usable.""",
        only_one_model_call=False,
        max_toolcalls=5,
        toolbased_answer_prompt="""Beantworte die ursprüngliche Nutzerfrage freundlich sachlich.
        
        Dir liegen dafür Ergebnisse aus Toolanfragen vor. Beziehe diese in deine Antwort ein.
        Beachte dabei, dass du Toolergebnisse in menschenlesbarer Form integrierst:
        - Liefern die Tools zusammenhängende Sätze, bediene dich der Information aus diesen Sätzen.
        - Liefern die Toolantworten strukturiere Daten, so gib dem Nutzer einen vollständigen Antwortsatz aus, in menschlicher Sprache:
        -- Zahlen, Währungen, und technische Infor dürfen enthalten sein.
        -- JSON-Strukturen oder technische strukturelle Trennzeichen nicht.
        """
    ),
    tool_schemas=[schema_add, schema_structured_dict, schema_structured_pydantic],
)
