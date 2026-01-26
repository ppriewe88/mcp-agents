import logging
from typing import Any, Callable, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallLimitMiddleware,
    after_agent,
    after_model,
    hook_config,
    wrap_model_call,
)
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from agents import configure_logging
from agents.middleware.utils import (
    DetectedStatus,
    detect_loop_status,
)
from agents.middleware.validation import AgentResponseValidator
from agents.models.agents import (
    AbortionCodes,
    LoopStatus,
    PromptMarkers,
    ValidatedAgentResponse,
)
from agents.models.extended_state import CustomStateShared

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


############################################################### Log before and after model calls
class LoggingMiddlewareSync(AgentMiddleware):
    """Middleware for logging (debugging)."""

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Logs before model calls."""
        agent_name = state.get("agent_name")
        assert agent_name is not None
        logger.info(f"[AGENT {agent_name}] Agent call: model node")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Logs after model calls."""
        messages: List[HumanMessage | AIMessage | ToolMessage] = state["messages"]  # type: ignore[assignment]
        last_message = messages[-1]
        detected: DetectedStatus = detect_loop_status(messages)
        agent_name = state.get("agent_name")
        assert agent_name is not None
        logger.info(
            f"[AGENT {agent_name}] Agent (model node) response: {detected.type.value}"
        )
        if detected.abortion_code:
            logger.info(f"[AGENT {agent_name}] Abortion code: {detected.type.value}")
        if detected.type == LoopStatus.TOOLCALL_REQUEST:
            assert hasattr(last_message, "tool_calls")
            for toolcall in last_message.tool_calls:
                logger.info(
                    f"[AGENT {agent_name}] Requested toolcall: {toolcall['name']}"
                )
        return None

############################################################### evaluate toolcalls for error handling
class AbortOnToolErrors(AgentMiddleware):
    """Checks results of toolcalls for error categories. Returns error marker for abortion cases."""

    @hook_config(can_jump_to=["end"])  # can jump to end (but nowhere else!)
    def before_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        """Enforces agent to jump to end, if model call has been made already.

        Args:
            state (AgentState): agent state containing all past messages.
            runtime (Any): Runtime context provided by the agent loop.

        Returns:
            dict[str, Any] | None: A jump directive when the limit is exceeded,
            otherwise None to continue execution.
        """
        messages: List[HumanMessage | AIMessage | ToolMessage] = state["messages"]  # type: ignore[assignment]
        detected = detect_loop_status(messages)
        if detected.type == LoopStatus.ABORTED and detected.abortion_code == AbortionCodes.TOOL_ERROR:
            last_message = state["messages"][-1]
            agent_name = state.get("agent_name")
            assert agent_name is not None
            logger.info(
                f"[AGENT {agent_name}] Postprocessed MCP Tool result of {last_message.name} is unfixable error! Jump to end!"
            )
            return {
                "toolcall_error": True,
                "error_toolname": f"{last_message.name}",
                "jump_to": "end"
                }
        return None

############################################################### Count Modelcalls
class ModelCallCounterMiddlewareSync(AgentMiddleware[CustomStateShared]):
    """Counts model calls."""

    state_schema = CustomStateShared

    def after_model(self, state: CustomStateShared, runtime) -> dict[str, Any] | None:
        """Increments the model call counter after each model invocation.

        This ensures that subsequent model calls can be blocked once the limit is reached.
        The updated counter is merged back into the agent state.

        Args:
            state (CustomStateShared): Shared state holding the current count of model calls.
            runtime (Any): Runtime context supplied by the agent loop.

        Returns:
            dict[str, Any] | None: Updated state containing the incremented model_call_count, or None if unchanged.
        """
        count: int = state.get("model_call_count") or 0
        logger.info(
            f"[AGENT {state['agent_name']}] Current count of model calls (after model node): {count + 1}"
        )
        return {"model_call_count": count + 1}

############################################################### Count Modelcalls
class OnlyOneModelCallMiddlewareSync(AgentMiddleware[CustomStateShared]):
    """Enforces that the agent performs at most one model call during a run.

    The middleware tracks how many model calls occurred and
    forces an immediate jump to the `end` node once the first model call has been made.
    This produces two possible outcomes:
    (1) If the model chooses a tool call, the tool node is executed after the model node,
    resulting in a ToolMessage (either with data or an error).
    (2) If the model decides not to call a tool, the result is only an AIMessage from the model node,
    with no tool execution.
    In both cases, the middleware prevents any further agentic iterations.
    """

    state_schema = CustomStateShared

    @hook_config(can_jump_to=["end"])  # can jump to end (but nowhere else!)
    def before_model(self, state: CustomStateShared, runtime) -> dict[str, Any] | None:
        """Enforces agent to jump to end, if model call has been made already.

        Args:
            state (CustomStateShared): Shared agent state containing the current model_call_count.
            runtime (Any): Runtime context provided by the agent loop.

        Returns:
            dict[str, Any] | None: A jump directive when the limit is exceeded,
            otherwise None to continue execution.
        """
        count: int = state.get("model_call_count") or 0
        agent_name = state.get("agent_name")
        assert agent_name is not None
        logger.info(
            f"[AGENT {agent_name}] Current count of model calls (before model node): {count}"
        )
        if count >= 1:
            return {
                "model_call_limit_reached": True,
                "jump_to": "end"}
        return None

############################################################### limit toolcalls globally
def global_toolcall_limit_sync(max_toolcalls: int):
    """Creates a synchronous middleware that limits how many tool calls an agent may perform in a run.

    The returned middleware enforces a global cap on tool invocations and
    stops further tool execution once the limit is reached.
    Note: 'stopping' means the model gives itself text feedback to stop calling tools.
    This can be overwritten by forceful user inputs on text basis (unlikely though)

    Args:
        max_toolcalls (int): Maximum number of allowed tool calls (over all tools) for the entire agent run.

    Returns:
        ToolCallLimitMiddleware: A configured middleware enforcing the specified tool call limit.
    """
    return ToolCallLimitMiddleware(thread_limit=None, run_limit=max_toolcalls)

############################################################### choose prompt dynamically
def override_final_agentprompt_async(
        toolbased_answer_prompt: str,
        direct_answer_prompt: Optional[str] = None
        ):
    """Creates middleware that replaces the system prompt once the agent reaches its final answering phase.

    The middleware detects when tooling is finished and the model is producing a final answer
    with no further tool calls, then reruns the model (for final answer) using a specified summary prompt.

    Args:
        final_prompt (str): The system prompt to use for the final summarizing model call.

    Returns:
        Callable: An async middleware function that overrides the system prompt during the model's final response phase.
    """

    @wrap_model_call  # type: ignore[arg-type]
    async def change_prompt_for_final_answer_async(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Change prompt when tooling is done."""
        messages: List[HumanMessage | AIMessage | ToolMessage] = request.state["messages"]  # type: ignore[assignment]
        agent_name = request.state.get("agent_name")
        assert agent_name is not None

        ######### generate the next model response (baseline for prompt switch)
        original_response: ModelResponse = await handler(request)  # type: ignore[misc]
        assert isinstance(original_response.result, list)
        assert isinstance(original_response.result[0], AIMessage)

        message = original_response.result[0]

        ######### detect loop status for the candidate trace (state + new AIMessage)
        next_messages: List[HumanMessage | AIMessage | ToolMessage] = [*messages, message]
        next_detected: DetectedStatus = detect_loop_status(next_messages)
        next_type = next_detected.type
        
        ######### Vessels
        remade_response: ModelResponse
        prompt_switched: bool = False

        ### DIRECT ANSWER WITHOUT PRIOR TOOLCALLS
        if next_type == LoopStatus.DIRECT_ANSWER and direct_answer_prompt is not None:
            new_prompt = SystemMessage(content=direct_answer_prompt)
            logger.info(
                f"[AGENT {agent_name}] Agent is answering directly (no prior toolcalls), switched direct_answer prompt: {bool(direct_answer_prompt)}"
            )
            prompt_switched = True
            remade_response = await handler(request.override(system_message=new_prompt)) # type: ignore[misc]
            assert isinstance(remade_response.result, list)
            assert isinstance(remade_response.result[0], AIMessage)
            remade_response.result[0].response_metadata["used_prompt"] = PromptMarkers.DIRECT_ANSWER.value
            return remade_response

        ### FINAL ANSWER WITH PRIOR TOOLCALLS
        if next_type == LoopStatus.TOOL_BASED_ANSWER:
            logger.info(
                f"[AGENT {agent_name}] Agent is answering with prior toolcalls made, switch prompt"
            )
            new_prompt = SystemMessage(content=toolbased_answer_prompt)
            prompt_switched = True
            remade_response = await handler(request.override(system_message=new_prompt))  # type: ignore[misc]
            remade_response.result[0].response_metadata["used_prompt"] = PromptMarkers.TOOLBASED_ANSWER.value
            return remade_response

        ### NO SWITCH (= answer with TOOLCALL REQUESTS or DIRECT ANSWER WITHOUT PROMPT SWITCH)
        original_response.result[0].response_metadata["used_prompt"] = PromptMarkers.INITIAL.value
        logger.info(
                f"[AGENT {agent_name}] Agent is answering. Loop status: {next_type.value}. Prompt switched: {prompt_switched}"
            )
        return original_response


    @after_model(state_schema=CustomStateShared)
    def document_final_prompt(
        state: CustomStateShared, runtime: Runtime
    ) -> dict[str, Any]:
        """If last message was generated (after prompt switch), mark prompt as switched."""
        messages: List[HumanMessage | AIMessage | ToolMessage] = state["messages"]  # type: ignore[assignment]
        last_message = messages[-1]
        last_prompt_used = last_message.response_metadata["used_prompt"]

        detected: DetectedStatus = detect_loop_status(messages)
        status = detected.type

        if status == LoopStatus.DIRECT_ANSWER:
            return {
                "final_agentprompt_switched": last_prompt_used == PromptMarkers.DIRECT_ANSWER.value,
                "final_agentprompt_used": last_prompt_used,
            }

        if status == LoopStatus.TOOL_BASED_ANSWER:
            return {
                "final_agentprompt_switched": True,
                "final_agentprompt_used": last_prompt_used,
            }

        return {
            "final_agentprompt_switched": False,
            "final_agentprompt_used": last_prompt_used,
        }

    return [
        change_prompt_for_final_answer_async,
        document_final_prompt,
    ]

############################################################### validate answer (after agent)
def configured_validator_async(
    directanswer_validation_prompt: Optional[str] = None,
):
    """Creates async post-agent middleware that validates direct agent (without toolcalls) output for usability.

    The middleware filters and type-checks message objects,
    then runs AgentResponseValidator to determine whether a usable agent output exists.

    Args:
        system_prompt_usability (str): System prompt used for the usability validation model call.

    Returns:
        Callable: An async middleware function that stores the validated agent output in the state.
    """

    @after_agent(state_schema=CustomStateShared)
    async def validate_agent_output_async(
        state: CustomStateShared, runtime: Runtime
    ) -> dict[str, Any]:
        """Validates the agent output and stores the validation result in the shared state.

        The middleware ensures message type integrity,
        then computes a validated agent output string or None via AgentResponseValidator.

        Args:
            state (CustomStateShared): Shared agent state containing messages and domain metadata.
            runtime (Runtime): Runtime context provided by the agent execution loop.

        Returns:
            dict[str, Any]: A dictionary containing `validated_agent_output` (str | None) and `validated_messages` (list).
        """
        ####################### check state & message types (for linter)
        assert isinstance(state, dict) and "messages" in state
        messages_raw = state["messages"]
        available_messages = [
            message
            for message in messages_raw
            if isinstance(message, AIMessage)
            or isinstance(message, HumanMessage)
            or isinstance(message, ToolMessage)
            or isinstance(message, SystemMessage)
        ]
        assert len(messages_raw) == len(available_messages)

        ###################### validator instance
        validator = AgentResponseValidator(
            system_prompt_usability=directanswer_validation_prompt
            )

        ###################### validate
        agent_name = state.get("agent_name")
        assert agent_name is not None
        logger.info(f"[AGENT {agent_name}] Run validation module")
        agent_output: ValidatedAgentResponse = await validator.validate_agent_response(
            available_messages,
        )

        assert agent_output.type is not None
        logger.info(f"[AGENT {agent_name}] Validation module end")
        return {
            "agent_output_aborted": not agent_output.valid,
            "agent_output_abortion_reason": (
                agent_output.abortion_code.value
                if agent_output.abortion_code is not None
                else None
            ),
            "agent_output_description": agent_output.type.value,
            "validated_agent_output": agent_output.response,
        }

    return validate_agent_output_async

#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
################################################################################################################################################################# TEST

if __name__ == "__main__":
    import asyncio

    ########################################################################## AGENT
    from agents.llm.client import model
    # get llm instance
    llm = model

    ########################################################################### tools
    @tool("get_customer_data", description="Liefert Kundendaten zur Lösung von Problemen.")
    def get_customer_data(request: str) -> str:
        """Doc."""
        # Intern rufen wir die partial-Funktion auf:
        result = (
            f"Ihre Anfrage: {request}. Antwort: Ihre Kundendaten: Preis = 5, Produkt = Hundefutter"
        )
        return result

    PROMPT = """"Du bist ein Tool-Agent, der Tools erhält, die möglicherweise zur Beantwortung einer NUTZERANFRAGE nützlich sind.

            Deine Aufgaben:
            <Toolaufrufe>
            - Entscheide, ob du Tools aufrufst.
            - Beachte dabei genau die Beschreibung des Tools.
            - Insbesondere die Abschnitte "USE WHEN" und "DO NOT USE WHEN" solltest du für deine Entscheidung berücksichtigen, ob ein Tool aufzurufen ist, oder nicht.
            - Wenn du ein Tool aufrufst, müssen die Inputs für den Aufruf zwingend dem Text entnommen werden. Wenn nicht alle Inhalte gegeben sind, rufe das Tool nicht auf.
            </Toolaufrufe>

            <Antwortgenerierung>
            - Wenn du keine weiteren Toolcalls machen kannst, fasse deine bisherigen Ergebnisse zusammen. Erwähne kurz, dass du das Limit erreicht hast (gib "TOOLLIMIT" als Stichwort aus).
            </Antwortgenerierung>
            """
    query = "Ich habe ein Problem mit dem Kunden, wie kann ich das lösen?"
    query = "Ich will meinen Vertrag kuendigen. Vertragsnummer ist 123."
    query = "Rufe alle vorhandenen Tools jeweils 2 mal mit verschiedenen Inputs auf. Das Ergebnis ist erstmal egal. Rufe dann alle Tools nochmal mit 2 verschiedenen Inputs auf."
    query = "Frage die Kundendaten dreimal hintereinander ab, ganz egal was passiert."

    async def test_agent_call_sync():
        """Doc."""
        agent: CompiledStateGraph = create_agent(
            model=llm,
            tools=[get_customer_data],
            middleware=[
                LoggingMiddlewareSync(),
                global_toolcall_limit_sync(2),
            ],
            system_prompt=PROMPT,
        ) # type: ignore[annotation-unchecked]
        results = agent.invoke(
            {
                "messages": [HumanMessage(query)],
                "body": query,
                "subject": "Neue Mail",
                "division": "KFZ",
                "ticket_id": "123",
            }
        )
        for message in results["messages"]:
            print("### type: ", type(message))
            print("### message: ", message)


    asyncio.run(test_agent_call_sync())
    print("done")

