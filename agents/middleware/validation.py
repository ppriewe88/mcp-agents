import logging
from typing import List, Optional

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel, Field

from agents.llm.client import model
from agents.middleware.utils import (
    DetectedStatus,
    detect_loop_status,
)
from agents.models.agents import (
    AbortionCodes,
    LoopStatus,
    ValidatedAgentResponse,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ValidationOutput(BaseModel):
    """Type for output of validation."""

    usable: bool = Field(
        ..., description="True or false according to instructions in system prompt."
    )
    reasoning: str = Field(..., description="Short reasoning why the answer is usable or not.")


class AgentResponseValidator:
    """Validates agent response."""

    DEFAULT_SYSTEM_PROMPT = """Du beurteilst die Antwort eines Toolcalling-Agenten.

        <Anweisungen>:
        Bestimme, ob die Antwort 'usable' ist.

        Eine Antwort ist usable, wenn:
        - wenn sie klar ausdrückt, dass für einen Toolaufruf Informationen fehlen, und angibt, welche das sind.

        Eine Antwort ist nicht usable, wenn:
        - wenn sie nur sagt, dass kein Tool aufgerufen werden konnte oder gar keine Werkzeuge benutzt wurden.
        - wenn sie eventuell ganz andere Aussagen und Informationen beinhaltet.
        """

    DEFAULT_HUMAN_PROMPT = """ANTWORT DES AGENTEN:
        {agent_text}
        """

    def __init__(
        self,
        system_prompt_usability: Optional[str] = None,
    ):
        self.system_prompt_usability = system_prompt_usability or self.DEFAULT_SYSTEM_PROMPT
        self.human_prompt_usability = self.DEFAULT_HUMAN_PROMPT
        self.llm = model

    def _build_usability_chain(self):
        """Prepare chain."""
        llm_with_structured_output = self.llm.with_structured_output(ValidationOutput)

        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.system_prompt_usability),
                HumanMessagePromptTemplate.from_template(self.human_prompt_usability),
            ],
            input_variables=["agent_text"],
        )

        chain = prompt | llm_with_structured_output
        return chain

    def _validate_usability_of_direct_answer(self, agent_response: str) -> ValidationOutput:
        """Validate AIMessage for useability."""
        chain = self._build_usability_chain()
        result: ValidationOutput = chain.invoke(
            {
                "agent_text": agent_response,
            }
        )
        logger.debug(f"[VALIDATION] last message is useable: {result.usable}")
        return result

    async def validate_agent_response(
        self,
        messages: List[AIMessage | ToolMessage | HumanMessage],
        allow_direct_answers: bool
    ) ->  ValidatedAgentResponse:
        """Validates agent response.

        Distinguishes different cases for agent response content.
        For each case, goes through different validation logic.
        """
        ############################## extract relevant messages for case distinctions
        last_message = messages[-1]
        tool_messages: List[ToolMessage] = [
            message for message in messages if isinstance(message, ToolMessage)
        ]
        ######################## detect agent answer type
        detected: DetectedStatus = detect_loop_status(messages)
        answer_type = detected.type
        abortion_code = detected.abortion_code or AbortionCodes.UNKNOWN

        # ABORTED (known or unknown reason from detector)
        if answer_type == LoopStatus.ABORTED:
            logger.debug(f"[VALIDATION] current message has been aborted (reason: {abortion_code})")
            return ValidatedAgentResponse(
                response=None,
                valid=False,
                abortion_code=abortion_code,
                type=LoopStatus.ABORTED,
            )

        # TOOLCALLS_ONLY
        if answer_type == LoopStatus.TOOLCALL_CONTENTS:
            logger.debug(
                "[VALIDATION] last message is ToolMessage. Agentic output consists of tool contents only (no postprocessing by agent)."
            )
            tool_content = "\n".join(msg.text for msg in tool_messages)
            return ValidatedAgentResponse(
                response=tool_content,
                valid=True,
                type=LoopStatus.TOOLCALL_CONTENTS,
            )

        # DIRECT_ANSWER
        if answer_type == LoopStatus.DIRECT_ANSWER:
            if not allow_direct_answers:
                logger.debug("[VALIDATION] No direct answers allowed. Abort")
                return ValidatedAgentResponse(
                    response=None,
                    valid=False,
                    abortion_code=AbortionCodes.DIRECT_ANSWERS_FORBIDDEN,
                    type=LoopStatus.ABORTED,
                )
            else:
                logger.debug("[VALIDATION] Direct answers allowed. Validating direct answer.")
                usability_check = self._validate_usability_of_direct_answer(last_message.text)
                if not usability_check.usable:
                    return ValidatedAgentResponse(
                        response = None,
                        valid = False,
                        abortion_code = AbortionCodes.DIRECT_ANSWER_UNUSABLE,
                        type = LoopStatus.ABORTED)
                else:
                    return ValidatedAgentResponse(
                        response = last_message.text,
                        valid = True,
                        type = LoopStatus.DIRECT_ANSWER
                        )

        # TOOLCALLS_WITH_FINAL_ANSWER:
        if answer_type == LoopStatus.TOOL_BASED_ANSWER:
            logger.debug(
                "[VALIDATION] last message is AIMessage with prior toolcalls made (with postprocessing by agent)"
            )
            logger.debug("[VALIDATION] Currently further checks done (NOT YET IMPLEMENTED)")
            return ValidatedAgentResponse(
                        response = last_message.text,
                        valid = True,
                        type = LoopStatus.TOOL_BASED_ANSWER
                        )

        # Fallback for safety reasons (should not happen if detector is exhaustive)
        return ValidatedAgentResponse(
            response=None,
            valid=False,
            abortion_code=AbortionCodes.UNKNOWN,
            type=LoopStatus.ABORTED,
        )
