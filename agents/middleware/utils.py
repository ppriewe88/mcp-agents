from typing import List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from agents.config import POSTPROCESSING_ERROR_MARKER
from agents.models.agents import AbortionCodes, LoopStatus


class DetectedStatus(BaseModel):
    type: LoopStatus
    abortion_code: AbortionCodes | None = None

def detect_loop_status(
        messages: List[HumanMessage | ToolMessage | AIMessage]
) -> DetectedStatus:
    """Detect the current loop status of the agent based on the message trace.

    The function classifies the agent’s current execution state by inspecting the last message
    and the overall message history. Each possible loop status is explicitly and redundantly
    defined for clarity and maintainability.

    The following states are distinguished:

    - PENDING:
      The agent has not yet produced a model response. The last message is a HumanMessage,
      indicating the first iteration of the loop.

    - TOOLCALL_REQUEST:
      The last message is an AIMessage that contains one or more tool calls. The agent is
      requesting tool execution and the loop will continue.

    - TOOLCALLS_ONLY:
      The last message is a ToolMessage without an error marker. This typically occurs when
      the model call limit was reached and no final natural-language answer was produced.

    - DIRECT_ANSWER:
      The last message is an AIMessage with content and without tool calls, and no ToolMessage
      exists in the message history. This represents a direct answer produced without any
      tool usage.

    - TOOLCALLS_WITH_FINAL_ANSWER:
      The last message is an AIMessage with content and without tool calls, and at least one
      ToolMessage exists in the message history. This represents a final answer that was
      generated after one or more tool calls.

    - ABORTED:
      The loop was terminated due to an unrecoverable error (e.g. a tool execution error) or
      an unknown or invalid state.

    Args:
        messages (List[HumanMessage | ToolMessage | AIMessage]):
            Full message trace of the agent execution loop, ordered chronologically.

    Returns:
        DetectedStatus:
            An object containing the detected LoopStatus and, if applicable, an abortion code
            describing the reason for termination.
    """
    last_message = messages[-1]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    ### PENDING
    if isinstance(last_message, HumanMessage):
        return DetectedStatus(type = LoopStatus.PENDING)

    ### TOOL ERROR
    # last message is ToolMessage, and contains error markers
    # OCCURS, when mcp toolcall was errorfull and marked by postprocessor
    # NOTICE: Last message = current message, when middleware AbortOnToolErrors is active!
    if isinstance(last_message, ToolMessage) and last_message.content == POSTPROCESSING_ERROR_MARKER:
        return DetectedStatus(
            type = LoopStatus.ABORTED,
            abortion_code = AbortionCodes.TOOL_ERROR)

    ### TOOLCALL_REQUEST
    # last message is AIMessage with toolcall request
    if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
        return DetectedStatus(type=LoopStatus.TOOLCALL_REQUEST)

    ### TOOLCALLS ONLY
    # last message is ToolMessage, no errors
    # OCCURS, when modelcall limit was reached and last model call contained toolcall only
    if isinstance(last_message, ToolMessage):
        return DetectedStatus(type = LoopStatus.TOOLCALL_CONTENTS)

    ### DIRECT ANSWER
    # last message is AIMessage, but no toolcalls have been made at all
    # ONLY, when only one model call was made (with no tool calls).
    # If direct answers allowed, validate. Else, abort
    if (
        isinstance(last_message, AIMessage)
        and len(tool_messages) == 0
        and len(last_message.tool_calls) == 0
    ):
        return DetectedStatus(type = LoopStatus.DIRECT_ANSWER)

    ### TOOLCALLS WITH FINAL ANSWER
    # last message is summary of toolcalls (AIMessage with content)
    # occurs when agent answers, AND toolcalls have been made before!
    # occurs when react loop closes with direct answer after toolcalls
    if (
            # loop did end with model response
            isinstance(last_message, AIMessage)
            # model response has content (-> is direct answer)
            and len(last_message.tool_calls) == 0
            # toolcalls have been made (-> model refers to received toolcall data)
            and len(tool_messages) > 0
        ):
        return  DetectedStatus(type = LoopStatus.TOOL_BASED_ANSWER)

    ### FALLBACK: None of the above cases was found
    return DetectedStatus(
        type = LoopStatus.ABORTED,
        abortion_code = AbortionCodes.UNKNOWN)
