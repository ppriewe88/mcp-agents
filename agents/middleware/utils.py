from typing import List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from agents.models.agents import AbortionCodes, LoopStatus, MiscMarkers


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
    if (
        isinstance(last_message, ToolMessage)
        and last_message.content == MiscMarkers.POSTPROCESSING_ERRORMARKER.value
    ):
        return DetectedStatus(
            type = LoopStatus.ABORTED,
            abortion_code = AbortionCodes.TOOL_ERROR)

    ### TOOLCALL_REQUEST
    if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
        return DetectedStatus(type=LoopStatus.TOOLCALL_REQUEST)

    ### TOOLCALLS ONLY
    if isinstance(last_message, ToolMessage):
        return DetectedStatus(type = LoopStatus.TOOLCALL_CONTENTS)

    ### DIRECT ANSWER
    if (
        isinstance(last_message, AIMessage)
        and len(tool_messages) == 0
        and len(last_message.tool_calls) == 0
    ):
        return DetectedStatus(type = LoopStatus.DIRECT_ANSWER)

    ### TOOLCALLS WITH FINAL ANSWER
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
