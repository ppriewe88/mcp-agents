from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal

from mcp.types import Tool
from pydantic import BaseModel, Field, model_validator

##################################################################### Tool models


class MCPTool(Tool):
    """Type to explicitly name (or further specify) MCPTools in protocol format."""

    pass


class OpenAIToolParameters(BaseModel):
    """Type for parameters of mcp tools in openai format.

    Notice: additionalProperties set to False to enforce strict mode of tool calling.
    """

    type: Literal["object"] = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    additionalProperties: bool = False  # noqa: N815


class OpenAIFunction(BaseModel):
    """Type for mcp tools in Openai format - inner part.

    Notice: strict set to True to enforce strict mode of tool calling.
    """

    name: str
    description: str = ""
    parameters: OpenAIToolParameters
    strict: bool = True


class OpenAITool(BaseModel):
    """Type for mcp tools in Openai format - outer hull."""

    type: Literal["function"] = "function"
    function: OpenAIFunction

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, data: Any):
        """Normalizes internally to {"type":"function","function":{...}}."""
        if isinstance(data, dict):
            # flat format -> create nested structure
            has_flat = any(k in data for k in ("name", "parameters", "description", "strict"))
            if has_flat:
                return {
                    "type": "function",
                    "function": {
                        "name": data.get("name"),
                        "description": data.get("description", ""),
                        "parameters": data.get("parameters"),
                        "strict": data.get("strict", True),
                    },
                }
        return data


class MCPToolDecision(BaseModel):
    """Type for llm decision for toolcalls."""

    name: str = Field(min_length=1)
    args: Dict[str, Any]
    id: str = Field(min_length=1)

class MCPErrorCode(str, Enum):
    """Error codes for MCP."""

    CLIENT = "MCP: CLIENT SETUP ERROR"
    LIST_TOOLS = "MCP: LIST TOOLS ERROR"
    CONVERSION = "MCP: TOOL CONVERSION ERROR"
    TOOLING = "MCP: TOOL ERROR"
    UNKNOWN = "MCP: UNKNOWN ERROR"

@dataclass
class MCPError(Exception):
    """Error for MCP."""

    message: str
    code: MCPErrorCode = MCPErrorCode.UNKNOWN

    def __post_init__(self):
        """Post init for MCP error."""
        super().__init__(self.message)

    def __str__(self) -> str:
        """String representation of MCP error."""
        return f"{self.code}: {self.message}"