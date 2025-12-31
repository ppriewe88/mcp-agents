from abc import ABC, abstractmethod
from typing import List

from mcp.types import CallToolResult

from agents.models.client import MCPToolDecision, OpenAITool


class BaseMCPClient(ABC):
    """Model for abstract mcp client. Serves to define signatures."""

    @abstractmethod
    def __init__(self):
        """Constructor."""
        raise NotImplementedError

    @abstractmethod
    async def connect(self) -> bool:
        """Method to connect to mcp server."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> bool:
        """Method to close connection to mcp server."""
        raise NotImplementedError

    @abstractmethod
    async def get_tools(self) -> List[OpenAITool]:
        """Method to get tools from mcp server."""
        raise NotImplementedError

    @abstractmethod
    async def call_tools(self, tooling_decision: List[MCPToolDecision]) -> List[CallToolResult]:
        """Method to call tools on mcp server."""
        raise NotImplementedError
