"""
run with: mcp dev server.py
"""
import logging
import os
from typing import Annotated

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()
MCP_SERVER_ENDPOINT: str = os.getenv("MCP_SERVER_ENDPOINT", "http://127.0.0.1:8000")


# Create an MCP server
mcp = FastMCP(
    name="dummy_server",
    host="0.0.0.0",
    port=8000,
    stateless_http=True,
)

# Add a simple calculator tool
@mcp.tool(
    name="add_numbers",
    description="""Adds two integers and returns their sum.
    This is a longer placeholder text. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.""",)
def add_typed(a: Annotated[int, Field(..., 
                   description="The first number to add")], 
        b: Annotated[int, Field(..., 
                   description="The second number to add")] = 2) -> int:
    """Add two numbers together"""
    return a + b

@mcp.tool()
def get_birthday_santaclaus(query:str) -> str:
    """say hi back"""
    return "Hallo! Die geheime Information ist: Der Weihnachtsmann ist am 31.12.1570 geboren!"

@mcp.tool()
def more_infos_on_santa(birth_year: str) -> str:
    """say hi back"""
    answer = f"Datum: {birth_year}, Name: Thaddäus!"
    return answer 

@mcp.tool()
def shopping_list(name:str) -> list:
    """say hi back"""
    return ["sugar", "flour", "butter"]


# Run the server
if __name__ == "__main__":
    mcp.run(transport="sse")