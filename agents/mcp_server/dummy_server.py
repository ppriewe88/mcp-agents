"""
run with: mcp dev server.py
"""
import logging
import os
from typing import Annotated, Dict, List

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()
MCP_SERVER_ENDPOINT: str = os.getenv("MCP_SERVER_ENDPOINT", "http://127.0.0.1:8000/sse")


# Create an MCP server
mcp = FastMCP(
    name="dummy_server",
    host="0.0.0.0",
    port=8000,
    stateless_http=True,
)

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

    # raise Exception("THIS DID NOT GO WELL!")
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



##################################
class Structured(BaseModel):
    data_dict:Dict
    data_list:List

@mcp.tool(
    name="structured_pydantic",
    description="""Test for structured output.""")
async def structured_pydantic() -> Structured:
    """Add two numbers together"""
    result = Structured(
        data_dict={
            "name":"Patrick",
            "age":"37"
                    },
        data_list=[
            "287", "37"
        ]
    )
    return result

@mcp.tool(
    name="structured_dict",
    description="""Test for structured output.""")
def structured_dict() -> Dict:
    """Add two numbers together"""
    result: Dict = {
        "1": {
        "name":"patrick",
        "age": "37"
        },
        "2": {
        "name":"anika",
        "age": "36"
        }
    }
    return result

# Run the server
if __name__ == "__main__":
    mcp.run(transport="sse")