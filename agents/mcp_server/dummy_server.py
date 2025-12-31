"""
run with: mcp dev server.py
"""
import logging
import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

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
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

@mcp.tool()
def get_birthday_santaclaus(query:str) -> str:
    """say hi back"""
    return "Hallo! Die geheime Information ist: Der Weihnachtsmann ist am 31.12.1570 geboren!"

@mcp.tool()
def summarize(birth_year:str) -> str:
    """say hi back"""
    answer = f"Datum: {birth_year}, Name: Thaddäus!"
    return answer 

@mcp.tool()
def shopping_list(name:str) -> list:
    """say hi back"""
    return [1,2,3]


# Run the server
if __name__ == "__main__":
    mcp.run(transport="sse")