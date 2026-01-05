from fastapi import FastAPI
from agents.mcp_client.client import MCPClient
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from agents.models.api import (
    GetToolsRequest,
    StreamAgentRequest
    )
from agents.models.agents import AgentConfig, AgentRegistryEntry
from agents.models.tools import ToolSchema
from agents.factory.factory import AgentFactory
from typing import List
import logging 
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import asyncio
from typing import AsyncGenerator
from langchain.messages import AIMessageChunk

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###################################################################### GET TOOLS
@app.post("/get_tools")
async def get_tools(req: GetToolsRequest):
    """Excample for input: http://127.0.0.1:8000/sse."""
    server_url = req.server_url
    client = MCPClient(mcp_server_endpoint=server_url)
    tools = await client.get_tools()
    
    dumped_tools = [tool.model_dump() for tool in tools]
    json_response = JSONResponse(content=dumped_tools)
    return json_response

###################################################################### CALL AGENT


async def word_stream(answer: str) -> AsyncGenerator[bytes, None]:
    # split() removes whitespace _> send empty space after each word
    words = answer.split()
    for i, w in enumerate(words):
        chunk = w if i == len(words) - 1 else (w + " ")
        yield chunk.encode("utf-8")
        await asyncio.sleep(0.05)


@app.post("/stream-test2")
async def stream_test(payload: StreamAgentRequest):
    # Plain text stream
    answer:str = ""
    try:
        message: str = payload.message
        agent_config: AgentConfig = payload.agent_config
        tool_schemas: List[ToolSchema] = payload.tool_schemas
        agent_reg_entry = AgentRegistryEntry(
            description=agent_config.description,
            config=agent_config,
            tool_schemas=tool_schemas
        )
    except Exception as error:
        answer = "Agent setup failed! Check data schemas (frontend/backend communication)!"
        logger.error(f"[API] {answer}. \n error: {error}")

    try:
        factory = AgentFactory()
        result = await factory.run_frontend_agent(
                name="", 
                entry = agent_reg_entry, 
                query=message
            )
    except Exception as error:
        answer = "Agent run failed! Check tools and if mcp server is running!"
        logger.error(f"[API] {answer}. \n error: {error}")

    try:
        if result["agent_output_aborted"]:
            answer = f"Agent response was aborted for the following reason: {result["agent_output_abortion_reason"]}"
        else: 
            answer = result["validated_agent_output"]
    except Exception as error:
        answer = "Agent output not in expected format! Check types/schemas in backend!"
        logger.error(f"[API] {answer}. \n error: {error}")

    return StreamingResponse(word_stream(answer), media_type="text/plain")

###################################################################### STREAM AGENT
async def agent_text_stream(agent, input_message) -> AsyncGenerator[bytes, None]:
    async for mode, data in agent.astream(
        {"messages": [input_message]},
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            token, metadata = data
            if isinstance(token, AIMessageChunk) and token.text:
                # Token-Text als bytes streamen
                yield token.text.encode("utf-8")

@app.post("/stream-test")
async def invoke_stream(payload: StreamAgentRequest):
    answer:str = ""
    try:
        message: str = payload.message
        agent_config: AgentConfig = payload.agent_config
        tool_schemas: List[ToolSchema] = payload.tool_schemas
        agent_reg_entry = AgentRegistryEntry(
            description=agent_config.description,
            config=agent_config,
            tool_schemas=tool_schemas
        )
    except Exception as error:
        answer = "Agent setup failed! Check data schemas (frontend/backend communication)!"
        logger.error(f"[API] {answer}. \n error: {error}")

    try:
        factory = AgentFactory()
        agent = factory._charge_agent(
            name="Test",
            entry=agent_reg_entry
        )
    except Exception as error:
        answer = "Agent charging failed!"
        logger.error(f"[API] {answer}. \n error: {error}")

    stream = StreamingResponse(
        agent.astream(message),
        media_type="text/plain",
    )
    return stream


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agents.mcp_api.api:app",
        host="127.0.0.1",
        port=3001,
        reload=True,
    )