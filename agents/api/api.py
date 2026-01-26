import logging
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from agents.api.utils import assemble_agent, use_test_agent
from agents.factory.factory import RunnableAgent
from agents.mcp_client.client import MCPClient
from agents.models.api import GetToolsRequest, StreamAgentRequest, ChatMessage

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

TEST_AGENTS_AS_TOOL: bool = True

@app.post("/stream-test")
async def stream_test(payload: StreamAgentRequest):
    # Plain text stream
    messages: List[ChatMessage]
    agent: RunnableAgent
    try:
        messages = payload.messages
        if TEST_AGENTS_AS_TOOL:
            agent = use_test_agent()
        else:
            agent = assemble_agent(payload)
        stream = StreamingResponse(
            agent.outer_astream(messages),
            media_type="application/x-ndjson",
        )
    except Exception as error:
        stream = "Ooops, something went wrong in the backend...!"
        logger.error(f"[API] {stream}. \n error: {error}")

    return stream



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agents.api.api:app",
        host="127.0.0.1",
        port=3001,
        reload=True,
    )