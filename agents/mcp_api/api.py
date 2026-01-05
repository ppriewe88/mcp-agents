from fastapi import FastAPI
from agents.mcp_client.client import MCPClient
from agents.factory.utils import artificial_stream
from agents.mcp_api.utils import assemble_agent
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from agents.models.api import (
    GetToolsRequest,
    StreamAgentRequest
    )
from agents.factory.factory import ConfiguredAgent
from typing import Literal
import logging 
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

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

MODE: Literal["true_stream" , "simulated_stream"] = "true_stream"

@app.post("/stream-test")
async def stream_test(payload: StreamAgentRequest):
    # Plain text stream
    if MODE == "simulated_stream":
        answer:str = ""
        try:
            message: str = payload.message
            agent:ConfiguredAgent = assemble_agent(payload)
            result = await agent.run(query=message)
            if result["agent_output_aborted"]:
                answer = f"Agent response was aborted for the following reason: {result["agent_output_abortion_reason"]}"
            else: 
                answer = result["validated_agent_output"]
            stream = StreamingResponse(artificial_stream(answer), media_type="text/plain")
        except Exception as error:
            answer = "Ooops, something went wrong in the backend...!"
            logger.error(f"[API] {answer}. \n error: {error}")

        return stream
    if MODE == "true_stream":
        answer:str = ""
        try:
            message: str = payload.message
            agent: ConfiguredAgent = assemble_agent(payload)
            stream = StreamingResponse(
                agent.astream(message),
                media_type="text/plain",
            )
        except Exception as error:
            answer = "Agent charging failed!"
            logger.error(f"[API] {answer}. \n error: {error}")

        return stream



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agents.mcp_api.api:app",
        host="127.0.0.1",
        port=3001,
        reload=True,
    )