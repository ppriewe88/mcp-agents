from fastapi import FastAPI
from agents.mcp_client.client import MCPClient
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

class GetToolsRequest(BaseModel):
    server_url: str

@app.post("/get_tools")
async def get_tools(req: GetToolsRequest):
    """Excample for input: http://127.0.0.1:8000/sse."""
    server_url = req.server_url
    client = MCPClient(mcp_server_endpoint=server_url)
    tools = await client.get_tools()
    
    dumped_tools = [tool.model_dump() for tool in tools]
    json_response = JSONResponse(content=dumped_tools)
    return json_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agents.mcp_client.api:app",
        host="127.0.0.1",
        port=3001,
        reload=True,
    )