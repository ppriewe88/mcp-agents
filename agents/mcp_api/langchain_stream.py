import asyncio
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage


def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"


agent = create_agent("openai:gpt-5.2", tools=[get_weather])


def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)
    # N.B. all content is available through token.content_blocks


def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")
    if isinstance(message, AIMessage) and not message.tool_calls:
        print(f"Model response: {message.content}")




async def run():
    input_message = {"role": "user", "content": "What is the weather in Boston? and please add 'Best regards'."}
    async for stream_mode, data in agent.astream(
        {"messages": [input_message]},
        stream_mode=["messages", "updates"],  
    ):
        if stream_mode == "messages":
            token, metadata = data
            if isinstance(token, AIMessageChunk):
                _render_message_chunk(token)  
        if stream_mode == "updates":
            for source, update in data.items():
                if source in ("model", "tools"):  # `source` captures node name
                    _render_completed_message(update["messages"][-1])

if __name__=="__main__":
    import asyncio
    asyncio.run(run())