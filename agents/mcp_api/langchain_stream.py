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



 # ################################################################ V1
        #     if stream_mode != "updates":
        #         continue

        #     # data: dict[source, update]
        #     for source, update in data.items():

        #         # NO STREAM CONDITION 1: no value in update
        #         if not isinstance(update, dict):
        #             continue
                
        #         # 1) Validator abort
        #         if update.get("agent_output_aborted") is True:
        #             reason = update.get("agent_output_abortion_reason") or "validation rejected"
        #             yield f"[ABORTED:{reason}]".encode("utf-8")
        #             return

        #         # NO STREAM CONDITION 3: No messages contained in update
        #         msgs = update.get("messages")
        #         if not msgs:
        #             continue

        #         last: AnyMessage = msgs[-1]

        #         # STREAM CONDITION 4: Check for toolcalls
        #         if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        #             for tc in last.tool_calls:
        #                 tc_id = tc.get("id") or f"{tc.get('name')}::{hash(str(tc.get('args')))}"
        #                 # NO STREAM CONDITION 4.1: emitted toolcalls....?
        #                 if tc_id in emitted_toolcall_ids:
        #                     continue
        #                 # STREAM CONDITION 4.2: emitted toolcalls....?
        #                 emitted_toolcall_ids.add(tc_id)
        #                 tool_name = tc.get("name", "unknown_tool")
        #                 yield f"[TOOL:{tool_name}]".encode("utf-8")
        #             continue

        #         # STREAM CONDITION 5: Check for toolcall results
        #         if isinstance(last, ToolMessage):
        #             yield f"[TOOL_DONE]".encode("utf-8")
        #             continue
                
        #         # STREAM CONDITION 6: Check for toolcall results
        #         validated: Optional[Any] = update.get("validated_agent_output")
        #         if validated is None or emitted_final:
        #             continue
                
        #         print("HIER ENDE???")

        #         # 3) Kandidat für finale Antwort (AIMessage ohne tool_calls)
        #         if isinstance(last, AIMessage) and not getattr(last, "tool_calls", None):
        #             # Wir merken uns die jeweils letzte "finale" AIMessage
        #             final_ai_message = last

        #         # Nach Ende des Graph-Runs: finale Antwort genau einmal ausgeben
        #         if final_ai_message and final_ai_message.content:
        #             # content kann str oder list sein je nach Modell; handle str zuerst
        #             if isinstance(final_ai_message.content, str):
        #                 yield final_ai_message.content.encode("utf-8")
        #             else:
        #                 # Fallback: grob serialisieren oder blocks flatten
        #                 yield str(final_ai_message.content).encode("utf-8")

        ################################################################ V0    
            # if stream_mode == "messages":
            #     token, metadata = data

            #     node = metadata.get("langgraph_node") or metadata.get("node") or metadata.get("name")
            #     print(f"\n[chunk from node={node}] ", end="")

            #     if isinstance(token, AIMessageChunk):
            #         self._render_message_chunk(token)

            #         # IMPORTANT: Only stream plain token text (prevents "too much yielding")
            #         if token.text:
            #             yield token.text.encode("utf-8")

            # elif stream_mode == "updates":
            #     items = data.items()
            #     for source, update in data.items():
            #         if source in ("model", "tools"):
            #             last_msg = update["messages"][-1]
            #             self._render_completed_message(last_msg)
    
    # def _render_message_chunk(self, token: AIMessageChunk) -> None:
    #         if token.text:
    #             print(token.text, end="|")
    #         if getattr(token, "tool_call_chunks", None):
    #             print(token.tool_call_chunks)

    # def _render_completed_message(self, message: AnyMessage) -> None:
    #     if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
    #         print(f"Tool calls: {message.tool_calls}")
    #     if isinstance(message, ToolMessage):
    #         print(f"Tool response: {message.content_blocks}")
    #     if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
    #         print(f"Model response: {message.content}")