import asyncio

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.config import get_stream_writer


# -------------------------
# 1) "Standard" Tool
# -------------------------
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# -------------------------
# 2) Innerer Agent (hat get_weather)
# -------------------------
inner_agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)


# -------------------------
# 3) Tool: Inneren Agenten laufen lassen
#    und seine Events in den äußeren Stream hochreichen
# -------------------------
@tool
async def run_inner_agent(user_query: str) -> str:
    """
    Runs the inner agent with streaming and forwards inner progress into the
    OUTER agent stream via custom events.
    Returns final inner result as a normal tool return (string).
    """
    writer = get_stream_writer()

    emitted_toolcall_ids: set[str] = set()
    emitted_final = False
    final_text: str | None = None
    
    # Optional: "Start" Marker
    writer({"type": "inner_agent", "event": "start", "query": user_query})

    final_text: str | None = None

    # Wichtig:
    # - Wir streamen den inneren Agenten hier selbst.
    # - Alles, was "live" nach außen soll, senden wir über writer(...) als custom.
    #
    # stream_mode kann kombiniert werden; dann liefert astream (mode, chunk). :contentReference[oaicite:3]{index=3}
    async for mode, chunk in inner_agent.astream(
        {"messages": [{"role": "user", "content": user_query}]},
        stream_mode=["updates", "messages", "custom"],
    ):
        if mode == "updates":
            # chunk: Dict[step_name, {"messages": [...]}]
            # Wir machen daraus eine leicht serialisierbare Struktur.
            for step, data in chunk.items():
                msg = data["messages"][-1]

                # content_blocks ist in den Docs das stabile Format für Streaming-Ausgaben
                # (Tool Calls / Text / etc.). :contentReference[oaicite:4]{index=4}
                content_blocks = getattr(msg, "content_blocks", None)
                writer({
                    "type": "inner_agent",
                    "event": "update",
                    "step": step,
                    "content_blocks": content_blocks,
                })

                # Finale inner Antwort typischerweise im letzten "model" Update ohne ToolCall.
                # Minimaler Pragmatismus: wenn der letzte model-step Text enthält, merken wir ihn.
                if step == "model" and content_blocks:
                    # content_blocks ist oft [{"type":"text","text":"..."}] oder ToolCall-Blöcke
                    text_parts = [
                        b.get("text", "")
                        for b in content_blocks
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    if text_parts:
                        final_text = "".join(text_parts).strip()

        elif mode == "messages":
            # chunk: (token_chunk, metadata) :contentReference[oaicite:5]{index=5}
            token, metadata = chunk  # type: ignore[misc]
            # Optional filtern: nur inner "model" Tokens
            node = metadata.get("langgraph_node")
            if node == "model":
                # token.content oder token.content_blocks kann leer sein je nach Chunk
                token_text = getattr(token, "content", "") or ""
                if token_text:
                    writer({
                        "type": "inner_agent",
                        "event": "token",
                        "node": node,
                        "text": token_text,
                    })

        elif mode == "custom":
            # Relevant, if 
            writer({
                "type": "inner_agent",
                "event": "custom",
                "data": chunk,
            })

    writer({"type": "inner_agent", "event": "end"})

    # Tool MUSS normal zurückgeben (kein Generator), damit outer Agent eine ToolMessage bekommt.
    return final_text or "(inner agent produced no final text)"


# -------------------------
# 4) Äußerer Agent (hat als Tool den inneren Agenten)
# -------------------------
outer_agent = create_agent(
    model="gpt-5-nano",
    tools=[run_inner_agent],
)


# -------------------------
# 5) Konsumieren: outer_agent.astream mit updates+custom(+messages)
#    -> custom enthält forwarded inner-agent Updates/Tokens
# -------------------------
async def main():
    user_prompt = "Ask the inner agent: What's the weather in SF? Use its answer."

    async for mode, chunk in outer_agent.astream(
        {"messages": [{"role": "user", "content": user_prompt}]},
        stream_mode=["updates", "custom", "messages"],
    ):
        if mode == "custom":
            # HIER "wirken" die inneren Updates im äußeren Stream:
            # du bekommst sie live, während das Tool läuft.
            print("CUSTOM:", chunk)

        elif mode == "updates":
            # Outer step updates (model/tools/model) :contentReference[oaicite:6]{index=6}
            print("UPDATES:", chunk)

        elif mode == "messages":
            # Outer tokens (optional)
            token, metadata = chunk
            if metadata.get("langgraph_node") == "model":
                txt = getattr(token, "content", "") or ""
                if txt:
                    print("OUTER TOKEN:", txt, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
