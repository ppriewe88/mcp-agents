from typing import AsyncGenerator
import asyncio

async def artificial_stream(answer: str, pause:float) -> AsyncGenerator[bytes, None]:
    words = answer.split()
    for i, w in enumerate(words):
        chunk = w if i == len(words) - 1 else f"{w} "
        yield chunk.encode("utf-8")
        await asyncio.sleep(pause)