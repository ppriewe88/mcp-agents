from typing import AsyncGenerator
import asyncio

async def artificial_stream(answer: str, pause:float) -> AsyncGenerator[str, None]:
    words = answer.split()
    for i, w in enumerate(words):
        chunk = w if i == len(words) - 1 else f"{w} "
        yield chunk
        await asyncio.sleep(pause)