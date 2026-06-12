"""Runs the NLP server."""

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, Request

app = FastAPI()
logger = logging.getLogger(__name__)
manager = None
manager_lock = asyncio.Lock()


class _LoadState:
    """Tracks corpus-loading state for async, pollable behavior."""

    def __init__(self) -> None:
        self.status: str = "idle"  # idle | loading | loaded | failed
        self.error: Optional[str] = None
        self.task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()


load_state = _LoadState()


async def _get_manager():
    """Lazy-load the heavy NLP stack after /health is already responsive."""
    global manager
    if manager is None:
        async with manager_lock:
            if manager is None:
                from nlp_manager import NLPManager

                manager = NLPManager()
    return manager


def _do_load(documents) -> bool:
    """Synchronous corpus load. Runs on a worker thread."""
    assert manager is not None
    manager.load_corpus(documents)
    return manager.loaded


async def _load_task(documents) -> None:
    try:
        await _get_manager()
        ok = await asyncio.to_thread(_do_load, documents)
        load_state.status = "loaded" if ok else "failed"
    except Exception as e:
        logger.exception("Corpus load failed")
        load_state.status = "failed"
        load_state.error = str(e)


@app.post("/nlp")
async def nlp(request: Request) -> dict[str, list[dict[str, list[str] | str]]]:
    inputs_json = await request.json()
    first = inputs_json["instances"][0]

    # Load: any request carrying `documents` kicks off the load.
    if first.get("documents") is not None:
        async with load_state.lock:
            if load_state.status == "idle":
                load_state.status = "loading"
                load_state.task = asyncio.create_task(_load_task(first["documents"]))
            return {"predictions": [{"status": load_state.status}]}
    # Poll: returns current status (subsequent polls).
    if first.get("poll") is not None:
        return {"predictions": [{"status": load_state.status}]}

    # Batch all questions in one call so vLLM can do continuous batching across them.
    manager = await _get_manager()
    questions = [instance["question"] for instance in inputs_json["instances"]]
    predictions = await asyncio.to_thread(manager.qa_batch, questions)

    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    return {"message": "health ok"}
