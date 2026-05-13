"""Runs the NLP server.

Endpoint contract (Wiki, May 2026 final NLP eval):
- Corpus load: POST /nlp with {"instances":[{"documents":[...]}]}
  Blocks until the manager has indexed the corpus, then returns
  {"predictions": [{"status": "loaded"}]}.
- Poll (optional, for legacy clients): POST /nlp with {"instances":[{"poll":"true"}]}
  -> {"predictions": [{"status": "<state>"}]}
- QA: POST /nlp with {"instances":[{"question":"..."}, ...]}
  -> {"predictions": [{"documents":["DOC-0001",...], "answer":"..."}, ...]}

We deliberately block on the load request rather than handing back a "loading"
placeholder. The new eval shape is dict-shaped everywhere; a bare-string
intermediate would score 0 on any client that doesn't poll. BGE-small + a
moderate corpus indexes in well under a request-timeout's worth of seconds on
GPU, so synchronous is the safer default.
"""

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, Request
from nlp_manager import NLPManager

app = FastAPI()
manager = NLPManager()
logger = logging.getLogger(__name__)


class _LoadState:
    def __init__(self) -> None:
        self.status: str = "idle"  # idle | loading | loaded | failed
        self.error: Optional[str] = None
        self.lock = asyncio.Lock()


load_state = _LoadState()


def _do_load(documents) -> bool:
    manager.load_corpus(documents)
    return manager.loaded


@app.post("/nlp")
async def nlp(request: Request) -> dict:
    inputs_json = await request.json()
    instances = inputs_json["instances"]
    first = instances[0]

    # Corpus load: block until indexed, return dict on success.
    if first.get("documents") is not None:
        async with load_state.lock:
            if load_state.status != "loaded":
                load_state.status = "loading"
                try:
                    ok = await asyncio.to_thread(_do_load, first["documents"])
                    load_state.status = "loaded" if ok else "failed"
                except Exception as e:
                    logger.exception("Corpus load failed")
                    load_state.status = "failed"
                    load_state.error = str(e)
        return {"predictions": [{"status": load_state.status}]}

    # Poll (legacy clients).
    if first.get("poll") is not None:
        return {"predictions": [{"status": load_state.status}]}

    predictions = await asyncio.to_thread(
        manager.qa_batch, [inst["question"] for inst in instances]
    )
    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    return {"message": "health ok"}
