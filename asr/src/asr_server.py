"""Runs the ASR server.

Backend selection is via the `ASR_BACKEND` env var:
  - `nemo` (default):    NVIDIA NeMo ASR checkpoint from `asr_manager.py` (shipped)
  - `whisper`:           faster-whisper distil-large-v3 from `asr_manager_fasterwhisper.py`

Defaulting to nemo keeps the shipped image bit-identical when this file is
deployed without the env var set (the Dockerfile also sets ASR_BACKEND=nemo
explicitly). The whisper path is the retained A/B variant so the existing
Dockerfile and build pipeline are unaffected.
"""

import base64
import os

from fastapi import FastAPI, Request

_BACKEND = os.environ.get("ASR_BACKEND", "nemo").lower()

if _BACKEND == "whisper":
    from asr_manager_fasterwhisper import ASRManager as _ManagerCls
else:
    from asr_manager import ASRManager as _ManagerCls

app = FastAPI()
manager = _ManagerCls()


@app.post("/asr")
async def asr(request: Request) -> dict[str, list[str]]:
    """Performs ASR on audio files.

    Args:
        request: The API request. Contains a list of audio files, encoded in
            base-64.

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `str` transcriptions, in the same order as which appears in `request`.
    """

    inputs_json = await request.json()
    audio_bytes_list = [
        base64.b64decode(instance["b64"]) for instance in inputs_json["instances"]
    ]
    predictions = manager.asr_batch(audio_bytes_list)
    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for the server."""
    return {"message": "health ok"}
