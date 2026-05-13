"""Runs the AE server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


from ae_manager import AEManager
from fastapi import FastAPI, Request


def _make_manager():
    """Prefer the learned policy if a checkpoint is present; fall back to the
    heuristic planner. Failures during policy load are logged and degrade
    gracefully so a missing/broken checkpoint can't take the service down."""

    try:
        from policy_manager import PolicyAEManager  # noqa: WPS433 (local import)
        return PolicyAEManager()
    except FileNotFoundError as exc:
        print(f"AE: no policy checkpoint — using heuristic planner ({exc})")
    except Exception as exc:  # noqa: BLE001
        print(f"AE: policy load failed — falling back to heuristic planner ({exc!r})")
    return AEManager()


app = FastAPI()
manager = _make_manager()


@app.post("/ae")
async def ae(request: Request) -> dict[str, list[dict[str, int]]] | dict[str, str]:
    """Feeds an observation into the AE model.

    Returns action taken given current observation (int). Empty POSTs are treated
    as a reset signal because the provided local test currently sends one.
    """

    try:
        input_json = await request.json()
    except Exception:
        await reset(request)
        return {"message": "reset ok"}

    predictions = []
    # each is a dict with one key "observation" and the value as a dictionary observation
    for instance in input_json.get("instances", []):
        observation = instance["observation"]
        # reset environment on a new round
        if observation.get("step") == 0:
            await reset(request)
        predictions.append({"action": manager.ae(observation)})
    return {"predictions": predictions}


@app.post("/reset")
@app.get("/reset")
async def reset(_: Request) -> None:
    """Resets the `AEManager` for a new round."""

    # The Docker container is not restarted between rounds (during Qualifiers).
    # Your model is reset via this endpoint by creating a new instance. You
    # should avoid storing persistent state information outside your
    # `AEManager` instance; but if you must, you should also reset it here.

    global manager  # pylint: disable=global-statement
    manager = _make_manager()

    return


@app.get("/health")
def health() -> dict[str, str]:
    """Health check function for your model."""
    return {"message": "health ok"}
