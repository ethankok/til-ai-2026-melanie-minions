"""Runs the AE server.

Mode selection is controlled by the ``AE_MODE`` env var:

- ``confidence_policy_hybrid`` (default, deployed): heuristic-first planner;
  the raw PPO policy is consulted only on low-confidence heuristic ticks.
  Falls back to heuristic if no policy checkpoint is present.
- ``policy``: pure neural policy. Falls back to heuristic if no
  checkpoint is present.
- ``option_hybrid``: neural 8-way option selector plus heuristic planner
  execution. Falls back to heuristic if no option checkpoint is present.
- ``tactical_hybrid``: neural 12-way tactical option selector plus planner
  execution. Falls back to heuristic if no tactical checkpoint is present.
- ``opening_hybrid``: divergence-gated opening prefix in front of an inner
  planner selected by ``AE_OPENING_PLANNER``.
- ``heuristic``: pure rule-based planner. No torch import, no model
  load, cheapest possible per-call latency.

Set ``AE_MODE=heuristic`` to ship the planner-v3b-equivalent build for a
clean speed-focused A/B, without rebuilding the image.

(Retired modes ``hybrid``, ``macro_hybrid``, ``confidence_hybrid`` and
``scripted_hybrid`` were removed with their manager modules — see git
history and ae/NOTES.md for the negative results.)
"""


import os
from pathlib import Path

from ae_manager import AEManager
from fastapi import FastAPI, Request


def _read_mode() -> str:
    """Resolve the runtime mode, with three fallbacks:

    1. ``AE_MODE`` env var (set in Dockerfile or by docker run -e).
    2. ``.ae_mode`` file next to this source (handy when ``til build``
       won't let you pass env vars / build args — just write the mode
       into the file and rebuild).
    3. Default ``confidence_policy_hybrid`` (the deployed mode; degrades to
       the heuristic when no policy checkpoint is present).
    """

    mode = os.environ.get("AE_MODE")
    if mode and mode.strip():
        return mode.strip().lower()
    mode_file = Path(__file__).resolve().parent / ".ae_mode"
    if mode_file.exists():
        try:
            content = mode_file.read_text().strip().lower()
            if content:
                return content
        except OSError:
            pass
    return "confidence_policy_hybrid"


def _make_manager():
    """Construct the AE manager described by ``AE_MODE``.

    Always degrades gracefully: if anything in the policy/hybrid path
    fails (missing checkpoint, torch import error, etc.) we fall back to
    the heuristic so the service can't refuse to start."""

    mode = _read_mode()

    if mode == "heuristic":
        print("AE: mode=heuristic — using rule-based planner only")
        return AEManager()

    if mode == "policy":
        try:
            from policy_manager import PolicyAEManager  # noqa: WPS433
            return PolicyAEManager()
        except FileNotFoundError as exc:
            print(f"AE: mode=policy but no checkpoint — falling back to heuristic ({exc})")
        except Exception as exc:  # noqa: BLE001
            print(f"AE: mode=policy load failed — falling back to heuristic ({exc!r})")
        return AEManager()

    if mode == "option_hybrid":
        try:
            from option_hybrid_manager import OptionHybridAEManager  # noqa: WPS433
            print("AE: mode=option_hybrid — option policy + planner execution")
            return OptionHybridAEManager()
        except FileNotFoundError as exc:
            print(f"AE: mode=option_hybrid but no option checkpoint — using heuristic ({exc})")
        except Exception as exc:  # noqa: BLE001
            print(f"AE: mode=option_hybrid init failed — using heuristic ({exc!r})")
        return AEManager()

    if mode == "tactical_hybrid":
        try:
            from tactical_hybrid_manager import TacticalHybridAEManager  # noqa: WPS433
            print("AE: mode=tactical_hybrid — tactical option policy + planner execution")
            return TacticalHybridAEManager()
        except FileNotFoundError as exc:
            print(f"AE: mode=tactical_hybrid but no tactical checkpoint — using heuristic ({exc})")
        except Exception as exc:  # noqa: BLE001
            print(f"AE: mode=tactical_hybrid init failed — using heuristic ({exc!r})")
        return AEManager()

    if mode == "confidence_policy_hybrid":
        try:
            from confidence_policy_hybrid_manager import ConfidencePolicyHybridAEManager  # noqa: WPS433
            print("AE: mode=confidence_policy_hybrid — raw PPO policy consulted only on low-confidence heuristic ticks")
            return ConfidencePolicyHybridAEManager()
        except FileNotFoundError as exc:
            print(f"AE: mode=confidence_policy_hybrid but no policy checkpoint — using heuristic ({exc})")
        except Exception as exc:  # noqa: BLE001
            print(f"AE: mode=confidence_policy_hybrid init failed — using heuristic ({exc!r})")
        return AEManager()

    if mode == "opening_hybrid":
        # Divergence-gated Novice opening prefix in front of a planner. The inner
        # planner is selected by AE_OPENING_PLANNER (default confidence_policy_hybrid
        # so the combined opening+confpol build ships; degrades to the heuristic if
        # the policy checkpoint is absent). The opening only fires on the spawns
        # locked in openings_gate.json; all other spawns run the planner from tick 0.
        try:
            from opening_hybrid_manager import OpeningHybridManager  # noqa: WPS433
            inner_name = os.environ.get("AE_OPENING_PLANNER", "confidence_policy_hybrid").strip().lower()
            planner = None
            if inner_name == "confidence_policy_hybrid":
                try:
                    from confidence_policy_hybrid_manager import ConfidencePolicyHybridAEManager  # noqa: WPS433
                    planner = ConfidencePolicyHybridAEManager()
                    print("AE: opening_hybrid inner planner = confidence_policy_hybrid")
                except Exception as exc:  # noqa: BLE001
                    print(f"AE: opening_hybrid confpol planner failed — using heuristic planner ({exc!r})")
            if planner is None:
                planner = AEManager()
                print("AE: opening_hybrid inner planner = heuristic")
            return OpeningHybridManager(planner=planner)
        except Exception as exc:  # noqa: BLE001
            print(f"AE: opening_hybrid init failed — using heuristic ({exc!r})")
        return AEManager()

    # Unknown mode: fall back to the heuristic so the service can't refuse
    # to start (retired modes from old builds land here too).
    print(f"AE: unknown mode {mode!r} — using rule-based planner only")
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
