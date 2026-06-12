"""Inference-side reader for the offline playbook.

Loads the `.npz` produced by `training/ae/build_playbook.py` once at startup
and exposes a single O(1) lookup: ``Playbook.lookup(location, direction, step)
-> int | None``.

The playbook is a state→action override layer. When it has high-confidence
data for the agent's current bucket, it returns the offline-mined action;
otherwise it returns ``None`` and the heuristic takes over. This is the
Tier-1 #1 implementation — exploits the fact that novice mode runs on a
fixed map (seed 88 for entities, seed 19 for walls), so empirical "best
action at this state" is well defined.

Disable with ``AE_USE_PLAYBOOK=0``. The Dockerfile bakes the playbook in
at ``/workspace/models/playbook.npz``; the local dev path is
``ae/models/playbook.npz``. If neither exists, lookups silently return
``None`` (no playbook → all-heuristic fallback).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


_PLAYBOOK_ENV_PATH = "AE_PLAYBOOK_PATH"
_PLAYBOOK_ENABLE = "AE_USE_PLAYBOOK"


def _candidate_paths() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [
        here / "models" / "playbook.npz",
        here.parent / "models" / "playbook.npz",
        Path("/workspace/models/playbook.npz"),
    ]


def _resolve_path() -> Path | None:
    override = os.environ.get(_PLAYBOOK_ENV_PATH)
    if override:
        p = Path(override)
        return p if p.exists() else None
    for p in _candidate_paths():
        if p.exists():
            return p
    return None


def _flag_enabled() -> bool:
    raw = os.environ.get(_PLAYBOOK_ENABLE, "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def pack_state_key(x: int, y: int, direction: int, step: int) -> int:
    """Mirror of `training/ae/simulate.py:pack_state_key`. Keep in sync."""
    x = int(x) & 0xFF
    y = int(y) & 0xFF
    d = int(direction) & 0xF
    s = int(step) & 0xFFFF
    return (s << 20) | (d << 16) | (y << 8) | x


class Playbook:
    """Loaded playbook with O(1) lookup."""

    def __init__(self, path: Path) -> None:
        data = np.load(path, allow_pickle=False)
        keys = data["keys"]
        actions = data["actions"]
        # Dict lookup; expected size is < 200K so memory cost is tiny.
        self._table: dict[int, int] = {
            int(k): int(a) for k, a in zip(keys, actions)
        }
        self.fallback = float(data["fallback_baseline"]) if "fallback_baseline" in data.files else 0.0
        self.size = len(self._table)

    def lookup(self, location, direction: int, step: int) -> int | None:
        if location is None:
            return None
        try:
            x = int(location[0])
            y = int(location[1])
        except (TypeError, IndexError, ValueError):
            return None
        key = pack_state_key(x, y, direction, step)
        action = self._table.get(key)
        if action is None:
            return None
        # The (x, y, dir, step) key is too coarse for movement actions — different
        # belief states share the same key. Bomb spots are map-geometry-tied
        # regardless of belief, so default to bomb-only. AE_PLAYBOOK_FILTER=all disables this.
        filter_mode = os.environ.get("AE_PLAYBOOK_FILTER", "bomb_only").strip().lower()
        if filter_mode == "all":
            return action
        if filter_mode == "bomb_only":
            return action if action == 5 else None  # 5 == PLACE_BOMB
        return action

    def __len__(self) -> int:
        return self.size


_playbook_cache: Playbook | None = None
_playbook_loaded = False


def get_playbook() -> Playbook | None:
    """Return the loaded playbook, or None if disabled / missing.

    Cached so that multiple AEManager instances share one in-memory copy.
    """
    global _playbook_cache, _playbook_loaded
    if _playbook_loaded:
        return _playbook_cache
    _playbook_loaded = True

    if not _flag_enabled():
        print("[playbook] disabled via AE_USE_PLAYBOOK", flush=True)
        return None

    path = _resolve_path()
    if path is None:
        print("[playbook] no playbook file found; lookups will return None", flush=True)
        return None

    try:
        pb = Playbook(path)
        _playbook_cache = pb
        print(f"[playbook] loaded {len(pb)} entries from {path}", flush=True)
        return pb
    except Exception as exc:  # noqa: BLE001
        print(f"[playbook] load failed ({exc}); proceeding without it", flush=True)
        return None
