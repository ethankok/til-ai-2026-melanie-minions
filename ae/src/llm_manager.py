"""LLM-as-player AE manager. EXPERIMENTAL.

Calls Anthropic's Claude (default: claude-sonnet-4-6) once per tick with a
textual encoding of the AE observation, parses an integer action 0-5, and
falls back to a safe legal action on parse/API failure. Latency is in the
seconds-per-tick range; this is for local-sim experiments only — it cannot
be shipped to the eval container which has no internet.

Backends (set via AE_LLM_BACKEND):
  sdk     : Use the `anthropic` Python SDK. Requires ANTHROPIC_API_KEY.
            Supports prompt caching → fastest/cheapest for Anthropic-native.
  pioneer : Use the Pioneer OpenAI-compatible gateway via urllib (no extra
            SDK dependency). Requires PIONEER_API_KEY.
  agy     : Shell out to `agy -p ...` (Gemini 3.5 Flash via subscription).
            No API key, no per-call cost. ~15s/call due to subprocess
            cold-start. Use this for budget-bound data collection.
  cli     : Shell out to `claude -p --model ...` Uses local Claude Code
            OAuth auth. No API key needed. Slowest (subprocess per call).

Env vars:
  AE_LLM_BACKEND       "sdk" | "pioneer" | "agy" | "cli"  (default "pioneer")
  ANTHROPIC_API_KEY    required for sdk backend
  PIONEER_API_KEY      required for pioneer backend
  AE_LLM_MODEL         default "claude-sonnet-4-6"
  AE_LLM_MAX_TOKENS    default 96
  AE_LLM_LOG_FILE      optional, append per-tick JSON traces here
  AE_LLM_CLI_PATH      override claude CLI path (default "claude")
  AE_LLM_CLI_TIMEOUT   subprocess timeout in seconds (default 60)
  AE_LLM_AGY_PATH      override agy CLI path (default "agy")
  AE_LLM_AGY_TIMEOUT   agy subprocess timeout in seconds (default 120)
  AE_LLM_PIONEER_URL   override Pioneer base URL (default https://api.pioneer.ai/v1)
  AE_LLM_HTTP_TIMEOUT  http timeout in seconds (default 60)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import anthropic
except ImportError as e:  # pragma: no cover
    anthropic = None  # type: ignore
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


# Viewcone channels — mirror til_environment.observation.ViewChannel
CH_VISIBLE = 0
CH_WALL_R, CH_WALL_D, CH_WALL_L, CH_WALL_U = 1, 2, 3, 4
CH_TILE_EMPTY = 5
CH_TILE_RECON, CH_TILE_MISSION, CH_TILE_RESOURCE = 6, 7, 8
CH_ALLY_AGENT, CH_ENEMY_AGENT = 9, 10
CH_ALLY_BASE, CH_ENEMY_BASE = 11, 12
CH_DESTR_R, CH_DESTR_D, CH_DESTR_L, CH_DESTR_U = 13, 14, 15, 16
CH_ALLY_BOMB, CH_ENEMY_BOMB = 17, 18
CH_ALLY_BOMB_TIMER, CH_ENEMY_BOMB_TIMER = 19, 20

DIR_NAMES = {0: "RIGHT(+x)", 1: "DOWN(+y)", 2: "LEFT(-x)", 3: "UP(-y)"}
ACTION_NAMES = ["FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT", "STAY", "PLACE_BOMB"]

# Agent's position inside its egocentric 7x5 viewcone.
# vision config: ahead=4, behind=2, left=2, right=2.
# Row layout: i=0..6 maps to (i - 2) world rows ahead-of-agent in agent's facing frame
#   row 0..1 = behind, row 2 = agent's row, row 3..6 = ahead.
# Col layout: j=0..4 maps to (j - 2). col 2 = agent col; col 0..1 = left; col 3..4 = right.
SELF_ROW, SELF_COL = 2, 2

# Direction constants (mirror AEManager)
DIR_RIGHT, DIR_DOWN, DIR_LEFT, DIR_UP = 0, 1, 2, 3


def _relative_to_world(loc_xy: tuple[int, int], direction: int, forward: int, side: int) -> tuple[int, int]:
    """Egocentric (forward, side) → world (x, y). Mirrors AEManager._relative_to_world."""
    x, y = loc_xy
    if direction == DIR_RIGHT:
        return (x + forward, y + side)
    if direction == DIR_DOWN:
        return (x - side, y + forward)
    if direction == DIR_LEFT:
        return (x - forward, y - side)
    return (x + side, y - forward)  # DIR_UP


SYSTEM_PROMPT = """You are playing the AE task in TIL-AI 2026, a 16x16 grid bomberman variant. You are a HIGHLY AGGRESSIVE bomber. Conservative play loses this game.

GOAL: maximize total reward over 200 steps. Rewards: destroy enemy base = +50, kill enemy = +15, collect mission = +5, resource = +2, recon = +1. Penalties: own base destroyed = -50, taking damage is negative.

CRITICAL STRATEGIC PRIORS (read carefully — these come from measured game data):
- Your own base WILL be destroyed by the enemy roughly every game (-50). You CANNOT prevent this reliably. Do not waste many ticks defending; spend them on offense.
- The strong baseline bot in this game places ~19 bombs per 200-step round. If you place <10, you are leaving 100s of points on the table. Bomb aggressively.
- Per-game maximum-score plays farm bombs: place a bomb, walk 2 cells, repeat. Each bomb that hits an enemy = +15. Each bomb that hits an enemy base = +50. Both happen often if you bomb at every opportunity.
- Trading 1 self-damage event (~-5) for 1 kill (+15) or 1 base hit (+50) is hugely positive EV.

ACTIONS (return ONE integer 0-5):
  0 FORWARD   - move one cell in facing direction (only if no wall blocks)
  1 BACKWARD  - move one cell opposite of facing (only if no wall blocks)
  2 TURN_LEFT - rotate 90 degrees counter-clockwise (does NOT move)
  3 TURN_RIGHT- rotate 90 degrees clockwise (does NOT move)
  4 STAY      - no-op (almost always a wasted tick — strongly avoid)
  5 PLACE_BOMB- drop a bomb at current cell (only if team_bombs > 0). Bomb fuse = 3 ticks. Blast radius = 2 in each cardinal direction (cross-shape). Damages walls, enemies, bases, and YOU/your base if in blast.

AGGRESSIVE PLAYBOOK (default to these):
- If team_bombs > 0 AND any visible enemy agent or enemy base is within 2 cells in any cardinal direction: PLACE_BOMB (5). The blast will hit them.
- If team_bombs > 0 AND any destructible wall ('D' in any direction flag) is adjacent: PLACE_BOMB (5). Breaks wall, may chain into enemy/items behind.
- If a mission ('m') or resource ('r') is visible in ahead-1 row: FORWARD (0) to grab it.
- If FORWARD is blocked but the path to a visible item turns: turn toward it (LEFT 2 or RIGHT 3).
- Use BACKWARD (1) only to escape your own bomb's blast radius (fuse 3 → you have 3 ticks to get 2 cells away).
- STAY (4) is almost always wrong. Pick movement or bomb instead.

LEGALITY RULES:
- You MUST respect the action_mask. Illegal actions are silently rejected and waste the tick.
- Never bomb without escape path: after PLACE_BOMB, you need 3 ticks to get 2+ cells away or behind a wall. If trapped, bomb anyway only if you'd kill an enemy/base in the trade.
- Movement is fail-on-wall. If FORWARD blocked, turn or bomb the destructible wall.

OBSERVATION FORMAT:
You'll see scalars (location/direction/health/bombs/base_health/legal-actions/step), then two viewcone grids:
  AGENT_VIEW (7 rows x 5 cols, egocentric, rendered so AHEAD is at the TOP: the row labeled 'ahead-4' is 4 cells in front of you, 'YOU' row is your tile, 'behind-2' is 2 cells behind. Col left-2..right-2.). FORWARD action moves you UP one row in this rendering.
  BASE_VIEW (square grid, centered on your team base. world-frame: +y is DOWN, +x is RIGHT). Used to spot enemies/bombs threatening your base.

Each cell token is 6 chars: [item][agent][wallR][wallD][wallL][wallU]
  item: ? unseen, . visible-empty, m mission, r resource, c recon, x enemy-base, b ally-base
  agent: . none, A ally, E enemy
  wall: - none, # solid, D destructible
Bombs are listed separately after the grid as: (row,col) ally|enemy timer=T

CONTINUITY:
You will see a MEMORY block each tick containing:
  - your last few actions
  - a PLAN you set in a previous tick (your stated goal)
  - a BELIEF MAP of items, enemy bases, and recently-seen enemies in WORLD coords
Use this to stay on plan and avoid re-exploring. If your stated plan no longer makes sense given the current observation, set a new one.

OUTPUT FORMAT (STRICT):
Reason in AT MOST one short sentence (under 20 words), then emit on the NEXT line your plan, then the action:

<plan>brief 6-12 word goal — e.g. "go grab mission at (12,9) then bomb enemy at (5,3)"</plan>
<action>N</action>

N must be one digit 0-5 and legal per action_mask. Stop after </action>. The PLAN persists into next tick — keep it stable unless invalidated."""


def _scalar(obs: dict, key: str, default: float = 0.0) -> float:
    v = obs.get(key, default)
    if isinstance(v, list):
        v = v[0] if v else default
    if hasattr(v, "item"):
        v = v.item()
    try:
        return float(v)
    except Exception:
        return float(default)


def _cell_token(cell: list) -> tuple[str, str | None]:
    """Encode one viewcone cell as 6-char token + optional bomb side-info.

    Returns (token, bomb_info_or_None). bomb_info is e.g. "ally t=2" or "enemy t=1".
    """
    if not isinstance(cell, (list, tuple)):
        return "??----", None

    def on(ch: int) -> bool:
        try:
            return float(cell[ch]) > 0.0
        except (IndexError, TypeError, ValueError):
            return False

    def val(ch: int) -> float:
        try:
            return float(cell[ch])
        except (IndexError, TypeError, ValueError):
            return 0.0

    # Item char (priority: bases > items > empty > unseen)
    if on(CH_ENEMY_BASE):
        item = "x"
    elif on(CH_ALLY_BASE):
        item = "b"
    elif on(CH_TILE_MISSION):
        item = "m"
    elif on(CH_TILE_RESOURCE):
        item = "r"
    elif on(CH_TILE_RECON):
        item = "c"
    elif on(CH_TILE_EMPTY):
        item = "."
    elif on(CH_VISIBLE):
        item = "."  # visible with no token = empty
    else:
        item = "?"

    # Agent
    if on(CH_ENEMY_AGENT):
        agent = "E"
    elif on(CH_ALLY_AGENT):
        agent = "A"
    else:
        agent = "."

    # Walls — destructible variant overrides solid
    def wall_char(solid_ch: int, destr_ch: int) -> str:
        if on(destr_ch):
            return "D"
        if on(solid_ch):
            return "#"
        return "-"

    wr = wall_char(CH_WALL_R, CH_DESTR_R)
    wd = wall_char(CH_WALL_D, CH_DESTR_D)
    wl = wall_char(CH_WALL_L, CH_DESTR_L)
    wu = wall_char(CH_WALL_U, CH_DESTR_U)

    token = item + agent + wr + wd + wl + wu

    # Bombs
    bomb_info = None
    if on(CH_ALLY_BOMB):
        t = int(val(CH_ALLY_BOMB_TIMER))
        bomb_info = f"ally t={t}"
    elif on(CH_ENEMY_BOMB):
        t = int(val(CH_ENEMY_BOMB_TIMER))
        bomb_info = f"enemy t={t}"

    return token, bomb_info


def _render_viewcone(
    view: list,
    label: str,
    self_marker: tuple[int, int] | None = None,
    flip_rows: bool = False,
    row_labels: dict[int, str] | None = None,
) -> str:
    """Render a viewcone as an ASCII grid with optional 'YOU' marker.

    flip_rows: if True, render with high-row-index at top (useful for
    egocentric views where row N=ahead and we want ahead-at-top).
    """
    if not view:
        return f"{label}: (empty)\n"

    rows = len(view)
    cols = len(view[0]) if rows else 0
    lines: list[str] = []
    bombs: list[str] = []

    lines.append(f"{label} ({rows}x{cols}):")
    header = "         " + " ".join(f"c{j:<5}" for j in range(cols))
    lines.append(header)

    row_order = range(rows - 1, -1, -1) if flip_rows else range(rows)
    for i in row_order:
        cells: list[str] = []
        for j in range(cols):
            tok, bomb = _cell_token(view[i][j])
            if self_marker is not None and (i, j) == self_marker:
                tok = tok[0] + "Y" + tok[2:]
            cells.append(tok)
            if bomb is not None:
                bombs.append(f"  (r{i},c{j}) {bomb}")
        tag = (row_labels or {}).get(i, f"r{i}")
        lines.append(f"{tag:>8}: " + " ".join(cells))

    if bombs:
        lines.append("bombs:")
        lines.extend(bombs)
    lines.append("")
    return "\n".join(lines)


def encode_observation(obs: dict) -> str:
    """Render an AE observation into a compact text prompt block."""
    step = int(_scalar(obs, "step"))
    loc = obs.get("location", [0, 0])
    base_loc = obs.get("base_location", [0, 0])
    if hasattr(loc, "tolist"):
        loc = loc.tolist()
    if hasattr(base_loc, "tolist"):
        base_loc = base_loc.tolist()
    direction = int(_scalar(obs, "direction"))
    health = _scalar(obs, "health")
    base_health = _scalar(obs, "base_health")
    team_bombs = int(_scalar(obs, "team_bombs"))
    team_resources = _scalar(obs, "team_resources")
    frozen = int(_scalar(obs, "frozen_ticks"))

    mask = obs.get("action_mask", [1, 1, 1, 1, 1, 1])
    if hasattr(mask, "tolist"):
        mask = mask.tolist()
    legal = [int(m) for m in mask]
    legal_str = ", ".join(
        f"{i}:{ACTION_NAMES[i]}={'YES' if legal[i] else 'NO'}"
        for i in range(min(6, len(legal)))
    )

    head = (
        f"STEP {step}/200\n"
        f"location=({loc[0]},{loc[1]}) world coords (x,y); +x=RIGHT, +y=DOWN\n"
        f"direction={direction} ({DIR_NAMES.get(direction, '?')})\n"
        f"health={health:.0f}/60   base_health={base_health:.0f}/100   base_location=({base_loc[0]},{base_loc[1]})\n"
        f"team_bombs={team_bombs}   team_resources={team_resources:.0f}   frozen_ticks={frozen}\n"
        f"legal_actions: {legal_str}\n"
    )

    agent_view = obs.get("agent_viewcone") or []
    base_view = obs.get("base_viewcone") or []
    if hasattr(agent_view, "tolist"):
        agent_view = agent_view.tolist()
    if hasattr(base_view, "tolist"):
        base_view = base_view.tolist()

    agent_row_labels = {
        6: "ahead-4",
        5: "ahead-3",
        4: "ahead-2",
        3: "ahead-1",
        2: "  YOU  ",
        1: "behind-1",
        0: "behind-2",
    }
    agent_block = _render_viewcone(
        agent_view,
        "AGENT_VIEW (egocentric, AHEAD at top; col left-2..right-2; FORWARD moves you UP one row)",
        self_marker=(SELF_ROW, SELF_COL),
        flip_rows=True,
        row_labels=agent_row_labels,
    )
    base_block = _render_viewcone(
        base_view,
        "BASE_VIEW (centered on team base. Row index increases with world +x, col index increases with world +y. Center cell = base. Agent cell is at row=base_radius+(your_x-base_x), col=base_radius+(your_y-base_y))",
        self_marker=None,
    )

    return head + "\n" + agent_block + "\n" + base_block


def _first_legal(mask: list[int]) -> int:
    for i, m in enumerate(mask):
        if int(m):
            return i
    return 4  # STAY as last resort


class BeliefMemory:
    """Reusable per-game belief + history + plan state.

    Maintains:
      - belief_items: world-coord -> (kind, last_seen_step)
      - belief_enemy_bases: set of world coords
      - belief_enemies: world-coord -> last_seen_step (stale >3 ticks)
      - belief_visited: set of cells the agent has stood on
      - history: rolling last-K (step, action)
      - plan: free-form text persisted across ticks
      - prev_observation: for last-tick delta detection

    Used by LLMAEManager (player mode) and the annotator collector.
    """

    def __init__(self, memory_window: int = 5) -> None:
        self.memory_window = memory_window
        self.reset()

    def reset(self) -> None:
        self.belief_items: dict[tuple[int, int], tuple[str, int]] = {}
        self.belief_enemy_bases: set[tuple[int, int]] = set()
        self.belief_enemies: dict[tuple[int, int], int] = {}
        self.belief_visited: set[tuple[int, int]] = set()
        self.history: list[dict] = []
        self.plan: str = ""
        self.prev_observation: dict | None = None

    def update(self, obs: dict) -> None:
        loc = obs.get("location") or [0, 0]
        if hasattr(loc, "tolist"):
            loc = loc.tolist()
        direction = int(_scalar(obs, "direction"))
        step = int(_scalar(obs, "step"))
        view = obs.get("agent_viewcone") or []
        if hasattr(view, "tolist"):
            view = view.tolist()
        self.belief_visited.add((int(loc[0]), int(loc[1])))

        for i, row in enumerate(view):
            for j, cell in enumerate(row):
                if not isinstance(cell, (list, tuple)):
                    continue
                try:
                    if float(cell[CH_VISIBLE]) <= 0.0:
                        continue
                except (IndexError, TypeError, ValueError):
                    continue
                forward = i - SELF_ROW
                side = j - SELF_COL
                world = _relative_to_world((int(loc[0]), int(loc[1])), direction, forward, side)
                if not (0 <= world[0] < 16 and 0 <= world[1] < 16):
                    continue

                def _on(ch: int) -> bool:
                    try:
                        return float(cell[ch]) > 0.0
                    except (IndexError, TypeError, ValueError):
                        return False

                if _on(CH_TILE_MISSION):
                    self.belief_items[world] = ("mission", step)
                elif _on(CH_TILE_RESOURCE):
                    self.belief_items[world] = ("resource", step)
                elif _on(CH_TILE_RECON):
                    self.belief_items[world] = ("recon", step)
                elif _on(CH_TILE_EMPTY) and world in self.belief_items:
                    self.belief_items.pop(world, None)

                if _on(CH_ENEMY_BASE):
                    self.belief_enemy_bases.add(world)
                if _on(CH_ENEMY_AGENT):
                    self.belief_enemies[world] = step

        stale_cutoff = step - 3
        self.belief_enemies = {p: t for p, t in self.belief_enemies.items() if t >= stale_cutoff}

    def format_block(self, obs: dict) -> str:
        loc = obs.get("location") or [0, 0]
        if hasattr(loc, "tolist"):
            loc = loc.tolist()

        lines: list[str] = ["MEMORY:"]
        plan = self.plan.strip() if self.plan else "(none yet — set one)"
        lines.append(f"  current_plan: {plan}")

        if self.history:
            recent = ", ".join(
                f"s{h['step']}:{ACTION_NAMES[h['action']][0:3]}"
                for h in self.history[-self.memory_window:]
            )
            lines.append(f"  recent_actions: {recent}")
        else:
            lines.append(f"  recent_actions: (start of game)")

        if self.prev_observation is not None:
            prev = self.prev_observation
            dh = _scalar(obs, "health") - _scalar(prev, "health")
            db = _scalar(obs, "base_health") - _scalar(prev, "base_health")
            dr = _scalar(obs, "team_resources") - _scalar(prev, "team_resources")
            tag_bits: list[str] = []
            if dh < 0:
                tag_bits.append(f"took_damage({dh:+.0f})")
            if db < 0:
                tag_bits.append(f"base_damaged({db:+.0f})")
            if dr > 0:
                tag_bits.append(f"gained_resources(+{dr:.0f})")
            lines.append(f"  last_tick_signals: {', '.join(tag_bits) if tag_bits else 'no change'}")

        items_by_kind: dict[str, list[tuple[int, int]]] = {"mission": [], "resource": [], "recon": []}
        for pos, (kind, _) in self.belief_items.items():
            items_by_kind.setdefault(kind, []).append(pos)
        for kind, positions in items_by_kind.items():
            if positions:
                positions.sort(key=lambda p: abs(p[0] - loc[0]) + abs(p[1] - loc[1]))
                shown = ", ".join(f"({p[0]},{p[1]})" for p in positions[:8])
                lines.append(f"  known_{kind}: {shown}")

        if self.belief_enemy_bases:
            shown = ", ".join(f"({p[0]},{p[1]})" for p in sorted(self.belief_enemy_bases))
            lines.append(f"  known_enemy_bases: {shown}")
        if self.belief_enemies:
            shown = ", ".join(f"({p[0]},{p[1]})@s{t}" for p, t in sorted(self.belief_enemies.items()))
            lines.append(f"  recent_enemies_seen: {shown}")

        lines.append(f"  cells_visited_so_far: {len(self.belief_visited)}/256")
        return "\n".join(lines) + "\n"

    def record_action(self, obs: dict, action: int) -> None:
        step = int(_scalar(obs, "step"))
        self.history.append({"step": step, "action": action})
        if len(self.history) > self.memory_window * 2:
            self.history = self.history[-self.memory_window:]
        self.prev_observation = obs

    def set_plan(self, plan: str) -> None:
        self.plan = plan[:200].strip()


class LLMAEManager:
    """AE manager that delegates each tick to an LLM call."""

    def __init__(self, model: str | None = None, max_tokens: int | None = None) -> None:
        self.model = model or os.environ.get("AE_LLM_MODEL", "claude-sonnet-4-6")
        self.max_tokens = max_tokens or int(os.environ.get("AE_LLM_MAX_TOKENS", "96"))
        self.backend = os.environ.get("AE_LLM_BACKEND", "pioneer").strip().lower()
        self.http_timeout = float(os.environ.get("AE_LLM_HTTP_TIMEOUT", "60"))

        self.client = None
        if self.backend == "sdk":
            if anthropic is None:
                raise ImportError(
                    f"anthropic SDK not available ({_IMPORT_ERR}). "
                    "Install with: pip install anthropic"
                )
            self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        elif self.backend == "pioneer":
            self.pioneer_key = os.environ.get("PIONEER_API_KEY")
            if not self.pioneer_key:
                raise RuntimeError("PIONEER_API_KEY env var not set")
            self.pioneer_url = os.environ.get(
                "AE_LLM_PIONEER_URL", "https://api.pioneer.ai/v1"
            ).rstrip("/")
        elif self.backend == "cli":
            self.cli_path = os.environ.get("AE_LLM_CLI_PATH", "claude")
            self.cli_timeout = float(os.environ.get("AE_LLM_CLI_TIMEOUT", "60"))
        elif self.backend == "agy":
            self.agy_path = os.environ.get("AE_LLM_AGY_PATH", "agy")
            self.agy_timeout = float(os.environ.get("AE_LLM_AGY_TIMEOUT", "120"))
        else:
            raise ValueError(
                f"Unknown AE_LLM_BACKEND={self.backend!r}; use 'sdk', 'pioneer', 'agy', or 'cli'"
            )

        log_path = os.environ.get("AE_LLM_LOG_FILE")
        self.log_path: Path | None = Path(log_path) if log_path else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Diagnostics
        self.calls = 0
        self.fallbacks_parse = 0
        self.fallbacks_illegal = 0
        self.fallbacks_api = 0
        self.total_latency = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_creation_tokens = 0

        # Stateful memory across ticks. Reset by _reset_memory().
        self._last_action: int = 4
        self._plan: str = ""
        self._history: list[dict] = []  # rolling last-K (step, action, reason)
        self._belief_items: dict[tuple[int, int], tuple[str, int]] = {}  # (x,y) -> (kind, last_seen_step)
        self._belief_enemy_bases: set[tuple[int, int]] = set()
        self._belief_enemies: dict[tuple[int, int], int] = {}  # (x,y) -> last_seen_step
        self._belief_visited: set[tuple[int, int]] = set()
        self._prev_observation: dict | None = None
        self._memory_window = 5

    def _reset_memory(self) -> None:
        self._last_action = 4
        self._plan = ""
        self._history = []
        self._belief_items = {}
        self._belief_enemy_bases = set()
        self._belief_enemies = {}
        self._belief_visited = set()
        self._prev_observation = None

    def _update_belief(self, obs: dict) -> None:
        """Walk the agent_viewcone, project visible cells to world coords, log items/enemies."""
        loc = obs.get("location") or [0, 0]
        if hasattr(loc, "tolist"):
            loc = loc.tolist()
        direction = int(_scalar(obs, "direction"))
        step = int(_scalar(obs, "step"))
        view = obs.get("agent_viewcone") or []
        if hasattr(view, "tolist"):
            view = view.tolist()
        self._belief_visited.add((int(loc[0]), int(loc[1])))

        for i, row in enumerate(view):
            for j, cell in enumerate(row):
                if not isinstance(cell, (list, tuple)):
                    continue
                try:
                    if float(cell[CH_VISIBLE]) <= 0.0:
                        continue
                except (IndexError, TypeError, ValueError):
                    continue
                forward = i - SELF_ROW
                side = j - SELF_COL
                world = _relative_to_world((int(loc[0]), int(loc[1])), direction, forward, side)
                if not (0 <= world[0] < 16 and 0 <= world[1] < 16):
                    continue

                def _on(ch: int) -> bool:
                    try:
                        return float(cell[ch]) > 0.0
                    except (IndexError, TypeError, ValueError):
                        return False

                if _on(CH_TILE_MISSION):
                    self._belief_items[world] = ("mission", step)
                elif _on(CH_TILE_RESOURCE):
                    self._belief_items[world] = ("resource", step)
                elif _on(CH_TILE_RECON):
                    self._belief_items[world] = ("recon", step)
                elif _on(CH_TILE_EMPTY) and world in self._belief_items:
                    self._belief_items.pop(world, None)

                if _on(CH_ENEMY_BASE):
                    self._belief_enemy_bases.add(world)
                if _on(CH_ENEMY_AGENT):
                    self._belief_enemies[world] = step
        # Drop enemy positions not seen for >3 ticks.
        stale_cutoff = step - 3
        self._belief_enemies = {p: t for p, t in self._belief_enemies.items() if t >= stale_cutoff}

    def _memory_block(self, obs: dict) -> str:
        loc = obs.get("location") or [0, 0]
        if hasattr(loc, "tolist"):
            loc = loc.tolist()
        step = int(_scalar(obs, "step"))

        lines: list[str] = ["MEMORY:"]
        plan = self._plan.strip() if self._plan else "(none yet — set one)"
        lines.append(f"  current_plan: {plan}")

        if self._history:
            recent = ", ".join(
                f"s{h['step']}:{ACTION_NAMES[h['action']][0:3]}" for h in self._history[-self._memory_window:]
            )
            lines.append(f"  recent_actions: {recent}")
        else:
            lines.append(f"  recent_actions: (start of game)")

        if self._prev_observation is not None:
            prev = self._prev_observation
            dh = _scalar(obs, "health") - _scalar(prev, "health")
            db = _scalar(obs, "base_health") - _scalar(prev, "base_health")
            dr = _scalar(obs, "team_resources") - _scalar(prev, "team_resources")
            tag_bits: list[str] = []
            if dh < 0:
                tag_bits.append(f"took_damage({dh:+.0f})")
            if db < 0:
                tag_bits.append(f"base_damaged({db:+.0f})")
            if dr > 0:
                tag_bits.append(f"gained_resources(+{dr:.0f})")
            lines.append(f"  last_tick_signals: {', '.join(tag_bits) if tag_bits else 'no change'}")

        items_by_kind: dict[str, list[tuple[int, int]]] = {"mission": [], "resource": [], "recon": []}
        for pos, (kind, _) in self._belief_items.items():
            items_by_kind.setdefault(kind, []).append(pos)
        for kind, positions in items_by_kind.items():
            if positions:
                # Cap to top 8 by manhattan distance from agent.
                positions.sort(key=lambda p: abs(p[0] - loc[0]) + abs(p[1] - loc[1]))
                shown = ", ".join(f"({p[0]},{p[1]})" for p in positions[:8])
                lines.append(f"  known_{kind}: {shown}")

        if self._belief_enemy_bases:
            shown = ", ".join(f"({p[0]},{p[1]})" for p in sorted(self._belief_enemy_bases))
            lines.append(f"  known_enemy_bases: {shown}")
        if self._belief_enemies:
            shown = ", ".join(f"({p[0]},{p[1]})@s{t}" for p, t in sorted(self._belief_enemies.items()))
            lines.append(f"  recent_enemies_seen: {shown}")

        visited_count = len(self._belief_visited)
        lines.append(f"  cells_visited_so_far: {visited_count}/256")

        return "\n".join(lines) + "\n"

    _PLAN_TAG_RE = re.compile(r"<plan>(.*?)</plan>", re.IGNORECASE | re.DOTALL)

    _ACTION_TAG_RE = re.compile(r"<action>\s*([0-5])\s*</action>", re.IGNORECASE)
    _LAST_DIGIT_RE = re.compile(r"[0-5]")

    def _parse_action(self, text: str) -> int | None:
        if not text:
            return None
        m = self._ACTION_TAG_RE.search(text)
        if m:
            return int(m.group(1))
        # Fallback: last 0-5 digit anywhere in the response (CoT-resistant).
        digits = self._LAST_DIGIT_RE.findall(text)
        if digits:
            return int(digits[-1])
        return None

    def _log(self, record: dict) -> None:
        if self.log_path is None:
            return
        try:
            with self.log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass

    def _call_sdk(self, prompt_body: str) -> tuple[str, bool, dict]:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": prompt_body}],
            )
            blocks = [b for b in resp.content if getattr(b, "type", None) == "text"]
            raw_text = "".join(getattr(b, "text", "") for b in blocks)
            usage: dict[str, Any] = {}
            u = getattr(resp, "usage", None)
            if u is not None:
                usage = {
                    "input_tokens": getattr(u, "input_tokens", 0) or 0,
                    "output_tokens": getattr(u, "output_tokens", 0) or 0,
                    "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                    "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
                }
            return raw_text, True, usage
        except Exception as exc:  # noqa: BLE001
            return f"<api_error: {exc!r}>", False, {}

    def _call_pioneer(self, prompt_body: str) -> tuple[str, bool, dict]:
        payload = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_body},
            ],
        }).encode()
        req = urllib.request.Request(
            f"{self.pioneer_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.pioneer_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.http_timeout) as r:
                body = json.loads(r.read().decode())
            choices = body.get("choices") or []
            if not choices:
                return f"<no_choices: {str(body)[:200]}>", False, {}
            text = choices[0].get("message", {}).get("content", "") or ""
            usage_raw = body.get("usage") or {}
            usage = {
                "input_tokens": usage_raw.get("prompt_tokens", 0) or 0,
                "output_tokens": usage_raw.get("completion_tokens", 0) or 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            }
            return text, True, usage
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode()[:200]
            except Exception:
                pass
            return f"<http_error {e.code}: {err_body}>", False, {}
        except Exception as exc:  # noqa: BLE001
            return f"<pioneer_exception: {exc!r}>", False, {}

    def _call_agy(self, prompt_body: str) -> tuple[str, bool, dict]:
        # agy has no --system-prompt flag; concatenate everything.
        combined = SYSTEM_PROMPT + "\n\n---OBSERVATION---\n\n" + prompt_body
        cmd = [self.agy_path, "-p", combined]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.agy_timeout,
                check=False,
            )
            if result.returncode != 0:
                return f"<agy_error rc={result.returncode}: {result.stderr.strip()[:200]}>", False, {}
            return result.stdout, True, {}
        except subprocess.TimeoutExpired:
            return "<agy_timeout>", False, {}
        except Exception as exc:  # noqa: BLE001
            return f"<agy_exception: {exc!r}>", False, {}

    def _call_cli(self, prompt_body: str) -> tuple[str, bool, dict]:
        cmd = [
            self.cli_path,
            "-p",
            "--model", self.model,
            "--system-prompt", SYSTEM_PROMPT,
            "--output-format", "text",
        ]
        try:
            result = subprocess.run(
                cmd,
                input=prompt_body,
                capture_output=True,
                text=True,
                timeout=self.cli_timeout,
                check=False,
            )
            if result.returncode != 0:
                return f"<cli_error rc={result.returncode}: {result.stderr.strip()[:200]}>", False, {}
            return result.stdout, True, {}
        except subprocess.TimeoutExpired:
            return "<cli_timeout>", False, {}
        except Exception as exc:  # noqa: BLE001
            return f"<cli_exception: {exc!r}>", False, {}

    def ae(self, observation: dict) -> int:
        mask = observation.get("action_mask", [1, 1, 1, 1, 1, 1])
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        legal = [int(m) for m in mask]

        self._update_belief(observation)

        memory_block = self._memory_block(observation)
        obs_text = encode_observation(observation)
        prompt_body = memory_block + "\n" + obs_text

        t0 = time.monotonic()
        if self.backend == "sdk":
            raw_text, api_ok, usage = self._call_sdk(prompt_body)
        elif self.backend == "pioneer":
            raw_text, api_ok, usage = self._call_pioneer(prompt_body)
        elif self.backend == "agy":
            raw_text, api_ok, usage = self._call_agy(prompt_body)
        else:
            raw_text, api_ok, usage = self._call_cli(prompt_body)

        if not api_ok:
            self.fallbacks_api += 1
        dt = time.monotonic() - t0
        self.calls += 1
        self.total_latency += dt
        self.total_input_tokens += usage.get("input_tokens", 0)
        self.total_output_tokens += usage.get("output_tokens", 0)
        self.total_cache_read_tokens += usage.get("cache_read_input_tokens", 0)
        self.total_cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)

        chosen: int | None = None
        if api_ok:
            chosen = self._parse_action(raw_text)
            if chosen is None:
                self.fallbacks_parse += 1

        if chosen is None or not (0 <= chosen <= 5) or not legal[chosen]:
            if chosen is not None and (0 <= chosen <= 5) and not legal[chosen]:
                self.fallbacks_illegal += 1
            chosen = _first_legal(legal)

        plan_match = self._PLAN_TAG_RE.search(raw_text or "")
        if plan_match:
            new_plan = plan_match.group(1).strip()[:200]
            if new_plan:
                self._plan = new_plan

        step = int(_scalar(observation, "step"))
        self._history.append({"step": step, "action": chosen})
        if len(self._history) > self._memory_window * 2:
            self._history = self._history[-self._memory_window:]
        self._prev_observation = observation
        self._last_action = chosen

        self._log({
            "step": step,
            "latency_s": round(dt, 3),
            "raw": raw_text[:200],
            "chosen": chosen,
            "legal": legal,
            "usage": usage,
            "plan": self._plan,
        })
        return chosen

    def diagnostics(self) -> dict:
        return {
            "llm_calls": self.calls,
            "llm_total_latency_s": round(self.total_latency, 2),
            "llm_mean_latency_s": round(self.total_latency / max(1, self.calls), 3),
            "llm_fallbacks_parse": self.fallbacks_parse,
            "llm_fallbacks_illegal": self.fallbacks_illegal,
            "llm_fallbacks_api": self.fallbacks_api,
            "llm_input_tokens": self.total_input_tokens,
            "llm_output_tokens": self.total_output_tokens,
            "llm_cache_read_tokens": self.total_cache_read_tokens,
            "llm_cache_creation_tokens": self.total_cache_creation_tokens,
            "llm_model": self.model,
            "llm_backend": self.backend,
        }
