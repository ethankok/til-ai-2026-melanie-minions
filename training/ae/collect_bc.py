"""Collect frame-stacked (observation, action) pairs from planner-v3b for BC.

Runs the bomberman env with our planner controlling agent 0 and scripted
opponents for the rest. Each saved sample is the **4-frame stacked**
encoding at the planner's step, so the BC dataset matches the v2 policy's
expected input shape.

Usage (legacy npz):
    python training/ae/collect_bc.py --games 200 --out training/ae/data/bc.npz

Usage (streaming memmap, low-RAM):
    python training/ae/collect_bc.py --games 8000 --stream-dir training/ae/data/bc_stream/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from tqdm import trange

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "ae" / "src"))

from ae_manager import AEManager  # noqa: E402
from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402

from encoder import FrameStacker, rasterize_belief  # noqa: E402
from opponents import MixedOpponent, OpponentFn, make_opponent  # noqa: E402


# ---------------------------------------------------------------------------
# Streaming writer
# ---------------------------------------------------------------------------

class _MemmapWriter:
    """Pre-allocated memmap writer for streaming BC collection.

    Pre-sizes all arrays to ``games * SAMPLES_PER_GAME`` and writes each
    sample directly to disk.  RAM stays O(1) regardless of dataset size.
    """

    SAMPLES_PER_GAME = 200  # empirically fixed-length episodes

    def __init__(self, out_dir: Path, games: int, n_frames: int,
                 with_belief: bool,
                 agent_shape, base_shape, belief_shape, scalar_shape) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        n = games * self.SAMPLES_PER_GAME
        self._n_max = n
        self._i = 0
        self._skipped = 0
        self._with_belief = with_belief
        self._n_frames = n_frames

        def _mm(name, dtype, shape):
            return np.lib.format.open_memmap(
                str(out_dir / f"{name}.npy"),
                mode="w+",
                dtype=dtype,
                shape=(n, *shape),
            )

        self._mm_agent  = _mm("agent_views",  np.float16, agent_shape)
        self._mm_base   = _mm("base_views",   np.float16, base_shape)
        self._mm_scalar = _mm("scalars",      np.float16, scalar_shape)
        self._mm_mask   = _mm("action_masks", np.uint8,   (6,))
        self._mm_action = _mm("actions",      np.int8,    ())
        if with_belief:
            self._mm_belief = _mm("beliefs", np.float16, belief_shape)
        else:
            self._mm_belief = None

        # Fields metadata for meta.json
        self._agent_shape  = agent_shape
        self._base_shape   = base_shape
        self._belief_shape = belief_shape
        self._scalar_shape = scalar_shape
        self._out_dir = out_dir

        # Incremental action counter for histogram
        self._action_counts = np.zeros(6, dtype=np.int64)

    def write(self, agent_view, base_view, scalar, action_mask, action,
              belief=None) -> None:  # belief may be passed positionally or as keyword
        """Write one sample directly to disk.  Counts and reports dropped overflow samples."""
        if self._i >= self._n_max:
            if self._skipped == 0:
                warnings.warn(
                    f"collect_bc streaming: more than {self._n_max} samples "
                    f"(games*{self.SAMPLES_PER_GAME}) — extra samples dropped.",
                    stacklevel=2,
                )
            self._skipped += 1
            return
        i = self._i
        self._mm_agent[i]  = agent_view.astype(np.float16)
        self._mm_base[i]   = base_view.astype(np.float16)
        self._mm_scalar[i] = scalar.astype(np.float16)
        self._mm_mask[i]   = action_mask.astype(np.uint8)
        self._mm_action[i] = np.int8(action)
        if self._mm_belief is not None and belief is not None:
            self._mm_belief[i] = belief.astype(np.float16)
        self._action_counts[int(action)] += 1
        self._i += 1

    def flush_and_finalize(self) -> int:
        """Flush memmaps and write meta.json. Returns actual sample count."""
        self._mm_agent.flush()
        self._mm_base.flush()
        self._mm_scalar.flush()
        self._mm_mask.flush()
        self._mm_action.flush()
        if self._mm_belief is not None:
            self._mm_belief.flush()

        fields: dict = {
            "agent_views":  {"shape": list(self._agent_shape),  "store_dtype": "float16"},
            "base_views":   {"shape": list(self._base_shape),   "store_dtype": "float16"},
            "scalars":      {"shape": list(self._scalar_shape), "store_dtype": "float16"},
            "action_masks": {"shape": [6],                      "store_dtype": "uint8"},
            "actions":      {"shape": [],                       "store_dtype": "int8"},
        }
        if self._with_belief:
            fields["beliefs"] = {"shape": list(self._belief_shape), "store_dtype": "float16"}

        if self._skipped > 0:
            print(
                f"[collect_bc] WARNING: dropped {self._skipped} overflow samples "
                f"(>{self._n_max} preallocated) — increase --games sizing",
                file=sys.stderr,
                flush=True,
            )

        meta = {
            "n_frames":    self._n_frames,
            "with_belief": self._with_belief,
            "n_samples":   self._i,
            "skipped":     self._skipped,
            "fields":      fields,
        }
        (self._out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return self._i

    @property
    def action_counts(self) -> np.ndarray:
        return self._action_counts


def _resolve_opponent_names(spec: str) -> list[str]:
    """Mirror of simulate.run_simulation()'s opponent resolution.

    Keeps `collect_bc.py` independent of simulate.py while matching its
    naming conventions ('mixed', 'library', 'cloudsuite', 'pressure2',
    legacy 'random', or an explicit comma-separated list).
    """

    if spec == "random":
        return ["random"] * 5
    if spec == "mixed":
        return ["mixed"] * 5
    if spec == "library":
        return ["greedy", "bomber", "defender", "hunter", "rusher"]
    if spec == "cloudsuite":
        return ["rusher", "hunter_sticky", "bomber_fast", "defender", "mixed"]
    if spec == "pressure2":
        return ["rusher_fast", "rusher_safe", "hunter_sticky", "bomber_fast", "base_bomber"]
    names = [n.strip() for n in spec.split(",") if n.strip()]
    if len(names) == 1:
        names = names * 5
    if len(names) != 5:
        raise ValueError(f"need 5 opponent names (got {len(names)}): {names}")
    return names


def _obs_to_python(obs) -> dict:
    out = {}
    for key, value in obs.items():
        if hasattr(value, "tolist"):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _run_game_loop(
    env,
    our_agent: str,
    other_ids: list[str],
    opponents: list,
    games: int,
    seed: int | None,
    n_frames: int,
    with_belief: bool,
    on_sample,
) -> tuple[int, float]:
    """Shared game loop for both the legacy (npz) and streaming (memmap) collectors.

    For each step where ``our_agent`` acts, ``on_sample`` is called with:
        on_sample(agent_view, base_view, scalar, action_mask, action, belief_or_None)
    All other logic (opponent stepping, mask safety, planner/stacker lifecycle)
    is identical between the two collection paths.

    Returns ``(n_collected, elapsed_seconds)``.
    """
    n_collected = 0
    start = time.time()
    for game in trange(games, desc="games"):
        if seed is not None:
            env.reset(seed=seed + game)
        else:
            env.reset()
        # Per-game reset for opponents that need it (Mixed picks a fresh
        # archetype each game; AEManager-based opponents zero their belief).
        for op in opponents:
            if hasattr(op, "reset_for_game"):
                op.reset_for_game()
            if hasattr(op, "_reset_memory"):
                op._reset_memory()
        planner = AEManager()
        stacker = FrameStacker(n_frames)

        for agent in env.agent_iter():
            obs, _reward, termination, truncation, _info = env.last()
            if termination or truncation:
                env.step(None)
                continue
            if agent == our_agent:
                obs_py = _obs_to_python(obs)
                if obs_py.get("step") == 0:
                    planner = AEManager()
                    stacker.reset()
                # Run the planner — this updates planner.* belief state
                # AND picks the action we'll BC against.
                action = planner.ae(obs_py)
                belief = rasterize_belief(planner, obs_py) if with_belief else None
                stacked = stacker.observe(obs_py, belief_map=belief)
                on_sample(
                    stacked["agent_view"],
                    stacked["base_view"],
                    stacked["scalars"],
                    stacked["action_mask"],
                    int(action),
                    belief,
                )
                n_collected += 1
            else:
                obs_py = _obs_to_python(obs)
                slot = other_ids.index(agent)
                op = opponents[slot]
                try:
                    action = int(op(obs_py))
                except Exception:
                    action = env.action_space(agent).sample()
                # Safety: if the opponent returned an illegal action, fall
                # back to the first legal one (mirrors simulate.py).
                mask = obs_py.get("action_mask")
                if mask is not None:
                    try:
                        if not int(mask[action]):
                            for i, m in enumerate(mask):
                                if int(m):
                                    action = i
                                    break
                    except Exception:
                        pass
            env.step(action)

    return n_collected, time.time() - start


def _print_action_histogram(counts: np.ndarray, total: int) -> None:
    labels = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STAY", "PLACE_BOMB"]
    print("Action distribution:")
    for label, count in zip(labels, counts):
        print(f"  {label:11s} {int(count):>7d}  ({int(count) / max(total, 1) * 100:5.1f}%)")


def collect_dataset(
    games: int,
    out_path: Path,
    novice: bool = True,
    seed: int | None = None,
    n_frames: int = 4,
    with_belief: bool = True,
    opponents_spec: str = "library",
) -> None:
    """Collect a BC dataset from planner-v3b rollouts (legacy in-RAM → npz path).

    When ``with_belief=True`` (default), each sample also includes the
    rasterized belief tensor at the planner's step. Setting it False
    keeps the file size down for legacy single-frame BC training.

    ``opponents_spec`` controls what the other 5 agents do during data
    collection. Use one of 'library' / 'cloudsuite' / 'pressure2' / 'mixed' / a
    5-comma-separated list to expose the planner to the same opponent mix the
    qualifier is likely to use. Legacy 'random' is still accepted for old
    reproductions. Mixing here makes the BC dataset cover a wider state
    distribution, which is how we avoid the
    bc-belief-hybrid failure mode (overfit to planner-vs-random).
    """
    config = default_config()
    config.env.novice = novice
    env = bomberman_env.basic_env(env_wrappers=[], cfg=config)
    our_agent = env.possible_agents[0]
    other_ids = list(env.possible_agents[1:])

    names = _resolve_opponent_names(opponents_spec)
    opponent_seed = seed if seed is not None else 0
    opponents: list[OpponentFn] = [
        make_opponent(n, seed=opponent_seed + 1000 + i) for i, n in enumerate(names)
    ]
    print(
        f"Controlling {our_agent} of {env.possible_agents}; "
        f"n_frames={n_frames}; with_belief={with_belief}; "
        f"opponents_spec={opponents_spec} -> {names}"
    )

    # In-RAM accumulators (legacy path — unchanged behavior)
    agent_views: list[np.ndarray] = []
    base_views: list[np.ndarray] = []
    scalars_list: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    beliefs: list[np.ndarray] = []
    actions: list[int] = []

    def _on_sample(av, bv, sc, am, act, bel):
        agent_views.append(av)
        base_views.append(bv)
        scalars_list.append(sc)
        action_masks.append(am)
        if with_belief and bel is not None:
            beliefs.append(bel)
        actions.append(act)

    n_collected, elapsed = _run_game_loop(
        env, our_agent, other_ids, opponents,
        games, seed, n_frames, with_belief, _on_sample,
    )
    env.close()

    print(
        f"Collected {n_collected:,} samples across {games} games in {elapsed:.1f}s "
        f"({n_collected / max(elapsed, 1):.1f} samples/sec)"
    )

    if not actions:
        raise SystemExit("No samples collected — env probably terminated immediately?")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(
        agent_views=np.stack(agent_views).astype(np.float32),
        base_views=np.stack(base_views).astype(np.float32),
        scalars=np.stack(scalars_list).astype(np.float32),
        action_masks=np.stack(action_masks).astype(np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        n_frames=np.asarray(n_frames, dtype=np.int32),
        with_belief=np.asarray(int(with_belief), dtype=np.int32),
    )
    if with_belief:
        save_kwargs["beliefs"] = np.stack(beliefs).astype(np.float32)
    np.savez_compressed(out_path, **save_kwargs)
    counts = np.bincount(actions, minlength=6)
    print(f"Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Shapes: agent_views={agent_views[0].shape}, base_views={base_views[0].shape}, "
          f"scalars={scalars_list[0].shape}"
          + (f", beliefs={beliefs[0].shape}" if with_belief and beliefs else ""))
    _print_action_histogram(counts, len(actions))


def collect_dataset_streaming(
    games: int,
    stream_dir: Path,
    novice: bool = True,
    seed: int | None = None,
    n_frames: int = 4,
    with_belief: bool = True,
    opponents_spec: str = "library",
    overwrite: bool = False,
) -> None:
    """Collect a BC dataset using disk-backed memmaps (low-RAM streaming path).

    Samples are written directly to ``stream_dir``/*.npy on every step — no
    end-of-run np.stack.  RAM stays O(1) in ``games``.  Suitable for 8000-game
    runs on an 8 GB machine.

    Output layout::

        stream_dir/
          agent_views.npy   float16  (games*200, 100, 7, 5)
          base_views.npy    float16  (games*200, 100, 7, 7)
          beliefs.npy       float16  (games*200, 11, 16, 16)  -- if with_belief
          scalars.npy       float16  (games*200, 68)
          action_masks.npy  uint8    (games*200, 6)
          actions.npy       int8     (games*200,)
          meta.json
    """
    # C2: refuse to silently clobber an existing stream dir unless --overwrite is set.
    _existing = stream_dir.is_dir() and (
        list(stream_dir.glob("*.npy")) or (stream_dir / "meta.json").exists()
    )
    if _existing and not overwrite:
        raise SystemExit(
            f"--stream-dir {stream_dir} already has data; "
            "pass --overwrite to replace it or choose a new dir"
        )
    if _existing and overwrite:
        import glob as _glob
        for _f in stream_dir.glob("*.npy"):
            _f.unlink()
        _meta = stream_dir / "meta.json"
        if _meta.exists():
            _meta.unlink()

    config = default_config()
    config.env.novice = novice
    env = bomberman_env.basic_env(env_wrappers=[], cfg=config)
    our_agent = env.possible_agents[0]
    other_ids = list(env.possible_agents[1:])

    names = _resolve_opponent_names(opponents_spec)
    opponent_seed = seed if seed is not None else 0
    opponents: list[OpponentFn] = [
        make_opponent(n, seed=opponent_seed + 1000 + i) for i, n in enumerate(names)
    ]
    print(
        f"Controlling {our_agent} of {env.possible_agents}; "
        f"n_frames={n_frames}; with_belief={with_belief}; "
        f"opponents_spec={opponents_spec} -> {names}"
    )
    print(f"Streaming → {stream_dir}  (pre-allocated for {games} games × "
          f"{_MemmapWriter.SAMPLES_PER_GAME} samples = "
          f"{games * _MemmapWriter.SAMPLES_PER_GAME:,} rows)")

    # We need concrete per-sample shapes to pre-allocate.
    # Run a single dummy reset to get them from the first stacked observation.
    _config2 = default_config()
    _config2.env.novice = novice
    _probe_env = bomberman_env.basic_env(env_wrappers=[], cfg=_config2)
    _probe_env.reset(seed=0)
    _probe_planner = AEManager()
    _probe_stacker = FrameStacker(n_frames)
    _probe_our = _probe_env.possible_agents[0]
    _agent_shape = _base_shape = _belief_shape = _scalar_shape = None
    for _agent in _probe_env.agent_iter():
        _obs, _, _term, _trunc, _ = _probe_env.last()
        if _term or _trunc:
            _probe_env.step(None)
            continue
        if _agent == _probe_our:
            _obs_py = _obs_to_python(_obs)
            _belief = rasterize_belief(_probe_planner, _obs_py) if with_belief else None
            _stacked = _probe_stacker.observe(_obs_py, belief_map=_belief)
            _agent_shape  = _stacked["agent_view"].shape
            _base_shape   = _stacked["base_view"].shape
            _scalar_shape = _stacked["scalars"].shape
            _belief_shape = _belief.shape if _belief is not None else None
            break
        _probe_env.step(0)
    _probe_env.close()

    if _agent_shape is None:
        raise RuntimeError("Could not probe observation shapes from a single env step")

    writer = _MemmapWriter(
        stream_dir, games, n_frames, with_belief,
        _agent_shape, _base_shape, _belief_shape, _scalar_shape,
    )

    n_collected, elapsed = _run_game_loop(
        env, our_agent, other_ids, opponents,
        games, seed, n_frames, with_belief, writer.write,
    )
    env.close()
    n_written = writer.flush_and_finalize()

    print(
        f"Collected {n_collected:,} samples across {games} games in {elapsed:.1f}s "
        f"({n_collected / max(elapsed, 1):.1f} samples/sec)"
    )
    print(f"Written {n_written:,} samples → {stream_dir}")
    print(f"Shapes: agent_views={_agent_shape}, base_views={_base_shape}, "
          f"scalars={_scalar_shape}"
          + (f", beliefs={_belief_shape}" if with_belief and _belief_shape else ""))
    _print_action_histogram(writer.action_counts, n_written)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect BC dataset from planner-v3b rollouts.",
    )
    parser.add_argument("--games", type=int, default=200)

    # Output mode: --out (legacy npz) XOR --stream-dir (memmap directory)
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument(
        "--out", type=str, default=None,
        help="Output .npz path (legacy in-RAM mode). Default when --stream-dir not given.",
    )
    out_group.add_argument(
        "--stream-dir", type=str, default=None,
        help=(
            "Output directory for streaming memmap mode (low-RAM). "
            "Writes float16/uint8/int8 .npy files + meta.json directly to disk; "
            "RAM stays flat regardless of dataset size. "
            "Mutually exclusive with --out."
        ),
    )

    parser.add_argument("--novice", action="store_true", default=True)
    parser.add_argument("--no-novice", dest="novice", action="store_false")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-frames", type=int, default=4)
    parser.add_argument("--no-belief", dest="with_belief", action="store_false",
                        help="Skip belief-map rasterization (legacy single-frame BC)")
    parser.add_argument(
        "--opponents", type=str, default="library",
        help=(
            "Opponent set used for the other 5 agents during data collection. "
            "Same vocabulary as simulate.py: 'mixed', 'library', 'cloudsuite', "
            "'pressure2', legacy 'random', or 5 comma-separated names."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help=(
            "Allow overwriting an existing --stream-dir. "
            "Stale .npy and meta.json files are removed before collection begins. "
            "Without this flag, collecting into a non-empty dir raises an error."
        ),
    )
    parser.set_defaults(with_belief=True)
    args = parser.parse_args()

    if args.stream_dir is not None:
        # --- Streaming memmap path ---
        stream_dir = Path(args.stream_dir)
        if not stream_dir.is_absolute():
            stream_dir = REPO_ROOT / stream_dir
        collect_dataset_streaming(
            args.games,
            stream_dir,
            novice=args.novice,
            seed=args.seed,
            n_frames=args.n_frames,
            with_belief=args.with_belief,
            opponents_spec=args.opponents,
            overwrite=args.overwrite,
        )
    else:
        # --- Legacy npz path (default) ---
        raw_out = args.out if args.out is not None else "training/ae/data/bc.npz"
        out_path = Path(raw_out)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        collect_dataset(
            args.games,
            out_path,
            novice=args.novice,
            seed=args.seed,
            n_frames=args.n_frames,
            with_belief=args.with_belief,
            opponents_spec=args.opponents,
        )


if __name__ == "__main__":
    main()
