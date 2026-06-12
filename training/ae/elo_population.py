"""Elo-rated snapshot population for AE PPO self-play.

Adapted from the 2024 Pommerman work on population-based self-play with
Elo matchmaking (arxiv 2407.00662). The hypothesis we're testing
(24 May 2026): the existing ``--preset full-rl`` league mode samples
frozen-policy opponents uniformly, which makes curriculum progression
brittle — the policy can plateau against opponents that are too easy
(no learning signal) or too hard (no positive trajectories). Elo
matchmaking biases sampling toward "appropriate-difficulty" opponents.

What this module provides:

    * :class:`EloPopulation` — a bounded pool of frozen policy snapshots,
      each with a tracked Elo rating. Sample-matched via Gaussian weight
      ``exp(-((elo - target) / sigma)**2)``.
    * :class:`LiveRating` — running Elo for the in-training policy.
    * Standard chess Elo update: K=32, expected score
      ``1 / (1 + 10**((opp - me) / 400))``.

What it deliberately does NOT do:
    * Track Elo for scripted opponents (rusher/hunter/etc). Their
      strength is roughly fixed; adding Elo would just add noise.
      Scripted opponents stay on the uniform-sample fast path.
    * Snapshot persistence to disk. Snapshots are large (PolicyNetwork
      weights); only the Elo metadata is serialised here. The PolicyNetwork
      objects themselves are held in the SnapshotPool by reference and
      cease to exist when training ends.

Standalone module — no torch imports, no train_ppo imports. Pure Python
data structures + math. Tested in isolation, then wired into SnapshotPool.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class EloEntry:
    """One snapshot's metadata. ``snapshot_id`` is the index in the
    parent pool's internal list at the time of insertion; we use it as
    a stable handle for Elo updates after rollouts."""

    snapshot_id: int
    rating: float
    n_games: int = 0
    added_at_update: int = 0  # bookkeeping for eviction policy


def _expected_score(my_elo: float, opp_elo: float) -> float:
    """Standard Elo expected score for the player rated my_elo."""
    return 1.0 / (1.0 + 10.0 ** ((opp_elo - my_elo) / 400.0))


def elo_delta(my_elo: float, opp_elo: float, my_actual: float, k: float = 32.0) -> float:
    """Symmetric Elo update: returns the delta to apply to my_elo.
    The opponent's delta is exactly the negative of this.

    ``my_actual`` should be in [0, 1]: 1 = clear win, 0 = clear loss,
    0.5 = draw. For AE we map score-margin via a sigmoid in the caller
    so a marginal win produces ~0.6 not ~1.0."""
    expected = _expected_score(my_elo, opp_elo)
    return k * (my_actual - expected)


def score_to_outcome(my_score: float, baseline_score: float, scale: float = 0.10) -> float:
    """Map a normalized game score margin to a [0, 1] outcome value.

    ``my_score`` and ``baseline_score`` are both in normalized-reward
    units (i.e. typical range 0.0-1.0 like the validate_cloud_suite
    output). The sigmoid widens around the baseline so that small
    margins don't move Elo much. Margin > 2*scale = ~88% outcome;
    margin = 0 = 50% outcome; margin < -2*scale = ~12% outcome."""
    margin = (my_score - baseline_score) / max(scale, 1e-6)
    return 1.0 / (1.0 + math.exp(-margin))


@dataclass
class LiveRating:
    """Running Elo for the in-training policy. Initialized at 1200
    (chess-default novice) so that scripted opponents at ~1200-1500 give
    us early uphill battles and produce learning signal even pre-warmup."""

    rating: float = 1200.0
    n_games: int = 0

    def update(self, opp_elo: float, my_actual: float, k: float = 32.0) -> float:
        delta = elo_delta(self.rating, opp_elo, my_actual, k)
        self.rating += delta
        self.n_games += 1
        return delta


class EloPopulation:
    """Bounded pool of frozen policy snapshots, each with an Elo rating.

    Sampling: weighted by Gaussian distance from ``target_elo``. This
    implements the "play against opponents of similar strength" idea
    from the 2024 Pommerman paper. ``sigma`` controls how strict the
    matching is — small sigma = strict (only similar Elo), large sigma
    = lax (approaches uniform).

    Eviction (when at capacity): drop the OLDEST entry strictly below the
    pool's median Elo. This preserves both old-diversity (some weak
    historical anchors) and recency (always have current-ish strong
    opponents). If everyone is at or above median (rare), drop oldest
    period.
    """

    def __init__(self, max_size: int = 8, sigma: float = 200.0, k: float = 32.0):
        self.max_size = max(1, int(max_size))
        self.sigma = float(sigma)
        self.k = float(k)
        self._entries: list[EloEntry] = []
        self._snapshots: list[Any] = []  # type: ignore[type-arg]
        self._next_id = 0

    # ------------------------------------------------------------------
    # Population management
    # ------------------------------------------------------------------
    def add(self, snapshot: Any, initial_elo: float, at_update: int = 0) -> int:
        """Add a snapshot to the pool. Returns the assigned snapshot_id.

        ``initial_elo`` should usually be the current live policy's Elo
        at promotion time — that's the standard self-play recipe."""
        if len(self._entries) >= self.max_size:
            self._evict()
        snap_id = self._next_id
        self._next_id += 1
        self._entries.append(
            EloEntry(snapshot_id=snap_id, rating=float(initial_elo), added_at_update=at_update)
        )
        self._snapshots.append(snapshot)
        return snap_id

    def _evict(self) -> None:
        if not self._entries:
            return
        ratings = sorted(e.rating for e in self._entries)
        median = ratings[len(ratings) // 2]
        candidates = [
            i for i, e in enumerate(self._entries) if e.rating < median
        ]
        if not candidates:
            # All at or above median: drop the literally-oldest.
            idx = min(range(len(self._entries)), key=lambda i: self._entries[i].added_at_update)
        else:
            idx = min(candidates, key=lambda i: self._entries[i].added_at_update)
        del self._entries[idx]
        del self._snapshots[idx]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_matched(self, target_elo: float) -> tuple[Any, int] | None:
        """Return (snapshot, snapshot_id) sampled with Gaussian weight on
        |rating - target_elo|. Returns None if the pool is empty."""
        if not self._entries:
            return None
        weights = [
            math.exp(-(((e.rating - target_elo) / self.sigma) ** 2))
            for e in self._entries
        ]
        total = sum(weights)
        if total <= 0.0:
            # Pathological: fall back to uniform.
            idx = random.randrange(len(self._entries))
        else:
            r = random.random() * total
            acc = 0.0
            idx = len(self._entries) - 1
            for i, w in enumerate(weights):
                acc += w
                if r <= acc:
                    idx = i
                    break
        entry = self._entries[idx]
        return self._snapshots[idx], entry.snapshot_id

    # ------------------------------------------------------------------
    # Elo update
    # ------------------------------------------------------------------
    def update(
        self,
        snapshot_id: int,
        live: LiveRating,
        my_actual: float,
    ) -> float | None:
        """Update one snapshot's Elo and the live rating symmetrically
        based on a single game outcome. Returns the live policy's delta,
        or None if snapshot_id wasn't found (e.g. evicted mid-update)."""
        idx = self._find(snapshot_id)
        if idx is None:
            return None
        entry = self._entries[idx]
        delta_live = elo_delta(live.rating, entry.rating, my_actual, self.k)
        # Symmetric: opponent moves by -delta_live.
        entry.rating -= delta_live
        entry.n_games += 1
        live.rating += delta_live
        live.n_games += 1
        return delta_live

    def _find(self, snapshot_id: int) -> int | None:
        for i, e in enumerate(self._entries):
            if e.snapshot_id == snapshot_id:
                return i
        return None

    # ------------------------------------------------------------------
    # Introspection / persistence
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        if not self._entries:
            return {"size": 0}
        ratings = [e.rating for e in self._entries]
        return {
            "size": len(self._entries),
            "min": min(ratings),
            "max": max(ratings),
            "mean": sum(ratings) / len(ratings),
            "spread": max(ratings) - min(ratings),
        }

    def __len__(self) -> int:
        return len(self._entries)

    def save_metadata(self, path: str | Path) -> None:
        """Persist Elo metadata only (not the PolicyNetwork snapshots).
        Useful for analysis; not used for resume since snapshots live in
        the SnapshotPool referenced from a checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_size": self.max_size,
            "sigma": self.sigma,
            "k": self.k,
            "next_id": self._next_id,
            "entries": [asdict(e) for e in self._entries],
        }
        path.write_text(json.dumps(data, indent=2))


# Self-test: run with `python training/ae/elo_population.py`
if __name__ == "__main__":
    random.seed(0)
    pool = EloPopulation(max_size=4, sigma=150.0)
    live = LiveRating(rating=1200.0)

    for i, elo in enumerate([1000.0, 1100.0, 1200.0, 1400.0]):
        sid = pool.add(snapshot=f"snap_{i}", initial_elo=elo, at_update=i * 10)
        print(f"added snap_{i} id={sid} elo={elo}")

    print(f"\npool stats: {pool.stats()}")
    print(f"live elo: {live.rating:.1f}\n")

    picks: dict[int, int] = {}
    for _ in range(20):
        snap, sid = pool.sample_matched(live.rating)
        picks[sid] = picks.get(sid, 0) + 1
    print("matchmaking picks (Gaussian sigma=150, target=1200):")
    for sid, n in sorted(picks.items()):
        idx = pool._find(sid)
        print(f"  id={sid} elo={pool._entries[idx].rating:.1f}: {n}/20")

    # 50 games where live policy actually IS rated ~1300; shows Elo convergence.
    print("\n50 simulated games, live actual skill ~1300:")
    live = LiveRating(rating=1200.0)
    for game in range(50):
        snap, sid = pool.sample_matched(live.rating)
        opp_elo = pool._entries[pool._find(sid)].rating
        true_p = _expected_score(1300.0, opp_elo)
        actual = 1.0 if random.random() < true_p else 0.0
        pool.update(sid, live, actual)
    print(f"  final live elo: {live.rating:.1f} (true skill 1300)")
    print(f"  final pool: {pool.stats()}")
