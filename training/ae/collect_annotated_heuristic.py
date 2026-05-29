"""Collect (obs, heuristic_action, llm_rationale, mc_return) tuples.

The heuristic AEManager plays the game as the "optimal" demonstrator. After
each tick, an LLM is given the same observation (plus a maintained belief
map / plan / history block) and asked to articulate WHY the heuristic's
chosen action is good. The resulting rationales become the auxiliary text
target for chain-of-thought-style BC distillation.

Output: one JSON object per tick to a .jsonl file:
    {
        "round_idx": int,
        "step": int,
        "loc": [x, y],
        "direction": int,
        "obs_text": str,           # textual encoding (LLM-readable)
        "memory_text": str,        # belief block as the LLM saw it
        "action": int,             # heuristic action 0-5
        "reward": float,           # reward observed AFTER this action
        "mc_return": float,        # filled at end of round
        "rationale": str,          # 2-3 sentence LLM justification
        "rationale_latency_s": float,
        "rationale_ok": bool,
    }

Usage:
    PYTHONHASHSEED=0 AE_LLM_BACKEND=agy \\
      python training/ae/collect_annotated_heuristic.py \\
        --rounds 5 --opponents mixed --seed 100 \\
        --out training/ae/data/annotated/pilot_agy_n5.jsonl

The script is resumable: re-running with the same --out skips rounds
already present.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable, *sys.argv])

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AE_SRC = REPO_ROOT / "ae" / "src"
TIL_AE = REPO_ROOT / "til-26-ae"
TRAINING_AE = REPO_ROOT / "training" / "ae"
for p in (str(AE_SRC), str(TIL_AE), str(TRAINING_AE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ae_manager import AEManager  # noqa: E402
from llm_manager import (  # noqa: E402
    ACTION_NAMES,
    BeliefMemory,
    encode_observation,
)
from opponents import make_opponent, resolve_opponent_spec  # noqa: E402

from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402


ANNOTATOR_SYSTEM = """You are an expert player coach for the TIL-AI 2026 AE task — a 16x16 partially-observable bomberman variant. A strong rule-based player just chose an action in the situation below. Your job is to articulate WHY that action is the correct choice in 2-3 sentences. Reference specific observation features (item positions, walls, enemy threats, etc).

The coach reasoning will be used as a training target for a smaller neural policy that will learn to predict the rule-based player's actions. Good rationales are concrete, mention world coordinates when relevant, and connect observation features to the action's expected reward.

ACTIONS:
  0 FORWARD, 1 BACKWARD, 2 TURN_LEFT, 3 TURN_RIGHT, 4 STAY, 5 PLACE_BOMB

REWARD STRUCTURE: enemy_base destroyed = +50, kill = +15, mission = +5, resource = +2, recon = +1. Own base destroyed = -50; taking damage is negative.

OBSERVATION FORMAT:
You'll see a MEMORY block (belief map, recent actions, last-tick signals), then a STEP block (scalars + AGENT_VIEW + BASE_VIEW grids). Cell tokens are 6 chars: [item][agent][wallR][wallD][wallL][wallU]. Items: ? unseen, . empty, m mission, r resource, c recon, x enemy-base, b ally-base. Agent: . none, A ally, E enemy. Wall: - none, # solid, D destructible.

OUTPUT FORMAT (STRICT):
<rationale>2-3 sentence justification of the action. Concrete, references coords/features.</rationale>

No other text. Stop after </rationale>."""


# ---------------------------------------------------------------------------
# LLM calling — minimal duplicate of LLMAEManager backends, but with the
# annotator system prompt and a different parser.
# ---------------------------------------------------------------------------

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


class Annotator:
    def __init__(self) -> None:
        self.backend = os.environ.get("AE_LLM_BACKEND", "agy").strip().lower()
        self.model = os.environ.get("AE_LLM_MODEL", "claude-sonnet-4-6")
        self.http_timeout = float(os.environ.get("AE_LLM_HTTP_TIMEOUT", "60"))
        if self.backend == "pioneer":
            self.pioneer_key = os.environ.get("PIONEER_API_KEY")
            if not self.pioneer_key:
                raise RuntimeError("PIONEER_API_KEY env var not set")
            self.pioneer_url = os.environ.get(
                "AE_LLM_PIONEER_URL", "https://api.pioneer.ai/v1"
            ).rstrip("/")
        elif self.backend == "agy":
            self.agy_path = os.environ.get("AE_LLM_AGY_PATH", "agy")
            self.agy_timeout = float(os.environ.get("AE_LLM_AGY_TIMEOUT", "120"))
        else:
            raise ValueError(
                f"AE_LLM_BACKEND={self.backend!r} not supported by annotator; "
                "use 'agy' or 'pioneer'"
            )

    def call(self, prompt_body: str) -> tuple[str, bool]:
        if self.backend == "pioneer":
            return self._call_pioneer(prompt_body)
        return self._call_agy(prompt_body)

    def _call_pioneer(self, prompt_body: str) -> tuple[str, bool]:
        payload = json.dumps({
            "model": self.model,
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": ANNOTATOR_SYSTEM},
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
                return f"<no_choices>", False
            return choices[0].get("message", {}).get("content", "") or "", True
        except urllib.error.HTTPError as e:
            err = ""
            try:
                err = e.read().decode()[:200]
            except Exception:
                pass
            return f"<http_error {e.code}: {err}>", False
        except Exception as exc:  # noqa: BLE001
            return f"<pioneer_exception: {exc!r}>", False

    def _call_agy(self, prompt_body: str) -> tuple[str, bool]:
        combined = ANNOTATOR_SYSTEM + "\n\n---\n\n" + prompt_body
        try:
            result = subprocess.run(
                [self.agy_path, "-p", combined],
                capture_output=True,
                text=True,
                timeout=self.agy_timeout,
                check=False,
            )
            if result.returncode != 0:
                return f"<agy_error rc={result.returncode}: {result.stderr.strip()[:200]}>", False
            stdout = result.stdout or ""
            if not stdout.strip():
                # agy returns rc=0 with empty stdout when quota-throttled; treat as failure
                return "<agy_empty_response (likely quota/rate-limit)>", False
            return stdout, True
        except subprocess.TimeoutExpired:
            return "<agy_timeout>", False
        except Exception as exc:  # noqa: BLE001
            return f"<agy_exception: {exc!r}>", False


import re

RATIONALE_RE = re.compile(r"<rationale>(.*?)</rationale>", re.IGNORECASE | re.DOTALL)


def parse_rationale(raw: str) -> str:
    if not raw:
        return ""
    m = RATIONALE_RE.search(raw)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner
    # Tag missing or empty body — fall back to the whole response, minus stray tags
    cleaned = re.sub(r"</?rationale>", "", raw, flags=re.IGNORECASE).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def run_round(
    env,
    heuristic: AEManager,
    opponents,
    annotator: Annotator,
    round_idx: int,
    seed: int,
    out_fp,
) -> dict:
    env.reset(seed=seed)
    agent_id_us = env.possible_agents[0]
    other_ids = list(env.possible_agents[1:])

    if hasattr(heuristic, "_reset_memory"):
        heuristic._reset_memory()
    for op in opponents:
        if hasattr(op, "reset_for_game"):
            op.reset_for_game()
        if hasattr(op, "_reset_memory"):
            op._reset_memory()

    belief = BeliefMemory()
    samples: list[dict] = []
    cumulative = 0.0
    prev_cumulative = 0.0

    for agent in env.agent_iter():
        observation, reward, term, trunc, info = env.last()
        if agent == agent_id_us:
            cumulative += float(reward)

        if term or trunc:
            env.step(None)
            continue

        obs_native = {
            k: v if type(v) in (int, float) else (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in observation.items()
        }

        if agent == agent_id_us:
            # Reward delta for the prior sample = cumulative - prev_cumulative
            if samples:
                samples[-1]["reward"] = cumulative - prev_cumulative
            prev_cumulative = cumulative

            belief.update(obs_native)
            memory_text = belief.format_block(obs_native)
            obs_text = encode_observation(obs_native)
            action = int(heuristic.ae(obs_native))
            belief.record_action(obs_native, action)

            prompt = (
                memory_text + "\n" + obs_text + "\n"
                + f"---\nThe expert player chose action {action} ({ACTION_NAMES[action]}). "
                f"In 2-3 sentences, explain why this is the correct play in this situation."
            )

            t0 = time.monotonic()
            raw, ok = annotator.call(prompt)
            dt = time.monotonic() - t0
            rationale = parse_rationale(raw) if ok else ""
            # Mark ok=False if rationale ended up empty (model returned nothing useful)
            if ok and not rationale.strip():
                ok = False

            loc = obs_native.get("location") or [0, 0]
            direction = int(_scalar(obs_native, "direction"))
            step = int(_scalar(obs_native, "step"))

            sample = {
                "round_idx": round_idx,
                "step": step,
                "loc": [int(loc[0]), int(loc[1])],
                "direction": direction,
                "obs_text": obs_text,
                "memory_text": memory_text,
                "action": action,
                "reward": 0.0,        # filled on next iteration
                "mc_return": 0.0,     # filled at end of round
                "rationale": rationale,
                "rationale_raw_tail": raw[-200:] if not ok else "",
                "rationale_latency_s": round(dt, 2),
                "rationale_ok": ok,
            }
            samples.append(sample)

            env.step(action)
        else:
            slot = other_ids.index(agent)
            op = opponents[slot]
            a = int(op(obs_native))
            mask = obs_native.get("action_mask")
            if mask is not None:
                try:
                    if not int(mask[a]):
                        for k, m in enumerate(mask):
                            if int(m):
                                a = k
                                break
                except Exception:
                    pass
            env.step(a)

    # Final reward attribution: the last sample's reward delta from end-of-game
    if samples:
        samples[-1]["reward"] = cumulative - prev_cumulative

    # MC returns
    total = 0.0
    for s in reversed(samples):
        total += s["reward"]
        s["mc_return"] = round(total, 4)

    # Flush
    for s in samples:
        out_fp.write(json.dumps(s) + "\n")
    out_fp.flush()

    ok_count = sum(1 for s in samples if s["rationale_ok"])
    return {
        "round_idx": round_idx,
        "round_seed": seed,
        "round_score": round(cumulative / 1000.0, 4),
        "samples": len(samples),
        "rationale_ok": ok_count,
        "rationale_fail": len(samples) - ok_count,
        "mean_latency_s": round(
            sum(s["rationale_latency_s"] for s in samples) / max(1, len(samples)), 2
        ),
    }


def _reannotate(in_path: Path, out_path: Path, annotator: Annotator) -> int:
    """Re-call the annotator on samples in `in_path` whose rationale_ok=False;
    pass good samples through unchanged. Writes to `out_path` (jsonl)."""
    if not in_path.exists():
        print(f"[reannotate] input missing: {in_path}", file=sys.stderr)
        return 1

    samples: list[dict] = []
    with in_path.open() as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except Exception:
                pass
    total = len(samples)
    bad = [i for i, s in enumerate(samples) if not s.get("rationale_ok")]
    print(f"[reannotate] {total} samples loaded; {len(bad)} need re-annotation")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resumable: if out_path already has entries with rationale_ok=True for indices
    # already processed, keep them. Simplest: always write fresh from scratch.
    t_start = time.monotonic()
    ok_count = 0
    with out_path.open("w") as out_fp:
        for idx, s in enumerate(samples):
            if s.get("rationale_ok"):
                out_fp.write(json.dumps(s) + "\n")
                continue
            obs_text = s.get("obs_text", "")
            memory_text = s.get("memory_text", "")
            action = int(s.get("action", 4))
            prompt = (
                memory_text + "\n" + obs_text + "\n"
                + f"---\nThe expert player chose action {action} ({ACTION_NAMES[action]}). "
                f"In 2-3 sentences, explain why this is the correct play in this situation."
            )
            t0 = time.monotonic()
            raw, ok = annotator.call(prompt)
            dt = time.monotonic() - t0
            rationale = parse_rationale(raw) if ok else ""
            if ok and not rationale.strip():
                ok = False
            s["rationale"] = rationale
            s["rationale_raw_tail"] = raw[-200:] if not ok else ""
            s["rationale_latency_s"] = round(dt, 2)
            s["rationale_ok"] = ok
            if ok:
                ok_count += 1
            out_fp.write(json.dumps(s) + "\n")
            out_fp.flush()
            done = bad.index(idx) + 1 if idx in bad else 0
            if done and done % 25 == 0:
                eta = (time.monotonic() - t_start) * (len(bad) - done) / done
                print(
                    f"[reannotate] {done}/{len(bad)} re-done  ok_now={ok_count}/{done}  "
                    f"elapsed={time.monotonic()-t_start:.0f}s  eta~{eta:.0f}s",
                    flush=True,
                )

    elapsed = time.monotonic() - t_start
    final_ok = sum(1 for s in samples if s.get("rationale_ok"))
    print(f"[reannotate] done in {elapsed:.0f}s. Total ok now: {final_ok}/{total}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--opponents", type=str, default="mixed")
    p.add_argument("--seed", type=int, default=100, help="base seed; round k uses seed+k")
    p.add_argument("--out", type=Path, required=True, help=".jsonl output (resumable)")
    p.add_argument("--non-novice", action="store_true")
    p.add_argument("--reannotate-empty", action="store_true",
                   help="instead of running new rounds, re-call annotator on samples in --in whose rationale_ok=False; write all samples to --out")
    p.add_argument("--in", dest="in_path", type=Path, default=None,
                   help="input .jsonl (with --reannotate-empty)")
    args = p.parse_args(argv)

    if args.reannotate_empty:
        if args.in_path is None:
            print("--in required with --reannotate-empty", file=sys.stderr)
            return 1
        annotator = Annotator()
        print(f"[annotator] backend={annotator.backend} model={annotator.model}")
        return _reannotate(args.in_path, args.out, annotator)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Resumable: detect existing rounds in the output file
    completed_rounds: set[int] = set()
    if args.out.exists():
        try:
            with args.out.open() as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        completed_rounds.add(int(rec.get("round_idx", -1)))
                    except Exception:
                        pass
            print(f"[resume] found {len(completed_rounds)} completed rounds in {args.out}")
        except OSError:
            pass

    cfg = default_config()
    cfg.env.novice = not args.non_novice
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    heuristic = AEManager()

    names = resolve_opponent_spec(args.opponents)
    opponents = [make_opponent(n, seed=args.seed + 1000 + i) for i, n in enumerate(names)]

    annotator = Annotator()
    print(f"[annotator] backend={annotator.backend} model={annotator.model}")
    print(f"[setup] opponents={names} rounds={args.rounds} seed={args.seed}")

    t_start = time.monotonic()
    with args.out.open("a") as out_fp:
        for r in range(args.rounds):
            if r in completed_rounds:
                print(f"[round {r+1}/{args.rounds}] already collected — skip")
                continue
            seed = args.seed + r
            t0 = time.monotonic()
            stats = run_round(env, heuristic, opponents, annotator, r, seed, out_fp)
            dt = time.monotonic() - t0
            print(
                f"[round {r+1}/{args.rounds}] seed={seed} score={stats['round_score']:.4f} "
                f"samples={stats['samples']} rationale_ok={stats['rationale_ok']}/{stats['samples']} "
                f"mean_lat={stats['mean_latency_s']}s round_wall={dt:.0f}s "
                f"total_wall={time.monotonic()-t_start:.0f}s",
                flush=True,
            )

    env.close()
    print(f"[done] total wall-clock {time.monotonic()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
