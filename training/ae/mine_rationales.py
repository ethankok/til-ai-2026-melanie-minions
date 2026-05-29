"""Mine LLM rationales of heuristic actions for heuristic-improvement leads.

Input: a .jsonl produced by collect_annotated_heuristic.py, where each row is
a heuristic action + an LLM rationale explaining why it's good.

The LLM was told the action is the *expert's* choice and asked to justify it.
So when the rationale hedges ("forced", "only option", "suboptimal", "ideally
X but"), that's a signal the heuristic may be in a weak spot — the model
couldn't fully endorse the move. We surface those, plus other diagnostic
buckets, so a human can turn the good leads into testable heuristic tweaks
(validated downstream with multi_seed_eval.py).

This does NOT change any policy. It's a read-only analysis tool.

Usage:
    python training/ae/mine_rationales.py \
        --in training/ae/data/annotated/pilot_mixed_r5_final.jsonl \
        [--top 25] [--out report.md]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ACTION_NAMES = ["FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT", "STAY", "PLACE_BOMB"]

# Phrases that suggest the model could not fully endorse the expert action —
# i.e. the heuristic may be making a forced/defensive/suboptimal move.
HEDGE_PATTERNS = [
    r"\bforced\b", r"\bonly option\b", r"\bno (?:other )?choice\b", r"\bno better\b",
    r"\bhad to\b", r"\bsub-?optimal\b", r"\bnot ideal\b", r"\bideally\b",
    r"\bwould be better\b", r"\bbetter to\b", r"\binstead\b", r"\bunfortunately\b",
    r"\bwasted?\b", r"\bwaste\b", r"\btrapped\b", r"\bstuck\b", r"\blimited\b",
    r"\bcannot\b", r"\bunable\b", r"\bavoid(?:ing)? (?:damage|death|threat)\b",
    r"\bretreat\b", r"\bdefensive\b", r"\bfallback\b", r"\bhowever\b",
    r"\balthough\b", r"\bbut (?:there|the|it|this)\b", r"\bsuboptimal\b",
    r"\bdead ?end\b", r"\bbacktrack\b", r"\bre-?explore\b", r"\bnothing better\b",
]
HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)

# Phrases tied specifically to base-defense reasoning (our biggest reward leak
# per the simulate diagnostics: base_damage / own_base_destroyed dominate losses)
DEFENSE_RE = re.compile(
    r"\b(defend|base (?:is )?(?:under )?(?:threat|attack|danger)|protect (?:the |our )?base|"
    r"enemy (?:near|approaching) (?:our |the )?base|own base)\b",
    re.IGNORECASE,
)


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hedges(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in HEDGE_RE.finditer(text or "")})


def fmt_row(r: dict, extra: str = "") -> str:
    a = int(r.get("action", 4))
    aname = ACTION_NAMES[a] if 0 <= a < len(ACTION_NAMES) else str(a)
    head = (
        f"[r{r['round_idx']} s{r['step']}] loc={r.get('loc')} dir={r.get('direction')} "
        f"act={a}({aname}) reward={r.get('reward', 0):.0f} mc={r.get('mc_return', 0):.0f}"
    )
    if extra:
        head += f"  {extra}"
    return head + "\n    " + (r.get("rationale") or "").strip().replace("\n", " ")[:500]


def section(title: str) -> str:
    return f"\n{'='*72}\n{title}\n{'='*72}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--top", type=int, default=25, help="max items per bucket")
    p.add_argument("--out", type=Path, default=None, help="optional .md report (also prints to stdout)")
    args = p.parse_args(argv)

    rows = load(args.in_path)
    ok = [r for r in rows if r.get("rationale_ok")]

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)

    emit(f"# Rationale mining — {args.in_path.name}")
    emit(f"\nTotal samples: {len(rows)}   with rationale: {len(ok)}")
    if not ok:
        emit("\nNo usable rationales. Nothing to mine.")
        report = "\n".join(lines)
        print(report)
        if args.out:
            args.out.write_text(report)
        return 0

    mcs = [r["mc_return"] for r in rows]
    emit(
        f"mc_return: min={min(mcs):.0f} p10={statistics.quantiles(mcs, n=10)[0]:.0f} "
        f"p50={statistics.median(mcs):.0f} p90={statistics.quantiles(mcs, n=10)[8]:.0f} max={max(mcs):.0f}"
    )

    # --- Bucket 1: hedged rationales (model couldn't fully endorse the move) ---
    hedged = []
    for r in ok:
        h = hedges(r.get("rationale", ""))
        if h:
            hedged.append((r, h))
    # Rank by: more hedge hits first, then lower mc_return (worse outcome = more interesting)
    hedged.sort(key=lambda x: (-len(x[1]), x[0]["mc_return"]))

    emit(section(f"① HEDGED RATIONALES ({len(hedged)} of {len(ok)}) — candidate heuristic weak spots"))
    emit("The model was asked to justify the EXPERT action; hedging suggests a forced/")
    emit("suboptimal move. Lower mc_return + more hedge-words ranked first.\n")
    from collections import Counter
    hedge_freq = Counter(h for _, hs in hedged for h in hs)
    emit("hedge-word frequency: " + ", ".join(f"{w}={c}" for w, c in hedge_freq.most_common(15)))
    emit("")
    for r, h in hedged[: args.top]:
        emit(fmt_row(r, extra=f"hedges={h}"))

    # --- Bucket 2: defense reasoning (our biggest reward leak) ---
    defense = [r for r in ok if DEFENSE_RE.search(r.get("rationale", ""))]
    defense.sort(key=lambda r: r["mc_return"])
    emit(section(f"② BASE-DEFENSE RATIONALES ({len(defense)}) — biggest reward leak is base loss"))
    emit("Where the model framed the move as base defense. Lowest mc_return first.\n")
    for r in defense[: args.top]:
        emit(fmt_row(r))

    # --- Bucket 3: PLACE_BOMB decisions (highest value/risk action) ---
    bombs = [r for r in ok if int(r.get("action", 4)) == 5]
    bombs.sort(key=lambda r: -r["mc_return"])
    emit(section(f"③ PLACE_BOMB RATIONALES ({len(bombs)}) — highest value+risk action"))
    emit("Sorted high→low mc_return. Compare the model's bomb justification vs outcome.\n")
    for r in bombs[: args.top]:
        emit(fmt_row(r))

    # --- Bucket 4: STAY decisions (prompt says STAY is usually wasted) ---
    stays = [r for r in ok if int(r.get("action", 4)) == 4]
    stays.sort(key=lambda r: r["mc_return"])
    emit(section(f"④ STAY RATIONALES ({len(stays)}) — STAY is usually a wasted tick"))
    emit("Why did the expert idle? Genuine reasons vs heuristic dithering.\n")
    for r in stays[: args.top]:
        emit(fmt_row(r))

    # --- Bucket 5: lowest-return states overall ---
    worst = sorted(ok, key=lambda r: r["mc_return"])[: args.top]
    emit(section(f"⑤ LOWEST mc_return STATES (top {len(worst)}) — what was the plan when losing?"))
    for r in worst:
        emit(fmt_row(r))

    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"\n[wrote report -> {args.out}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
