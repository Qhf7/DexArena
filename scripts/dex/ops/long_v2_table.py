"""Version-2 long-horizon scores: per seed, AND over the family's subtasks.

A (specialists): each subtask's own 50-episode evaluation ($DEX_OUT/evals/groot/<run>/checkpoint-12000,
seeds 10001-10050) -- seed s passes the long task iff every subtask passed seed s.
B (family checkpoint): the same, from $DEX_OUT/evals/long_v2/family/<family>/<stem> written by
scripts/dex/eval_family_on_subtask.py. Prints a markdown table; --json dumps the numbers.

    python scripts/dex/ops/long_v2_table.py [--run-tag v4] [--json out.json]
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils.dex_long_tasks import LONG_TASKS
from utils.dex_tasks import resolve_dex_task

# $DEX_OUT is the directory that holds datasets/, runs/ and evals/ (docs/GROOT_DEXARENA.md).
OUT = Path(os.environ.get("DEX_OUT", "dex_out")) / "evals"


def per_seed(path):
    if not os.path.exists(path):
        return None
    return {int(r["environment_seed"]): bool(r["success"]) for r in map(json.loads, open(path))}


RUN_TAG = ""   # set from --run-tag: "" = the 2026-09-21 round, "v4" = per-episode-label retrains


def specialist_run(stem):
    tag = f"_{RUN_TAG}" if RUN_TAG else ""
    for suffix in ("_rig", "_fixcam", ""):
        p = OUT / "groot" / f"{stem}{suffix}{tag}" / "checkpoint-12000" / "episodes.jsonl"
        if p.exists():
            return p
    return None


def main():
    global RUN_TAG
    ap = argparse.ArgumentParser(); ap.add_argument("--json", default=None)
    ap.add_argument("--run-tag", default="", help="v4: the v4 specialists (A) and v4 families (B, evals/long_v2_v4)")
    a = ap.parse_args(); RUN_TAG = a.run_tag
    v2dir = f"long_v2_{RUN_TAG}" if RUN_TAG else "long_v2"
    rows, dump = [], {}
    for t in LONG_TASKS:
        stems = [resolve_dex_task(s.subtask).short_name for s in t.stages]
        A = [per_seed(specialist_run(st)) for st in stems]
        B = [per_seed(OUT / v2dir / "family" / t.family / st / "episodes.jsonl") for st in stems]
        def combine(parts):
            if any(p is None for p in parts):
                return None, [None if p is None else sum(p.values()) for p in parts]
            seeds = set.intersection(*(set(p) for p in parts))
            ok = sum(all(p[s] for p in parts) for s in seeds)
            return (ok, len(seeds)), [sum(p[s] for s in seeds) for p in parts]
        a_res, a_sub = combine(A); b_res, b_sub = combine(B)
        rows.append((t.family, stems, a_res, a_sub, b_res, b_sub))
        dump[t.family] = {"stages": stems, "A": a_res, "A_subtasks": a_sub, "B": b_res, "B_subtasks": b_sub}
    fmt = lambda r: "pending" if r is None else "%d%% (%d/%d)" % (round(100 * r[0] / r[1]), r[0], r[1])
    sub = lambda xs: "/".join("?" if x is None else str(x) for x in xs)
    print("| long task | stages | A: specialists, per-seed AND | A: passes per stage | B: family checkpoint, per-seed AND | B: passes per stage |")
    print("|---|---|---|---|---|---|")
    for fam, stems, ar, asub, br, bsub in rows:
        print(f"| {fam} | {' → '.join(stems)} | {fmt(ar)} | {sub(asub)} | {fmt(br)} | {sub(bsub)} |")
    if a.json:
        Path(a.json).write_text(json.dumps(dump, indent=1))


if __name__ == "__main__":
    main()
