"""Secondary experiment: MiniARC direct-vs-CoT sanity check.

Loads a small slice of MiniARC tasks, builds the standard ICL prompt
(train pairs as demos, then the test input), and prompts the model in
two conditions:
  - direct: produce the output grid only
  - cot:    think step by step, then produce the output grid

Scoring: exact-match on the full output grid.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import chat


ROOT = Path(__file__).resolve().parent.parent
MINIARC_PATH = ROOT / "code" / "cot_icl_eval" / "miniarc" / "miniarc.json"


def grid_to_text(g: list[list[int]]) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in g)


def parse_grid(raw: str) -> list[list[int]] | None:
    """Find the last consecutive run of all-integer lines in `raw`."""
    if raw is None:
        return None
    lines = raw.strip().splitlines()
    # collect runs of "integer-only" rows
    runs: list[list[list[int]]] = []
    current: list[list[int]] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            if current:
                runs.append(current)
                current = []
            continue
        toks = re.findall(r"-?\d+", ln)
        if toks and re.fullmatch(r"[\d\s,\-]+", ln):
            try:
                row = [int(t) for t in toks]
                current.append(row)
                continue
            except ValueError:
                pass
        if current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:
        return None
    # prefer the longest run; on ties, keep the last (later in output) one
    best = max(runs, key=lambda r: (len(r), runs.index(r)))
    if not best:
        return None
    # enforce rectangular: all rows same width
    W = len(best[0])
    if any(len(r) != W for r in best):
        return None
    return best


DIRECT_SYSTEM = (
    "You are solving a small ARC-style puzzle. You see a few "
    "(input -> output) examples; infer the rule and apply it to the test "
    "input. Output only the test output grid as integers separated by spaces, "
    "one row per line. Nothing else."
)

COT_SYSTEM = (
    "You are solving a small ARC-style puzzle. You see a few "
    "(input -> output) examples; infer the rule and apply it to the test "
    "input. Think briefly, then on the LAST lines of your response output "
    "the test output grid as integers separated by spaces, one row per line."
)


def build_user(task: dict) -> str:
    parts: list[str] = []
    for i, ex in enumerate(task["train"]):
        parts.append(f"Example {i + 1} input:")
        parts.append(grid_to_text(ex["input"]))
        parts.append(f"Example {i + 1} output:")
        parts.append(grid_to_text(ex["output"]))
        parts.append("")
    test = task["test"]
    parts.append("Test input:")
    parts.append(grid_to_text(test["input"]))
    parts.append("Test output:")
    return "\n".join(parts)


MODELS = [
    {"name": "gpt-4.1-mini", "provider": "openai", "model": "gpt-4.1-mini"},
    {"name": "claude-haiku-4.5", "provider": "openrouter", "model": "anthropic/claude-haiku-4.5"},
]


def run(n_tasks: int, out_csv: Path) -> None:
    data = json.loads(MINIARC_PATH.read_text())
    # data is a list of {"index": .., "data": {"train": [...], "test": [...]}}
    items = data[:n_tasks]
    print(f"[miniarc] loaded {len(items)} tasks")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "reasoning", "task_index", "correct",
            "input_tokens", "output_tokens", "cached", "raw_first_120"])
        w.writeheader()
        t0 = time.time()
        done = 0
        total = len(MODELS) * 2 * len(items)
        for m in MODELS:
            for reasoning in ("direct", "cot"):
                system = DIRECT_SYSTEM if reasoning == "direct" else COT_SYSTEM
                max_tokens = 256 if reasoning == "direct" else 700
                for it in items:
                    task = it["data"]
                    user = build_user(task)
                    r = chat(m["provider"], m["model"], system, user,
                             temperature=0.0, max_tokens=max_tokens)
                    pred = parse_grid(r.text)
                    gold = task["test"]["output"]
                    ok = int(pred == gold)
                    w.writerow({
                        "model": m["name"],
                        "reasoning": reasoning,
                        "task_index": it["index"],
                        "correct": ok,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "cached": int(r.cached),
                        "raw_first_120": (r.text or "")[:120].replace("\n", "\\n"),
                    })
                    done += 1
                    if done % 20 == 0 or done == total:
                        print(f"  [{done}/{total}] {m['name']}/{reasoning} "
                              f"elapsed={time.time() - t0:.0f}s")
                f.flush()
    print(f"[done] wrote {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tasks", type=int, default=25)
    ap.add_argument("--out", type=Path, default=Path("results/miniarc.csv"))
    args = ap.parse_args()
    run(args.n_tasks, args.out)
