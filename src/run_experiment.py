"""Phase 4 experiment driver.

Pipeline:
  1. Generate N grids, each with one target question (drawn round-robin across
     QUESTION_TYPES) and a pool of demo questions about the *same* grid.
  2. For each grid × reasoning × K, build a prompt with the first K demos and
     the target question. Call the LLM.
  3. Parse the answer, score against the gold, save every row to a CSV.

The same (grid, target_question) appears at every K — only the number of demos
preceding the target changes. This makes K-wise comparison paired within item.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import (
    QUESTION_TYPES, Grid, Question, make_grid, make_question, sample_demo_pool,
)
from prompting import build_prompt, is_correct, parse_answer
from llm_client import chat


REPETITION_LEVELS = [0, 1, 2, 4, 8]
REASONING_LEVELS = ["direct", "cot"]

MODELS = [
    {"name": "gpt-4.1-mini", "provider": "openai", "model": "gpt-4.1-mini"},
    {"name": "claude-haiku-4.5", "provider": "openrouter", "model": "anthropic/claude-haiku-4.5"},
]


def build_dataset(n_items: int, seed: int = 42) -> list[dict]:
    """Build a list of items, each a {grid, target_question, demo_pool}.

    Demo pool is pre-sampled with max(REPETITION_LEVELS) demos and the same
    pool is reused across K-levels: at K we use demos[:K]. This makes K=0..1..2
    a strict ladder, not independent draws.
    """
    rng = random.Random(seed)
    items: list[dict] = []
    for i in range(n_items):
        grid_seed = 10_000 + i
        grid = make_grid(seed=grid_seed)

        qtype = QUESTION_TYPES[i % len(QUESTION_TYPES)]
        # use a per-item rng so target generation is reproducible
        qrng = random.Random(grid_seed * 7919 + 1)
        target = make_question(grid, qtype, qrng)
        target_sig = f"{target.qtype}::{target.prompt}"

        demo_rng = random.Random(grid_seed * 31 + 2)
        demos = sample_demo_pool(
            grid, n=max(REPETITION_LEVELS), rng=demo_rng,
            exclude_signature=target_sig,
        )
        items.append({
            "item_id": i,
            "grid_id": grid.grid_id,
            "grid_seed": grid_seed,
            "grid": grid,
            "target": target,
            "demos": demos,
        })
    return items


def run(n_items: int, out_csv: Path, models: list[dict]) -> None:
    items = build_dataset(n_items)
    print(f"[setup] {len(items)} items, {len(REPETITION_LEVELS)} K levels, "
          f"{len(REASONING_LEVELS)} reasoning, {len(models)} models")

    # Compose all (model, reasoning, K, item) cells.
    rows: list[dict] = []
    t0 = time.time()
    total = len(models) * len(REASONING_LEVELS) * len(REPETITION_LEVELS) * len(items)
    done = 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "provider", "reasoning", "K", "item_id", "grid_id",
            "qtype", "target_prompt", "gold", "raw", "parsed", "correct",
            "input_tokens", "output_tokens", "cached", "error",
        ])
        writer.writeheader()

        for m in models:
            for reasoning in REASONING_LEVELS:
                # CoT models need more headroom for the chain
                max_tokens = 256 if reasoning == "cot" else 24
                for K in REPETITION_LEVELS:
                    for item in items:
                        grid: Grid = item["grid"]
                        target: Question = item["target"]
                        demos = item["demos"][:K]
                        bp = build_prompt(grid, demos, target, reasoning)
                        result = chat(
                            provider=m["provider"],
                            model=m["model"],
                            system=bp.system,
                            user=bp.user,
                            temperature=0.0,
                            max_tokens=max_tokens,
                        )
                        parsed = parse_answer(result.text, target.qtype)
                        ok = is_correct(parsed, target.answer)
                        row = {
                            "model": m["name"],
                            "provider": m["provider"],
                            "reasoning": reasoning,
                            "K": K,
                            "item_id": item["item_id"],
                            "grid_id": item["grid_id"],
                            "qtype": target.qtype,
                            "target_prompt": target.prompt,
                            "gold": target.answer,
                            "raw": (result.text or "")[:400].replace("\n", "\\n"),
                            "parsed": parsed if parsed is not None else "",
                            "correct": int(ok),
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                            "cached": int(result.cached),
                            "error": result.error or "",
                        }
                        rows.append(row)
                        writer.writerow(row)
                        done += 1
                        if done % 50 == 0 or done == total:
                            elapsed = time.time() - t0
                            print(f"  [{done}/{total}] "
                                  f"{m['name']}/{reasoning}/K={K} "
                                  f"acc-so-far={sum(r['correct'] for r in rows)/len(rows):.3f} "
                                  f"elapsed={elapsed:.0f}s")
                f.flush()

    print(f"[done] wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_items", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("results/main.csv"))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny smoke run: 3 items, openai only")
    args = ap.parse_args()
    if args.smoke:
        run(n_items=3, out_csv=Path("results/smoke.csv"), models=MODELS[:1])
    else:
        run(n_items=args.n_items, out_csv=args.out, models=MODELS)
