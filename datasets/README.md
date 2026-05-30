# Datasets

This directory documents the datasets available for the research project.
Most actual data is embedded in the cloned code repositories (see `code/`).
Only download instructions are kept here so that git stays small.

## Primary datasets (already in `code/cot_icl_eval/`)

These come bundled with the Curse-of-CoT repository and are immediately usable.
Each is a JSON list of pattern-induction tasks (input grid → output grid):

| Name      | Path                                         | Size | Pattern type                                          |
|-----------|----------------------------------------------|------|-------------------------------------------------------|
| ARC-AGI   | `code/cot_icl_eval/arc/arc.json`             | 835  | 2D grid (variable shape) symbolic transformations     |
| MiniARC   | `code/cot_icl_eval/miniarc/miniarc.json`     | 149  | 5×5 grids, hand-crafted novel patterns                |
| 1D-ARC    | `code/cot_icl_eval/1d_arc/1d_arc.json`       | 901  | 1×N grids; simpler pattern induction                  |
| RAVEN     | `code/cot_icl_eval/raven/`                   | 1259 | Numerical/symbolic Raven-style matrices               |
| List Func | `code/cot_icl_eval/listfunc/`                | 1250 | Numerical list-to-list patterns                       |
| SCAN/COGS | `code/cot_icl_eval/{scan,cogs}/`             | 1000 | Textual rule induction (less directly relevant)       |

Each entry has the shape:
```json
{ "index": 0, "data": { "train": [ {"input": [[...]], "output": [[...]]}, ... ],
                        "test":   { "input": [[...]], "output": [[...]] } } }
```

The `code/cot_icl_eval/utils/*_prompt.py` files contain ready-made templates
for *direct*, *CoT*, *ReAct*, and *ToT* prompting that can be reused.

## Reference: Official ARC-AGI

Cloned to `code/arc_agi_official/`. Same task format as above, organised as
one JSON file per task in `data/training/` (400 files) and `data/evaluation/`
(400 files). Used as the canonical source for the symbolic-grid benchmark.

## Polygon datasets (image-based)

Cloned to `code/shape_blind/`. CSVs in `code/shape_blind/CSVs_for_evaluation/`:

- `regular_polygons.csv` — triangle, square, pentagon, hexagon, heptagon, octagon
- `abstract_shapes.csv` — irregular / novel polygons unlikely to be in training
- `regular_polygon_pairs.csv` — multi-step counting tasks
- `arrow_on_plus_with_visual_cues.csv`, `heptagons_with_visual_cues.csv` — VC-CoT prompts
- `mathverse_revised.csv` — preprocessed subset

Images are zipped at `code/shape_blind/images.zip` (unzip on first use).

These are primarily for multimodal experiments. For our text-only setting,
we can describe the polygons textually (e.g. as vertex lists) instead.

## Synthetic generators

If we need novel geometries the model has provably not seen, the following
generators are available:

- `code/arc_1d/` (Rust) — generates 1D ARC-like tasks programmatically.
- `code/shape_blind/image_generation_code/` — Jupyter notebooks for generating
  abstract shapes, regular polygons, and visually-cued versions.
- We can also write a small Python generator that produces 2D shapes
  (polygons, free-form blobs, etc.) as ASCII grids or vertex lists, with
  controlled novelty and repetition.

## How to use these datasets in the experiment

1. **Pattern-induction (grid) baseline** — use `cot_icl_eval` JSON files
   directly; prompt templates already provided. Compare *Direct* vs *CoT*.
2. **Novel geometry addressing** — generate ASCII or vertex-list
   representations of unfamiliar shapes and ask the model to "address"
   them (name, refer to, identify a part). Compare with and without
   repetition in context, with and without CoT.
3. **Repetition control** — replicate examples in the prompt N times and
   measure how performance scales with N (links to the ICL-from-Repetitions
   self-reinforcement effect).

## Sample data

A handful of `.json` records from each source are kept in `samples/` for
quick inspection without re-loading the full corpora.
