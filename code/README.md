# Cloned Code Repositories

All repos are clones — no local modifications. Use them as references,
prompt templates, datasets, and (optionally) drop-in evaluation scripts.

## 1. CoT-ICL-Eval (`cot_icl_eval/`)
- URL: https://github.com/HKUST-KnowComp/CoT-ICL-Eval
- Paper: Curse of CoT (`papers/2504.05081_curse_of_cot.pdf`)
- Why we cloned it:
  - Ships **ready-to-use JSON datasets** for ARC, MiniARC, 1D-ARC, RAVEN,
    List-Function, SCAN, MiniSCAN, COGS, SALT (9 benchmarks).
  - Ships **prompt templates** under `utils/*_prompt.py` for four
    paradigms: `direct_*`, `cot_*`, `tot_*`, `react_*`. We can use these
    verbatim to compare "addressing without reasoning" vs. various
    reasoning strategies.
  - Their main result table (in `README.md`) gives us aggregate
    direct-vs-CoT numbers across 16 LLMs as a sanity-check baseline.
- Key files:
  - `arc/arc.json` (835 ARC-AGI tasks)
  - `miniarc/miniarc.json` (149 hand-crafted 5×5 tasks with patterns)
  - `1d_arc/1d_arc.json` (901 1D simplifications)
  - `utils/miniarc_prompt.py` etc. — pasteable prompt builders

## 2. To-CoT-or-not-to-CoT (`to_cot_or_not/`)
- URL: https://github.com/Zayne-sprague/To-CoT-or-not-to-CoT
- Paper: To CoT or Not to CoT (`papers/2409.12183_to_cot_or_not.pdf`)
- Why: meta-analysis methodology and a clean direct-vs-CoT evaluation
  harness over 20 datasets × 14 models. Reference design for our
  smaller comparison experiment.

## 3. ARC-AGI official (`arc_agi_official/`)
- URL: https://github.com/fchollet/ARC-AGI
- Why: canonical ARC-AGI source (400 train + 400 eval tasks, one JSON
  per task). Use this instead of `cot_icl_eval/arc/` if we want the
  full benchmark.

## 4. arc_1d (`arc_1d/`)
- URL: https://github.com/optozorax/arc_1d
- Why: **synthetic 1D ARC-task generator** (Rust). Useful if we need
  truly novel tasks that are guaranteed not to be in any pretraining set.

## 5. MINI-ARC (`mini_arc/`)
- URL: https://github.com/KSB21ST/MINI-ARC
- Why: the original MiniARC dataset (5×5 grids), plus a small
  inspection server. Lets us cross-check the version in CoT-ICL-Eval.

## 6. Shape-Blind (`shape_blind/`)
- URL: https://github.com/rsinghlab/Shape-Blind
- Paper: Forgotten Polygons (`papers/2502.15969_forgotten_polygons_shape_blind.pdf`)
- Why:
  - Provides **regular polygon images** (triangle → octagon) and
    **abstract / novel polygon images** in `CSVs_for_evaluation/`.
  - Image generation notebooks under `image_generation_code/` let us
    create *new* abstract shapes if we want a held-out novelty set.
  - Their VC-CoT prompt formulation (annotated vertices) is directly
    applicable to text-encoded shapes too — we can label vertices in
    a vertex-list representation as A, B, C, … and ask the model to
    address each.

## 7. VDLM (`vdlm/`)
- URL: https://github.com/MikeWangWZHL/VDLM
- Paper: Visually Descriptive Language Model (`papers/2404.06479_vdlm_vector_graphics.pdf`)
- Why: defines the **Primal Visual Description (PVD)** text format —
  `{type, vertices, color, style}` — a clean schema for describing 2D
  shapes to LLMs in pure text. We can adopt PVD (or a simplified
  variant) as our canonical input format for novel-geometry addressing
  tasks.
- Key file: `prompts.py` (prompt templates for shape-based reasoning).

## Installation status

None of these repos was installed into the project venv. They are
referenced for code/prompts/data only. If a future experiment needs
to actually *run* one (e.g. VDLM's SVG-to-PVD model), follow that
repo's own setup instructions — most pin specific torch/transformers
versions and should ideally be installed into their own sub-venv to
avoid clashing with our top-level pyproject.toml.

## Quick sanity checks performed

- `cot_icl_eval/{arc,miniarc,1d_arc}/*.json` load cleanly as Python
  lists with `index`, `data` (`train` + `test`), and (where applicable)
  `pattern` fields. Samples saved to `datasets/samples/`.
- `arc_agi_official/data/training/*.json` follow the standard ARC schema
  (`train`: list of `{input, output}` pairs; `test`: same).
- `shape_blind/CSVs_for_evaluation/` has six CSVs ready for evaluation
  scripts; image archive is zipped (`images.zip`) — unzip on first use.
