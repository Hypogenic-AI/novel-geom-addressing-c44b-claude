# Are repeated novel geometries addressable without reasoning?

A controlled study of whether LLMs can verbally address (point to, count,
locate features within) novel 2D ASCII-grid geometries, as a function of
(a) how many in-context demo Q/As about the same grid the prompt contains,
and (b) whether chain-of-thought is enabled.

## Headline findings

- **CoT > direct at every K cell** for both models tested
  (`gpt-4.1-mini`, `claude-haiku-4.5`). All 10 paired McNemar tests have
  p < 0.04; 9/10 are significant after Bonferroni.
- **CoT closes the gap from K=0**: ≥94% accuracy at zero in-context demos.
- **Direct prompting plateaus at 79–93%**, even with 8 same-grid demos.
- **The whole gap is driven by COUNT and ROWMAX**, the two question types
  that require scanning the grid. POINT, LOOKUP, NEIGHBOR are near-ceiling
  under both regimes.
- **Repetition helps direct prompting** (haiku 0.65 → 0.93 from K=0 to K=8,
  p < 10⁻⁶) but cannot fully bridge the COUNT failure.

So novel 2D geometries are *partially* addressable without reasoning:
random-access queries succeed; aggregation queries don't.

## Reproduce

```bash
# environment
uv venv && source .venv/bin/activate
uv add openai requests numpy pandas matplotlib scipy

# API keys (used: OPENAI_API_KEY, OPENROUTER_KEY)
export OPENAI_API_KEY=...
export OPENROUTER_KEY=...

# primary experiment (80 grids × 5 K × 2 reasoning × 2 models = 1600 calls)
python src/run_experiment.py --n_items 80 --out results/main.csv

# secondary MiniARC sanity check (100 calls)
python src/run_miniarc.py --n_tasks 25 --out results/miniarc.csv

# analysis: tables to results/, figures to figures/
python src/analyze.py
```

Results are disk-cached in `results/llm_cache/`; reruns are free.

## Files

| Path                      | Purpose                                                |
|---------------------------|--------------------------------------------------------|
| `planning.md`             | Pre-registered experimental plan                       |
| `src/geometry.py`         | Procedural grid + question oracle                      |
| `src/prompting.py`        | Prompt builder, tolerant answer parser                 |
| `src/llm_client.py`       | Cached LLM client (OpenAI + OpenRouter)                |
| `src/run_experiment.py`   | Main 2 × 2 × 5 sweep driver                            |
| `src/run_miniarc.py`      | Secondary MiniARC sanity check                         |
| `src/analyze.py`          | Statistics + figures                                   |
| `results/main.csv`        | 1600 raw outputs                                       |
| `results/summary_acc.csv` | Per-cell accuracy + bootstrap CIs                      |
| `results/mcnemar*.csv`    | Paired McNemar tests                                   |
| `figures/`                | Three publication-ready PNGs                           |
| `REPORT.md`               | Full report (motivation, setup, results, discussion)   |

## See also

`REPORT.md` for the full methodology, tables, figures, and discussion.
