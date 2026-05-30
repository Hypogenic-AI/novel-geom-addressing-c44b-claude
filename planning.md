# Planning: Are repeated novel geometries addressable without reasoning?

## Motivation & Novelty Assessment

### Why This Research Matters
The Liu et al. 2025 paper (arXiv 2501.00070, *Generative Patterns of Thought*-style work) shows
LLMs can pick up 2D geometric structure from in-context input. But a follow-on,
operationally useful capability is whether the model can *talk about* those geometries —
locate a feature, name a cell value, point to a position. If the answer is "yes, with
enough repetition, and without needing chain-of-thought," that has direct practical
implications for grid-based tool use (UI grounding, ARC-style reasoning, board-game
agents). If the answer is "no without explicit reasoning," it tells us that in-context
geometry induction is *perceptual* rather than *symbolic* and downstream grounding
needs scaffolding.

### Gap in Existing Work
From `literature_review.md`:
- **Curse of CoT** (Zheng et al. 2025) shows direct ≥ CoT on ICL pattern tasks, but the
  *repetition count* is treated as a confound, not an axis.
- **ICL from Repetitions** (Yan et al. 2024) shows surface-level self-reinforcement
  effects, but in pure generation, not in structured addressing.
- **Forgotten Polygons / Stuck in the Matrix** (2025) show shape and grid failures, but
  do not vary the amount of in-prompt repetition or perform a clean direct-vs-CoT
  contrast on *novel* shapes presented as text.

No existing paper holds the geometry fixed and varies (a) the number of in-prompt
demonstrations *about the same geometry* and (b) whether the model is asked to reason.

### Our Novel Contribution
A focused 2×K factorial experiment on **verbal addressing of novel 2D grid shapes**:
- Fix the geometry (a novel irregular shape on a small ASCII grid).
- Vary `K`, the number of in-context demo Q&A pairs about that same grid.
- Vary `reasoning ∈ {direct, CoT}`.
- Measure: exact-match accuracy on held-out addressing questions about the *same* grid.
- Cross-validate across two model families (OpenAI GPT-4.1-mini and an OpenRouter model)
  to control for API idiosyncrasies.

### Experiment Justification
- **Experiment 1 (primary)** — Synthetic novel-shape addressing: We control novelty,
  repetition count, and question type. Lets us isolate the effect of repetition and
  reasoning cleanly. Predicted (from Yan et al. and Curse-of-CoT): accuracy rises
  with K up to a few demos, CoT does not help and may hurt slightly.
- **Experiment 2 (secondary, scaled-down sanity check)** — A small slice of MiniARC
  with direct vs CoT prompting, no repetition manipulation, to confirm our setup
  reproduces the Curse-of-CoT direction on real novel patterns.

---

## Research Question
Can LLMs verbally address (refer to, locate, identify cell values within) novel 2D
geometries presented in a prompt — and how does this ability change as a function of
(a) the number of in-context demonstration questions about the same geometry and
(b) whether explicit chain-of-thought reasoning is elicited?

## Hypothesis Decomposition
- **H1 (repetition helps).** Accuracy on a held-out addressing question about a fixed
  novel geometry strictly increases with the number `K` of in-context demos about the
  same geometry, up to a saturation point.
- **H2 (direct ≥ CoT).** At every K, direct prompting matches or exceeds CoT prompting
  in exact-match accuracy on these addressing tasks (a stronger version of Curse-of-CoT
  on a controlled novel-geometry slice).
- **H3 (low-repetition is hard).** At K=0 (zero demos) and K=1, accuracy on novel
  geometries is substantially below ceiling, *even with* CoT — i.e. limited repetition
  is the dominant constraint, more than reasoning.

If H1+H2 both hold, the answer to the title is: **yes**, novel 2D geometries are
addressable without reasoning, *provided* enough in-context repetition.

## Proposed Methodology

### Approach
Build a synthetic, fully controlled task that mirrors the spirit of the Liu et al.
2D-geometry-in-context paper, but where the *output* is a verbal/symbolic address
rather than a continuous prediction. This lets us cleanly vary repetition and reasoning.

### Experimental Steps

1. **Task generator.** Procedurally generate novel "shapes" on a 7×7 grid using a
   small alphabet of cell tokens (`.` for empty plus 2–4 non-empty tokens like `R`,
   `B`, `G`). Each grid contains a connected, irregular polygon-like region of
   non-empty tokens. We generate via random-walk shape growth so each shape is
   distinct and almost-certainly out-of-training-distribution as a literal string.

2. **Question types.** Five families of "addressing" questions about a grid:
   - **POINT**: "What token is at row r, col c?" (answer: a single character)
   - **LOOKUP**: "At which (row, col) is the token `X`?" (answer: coordinate)
   - **COUNT**: "How many cells contain token `X`?" (integer)
   - **ROWMAX**: "Which row has the most non-empty cells?" (integer)
   - **NEIGHBOR**: "What is the token directly to the right of (r, c)?" (character)

   All questions have a unique, mechanically-checkable ground-truth answer.

3. **Prompt structure.** For each (grid, target-question) pair, build prompts of
   the form:
   ```
   Grid:
   . . R . . . .
   . R R R . . .
   ...
   Q: <demo question 1>
   A: <demo answer 1>
   Q: <demo question 2>
   A: <demo answer 2>
   ...
   Q: <target question>
   A:
   ```
   The number of `(Q, A)` demos before the target is `K ∈ {0, 1, 2, 4, 8}`. All demo
   questions are drawn from the same five families and all reference the *same* grid;
   their answers are the mechanical ground truth.

4. **Reasoning conditions.**
   - **Direct**: just "A: <answer>". For CoT we instead use "A: Let's think step by
     step. ... Final answer: <answer>" and an instruction in the system prompt.
   - We parse the final answer with a tolerant regex that strips quotes / whitespace
     and only accepts the *last* coordinate / character / integer that appears.

5. **Novelty controls.**
   - Each grid is procedurally generated with a fresh seed.
   - We do **not** describe the shape verbally in the prompt (no "this is a heptagon").
   - We use the same five non-empty tokens (`R`, `B`, `G`, `Y`) across grids so the
     model cannot rely on remembering token-specific patterns — only the spatial
     layout differs.

6. **Models.**
   - **Primary**: `gpt-4.1-mini` (OpenAI). Cheap, fast, modern.
   - **Secondary**: `anthropic/claude-haiku-4.5` via OpenRouter, as a second family.
     We will fall back to a different OpenRouter model if Haiku 4.5 is unavailable.

7. **Experiment 2 (sanity check).** Take 25 MiniARC tasks from
   `code/cot_icl_eval/miniarc/miniarc.json`. For each, run direct vs CoT (no
   repetition manipulation — the demos are intrinsic to the task). Report the gap.

### Baselines
- **Random / chance baseline.** Computed analytically per question type (e.g. POINT
  has 1/(alphabet size) ≈ 0.20).
- **K=0 direct baseline** is the natural lower-bound "no help from in-context demos".
- **K=8 direct** is the upper-bound of within-prompt addressing capacity for our budget.

### Evaluation Metrics
- **Exact-match accuracy** per (model × condition × K × question-type) cell.
- **95% bootstrap confidence intervals** (1000 resamples) on per-cell accuracy.
- **Paired comparison** (McNemar test) for direct vs CoT at each K (same items,
  paired binary outcomes).
- **Repetition curve**: accuracy vs K, fit a logarithmic curve; report slope.
- **Cost-normalised accuracy**: accuracy / (mean output tokens per response), to
  account for CoT's higher token cost.

### Statistical Analysis Plan
- Primary test for H2: McNemar's exact test at each K, Bonferroni-corrected across K.
- Primary test for H1: ordinal trend test (Cochran-Armitage / Spearman correlation of
  K vs accuracy across pooled items).
- Significance threshold α = 0.05.

## Expected Outcomes
- **H1 confirmed** if accuracy at K=8 ≥ K=0 with a statistically significant trend.
- **H2 confirmed** if direct accuracy ≥ CoT accuracy at K=8 (one-sided).
- **H3 confirmed** if accuracy at K=0 or K=1 is far from 1.0 even under CoT.

## Timeline and Milestones
- Phase 1 planning: done (this doc).
- Phase 2 env setup: ~5 min (install openai, requests, numpy, pandas, matplotlib).
- Phase 3 implementation: ~30 min (generator, prompting, harness, parser).
- Phase 4 experiments: ~30–60 min API time. Budget: ≤ 1500 calls per model
  (5 K-levels × 2 reasoning × ~150 items ≈ 1500). At gpt-4.1-mini prices
  (~$0.40/M in, ~$1.60/M out) this is well under $5.
- Phase 5 analysis + plots: ~20 min.
- Phase 6 report: ~20 min.

## Potential Challenges
- **Answer parsing.** Especially for CoT, models may verbose. Mitigation: strict
  "Final answer:" suffix in CoT prompts + tolerant regex parser + manual spot-check.
- **API errors / rate limits.** Mitigation: exponential-backoff retry, disk-cached
  responses keyed by hash, max-3 retries.
- **Tokenisation of grids.** Putting spaces between cells (`. R R R .`) prevents the
  tokenizer from merging cell symbols into bigrams.
- **Confound: question difficulty varies across K.** Mitigation: same target question
  *across* K levels for a given grid; only the demo padding changes. This makes
  comparison across K within-item paired.

## Success Criteria
- All 4 experimental conditions (direct/CoT × 5 K-levels) run for both models.
- Per-condition n ≥ 100 target questions.
- Repetition curve and direct-vs-CoT plot present in REPORT.md with CIs.
- At least one McNemar p-value reported.
- A clear, hedged narrative answer to the research question.
