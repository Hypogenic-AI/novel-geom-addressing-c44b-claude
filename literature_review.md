# Literature Review

**Research question.** Can LLMs verbally address 2D geometries through
prompting — especially with limited repetition — or is explicit reasoning
(chain-of-thought, decomposition, etc.) required?

By "address" we mean: refer to, identify, name, locate, or otherwise
correctly produce output about a 2D shape presented in the prompt. By
"novel geometries" we mean shapes the model has not encountered in
training (irregular polygons, ARC-style symbolic grids, ad-hoc spatial
configurations).

## 1. Research-area overview

Three streams of recent work bear directly on this question:

1. **CoT skepticism.** A growing body of evidence (Sprague et al. 2024;
   Zheng et al. 2025) shows chain-of-thought helps mainly on math and
   symbolic execution. On *pattern-induction* tasks where the rule must
   be inferred from a few demos, direct answering often beats CoT.
2. **Spatial / shape reasoning in LLMs and MLLMs.** Both text-only LLMs
   (Bai et al. 2025 — "Stuck in the Matrix") and multimodal models
   (Rudman et al. 2025 — "Forgotten Polygons") fail on tasks that
   require *actually counting* sides, vertices, or grid positions of
   unfamiliar shapes. They fall back on memorised label associations.
3. **Prompt-structure interventions.** Visualization-of-Thought (Wu et
   al. 2024), VDLM/PVD (Wang et al. 2025), VC-CoT (Rudman et al. 2025),
   and prefix-based prompting all show that the *form* of the prompt —
   re-rendering the grid, providing a vertex schema, labelling
   vertices — often matters more than whether explicit reasoning steps
   are produced.

The intersection — *novel* 2D geometries described *verbally*, varying
*amount of repetition*, with/without explicit reasoning — is essentially
the gap our research targets.

## 2. Key papers

### Zheng et al. 2025 — *The Curse of CoT* (TMLR'25, arXiv 2504.05081)
- **Setup.** 16 LLMs × 9 pattern-based ICL benchmarks (ARC-AGI, MiniARC,
  1D-ARC, RAVEN, SCAN, MiniSCAN, COGS, SALT, List-Function). Four
  prompting paradigms: Direct, CoT, ReAct, Tree-of-Thought.
- **Key finding.** Direct beats CoT by +20.42% relative on average,
  +41.88% on symbolic-grid tasks. Gap *widens* with more demonstrations.
  Even o1-style long-CoT models with 40× more inference tokens fail to
  recover the gap.
- **Mechanism.** A hybrid explicit/implicit reasoning model: CoT injects
  noise via flawed rationales (explicit fails) *and* increases contextual
  distance between demos and the answer (disrupts implicit ICL).
- **Why central for us.** This is the strongest published evidence that
  on novel symbolic-grid tasks (which include ARC), reasoning is *not*
  the bottleneck — in fact it hurts. Our experiment can replicate this
  on a controlled novel-geometry slice and extend it to a "limited
  repetition" axis.

### Sprague et al. 2025 — *To CoT or Not to CoT* (ICLR'25, arXiv 2409.12183)
- **Setup.** Meta-analysis of 100+ CoT papers; own evaluation on 20
  datasets × 14 LLMs across 5 reasoning categories.
- **Finding.** CoT helps substantially only on math + symbolic. On
  MMLU, 95% of CoT's gain comes from questions containing "=". On soft
  reasoning, CoT is neutral or harmful.
- **Why relevant.** Provides the *taxonomy* in which our research sits:
  "addressing 2D geometries" is closer to *spatial/pattern induction*
  than to *symbolic execution*, so by Sprague's taxonomy CoT is unlikely
  to help — testable prediction for our experiment.

### Rudman et al. 2025 — *Forgotten Polygons* (ACL Findings, arXiv 2502.15969)
- **Setup.** 13 MLLMs tested on regular polygons (triangle → octagon),
  irregular novel polygons, two-shape arithmetic, and a vision-encoder
  embedding probe.
- **Findings.** (a) <50% accuracy on identifying common regular polygons;
  near-zero on novel shapes. (b) Vision encoders cluster only common
  shapes; rare ones overlap → "shape-blindness". (c) Underlying LLMs
  *do* know the names and side-counts when asked text-only — the failure
  is at the perception layer. (d) **VC-CoT** (visually labelling vertices
  with letters and asking the model to enumerate them) jumps GPT-4o from
  7% → 93% on irregular-polygon side-counting.
- **Why relevant.** Directly tests our "novelty" axis on shapes. Their
  VC-CoT result is the canonical example of "addressing via labels
  enables zero-shot success without complex reasoning" — the strongest
  positive result for our hypothesis.

### Wu et al. 2024 — *Visualization-of-Thought* (NeurIPS, arXiv 2404.03622)
- ASCII 2D grids for navigation and tiling. VoT (re-render the grid each
  step) significantly improves over plain CoT and even over MLLMs.
- **Implication.** When reasoning *does* help, it's because the model is
  forced to refresh its internal grid representation. This suggests an
  experimental design: directly compare *re-rendering* vs. *generic
  CoT* vs. *direct* answering on novel grid shapes.

### Bai et al. 2025 — *Stuck in the Matrix* (arXiv 2510.20198)
- Tests Claude 3.7 Sonnet (No Thinking vs 16k Thinking tokens), GPT-4o,
  GPT-4.1 on 5 ASCII-grid spatial tasks scaled by grid size.
- Average 42.7% accuracy drop with scale; up to 84%. Includes the
  thinking/no-thinking contrast on a single model family — exactly the
  comparison our hypothesis demands.

### Wang et al. 2025 — *VDLM* (TMLR'25, arXiv 2404.06479)
- Defines **Primal Visual Description (PVD)**: a JSON-style text
  representation of 2D shapes as `{type, vertices, color, style}`.
- A text-only LLM consuming PVD beats GPT-4V on shape comparison, mazes,
  and other low-level reasoning tasks.
- **Take-away.** The *form* in which a shape is verbalised matters a lot;
  vertex-list / primitive form is a strong baseline for our experiment.

### Yan et al. 2024 — *Understanding ICL from Repetitions* (ICLR, arXiv 2310.00297)
- 10 repeats of any sequence push its predicted probability to ≈1.0,
  even for random tokens (self-reinforcement / token co-occurrence
  reinforcement).
- **Implication for us.** "Limited repetition" is exactly the
  intermediate regime where surface-pattern reinforcement starts to bite.
  Repetition will likely *help addressing accuracy* up to a point, then
  potentially lock in spurious patterns. We can test this directly.

### Zheng et al. 2024 — *Quantifying ICL Reasoning vs Memorization* (arXiv 2405.11880)
- Provides an axiomatic decomposition of what an LLM's ICL prediction
  attributes to memorisation vs. reasoning.
- **Use.** Vocabulary + metrics we can adopt when reporting which
  fraction of accuracy on our task comes from memorised priors vs.
  in-context induction.

### Chollet 2019 — *On the Measure of Intelligence* (arXiv 1911.01547)
- Origin of ARC-AGI. Frames novel-pattern induction as the canonical
  test of human-like generalisation. Sets up the broader question our
  hypothesis is a specific case of.

## 3. Common methodologies in this area

- **Pattern-induction prompting.** Show K input→output examples,
  ask for the output of a held-out input. K is usually 2–10.
- **Direct vs CoT prompting.** Same prompt; CoT adds a "reasoning"
  slot. Both Curse-of-CoT and To-CoT-or-Not-to-CoT use this contrast.
- **Reasoning-token sweeps.** On thinking-token models (Claude 3.7,
  o1, DeepSeek-R1), vary the budget (0 / few-K / many-K) and watch
  accuracy. *Stuck in the Matrix* uses this directly.
- **Text representation variants.** Vertex lists, ASCII grids, SVG,
  PVD JSON — all are used somewhere in the literature for "address
  this shape" tasks.
- **Novelty controls.** Use procedurally generated shapes (Shape-Blind
  abstract polygons; arc_1d generator; MiniARC's hand-crafted novel
  patterns) to ensure absence from pretraining.

## 4. Standard baselines

| Baseline | Source | Notes |
|----------|--------|-------|
| Zero-shot direct answer | most papers | trivial floor |
| Few-shot direct answer (K=2–10) | Curse of CoT | strongest in many cases |
| Zero/few-shot CoT ("think step by step") | Sprague et al. | usual reasoning ablation |
| ReAct / ToT | Curse of CoT | more elaborate reasoning frameworks; usually worse on ICL |
| Visualization-of-Thought | Wu et al. | re-render grid each step |
| VC-CoT (label vertices, enumerate) | Rudman et al. | strongest *prompt-structure* baseline for shapes |
| PVD-style structured input | Wang et al. | reformat the geometry rather than reformat the reasoning |
| "Thinking tokens" on/off | Stuck in the Matrix | clean within-model reasoning contrast |

## 5. Evaluation metrics

- **Exact-match accuracy** on the target output (the standard for ARC
  and Curse-of-CoT).
- **Partial-match metrics** like Modified Levenshtein Distance (used in
  ArtPerception for ASCII art recognition) — useful when "addressing"
  means producing a structured description rather than a single label.
- **Token-cost-normalised accuracy** — important if we're claiming
  "direct ≥ CoT" given CoT's much higher inference cost.
- **Sensitivity to repetition count** — accuracy as a function of
  number of in-prompt repetitions (matches our hypothesis directly).

## 6. Datasets we can use immediately

| Dataset | Source | Novel? | Format |
|---------|--------|--------|--------|
| ARC-AGI | Chollet 2019 / `code/arc_agi_official` | yes (held-out) | grid JSON |
| MiniARC | `code/cot_icl_eval/miniarc/` | yes, hand-crafted | 5×5 grid |
| 1D-ARC | `code/cot_icl_eval/1d_arc/` | yes | 1×N grid |
| Regular polygons | `code/shape_blind/.../regular_polygons.csv` | partly novel (rotation/colour) | image (we can transcribe) |
| Abstract polygons | `code/shape_blind/.../abstract_shapes.csv` | yes | image |
| arc_1d generator | `code/arc_1d/` | fully synthetic | grid |
| Shape-Blind generators | `code/shape_blind/image_generation_code/` | controllable | image |

## 7. Gaps and opportunities specific to our hypothesis

- **Verbal addressing of novel 2D shapes** (not just identification of
  named polygons) is under-studied. Most work either asks for a label or
  a numeric answer; few ask for *referring expressions* or *position
  descriptions* of geometric features.
- **Repetition as an axis** is rarely treated as a free variable. Yan et
  al. show the self-reinforcement effect but in a generation setting,
  not in a structured pattern-induction setting. Curse-of-CoT varies the
  demo count but treats it as a confound, not a hypothesis variable.
- **Within-model thinking-budget sweep on novel 2D-geometry tasks** is
  what Stuck-in-the-Matrix begins, but only for navigation-style tasks,
  and not aligned with the repetition axis.

## 8. Recommendations for our experiment

Based on the literature, the strongest design is:

1. **Task.** "Addressing" a 2D geometry presented as text. Two flavours:
   (a) ARC-style grid transformations (replicating Curse-of-CoT) and
   (b) novel polygon description (extending Forgotten-Polygons to a
   text-only, vertex-list setting using PVD).
2. **Conditions.**
   - *Reasoning*: direct / CoT / thinking-budget sweep on a model that
     supports it.
   - *Repetition*: number of identical demonstrations in the context,
     1, 2, 4, 8, 16. Holds task structure fixed; varies the surface
     repetition signal.
   - *Novelty*: 3 buckets — common shapes, rare-but-namable shapes,
     fully procedurally novel shapes.
3. **Baselines.** Direct (zero-shot), Direct (few-shot), CoT (zero-
   and few-shot). Optional: VoT-style re-rendering, VC-CoT-style
   vertex labels.
4. **Metrics.** Exact-match accuracy, plus accuracy-per-token; sweep
   over repetition count to test the *self-reinforcement → benefit*
   curve.
5. **Predictions from the literature.**
   - On novel symbolic-grid tasks: direct ≥ CoT (Curse of CoT, To-CoT).
   - On novel polygon-addressing: direct ≈ CoT unless labels are
     provided; structured representation (PVD / vertex labels) is the
     biggest lever (Forgotten Polygons, VDLM).
   - Repetition helps up to ~3–5 copies (analogy capture) and then
     plateaus or hurts via spurious surface reinforcement (ICL from
     Repetitions).

If the data confirms these predictions, the answer to our title is:
**yes**, novel 2D geometries are largely addressable without reasoning,
provided the *representation* and the *amount of repetition* are right.

Sources:
- [Curse of CoT (arXiv 2504.05081)](https://arxiv.org/abs/2504.05081)
- [To CoT or Not to CoT (arXiv 2409.12183)](https://arxiv.org/abs/2409.12183)
- [Forgotten Polygons (arXiv 2502.15969)](https://arxiv.org/abs/2502.15969)
- [Visualization-of-Thought (arXiv 2404.03622)](https://arxiv.org/abs/2404.03622)
- [Stuck in the Matrix (arXiv 2510.20198)](https://arxiv.org/abs/2510.20198)
- [VDLM (arXiv 2404.06479)](https://arxiv.org/abs/2404.06479)
- [ICL from Repetitions (arXiv 2310.00297)](https://arxiv.org/abs/2310.00297)
- [Quantifying ICL Reasoning vs Memorization (arXiv 2405.11880)](https://arxiv.org/abs/2405.11880)
- [On the Measure of Intelligence (arXiv 1911.01547)](https://arxiv.org/abs/1911.01547)
- [Exploring Spatial Reasoning Abilities of LLMs (arXiv 2312.01054)](https://arxiv.org/abs/2312.01054)
- [Chain-of-Thought Reasoning Without Prompting (arXiv 2402.10200)](https://arxiv.org/abs/2402.10200)
