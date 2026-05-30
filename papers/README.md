# Downloaded Papers

All PDFs are stored in this directory. The chunked versions for incremental
reading are in `pages/` (one PDF chunk per 3 source pages, plus a per-paper
manifest).

## Most directly relevant

### 1. The Curse of CoT: On the Limitations of Chain-of-Thought in In-Context Learning
- File: `2504.05081_curse_of_cot.pdf`
- Authors: Zheng et al. (HKUST & NVIDIA), TMLR 11/2025
- arXiv: 2504.05081
- Code: https://github.com/HKUST-KnowComp/CoT-ICL-Eval
- Why relevant: Directly tests whether reasoning is required for **pattern-based
  in-context learning** on ARC-AGI, MiniARC, 1D-ARC, RAVEN, and other symbolic
  grid tasks. Finds Direct Answering consistently beats CoT (relative +20.42%,
  +41.88% on symbolic grids). Identifies a "hybrid explicit-implicit" mechanism
  and shows long-CoT reasoning models (o1, R1) don't fix it despite 40× tokens.
- Key result for us: on symbolic grid tasks, asking the model to *not* reason
  produces *better* answers. This is the strongest existing evidence for our
  hypothesis that novel geometries may be addressable without reasoning.

### 2. To CoT or Not to CoT? Chain-of-Thought Helps Mainly on Math and Symbolic Reasoning
- File: `2409.12183_to_cot_or_not.pdf`
- Authors: Sprague et al. (UT Austin / JHU / Princeton), ICLR 2025
- arXiv: 2409.12183
- Code: https://github.com/Zayne-sprague/To-CoT-or-not-to-CoT
- Why relevant: Meta-analysis of 100+ CoT papers + own experiments on 20 datasets
  × 14 LLMs. CoT only helps substantially on math/symbolic. For non-symbolic
  tasks, direct answering is as good or better. 95% of MMLU CoT-gain is on
  questions containing "=". Gives a principled answer to "when is reasoning
  needed?" — orthogonal but complementary framing to our 2D-geometry question.

### 3. Forgotten Polygons: Multimodal Large Language Models are Shape-Blind
- File: `2502.15969_forgotten_polygons_shape_blind.pdf`
- Authors: Rudman et al. (Brown / TAU / NYU), ACL Findings 2025
- arXiv: 2502.15969
- Code: https://github.com/rsinghlab/Shape-Blind
- Why relevant: Tests whether MLLMs can address novel polygons. Finding: top
  models <50% accuracy on regular polygon ID, near-zero on irregular novel
  shapes — they rely on System-1 memorisation, not actual side-counting. Their
  VC-CoT (Visually-Cued CoT, prompting with annotations like vertex labels)
  jumps GPT-4o from 7% → 93% on counting irregular polygon sides. Directly
  addresses our "novelty" axis and shows that the right *kind* of prompt
  (anchoring on labels) matters more than reasoning per se.

### 4. Mind's Eye of LLMs: Visualization-of-Thought (VoT) Elicits Spatial Reasoning
- File: `2404.03622_visualization_of_thought.pdf`
- Authors: Wu et al. (Microsoft), NeurIPS 2024
- arXiv: 2404.03622
- Why relevant: Tasks include "visual tiling" and "visual navigation" on
  ASCII-grid 2D worlds — basically asking the model to address shapes/positions
  in a grid. VoT (re-rendering the grid at each step) significantly improves
  spatial reasoning over plain CoT. Suggests that explicit *re-visualisation*
  is what helps, not generic reasoning steps — relevant to interpreting any
  CoT-helps result we might observe.

### 5. Stuck in the Matrix: Probing Spatial Reasoning in Large Language Models
- File: `2510.20198_stuck_in_the_matrix.pdf`
- Authors: Bai, Cohen, Koss, Lichtenbaum (2025-10-23)
- arXiv: 2510.20198
- Why relevant: Tests GPT-4o, GPT-4.1, Claude-3.7 Sonnet (No Thinking / Medium
  Thinking) on ASCII 2D grids: quadrant ID, reflection, distance, word-search,
  tile-slide. Average 42.7% accuracy loss as grid scales; 84% worst case.
  Includes a direct "no thinking vs. thinking" contrast on Claude — a baseline
  for our reasoning-required question, restricted to "addressing" type tasks.

### 6. Visually Descriptive Language Model (VDLM) for Vector Graphics Reasoning
- File: `2404.06479_vdlm_vector_graphics.pdf`
- Authors: Wang et al. (UIUC / Stanford / TAMU / NU), TMLR 5/2025
- arXiv: 2404.06479
- Code: https://github.com/MikeWangWZHL/VDLM
- Why relevant: Defines Primal Visual Description (PVD) — a text representation
  of 2D shapes as `{type, vertices, color}`. Lets a *text-only* LLM beat
  GPT-4V on shape-comparison and maze tasks. Concrete template for how to
  verbally represent novel geometries to an LLM.

## Background / supporting

### 7. On the Measure of Intelligence (the ARC paper)
- File: `1911.01547_measure_of_intelligence_arc.pdf`
- Author: François Chollet (Google), arXiv 2019
- Why relevant: Original ARC-AGI definition. Defines the genre of "novel
  pattern induction from a few demos" that all our work sits in. Useful
  framing for "addressing without prior exposure".

### 8. Understanding In-Context Learning from Repetitions
- File: `2310.00297_icl_from_repetitions.pdf`
- Authors: Yan et al., ICLR 2024
- arXiv: 2310.00297
- Code: https://github.com/ElliottYan/understand-icl-from-repetition
- Why relevant: Establishes the *token co-occurrence reinforcement* effect.
  10 repeats of any sequence push p(repeat) → 1.0 even for random tokens.
  Directly relevant to the "repetition" half of our hypothesis: repetition
  in context creates surface-pattern shortcuts that may substitute for
  reasoning — for good or ill.

### 9. Quantifying In-Context Reasoning and Memorization Effects in LLMs
- File: `2405.11880_quantifying_icl_memorization.pdf`
- arXiv: 2405.11880
- Why relevant: Proposes an axiomatic decomposition of ICL contributions into
  *memorisation* vs *reasoning* effects. Gives vocabulary and metrics we can
  reuse when reporting where our LLM's accuracy comes from.

### 10. Chain-of-Thought Reasoning Without Prompting
- File: `2402.10200_cot_without_prompting.pdf`
- arXiv: 2402.10200, NeurIPS 2024
- Why relevant: Shows CoT-like paths often live in top-k alternative tokens
  even under direct-answer prompting. Implies that "no CoT" doesn't fully
  rule out latent reasoning — important caveat for our hypothesis testing.

### 11. Exploring and Improving the Spatial Reasoning Abilities of LLMs
- File: `2312.01054_exploring_spatial_reasoning.pdf`
- arXiv: 2312.01054
- Why relevant: Earlier work on the same axis; provides additional baselines
  and prefix-prompt techniques for spatial reasoning over text.

## Read status

- Deep read (chunks 1–3): #1, #2, #3, #4, #5, #6, #8
- Abstract / intro only: #7, #9, #10, #11

Chunked versions for all of #1, #3, #4, #5, #6, #8 are available under `pages/`
for the experiment-runner agent to re-read targeted sections.
