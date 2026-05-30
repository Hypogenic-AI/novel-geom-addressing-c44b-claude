"""Prompt builders for the addressing-task experiment.

Two reasoning conditions:
  - "direct": A: <answer>
  - "cot":    A: think briefly. Final answer: <answer>

The prompt always presents the grid, then K (Q, A) demo pairs about the same
grid (with mechanical ground-truth answers), then the target question.
"""
from __future__ import annotations

from dataclasses import dataclass

from geometry import Grid, Question


DIRECT_SYSTEM = (
    "You answer questions about a small 2D grid of tokens. "
    "Each cell is either '.' (empty) or one of the colour letters R, B, G, Y. "
    "Rows and columns are 0-indexed. "
    "Answer each question with ONLY the final answer — a single character, "
    "a single integer, or a (row, column) tuple — no explanation."
)

COT_SYSTEM = (
    "You answer questions about a small 2D grid of tokens. "
    "Each cell is either '.' (empty) or one of the colour letters R, B, G, Y. "
    "Rows and columns are 0-indexed. "
    "Think step by step about how to find the answer, then on the LAST line "
    "write exactly: 'Final answer: <answer>' where <answer> is a single "
    "character, a single integer, or a (row, column) tuple."
)


@dataclass
class BuiltPrompt:
    system: str
    user: str


def build_prompt(grid: Grid, demos: list[Question], target: Question,
                 reasoning: str) -> BuiltPrompt:
    """`demos` is the K in-context examples (already ground-truth-answered).
    `target` is the held-out question whose answer the model must produce.
    `reasoning ∈ {"direct", "cot"}`.
    """
    if reasoning == "direct":
        system = DIRECT_SYSTEM
    elif reasoning == "cot":
        system = COT_SYSTEM
    else:
        raise ValueError(reasoning)

    lines: list[str] = ["Grid:", grid.render(), ""]
    for d in demos:
        lines.append(f"Q: {d.prompt}")
        if reasoning == "direct":
            lines.append(f"A: {d.answer}")
        else:
            # For CoT demos we keep them concise: skip the reasoning, just the
            # "Final answer:" suffix. This avoids modelling reasoning chains that
            # we'd then need to ground-truth annotate.
            lines.append(f"A: Final answer: {d.answer}")
        lines.append("")
    lines.append(f"Q: {target.prompt}")
    lines.append("A:")
    user = "\n".join(lines)
    return BuiltPrompt(system=system, user=user)


# ─────────────────────────── Answer parsing ───────────────────────────


import re

_COORD_RE = re.compile(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?")
_INT_RE = re.compile(r"-?\d+")
_CHAR_RE = re.compile(r"[\.RBGY]")
_FINAL_RE = re.compile(r"final\s*answer\s*[:\-]?\s*(.+?)$", re.IGNORECASE | re.MULTILINE)


def parse_answer(raw: str, qtype: str) -> str | None:
    """Tolerantly extract the model's final answer for question of `qtype`.
    Returns None if parsing fails.
    """
    if raw is None:
        return None
    text = raw.strip()
    # if 'Final answer:' marker exists, look only after it
    m = _FINAL_RE.search(text)
    if m:
        text = m.group(1).strip()
    # remove only trailing sentence punctuation (not '.' since it is a valid token)
    text = text.strip().strip("!?,;").strip()

    if qtype == "LOOKUP":
        # take the LAST coord-like match in the residual text
        matches = list(_COORD_RE.finditer(text))
        if not matches:
            return None
        r, c = matches[-1].group(1), matches[-1].group(2)
        return f"({r}, {c})"

    if qtype in ("COUNT", "ROWMAX"):
        # last integer match
        matches = list(_INT_RE.finditer(text))
        if not matches:
            return None
        return matches[-1].group(0)

    if qtype in ("POINT", "NEIGHBOR"):
        # last character match from the allowed alphabet
        matches = list(_CHAR_RE.finditer(text))
        if not matches:
            return None
        return matches[-1].group(0)

    return None


def is_correct(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    # numeric normalisation
    return predicted.strip() == gold.strip()


if __name__ == "__main__":
    # quick parser sanity
    assert parse_answer("Final answer: (1, 0)", "LOOKUP") == "(1, 0)"
    assert parse_answer("Let me see... it is 3", "COUNT") == "3"
    assert parse_answer("Final answer: R", "POINT") == "R"
    assert parse_answer("Final answer: .", "NEIGHBOR") == "."
    assert parse_answer("3", "ROWMAX") == "3"
    print("parser ok")
