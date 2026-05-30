"""Novel 2D geometry generator and addressing-question oracle.

A "grid" is a list[list[str]] with cell tokens drawn from EMPTY + COLOURS.
A "shape" is the set of (row, col) cells that hold a non-empty token.

We grow shapes by a constrained random walk so the shape is connected and
irregular (i.e. a "novel polygon" not resembling a square/triangle/etc.).
Cells inside the shape are coloured by a per-grid colour assignment so the
model also has token-level information to address.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Literal

EMPTY = "."
COLOURS = ["R", "B", "G", "Y"]  # token alphabet for non-empty cells
ROWS = 7
COLS = 7

QuestionType = Literal["POINT", "LOOKUP", "COUNT", "ROWMAX", "NEIGHBOR"]
QUESTION_TYPES: list[QuestionType] = ["POINT", "LOOKUP", "COUNT", "ROWMAX", "NEIGHBOR"]


@dataclass
class Grid:
    cells: list[list[str]]  # ROWS x COLS, each in EMPTY | COLOURS
    seed: int
    shape_cells: list[tuple[int, int]] = field(default_factory=list)

    @property
    def grid_id(self) -> str:
        text = "\n".join("".join(row) for row in self.cells)
        return hashlib.sha1(text.encode()).hexdigest()[:10]

    def render(self) -> str:
        return "\n".join(" ".join(row) for row in self.cells)


def _grow_shape(rng: random.Random, target_size: int) -> list[tuple[int, int]]:
    """Random-walk connected blob of `target_size` cells inside a ROWS x COLS grid."""
    start = (rng.randrange(1, ROWS - 1), rng.randrange(1, COLS - 1))
    cells = {start}
    frontier = [start]
    while len(cells) < target_size and frontier:
        r, c = frontier[rng.randrange(len(frontier))]
        neighbours = [(r + dr, c + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        rng.shuffle(neighbours)
        added = False
        for nr, nc in neighbours:
            if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) not in cells:
                cells.add((nr, nc))
                frontier.append((nr, nc))
                added = True
                break
        if not added:
            # remove this exhausted frontier cell to keep the walk moving
            frontier = [p for p in frontier if p != (r, c)]
    return sorted(cells)


def make_grid(seed: int, shape_size: int | None = None) -> Grid:
    """Generate a grid with one connected irregular shape filled with COLOURS."""
    rng = random.Random(seed)
    size = shape_size if shape_size is not None else rng.randint(6, 12)
    shape = _grow_shape(rng, size)
    # Colour assignment: each shape-cell gets a random colour from COLOURS, but
    # bias so each colour appears at least once when shape is large enough.
    cells = [[EMPTY] * COLS for _ in range(ROWS)]
    needed = list(COLOURS)
    rng.shuffle(needed)
    for i, (r, c) in enumerate(shape):
        if i < len(needed):
            cells[r][c] = needed[i]
        else:
            cells[r][c] = rng.choice(COLOURS)
    return Grid(cells=cells, seed=seed, shape_cells=shape)


# ─────────────────────────── Question oracle ────────────────────────────


@dataclass
class Question:
    qtype: QuestionType
    prompt: str        # how it appears in the prompt after "Q: "
    answer: str        # canonical ground-truth answer string
    payload: dict      # debugging metadata


def _q_point(grid: Grid, rng: random.Random) -> Question:
    r, c = rng.randrange(ROWS), rng.randrange(COLS)
    val = grid.cells[r][c]
    return Question(
        qtype="POINT",
        prompt=f"What token is at row {r}, column {c}?",
        answer=val,
        payload={"r": r, "c": c},
    )


def _q_lookup(grid: Grid, rng: random.Random) -> Question:
    """Pick a non-empty token that has at least one occurrence; ask first row-major."""
    present = sorted({grid.cells[r][c] for r in range(ROWS) for c in range(COLS)
                      if grid.cells[r][c] != EMPTY})
    tok = rng.choice(present)
    # canonical "first" (row-major) location
    for r in range(ROWS):
        for c in range(COLS):
            if grid.cells[r][c] == tok:
                return Question(
                    qtype="LOOKUP",
                    prompt=f"What is the (row, column) of the first '{tok}' in row-major order?",
                    answer=f"({r}, {c})",
                    payload={"tok": tok, "r": r, "c": c},
                )
    raise RuntimeError("unreachable")


def _q_count(grid: Grid, rng: random.Random) -> Question:
    tok = rng.choice(COLOURS)
    n = sum(1 for r in range(ROWS) for c in range(COLS) if grid.cells[r][c] == tok)
    return Question(
        qtype="COUNT",
        prompt=f"How many cells contain the token '{tok}'?",
        answer=str(n),
        payload={"tok": tok, "n": n},
    )


def _q_rowmax(grid: Grid, rng: random.Random) -> Question:
    counts = [sum(1 for c in range(COLS) if grid.cells[r][c] != EMPTY) for r in range(ROWS)]
    best = max(counts)
    # ties broken by smallest row index (deterministic)
    row = counts.index(best)
    return Question(
        qtype="ROWMAX",
        prompt="Which row index has the most non-empty cells? "
               "(If tied, give the smallest row index.)",
        answer=str(row),
        payload={"counts": counts, "row": row},
    )


def _q_neighbor(grid: Grid, rng: random.Random) -> Question:
    """Token directly to the right of (r, c). Pick c so c+1 stays in-grid."""
    r = rng.randrange(ROWS)
    c = rng.randrange(COLS - 1)
    val = grid.cells[r][c + 1]
    return Question(
        qtype="NEIGHBOR",
        prompt=f"What token is directly to the right of the cell at row {r}, column {c}?",
        answer=val,
        payload={"r": r, "c": c},
    )


_GENERATORS = {
    "POINT": _q_point,
    "LOOKUP": _q_lookup,
    "COUNT": _q_count,
    "ROWMAX": _q_rowmax,
    "NEIGHBOR": _q_neighbor,
}


def make_question(grid: Grid, qtype: QuestionType, rng: random.Random) -> Question:
    return _GENERATORS[qtype](grid, rng)


def sample_demo_pool(grid: Grid, n: int, rng: random.Random,
                     exclude_signature: str | None = None) -> list[Question]:
    """Sample `n` demo questions about `grid`, balancing across question types,
    excluding any whose (qtype, prompt) signature matches `exclude_signature`."""
    out: list[Question] = []
    attempts = 0
    while len(out) < n and attempts < n * 30:
        attempts += 1
        # round-robin across types to keep demos diverse
        qtype = QUESTION_TYPES[len(out) % len(QUESTION_TYPES)]
        q = make_question(grid, qtype, rng)
        sig = f"{q.qtype}::{q.prompt}"
        if sig == exclude_signature:
            continue
        # also dedupe within demo pool
        if any(f"{x.qtype}::{x.prompt}" == sig for x in out):
            continue
        out.append(q)
    return out


if __name__ == "__main__":
    g = make_grid(seed=42)
    print(g.render())
    print()
    rng = random.Random(0)
    for qt in QUESTION_TYPES:
        q = make_question(g, qt, rng)
        print(f"[{q.qtype}] Q: {q.prompt}")
        print(f"         A: {q.answer}")
