"""Phase 5 analysis: load results/main.csv, compute accuracy curves,
McNemar tests, save tables and figures.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def bootstrap_ci(x: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"))
    samples = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar p-value (two-sided) for discordant pair counts b, c.

    b = number where A correct, B wrong.
    c = number where A wrong,   B correct.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # binomial with p=0.5
    p_one_side = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * p_one_side)


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "main.csv")
    df["K"] = df["K"].astype(int)
    df["correct"] = df["correct"].astype(int)
    return df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Acc, CI per (model, reasoning, K)."""
    rows = []
    for (model, reasoning, K), sub in df.groupby(["model", "reasoning", "K"]):
        x = sub["correct"].to_numpy()
        lo, hi = bootstrap_ci(x)
        rows.append({
            "model": model,
            "reasoning": reasoning,
            "K": int(K),
            "n": len(x),
            "acc": float(x.mean()),
            "ci_lo": lo, "ci_hi": hi,
            "mean_out_tokens": float(sub["output_tokens"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["model", "reasoning", "K"]).reset_index(drop=True)


def per_qtype_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, reasoning, K, qtype), sub in df.groupby(
            ["model", "reasoning", "K", "qtype"]):
        x = sub["correct"].to_numpy()
        lo, hi = bootstrap_ci(x)
        rows.append({
            "model": model,
            "reasoning": reasoning,
            "K": int(K),
            "qtype": qtype,
            "n": len(x),
            "acc": float(x.mean()),
            "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows).sort_values(
        ["model", "reasoning", "K", "qtype"]).reset_index(drop=True)


def mcnemar_table(df: pd.DataFrame) -> pd.DataFrame:
    """Paired direct-vs-CoT McNemar at each (model, K)."""
    rows = []
    for (model, K), sub in df.groupby(["model", "K"]):
        # pair on item_id
        d = sub[sub["reasoning"] == "direct"].set_index("item_id")["correct"]
        c = sub[sub["reasoning"] == "cot"].set_index("item_id")["correct"]
        common = d.index.intersection(c.index)
        d = d.loc[common].to_numpy()
        c = c.loc[common].to_numpy()
        # b = direct correct, cot wrong; c = direct wrong, cot correct
        b_cnt = int(((d == 1) & (c == 0)).sum())
        c_cnt = int(((d == 0) & (c == 1)).sum())
        p = mcnemar_exact(b_cnt, c_cnt)
        rows.append({
            "model": model,
            "K": int(K),
            "n_paired": len(d),
            "direct_acc": float(d.mean()),
            "cot_acc": float(c.mean()),
            "b_direct_only": b_cnt,
            "c_cot_only": c_cnt,
            "mcnemar_p": p,
        })
    return pd.DataFrame(rows).sort_values(["model", "K"]).reset_index(drop=True)


def repetition_mcnemar(df: pd.DataFrame, k_low: int = 0, k_high: int = 8) -> pd.DataFrame:
    """Paired K=k_low vs K=k_high McNemar test for each (model, reasoning)."""
    rows = []
    for (model, reasoning), sub in df.groupby(["model", "reasoning"]):
        a = sub[sub["K"] == k_low].set_index("item_id")["correct"]
        b = sub[sub["K"] == k_high].set_index("item_id")["correct"]
        common = a.index.intersection(b.index)
        a = a.loc[common].to_numpy()
        b = b.loc[common].to_numpy()
        only_low_correct = int(((a == 1) & (b == 0)).sum())
        only_high_correct = int(((a == 0) & (b == 1)).sum())
        p = mcnemar_exact(only_low_correct, only_high_correct)
        rows.append({
            "model": model,
            "reasoning": reasoning,
            "K_low": k_low, "K_high": k_high,
            "n_paired": len(a),
            "acc_low": float(a.mean()),
            "acc_high": float(b.mean()),
            "only_low_correct": only_low_correct,
            "only_high_correct": only_high_correct,
            "mcnemar_p": p,
        })
    return pd.DataFrame(rows)


def trend_test(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of K vs item-level correctness, per (model, reasoning)."""
    rows = []
    for (model, reasoning), sub in df.groupby(["model", "reasoning"]):
        rho, p = stats.spearmanr(sub["K"].to_numpy(), sub["correct"].to_numpy())
        rows.append({
            "model": model,
            "reasoning": reasoning,
            "n": len(sub),
            "spearman_rho_K_vs_correct": float(rho),
            "p_value": float(p),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────── plotting ───────────────────────────────


def plot_repetition_curves(df: pd.DataFrame, out: Path) -> None:
    summ = summary_table(df)
    models = sorted(summ["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]
    colours = {"direct": "tab:blue", "cot": "tab:orange"}
    markers = {"direct": "o", "cot": "s"}
    for ax, model in zip(axes, models):
        sub = summ[summ["model"] == model]
        for reasoning, s in sub.groupby("reasoning"):
            s = s.sort_values("K")
            ax.errorbar(s["K"], s["acc"],
                        yerr=[s["acc"] - s["ci_lo"], s["ci_hi"] - s["acc"]],
                        marker=markers[reasoning], capsize=3,
                        color=colours[reasoning], label=reasoning)
        ax.set_title(model)
        ax.set_xlabel("K (in-context demos about the same grid)")
        ax.set_xticks(sorted(df["K"].unique()))
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Exact-match accuracy")
    fig.suptitle("Verbal addressing of novel 2D geometries: accuracy vs repetition")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_qtype_breakdown(df: pd.DataFrame, out: Path) -> None:
    pq = per_qtype_table(df)
    # one panel per question type, x=K, lines per (model, reasoning)
    qtypes = sorted(pq["qtype"].unique())
    fig, axes = plt.subplots(1, len(qtypes), figsize=(3.4 * len(qtypes), 4),
                             sharey=True)
    if len(qtypes) == 1:
        axes = [axes]
    style = {
        ("gpt-4.1-mini", "direct"):     {"color": "tab:blue", "ls": "-"},
        ("gpt-4.1-mini", "cot"):        {"color": "tab:blue", "ls": "--"},
        ("claude-haiku-4.5", "direct"): {"color": "tab:orange", "ls": "-"},
        ("claude-haiku-4.5", "cot"):    {"color": "tab:orange", "ls": "--"},
    }
    for ax, qt in zip(axes, qtypes):
        sub = pq[pq["qtype"] == qt].sort_values("K")
        for (model, reasoning), s in sub.groupby(["model", "reasoning"]):
            st = style.get((model, reasoning), {})
            ax.plot(s["K"], s["acc"], marker="o", label=f"{model}/{reasoning}", **st)
        ax.set_title(qt)
        ax.set_xticks(sorted(df["K"].unique()))
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("K")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Accuracy by question type")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost_normalised(df: pd.DataFrame, out: Path) -> None:
    """Accuracy per output token: efficiency comparison direct vs CoT."""
    rows = []
    for (model, reasoning), sub in df.groupby(["model", "reasoning"]):
        acc = sub["correct"].mean()
        tok = sub["output_tokens"].mean()
        rows.append({"model": model, "reasoning": reasoning,
                     "acc": acc, "mean_out_tokens": tok,
                     "acc_per_100tokens": acc / max(tok, 1) * 100})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(d))
    ax.bar(x - 0.2, d["acc"], width=0.4, label="accuracy")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, d["mean_out_tokens"], width=0.4,
            label="mean output tokens", color="tab:gray", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{r}" for m, r in zip(d["model"], d["reasoning"])],
                       fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax2.set_ylabel("mean output tokens")
    ax.set_title("Accuracy and output-token cost")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ───────────────────────────── error analysis ─────────────────────────


def error_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Confusion-style table per qtype: parse_failure rate, miss-by-1 etc."""
    rows = []
    for (model, reasoning, qtype), sub in df.groupby(
            ["model", "reasoning", "qtype"]):
        wrong = sub[sub["correct"] == 0]
        n = len(sub)
        n_wrong = len(wrong)
        n_parse_fail = int((wrong["parsed"] == "").sum())
        # numeric off-by-one for COUNT / ROWMAX / LOOKUP
        off_by_one = 0
        if qtype in ("COUNT", "ROWMAX") and n_wrong:
            for _, r in wrong.iterrows():
                try:
                    if abs(int(r["parsed"]) - int(r["gold"])) == 1:
                        off_by_one += 1
                except Exception:
                    pass
        rows.append({
            "model": model,
            "reasoning": reasoning,
            "qtype": qtype,
            "n": n,
            "n_wrong": n_wrong,
            "error_rate": n_wrong / n if n else float("nan"),
            "parse_failures": n_parse_fail,
            "off_by_one": off_by_one,
        })
    return pd.DataFrame(rows)


# ────────────────────────────── main ─────────────────────────────────


def main() -> None:
    df = load()
    print(f"Loaded {len(df)} rows, {df['item_id'].nunique()} unique items")

    summ = summary_table(df)
    summ.to_csv(RESULTS / "summary_acc.csv", index=False)
    print("\n=== Accuracy by (model, reasoning, K) ===")
    print(summ.to_string(index=False))

    pq = per_qtype_table(df)
    pq.to_csv(RESULTS / "summary_acc_by_qtype.csv", index=False)

    mcn = mcnemar_table(df)
    mcn.to_csv(RESULTS / "mcnemar.csv", index=False)
    print("\n=== Direct vs CoT (McNemar) ===")
    print(mcn.to_string(index=False))

    rep = repetition_mcnemar(df)
    rep.to_csv(RESULTS / "mcnemar_repetition.csv", index=False)
    print("\n=== K=0 vs K=8 (McNemar) ===")
    print(rep.to_string(index=False))

    trend = trend_test(df)
    trend.to_csv(RESULTS / "trend.csv", index=False)
    print("\n=== Trend (Spearman K vs correctness) ===")
    print(trend.to_string(index=False))

    err = error_breakdown(df)
    err.to_csv(RESULTS / "error_breakdown.csv", index=False)
    print("\n=== Error breakdown ===")
    print(err.to_string(index=False))

    plot_repetition_curves(df, FIGURES / "repetition_curves.png")
    plot_qtype_breakdown(df, FIGURES / "qtype_breakdown.png")
    plot_cost_normalised(df, FIGURES / "cost_normalised.png")
    print(f"\nFigures saved to {FIGURES}/")

    # also write a one-shot JSON summary
    out_json = {
        "n_items": int(df["item_id"].nunique()),
        "models": sorted(df["model"].unique().tolist()),
        "K_levels": sorted(df["K"].unique().tolist()),
        "headline_accuracy": summ.to_dict(orient="records"),
        "mcnemar": mcn.to_dict(orient="records"),
        "trend": trend.to_dict(orient="records"),
    }
    (RESULTS / "summary.json").write_text(json.dumps(out_json, indent=2))


if __name__ == "__main__":
    main()
