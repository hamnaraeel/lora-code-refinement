#!/usr/bin/env python
"""Render the figures for the report and the demo.

Reads only files under ``artifacts/`` — the same sources the report uses — so a
figure can never disagree with the number printed next to it. Anything without
data is skipped with a message rather than drawn empty.

    python scripts/make_plots.py [--artifacts artifacts] [--out reports/figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# A single restrained palette, used consistently: base is grey, tuned is blue,
# so the reader learns the mapping once and it holds across every figure.
BASE_C, TUNED_C, ACCENT = "#94a3b8", "#2563eb", "#64748b"


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def plot_training_curves(artifacts: Path, out: Path) -> bool:
    runs = sorted(artifacts.glob("runs/*/metrics.jsonl"))
    if not runs:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("tab10")

    for i, path in enumerate(runs):
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        name = path.parent.name
        train = [(r["_step"], r["loss"]) for r in rows if "loss" in r]
        val = [(r["_step"], r["eval_loss"]) for r in rows if "eval_loss" in r]
        colour = cmap(i % 10)
        if train:
            ax.plot(*zip(*train), lw=1.2, alpha=0.45, color=colour)
        if val:
            ax.plot(*zip(*val), lw=1.8, marker="o", ms=4, color=colour, label=name)
            best_step, best_loss = min(val, key=lambda x: x[1])
            ax.scatter([best_step], [best_loss], s=90, facecolors="none",
                       edgecolors=colour, lw=1.8, zorder=5)

    _style(ax, "Training and validation loss  (solid = validation, faded = train, ring = best checkpoint)",
           "optimiser step", "loss")
    if len(runs) > 1:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=160)
    plt.close(fig)
    return True


def plot_headline(comparison: dict, out: Path) -> bool:
    p = comparison["paired_metrics"]
    keys = ["exact_match", "improved", "edit_line_f1", "token_f1"]
    labels = ["Exact\nmatch", "Moved toward\ngold", "Edit-line\nF1", "Token\nF1"]
    base = [p[k]["base"] for k in keys]
    tuned = [p[k]["tuned"] for k in keys]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = range(len(keys))
    w = 0.36
    ax.bar([i - w / 2 for i in x], base, w, label="base", color=BASE_C)
    ax.bar([i + w / 2 for i in x], tuned, w, label="fine-tuned", color=TUNED_C)
    for i, (b, t) in enumerate(zip(base, tuned)):
        ax.text(i - w / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, t, f"{t:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    _style(ax, f"Base vs fine-tuned  (n = {comparison['headline']['n']})", "", "score")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "headline_metrics.png", dpi=160)
    plt.close(fig)
    return True


def plot_by_category(comparison: dict, out: Path) -> bool:
    items = sorted(comparison["by_category"].items(), key=lambda kv: kv[1]["delta_exact_match"])
    if not items:
        return False
    names = [f"{k}  (n={v['n']})" for k, v in items]
    deltas = [v["delta_exact_match"] * 100 for _, v in items]

    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.42 * len(items) + 1.4)))
    colours = [TUNED_C if d > 0 else "#dc2626" if d < 0 else ACCENT for d in deltas]
    ax.barh(names, deltas, color=colours, height=0.62)
    ax.axvline(0, color="#334155", lw=1)
    for i, d in enumerate(deltas):
        ax.text(d + (0.6 if d >= 0 else -0.6), i, f"{d:+.0f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=8)
    _style(ax, "Exact-match change by feedback category", "percentage points", "")
    ax.grid(axis="y", alpha=0)
    ax.grid(axis="x", alpha=0.25, lw=0.7)
    fig.subplots_adjust(left=0.34, right=0.97, top=0.90, bottom=0.12)
    fig.savefig(out / "by_category.png", dpi=160)
    plt.close(fig)
    return True


def plot_forgetting(artifacts: Path, out: Path) -> bool:
    results = {}
    for path in artifacts.glob("forgetting/*.forgetting.json"):
        data = _load(path)
        if data:
            results[data["tag"]] = data["headline"]
    base = results.get("base")
    tuned = next((v for k, v in results.items() if k != "base"), None)
    if not (base and tuned):
        return False

    labels = ["Multiple-choice\naccuracy", "Instruction\nfollowing", "Spurious code\nfence (lower better)"]
    keys = ["mcq_accuracy", "instruction_pass_rate", "spurious_code_fence_rate"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(3)
    w = 0.36
    ax.bar([i - w / 2 for i in x], [base[k] for k in keys], w, label="base", color=BASE_C)
    ax.bar([i + w / 2 for i in x], [tuned[k] for k in keys], w, label="fine-tuned", color=TUNED_C)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    _style(ax, "General capability retention after fine-tuning", "", "rate")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "forgetting.png", dpi=160)
    plt.close(fig)
    return True


def plot_sweep(artifacts: Path, out: Path) -> bool:
    rows = []
    for path in sorted(artifacts.glob("runs/*/train_summary.json")):
        s = _load(path)
        if s and s.get("best_eval_loss") is not None:
            rows.append((s["run_name"], s["best_eval_loss"]))
    if len(rows) < 2:
        return False
    rows.sort(key=lambda r: r[1], reverse=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.42 * len(rows) + 1.2)))
    colours = [TUNED_C if i == len(rows) - 1 else BASE_C for i in range(len(rows))]
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=colours, height=0.62)
    for i, (_, v) in enumerate(rows):
        ax.text(v, i, f"  {v:.4f}", va="center", fontsize=8)
    _style(ax, "Hyperparameter sweep — best validation loss (blue = selected)", "validation loss", "")
    ax.grid(axis="y", alpha=0); ax.grid(axis="x", alpha=0.25, lw=0.7)
    fig.subplots_adjust(left=0.32, right=0.97, top=0.90, bottom=0.14)
    fig.savefig(out / "sweep.png", dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, default=Path("reports/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    comparison = _load(args.artifacts / "eval" / "comparison.json")
    done, skipped = [], []

    for name, fn in [
        ("training_curves", lambda: plot_training_curves(args.artifacts, args.out)),
        ("sweep", lambda: plot_sweep(args.artifacts, args.out)),
        ("forgetting", lambda: plot_forgetting(args.artifacts, args.out)),
        ("headline_metrics", lambda: comparison and plot_headline(comparison, args.out)),
        ("by_category", lambda: comparison and plot_by_category(comparison, args.out)),
    ]:
        (done if fn() else skipped).append(name)

    for n in done:
        print(f"  wrote  {args.out / (n + '.png')}")
    for n in skipped:
        print(f"  skipped {n} (no data yet)")


if __name__ == "__main__":
    main()
