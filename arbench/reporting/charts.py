"""Static charts (PNG) for the Markdown / HTML reports.

Design rules (validated palette, see README): one colour per model, fixed by config order
(colour follows the entity, never its rank); one y-axis per chart; thin bars with
surface gaps; recessive grid; values direct-labelled (the 3rd slot is < 3:1 contrast,
so labels + table views are mandatory relief).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("arbench.charts")

# validated categorical slots 1-3 (all-pairs CVD-safe); extra models fall back to slots 4-8
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"


def model_colors(models: list[str]) -> dict[str, str]:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def _style(ax, title: str, ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT2, fontsize=9)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def grouped_bars(
    groups: list[str],
    series: dict[str, list[float | None]],
    labels: dict[str, str],
    colors: dict[str, str],
    title: str,
    path: Path,
    errors: dict[str, list[tuple[float, float] | None]] | None = None,
    ylim: tuple[float, float] = (0, 1.05),
    ylabel: str = "score",
    pct: bool = True,
) -> Path:
    n = len(series)
    width = min(0.8 / max(n, 1), 0.28)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.5 + 1.3 * len(groups) * max(1, n) * 0.55), 4.2))
    for si, (model, vals) in enumerate(series.items()):
        xs = [gi + (si - (n - 1) / 2) * width for gi in range(len(groups))]
        ys = [v if v is not None else 0 for v in vals]
        bars = ax.bar(xs, ys, width=width, color=colors[model], label=labels[model], edgecolor=SURFACE, linewidth=2)
        tops = list(ys)
        if errors and errors.get(model):
            for k, (x, ci) in enumerate(zip(xs, errors[model])):
                if ci:
                    ax.plot([x, x], [ci[0], ci[1]], color=TEXT2, linewidth=0.8, alpha=0.6)
                    tops[k] = max(tops[k], ci[1])
        for b, v, top in zip(bars, vals, tops):
            if v is None:
                ax.text(b.get_x() + b.get_width() / 2, 0.01, "n/a", ha="center", va="bottom", fontsize=7, color=TEXT2)
                continue
            txt = f"{v * 100:.0f}" if pct else f"{v:.1f}"
            ax.text(
                b.get_x() + b.get_width() / 2,
                top + (ylim[1] - ylim[0]) * 0.01,
                txt,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=TEXT,
            )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=0 if len(groups) <= 5 else 25, ha="center" if len(groups) <= 5 else "right")
    ax.set_ylim(*ylim)
    _style(ax, title, ylabel)
    if n >= 2:
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0), labelcolor=TEXT)
    return _save(fig, path)


def leaderboard(models: list[str], scores: dict[str, float], labels: dict[str, str], colors: dict[str, str], path: Path):
    order = sorted(models, key=lambda m: scores[m])
    fig, ax = plt.subplots(figsize=(7, 0.7 + 0.6 * len(models)))
    bars = ax.barh(
        [labels[m] for m in order], [scores[m] for m in order], color=[colors[m] for m in order], height=0.55,
        edgecolor=SURFACE, linewidth=2,
    )
    for b, m in zip(bars, order):
        ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, f"{scores[m] * 100:.1f}", va="center",
                fontsize=9, color=TEXT)
    ax.set_xlim(0, 1.08)
    _style(ax, "ARBench composite score (0-100)")
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.spines["bottom"].set_visible(False)
    return _save(fig, path)


def quality_vs_speed(models, quality: dict, speed: dict, labels, colors, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in models:
        ax.scatter(speed[m], quality[m], s=110, color=colors[m], edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(labels[m], (speed[m], quality[m]), textcoords="offset points", xytext=(8, 6), fontsize=8.5,
                    color=TEXT)
    ax.set_xlabel("generation speed (tokens / s, higher is better)", color=TEXT2, fontsize=9)
    ax.set_ylim(0, 1.05)
    xmax = max(speed.values()) if speed else 1
    ax.set_xlim(0, xmax * 1.3 + 1)
    _style(ax, "Answer quality vs. speed", "quality (mean of task scores)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    return _save(fig, path)


def single_model_tasks(tasks: list[str], values: list[float | None], cis, color: str, title: str, path: Path):
    return grouped_bars(tasks, {"m": values}, {"m": ""}, {"m": color}, title, path, errors={"m": cis})
