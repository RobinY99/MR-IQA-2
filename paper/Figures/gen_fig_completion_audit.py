#!/usr/bin/env python3
"""Generate training and validation figures for the completion-credit audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


OUTPUT_DIR = Path(__file__).resolve().parent
EPOCHS = np.asarray([1, 2, 3, 5], dtype=int)

# Exact four-rank epoch means from the 2026-08-10 experiment audit.
TRAINING = {
    "Mask": {
        "reasoning": [0.206967, 0.273531, 0.290843, 0.306970],
        "rating": [0.908946, 0.933347, 0.938825, 0.957583],
        "judge_gain": [0.585749, 0.755868, 0.800368, 0.839973],
        "length": [100.565, 107.574, 108.395, 109.002],
    },
    "Without Mask": {
        "reasoning": [0.294613, 0.454589, 0.384219, 0.472984],
        "rating": [0.896697, 0.908073, 0.895361, 0.955658],
        "judge_gain": [0.803710, 1.197148, 1.034881, 1.216596],
        "length": [89.388, 99.390, 91.846, 85.257],
    },
}

# Exact 200-image validation metrics at the retained checkpoints.
VALIDATION = {
    "Mask": {
        "plcc": [0.877854, 0.916015, 0.928323, 0.935394],
        "srcc": [0.883833, 0.897803, 0.916390, 0.919533],
        "judge_gain": [0.685800, 0.764619, 0.793250, 0.805100],
        "norm_unique": [
            199 / 200 * 100,
            196 / 197 * 100,
            199 / 200 * 100,
            199 / 200 * 100,
        ],
        "template": [0.0, 0.0, 0.0, 0.0],
    },
    "Without Mask": {
        "plcc": [0.867164, 0.914423, 0.925644, 0.928980],
        "srcc": [0.878563, 0.899606, 0.909860, 0.915821],
        "judge_gain": [1.173096, 1.180600, 1.174619, 1.183700],
        "norm_unique": [
            181 / 197 * 100,
            1 / 200 * 100,
            1 / 197 * 100,
            1 / 200 * 100,
        ],
        "template": [100.0, 100.0, 100.0, 100.0],
    },
}

STYLES = {
    "Mask": {
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "Without Mask": {
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
    },
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def validate_data() -> None:
    for collection in (TRAINING, VALIDATION):
        if set(collection) != set(STYLES):
            raise ValueError("Every plotted configuration must define a style")
        for configuration, metrics in collection.items():
            for metric, values in metrics.items():
                array = np.asarray(values, dtype=float)
                if array.shape != EPOCHS.shape or not np.isfinite(array).all():
                    raise ValueError(
                        f"Invalid {metric} series for {configuration}: {array}"
                    )
                if metric in {"plcc", "srcc"} and not np.all(
                    (-1 <= array) & (array <= 1)
                ):
                    raise ValueError(f"Invalid correlation values for {configuration}")
                if metric in {"norm_unique", "template"} and not np.all(
                    (0 <= array) & (array <= 100)
                ):
                    raise ValueError(f"Invalid percentage values for {configuration}")


def style_axis(
    ax: plt.Axes,
    title: str,
    ylim: tuple[float, float],
    tick_format: str,
) -> None:
    ax.set_title(title, loc="left", pad=3)
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(*ylim)
    ax.set_xticks(EPOCHS)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FormatStrFormatter(tick_format))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


def plot_configurations(ax: plt.Axes, collection: dict, metric: str) -> list:
    handles = []
    for configuration, values in collection.items():
        style = STYLES[configuration]
        (line,) = ax.plot(
            EPOCHS,
            values[metric],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=configuration,
            zorder=3,
        )
        handles.append(line)
    return handles


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def build_training_figure() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.4), sharex=True)
    panels = (
        ("reasoning", r"(a) Reasoning reward $\uparrow$", (0.18, 0.50), "%.2f"),
        ("rating", r"(b) Rating reward $\uparrow$", (0.88, 0.97), "%.2f"),
        ("judge_gain", r"(c) Judge gain $\Delta s$ $\uparrow$", (0.50, 1.28), "%.1f"),
        ("length", "(d) Completion length", (78.0, 113.0), "%.0f"),
    )

    legend_handles = []
    for ax, (metric, title, ylim, tick_format) in zip(axes.flat, panels):
        handles = plot_configurations(ax, TRAINING, metric)
        if not legend_handles:
            legend_handles = handles
        style_axis(ax, title, ylim, tick_format)

    for ax in axes[1, :]:
        ax.set_xlabel("Epoch")

    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        frameon=False,
        handlelength=2.6,
        columnspacing=2.0,
        handletextpad=0.5,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.12,
        top=0.82,
        hspace=0.42,
        wspace=0.22,
    )
    save_figure(fig, "fig_completion_audit_training")


def build_validation_figure() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6), sharex=True)
    panels = (
        ("plcc", r"(a) Validation PLCC $\uparrow$", (0.85, 0.95), "%.2f"),
        ("srcc", r"(b) Validation SRCC $\uparrow$", (0.86, 0.94), "%.2f"),
        ("judge_gain", r"(c) Validation Judge gain $\uparrow$", (0.62, 1.23), "%.1f"),
    )

    legend_handles = []
    for ax, (metric, title, ylim, tick_format) in zip(axes.flat[:3], panels):
        handles = plot_configurations(ax, VALIDATION, metric)
        if not legend_handles:
            legend_handles = handles
        style_axis(ax, title, ylim, tick_format)

    health_ax = axes.flat[3]
    for configuration, values in VALIDATION.items():
        style = STYLES[configuration]
        health_ax.plot(
            EPOCHS,
            values["norm_unique"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=configuration,
            zorder=3,
        )
    style_axis(
        health_ax,
        "(d) Normalized-solution uniqueness (%)",
        (-4.0, 104.0),
        "%.0f",
    )

    for ax in axes[1, :]:
        ax.set_xlabel("Checkpoint epoch")

    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        frameon=False,
        handlelength=2.6,
        columnspacing=2.0,
        handletextpad=0.5,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.11,
        top=0.81,
        hspace=0.58,
        wspace=0.24,
    )
    save_figure(fig, "fig_completion_audit_validation")


def main() -> None:
    configure_style()
    validate_data()
    build_training_figure()
    build_validation_figure()
    print("Generated completion-credit training and validation audit figures.")


if __name__ == "__main__":
    main()
