#!/usr/bin/env python3
"""Generate the aligned-validation trajectories for the ablation study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


OUTPUT_DIR = Path(__file__).resolve().parent
EPOCHS = np.array([1, 2, 3], dtype=int)

RESULTS = {
    "T1": {
        "label": "T1: 3VL-2B DAPO",
        "plcc": [0.7715, 0.8245, 0.8310],
        "srcc": [0.7378, 0.8103, 0.8167],
        "mae": [0.9689, 0.8195, 0.8683],
        "uscore": [5.5, 11.5, 12.0],
    },
    "T2": {
        "label": "T2: 2B DAPO i1",
        "plcc": [0.8822, 0.8862, 0.8897],
        "srcc": [0.8733, 0.8781, 0.8805],
        "mae": [0.2714, 0.5993, 0.6905],
        "uscore": [30.0, 33.0, 36.0],
    },
    "T3": {
        "label": "T3: 2B DAPO i4",
        "plcc": [0.8717, 0.9054, 0.9104],
        "srcc": [0.8783, 0.9018, 0.9035],
        "mae": [0.4834, 0.2924, 0.2645],
        "uscore": [14.0, 18.5, 21.5],
    },
    "T4": {
        "label": "T4: 2B GRPO i1",
        "plcc": [0.8525, 0.8872, 0.8861],
        "srcc": [0.8597, 0.8828, 0.8834],
        "mae": [0.7822, 0.7270, 0.8449],
        "uscore": [19.5, 35.5, 34.0],
    },
    "T5": {
        "label": "T5: 2B visual-global",
        "plcc": [0.9102, 0.9273, 0.9287],
        "srcc": [0.9017, 0.9178, 0.9207],
        "mae": [0.2096, 0.2384, 0.2612],
        "uscore": [21.5, 32.0, 35.0],
    },
    "T6": {
        "label": "T6: 2B visual-local",
        "plcc": [0.9017, 0.9152, 0.9169],
        "srcc": [0.8905, 0.9043, 0.9058],
        "mae": [0.6933, 0.7441, 0.5868],
        "uscore": [15.0, 33.0, 39.0],
    },
    "T7": {
        "label": "T7: 4B local-six",
        "plcc": [0.9089, 0.9285, 0.9351],
        "srcc": [0.8884, 0.9186, 0.9236],
        "mae": [0.9794, 0.5659, 0.2280],
        "uscore": [20.0, 41.0, 41.5],
    },
}

# Okabe-Ito-derived colors plus dark gray for the GRPO control. Marker and line
# style provide redundant encoding for grayscale and color-vision accessibility.
STYLES = {
    "T1": ("#0072B2", "o", "-"),
    "T2": ("#56B4E9", "s", "--"),
    "T3": ("#009E73", "^", "-."),
    "T4": ("#4D4D4D", "D", ":"),
    "T5": ("#D55E00", "v", (0, (5, 1))),
    "T6": ("#CC79A7", "P", (0, (3, 1, 1, 1))),
    "T7": ("#E69F00", "X", (0, (1, 1))),
}

# Match the table's natural top-to-bottom experiment numbering.
DISPLAY_ORDER = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")

PANELS = (
    ("plcc", r"(a) PLCC $\uparrow$", (0.75, 0.95), "%.2f"),
    ("srcc", r"(b) SRCC $\uparrow$", (0.72, 0.94), "%.2f"),
    ("mae", r"(c) MAE $\downarrow$", (0.15, 1.02), "%.1f"),
    ("uscore", "(d) U-score (%)", (0.0, 45.0), "%.0f"),
)


def validate_results() -> None:
    """Fail early when an edited data series violates the figure contract."""

    if set(RESULTS) != set(STYLES):
        raise ValueError("RESULTS and STYLES must define the same variants")
    if set(DISPLAY_ORDER) != set(RESULTS):
        raise ValueError("DISPLAY_ORDER must contain every variant exactly once")
    if len(RESULTS) != 7 or len(EPOCHS) != 3:
        raise ValueError("Expected seven variants evaluated at three epochs")

    for variant, values in RESULTS.items():
        for metric in ("plcc", "srcc", "mae", "uscore"):
            series = np.asarray(values[metric], dtype=float)
            if series.shape != EPOCHS.shape or not np.isfinite(series).all():
                raise ValueError(f"Invalid {metric} series for {variant}")
            if metric in {"plcc", "srcc"} and not np.all((-1 <= series) & (series <= 1)):
                raise ValueError(f"Correlation outside [-1, 1] for {variant}")
            if metric == "mae" and np.any(series < 0):
                raise ValueError(f"Negative MAE for {variant}")
            if metric == "uscore" and not np.all((0 <= series) & (series <= 100)):
                raise ValueError(f"U-score outside [0, 100] for {variant}")


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
            "lines.linewidth": 1.35,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def build_figure() -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.25), sharex=True)

    legend_handles = []
    for ax, (metric, title, ylim, tick_format) in zip(axes.flat, PANELS):
        for variant in DISPLAY_ORDER:
            values = RESULTS[variant]
            color, marker, linestyle = STYLES[variant]
            metric_values = np.asarray(values[metric], dtype=float)
            (line,) = ax.plot(
                EPOCHS[: metric_values.size],
                metric_values,
                color=color,
                marker=marker,
                linestyle=linestyle,
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=values["label"],
                zorder=3,
            )
            if metric == "plcc":
                legend_handles.append(line)

        ax.set_title(title, loc="left", pad=3)
        ax.set_xlim(0.88, 3.12)
        ax.set_ylim(*ylim)
        ax.set_xticks(EPOCHS)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FormatStrFormatter(tick_format))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)

    for ax in axes[1, :]:
        ax.set_xlabel("Epoch")

    # Matplotlib fills multi-row legends by columns. Reorder handles so the
    # rendered rows follow DISPLAY_ORDER from left to right.
    legend_order = [0, 4, 1, 5, 2, 6, 3]
    ordered_handles = [legend_handles[index] for index in legend_order]
    ordered_labels = [handle.get_label() for handle in ordered_handles]
    fig.legend(
        handles=ordered_handles,
        labels=ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.25,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.12, top=0.80, hspace=0.42, wspace=0.22)
    return fig


def main() -> None:
    validate_results()
    figure = build_figure()
    pdf_path = OUTPUT_DIR / "fig_ablation_epochs.pdf"
    png_path = OUTPUT_DIR / "fig_ablation_epochs.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=300)
    plt.close(figure)
    print(
        "Generated 2x2 ablation trajectories: "
        f"{len(RESULTS)} variants, {len(EPOCHS)} epochs, {len(PANELS)} metrics."
    )
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
