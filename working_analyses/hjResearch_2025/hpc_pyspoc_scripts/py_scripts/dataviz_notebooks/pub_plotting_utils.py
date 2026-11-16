"""plotting utilities."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot_explained_variance(evr, FIG_DIR,n_components=100, threshold=0.90, save_prefix=None, figsize=(4.8, 3.2)):
    """Cumulative (left axis) and per-component (right axis) explained variance; marks `threshold`.
    Returns fig, ax, k_thresh (components needed to reach threshold, or None)."""
    evr = np.asarray(evr, dtype=float)[:n_components]
    cum, k = np.cumsum(evr), np.arange(1, len(evr) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()
    ax.plot(k, cum, color="C0", lw=1.8, label="cumulative")
    ax2.bar(k, evr, color="C1", width=0.9, label="per component", zorder=0)
    
    k_thresh = int(np.searchsorted(cum, threshold) + 1) if cum[-1] >= threshold else None
    ax.axhline(threshold, ls="--", lw=1, color="k")
    if k_thresh:
        ax.axvline(k_thresh, ls=":", lw=1, color="k")
        ax.annotate(f"{k_thresh} PCs → {threshold:.0%}", (k_thresh, threshold),
                    xytext=(6, -14), textcoords="offset points", fontsize=8)

    ax.set(xlabel="Principal component", ylabel="Cumulative explained variance", ylim=(0, 1.02))
    ax2.set(ylabel="Per-component explained variance", ylim=(0, evr.max() * 1.1))
    for a, c in [(ax, "C0"), (ax2, "C1")]:
        a.yaxis.set_major_formatter(PercentFormatter(1.0)); a.yaxis.label.set_color(c)
        a.spines["top"].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)

    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    
    if save_prefix:
        plt.savefig(FIG_DIR / f"{save_prefix}.png", dpi=400, bbox_inches="tight")

    return fig, ax, k_thresh
