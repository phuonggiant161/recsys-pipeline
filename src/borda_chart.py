"""Borda ranking bubble map: top algorithm per (AIU level, user group), from Bang 5 (u<5) and Bang 6 (u5+).
Bubble size grows with the winning Borda count."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

ALGOS = ["VSM", "ItemKNN", "FunkSVD", "BPR", "NeuMF", "LightGCN", "SASRec", "Two-Tower"]
AIU = ["<2", "4", "9", "15", ">20"]

# Bang 5. Borda count u<5 (no row for AIU >20)
T5 = {
    "<2": [5.75, 2.00, 4.50, 2.75, 4.25, 2.50, 4.50, 1.75],
    "4":  [4.25, 0.50, 3.75, 1.00, 4.00, 4.50, 5.25, 4.75],
    "9":  [3.00, 1.00, 3.25, 2.00, 4.00, 4.50, 5.75, 4.50],
    "15": [2.25, 1.25, 1.75, 5.50, 2.75, 4.25, 5.50, 4.75],
}
# Bang 6. Borda count u5+
T6 = {
    "<2":  [4.63, 3.06, 3.19, 3.15, 4.27, 2.44, 3.60, 3.67],
    "4":   [5.17, 1.88, 2.79, 1.40, 2.81, 5.17, 4.08, 4.71],
    "9":   [3.31, 1.81, 2.31, 3.06, 2.81, 4.69, 6.06, 3.94],
    "15":  [2.88, 3.00, 2.00, 3.50, 2.38, 5.00, 7.00, 2.25],
    ">20": [1.88, 3.88, 1.42, 2.54, 1.92, 5.38, 7.00, 4.00],
}

COLOR = {"VSM": "#2a78d6", "SASRec": "#d6542a", "LightGCN": "#1f9e74", "BPR": "#8a5cd6"}
AXIS, INK = "#1b5e83", "#1f2328"


def size(score):
    # radius grows linearly with the Borda count, so bigger bubble = higher score
    return (score * 9.5) ** 2


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(12, 5.6))

ax.set_xlim(-0.7, len(AIU) - 0.3)
ax.set_ylim(-0.6, 1.6)
ax.set_xticks(range(len(AIU)), AIU)
ax.set_yticks([0, 1], ["u1_5", "U5+"])
ax.set_xlabel("AIU", fontsize=12, color=INK, labelpad=8)
ax.set_ylabel("User group", fontsize=12, color=INK, labelpad=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
    ax.spines[s].set_linewidth(1.4)
ax.tick_params(colors=INK, length=0, pad=8, labelsize=12)

fig.tight_layout()

# data units per point along x, so tied bubbles can be placed edge to edge
x0, x1 = ax.get_xlim()
x_per_pt = (x1 - x0) / (ax.get_window_extent().width * 72 / fig.dpi)

for yi, table in enumerate([T5, T6]):
    for xi, aiu in enumerate(AIU):
        if aiu not in table:
            continue
        scores = table[aiu]
        top = max(scores)
        winners = [a for a, s in zip(ALGOS, scores) if s == top]
        # tied winners sit side by side, touching
        r = size(top) ** 0.5 / 2 * x_per_pt
        offsets = [0.0] if len(winners) == 1 else [-r, r]
        for algo, dx in zip(winners, offsets):
            c = COLOR[algo]
            ax.scatter(xi + dx, yi, s=size(top), facecolors=to_rgba(c, 0.25),
                       edgecolors=c, linewidths=1.8, zorder=2)
            ax.text(xi + dx, yi, algo, ha="center", va="center",
                    fontsize=9, weight="bold", color=INK, zorder=3)

fig.savefig(OUT / "borda_ranking_chart.png", dpi=200)
fig.savefig(OUT / "borda_ranking_chart.pdf")
print("saved", OUT)
