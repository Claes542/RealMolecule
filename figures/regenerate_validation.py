#!/usr/bin/env python3
"""Regenerate validation_summary.pdf without the NaCl bar.

NaCl moved out of the deviation-vs-experiment figure because the ~50% T_m
overshoot is an uncalibrated Brownian-dynamics time-mapping issue, not a
wave-equation accuracy issue. Discussed in text in Section 7.8.
"""
import matplotlib.pyplot as plt
import numpy as np

# (label, deviation %, color)
data = [
    ("H$_2$ E$_{tot}$ vs KW",          0.3,  "#ff9800"),
    ("H$_2$ D$_e$ vs exp",             3.0,  "#ff9800"),
    ("Atom E$_{tot}$ Li-Rn",           1.0,  "#42a5f5"),
    ("NaH atomisation",                3.0,  "#42a5f5"),
    ("H$_2$O atomisation",             5.0,  "#42a5f5"),
    ("CH$_4$ atomisation",             7.0,  "#42a5f5"),
    ("SiH$_4$ atomisation",            9.0,  "#42a5f5"),
    ("GeH$_4$ atomisation",            8.0,  "#42a5f5"),
    ("S66 dimer geometries",           3.0,  "#26a69a"),
    ("Mini-protein topology",          15.0, "#9575cd"),
]

fig, ax = plt.subplots(figsize=(9, 4.2))
labels = [d[0] for d in data]
vals   = [d[1] for d in data]
colors = [d[2] for d in data]
xpos   = np.arange(len(data))

bars = ax.bar(xpos, vals, color=colors, edgecolor="black", linewidth=0.6)
ax.set_xticks(xpos)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Deviation from experimental / reference (%)", fontsize=10)
ax.set_ylim(0, 20)
ax.axhline(10, color="gray", linewidth=0.5, linestyle=":")
ax.axhline( 5, color="gray", linewidth=0.5, linestyle=":")
ax.set_axisbelow(True)
ax.grid(axis="y", color="#eee", linewidth=0.5)

# Legend by colour
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor="#ff9800", label="H$_2$ covalent bond (Level 1, parameter-free)"),
    Patch(facecolor="#42a5f5", label="Atom + closed-shell molecular energetics"),
    Patch(facecolor="#26a69a", label="Geometric (S66 / helix)"),
    Patch(facecolor="#9575cd", label="Protein folding (fraction of missing native H-bonds)"),
]
ax.legend(handles=legend_elems, loc="upper left", fontsize=8, frameon=True)

plt.tight_layout()
plt.savefig("validation_summary.pdf", bbox_inches="tight")
plt.savefig("validation_summary.png", bbox_inches="tight", dpi=160)
print("Wrote validation_summary.pdf, .png")
