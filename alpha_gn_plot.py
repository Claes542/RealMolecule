"""
Geiger-Nuttall plot for the alpha-decay paper:  log10(t_1/2) vs 1/sqrt(Q),
parameter-free WKB Coulomb-barrier (Gamow) model against experiment.
Writes geiger_nuttall.pdf.
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alpha_decay_gamow import gamow_halflife, YEAR

# (name, Z, A, Q_MeV, t_exp_s)
NUC = [
    ("232Th", 90, 232, 4.083, 1.40e10 * YEAR),
    ("238U",  92, 238, 4.270, 4.47e9  * YEAR),
    ("236U",  92, 236, 4.573, 2.34e7  * YEAR),
    ("230Th", 90, 230, 4.770, 7.54e4  * YEAR),
    ("234U",  92, 234, 4.858, 2.45e5  * YEAR),
    ("226Ra", 88, 226, 4.871, 1.60e3  * YEAR),
    ("228Th", 90, 228, 5.520, 1.91    * YEAR),
    ("238Pu", 94, 238, 5.593, 87.7    * YEAR),
    ("230U",  92, 230, 5.993, 20.8    * 86400),
    ("242Cm", 96, 242, 6.216, 162.8   * 86400),
    ("220Rn", 86, 220, 6.405, 55.6),
    ("216Po", 84, 216, 6.906, 0.145),
    ("214Po", 84, 214, 7.833, 1.64e-4),
    ("212Po", 84, 212, 8.954, 2.99e-7),
]

x   = [1.0 / math.sqrt(Q) for _, _, _, Q, _ in NUC]
ye  = [math.log10(te)     for _, _, _, _, te in NUC]
yg  = [math.log10(gamow_halflife(Z, A, Q)) for _, Z, A, Q, _ in NUC]

fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.plot(x, ye, "o", ms=6, color="#c62828", label="experiment", zorder=3)
ax.plot(x, yg, "s", ms=5, mfc="none", mec="#1565c0", mew=1.4,
        label="Gamow Coulomb barrier", zorder=2)
# straight-line guide through the Gamow points
import numpy as np
a, b = np.polyfit(x, yg, 1)
xs = np.array([min(x), max(x)])
ax.plot(xs, a * xs + b, "-", lw=1.0, color="#1565c0", alpha=0.6, zorder=1)

# a few labels
for name, _, _, Q, te in [NUC[0], NUC[-1]]:
    ax.annotate(name, (1/math.sqrt(Q), math.log10(te)),
                textcoords="offset points", xytext=(8, -2), fontsize=8)

ax.set_xlabel(r"$Q_\alpha^{-1/2}\ \ (\mathrm{MeV}^{-1/2})$")
ax.set_ylabel(r"$\log_{10}\, t_{1/2}\ \ (\mathrm{s})$")
ax.set_title("Geiger--Nuttall: one Coulomb barrier, 25 decades", fontsize=10)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(True, ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig("geiger_nuttall.pdf")
print("wrote geiger_nuttall.pdf   (Gamow slope a =", round(a, 1), ")")
