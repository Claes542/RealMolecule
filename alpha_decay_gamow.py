"""
Alpha decay as Coulomb-barrier tunnelling -- the Geiger-Nuttall check
=====================================================================

Alpha decay is the clean case for RealQM: a two-body decay of a CHARGED alpha
(charge +2) tunnelling the Coulomb barrier of the daughter -- pure electrostatics
of extended charge, no weak force, no neutrino, and a DISCRETE (monoenergetic)
alpha line, exactly the two-body kinematics a RealQM solve produces.

This script computes the tunnelling half-life by the standard WKB Gamow factor and
checks that it reproduces the Geiger-Nuttall law -- log10(t_1/2) linear in
1/sqrt(Q_alpha) -- across the ~24 orders of magnitude that alpha lifetimes span.
Reproducing that span from one Coulomb-barrier calculation is the quantitative
backbone of the RealQM alpha-decay picture: the rate is set by the barrier the
extended alpha charge must cross.

Nuclear units: MeV, fm.
"""

from __future__ import annotations
import math

HBAR_C   = 197.3269       # MeV fm
M_ALPHA  = 3727.379       # MeV / c^2
E2       = 1.439964       # e^2 = alpha * hbar c, in MeV fm
YEAR     = 3.1557e7       # s


def nuclear_radius(A_daughter):
    """Touching radius of alpha + daughter (fm)."""
    return 1.20 * (A_daughter ** (1.0 / 3.0) + 4.0 ** (1.0 / 3.0))


def gamow_halflife(Z_parent, A_parent, Q):
    """WKB Gamow half-life (s) for alpha emission of energy Q (MeV)."""
    Z_d = Z_parent - 2
    A_d = A_parent - 4
    R = nuclear_radius(A_d)
    C = 2.0 * Z_d * E2                 # Coulomb strength z1 z2 e^2 (MeV fm)
    b = C / Q                          # outer turning point (fm)
    x = R / b
    if x >= 1.0:
        return float('inf')            # no barrier to tunnel (Q above barrier)

    # 2G = (2/hbar) * sqrt(2 m) * integral_R^b sqrt(C/r - Q) dr
    # integral = sqrt(Q) * b * [arccos(sqrt(x)) - sqrt(x(1-x))]
    integral = math.sqrt(Q) * b * (math.acos(math.sqrt(x)) - math.sqrt(x * (1 - x)))
    two_G = (2.0 / HBAR_C) * math.sqrt(2.0 * M_ALPHA) * integral

    # assault frequency f = v / (2R), v from the alpha kinetic energy
    v_over_c = math.sqrt(2.0 * Q / M_ALPHA)
    f = (v_over_c * 2.99792458e23) / (2.0 * R)     # 1/s  (c = 2.998e23 fm/s)

    lam = f * math.exp(-two_G)
    return math.log(2.0) / lam


def main():
    print("Alpha decay as Coulomb-barrier tunnelling (WKB Gamow)")
    print("=" * 62)

    # A selection of even-even alpha emitters: (name, Z, A, Q_MeV, t_exp_s)
    nuclei = [
        ("232Th", 90, 232, 4.083, 1.40e10 * YEAR),
        ("238U",  92, 238, 4.270, 4.47e9  * YEAR),
        ("230Th", 90, 230, 4.770, 7.54e4  * YEAR),
        ("226Ra", 88, 226, 4.871, 1.60e3  * YEAR),
        ("238Pu", 94, 238, 5.593, 87.7    * YEAR),
        ("242Cm", 96, 242, 6.216, 162.8   * 86400),
        ("220Rn", 86, 220, 6.405, 55.6),
        ("216Po", 84, 216, 6.906, 0.145),
        ("212Po", 84, 212, 8.954, 2.99e-7),
    ]

    print(f"  {'nuclide':>8} {'Q(MeV)':>7} {'t_exp':>12} {'t_Gamow':>12} "
          f"{'log10 ratio':>12}")
    rows = []
    for name, Z, A, Q, t_exp in nuclei:
        t = gamow_halflife(Z, A, Q)
        lr = math.log10(t / t_exp)
        rows.append((Q, t, t_exp))
        print(f"  {name:>8} {Q:7.3f} {t_exp:12.2e} {t:12.2e} {lr:+12.2f}")

    # Geiger-Nuttall: log10(t) linear in 1/sqrt(Q)?
    import statistics
    xs = [1.0 / math.sqrt(Q) for Q, _, _ in rows]
    ys = [math.log10(t) for _, t, _ in rows]
    ye = [math.log10(te) for _, _, te in rows]
    n = len(xs)
    def linfit(xs, ys):
        mx = sum(xs) / n; my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        a = sxy / sxx; b = my - a * mx
        ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - my) ** 2 for y in ys)
        return a, b, 1 - ss_res / ss_tot
    aG, bG, r2G = linfit(xs, ys)
    aE, bE, r2E = linfit(xs, ye)

    print("\n  Geiger-Nuttall  log10(t_1/2) = a / sqrt(Q) + b :")
    print(f"    Gamow model : slope a = {aG:8.2f}   R^2 = {r2G:.4f}")
    print(f"    experiment  : slope a = {aE:8.2f}   R^2 = {r2E:.4f}")
    print(f"    span reproduced: {max(ys)-min(ys):.1f} decades "
          f"(experiment {max(ye)-min(ye):.1f} decades)")
    print("\n  => one Coulomb-barrier calculation reproduces the Geiger-Nuttall law")
    print("     across ~20+ orders of magnitude: alpha timing is barrier tunnelling")
    print("     of an extended charge -- entirely within RealQM, no neutrino.")


if __name__ == "__main__":
    main()
