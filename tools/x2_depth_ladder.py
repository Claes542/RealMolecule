"""
DOES THE CLAMP'S BOND ACCURACY SURVIVE A DEEP PLATEAU?  The unconfounded test.

THE PROBLEM.  The clamp beat hard depletion on the alkali dimers (bonds 4.9% vs 8.5%, binding
-3..+22% vs +36..+101%, tools/x2_clamp.py) but every alkali plateau is SHALLOW: Zx/r_c = 0.21-0.33
Ha.  Real molecules sit ten times deeper.  The obvious extension -- reduced O2 at Zx = 2, r_c =
0.787, depth 2.54 Ha -- was run in 3D (mol_fast, clamp) and gave a bond 25% too long.  But that
number is CONFOUNDED: reduced O2 keeps 2 of oxygen's 6 valence electrons, so it is a 4-electron
diatomic scored against a 12-electron one.  Depth and valence truncation moved together.

THE SEPARATION.  The halogens are singly bonded, so Zx = 1 is the COMPLETE bonding valence of X2 --
no truncation at all -- yet the plateau is deep because the atomic IE is large:

    I 1.30,  Br 2.12,  Cl 4.34 Ha      against the alkalis' 0.21-0.33

Cl is deeper than reduced oxygen.  So this ladder varies DEPTH ALONE, at fixed occupancy, in real
single-bonded molecules, using the same monovalent code that produced the alkali result.  If the
bond error grows along Li -> I -> Br -> Cl, the clamp degrades with depth and the O2 result is about
depth.  If it stays flat, the O2 error belongs to the valence truncation instead.

THE CEILING (found while calibrating, and worth recording).  A monovalent clamped kernel cannot
reproduce an IE above 13.6057 eV -- one Rydberg, its r_c -> 0 limit.  F (17.42 eV) has no solution;
neither would N (14.53) or O (13.618, just 0.012 eV over).  So Zx > 1 for the second row is not a
house convention, it is FORCED by the construction.

r_c is fitted at the SAME h as the dimer (x2_clamp.fit_rc), not at the converged radial value, so
the atom and dimer share their discretisation and E_bind = 2E(X) - E(X2) cancels it -- the house
rule, and what the alkali table did.

usage:  python3 tools/x2_depth_ladder.py [h] [L]
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x2_clamp import atom_clamp, x2_clamp, fit_rc

HA_EV = 27.211386

# measured IE (eV), R_e (bohr), D_e (eV).  From model knowledge -- UNVERIFIED, and the only
# external input here.  Ordered by plateau depth, shallow first.
MEAS = [("Li2", 5.391715, 5.051, 1.05),
        ("Na2", 5.139077, 5.818, 0.75),
        ("K2",  4.340663, 7.379, 0.55),
        ("I2",  10.45126, 5.038, 1.56),
        ("Br2", 11.81380, 4.311, 1.99),
        ("Cl2", 12.96763, 3.756, 2.51)]


def scan(rc, h, L, dlo, dhi):
    """Minimise over commensurate D = (2i+1)h, coarse then refine."""
    allD = [(2 * i + 1) * h for i in range(1, 80)]
    allD = [D for D in allD if dlo <= D <= dhi]
    vals = {}
    for k in range(0, len(allD), 3):
        vals[k] = x2_clamp(rc, allD[k], L, L, h)
    kb = min(vals, key=vals.get)
    for k in range(max(0, kb - 3), min(len(allD), kb + 4)):
        if k not in vals:
            vals[k] = x2_clamp(rc, allD[k], L, L, h)
    kb = min(vals, key=vals.get)
    return allD[kb], vals[kb], kb in (0, len(allD) - 1)


if __name__ == '__main__':
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.175
    L = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0
    print("CLAMP BOND ERROR vs PLATEAU DEPTH, at fixed monovalent occupancy.  h=%.3f L=%.0f" % (h, L))
    print("Zx = 1 is the COMPLETE bonding valence of a single bond, so no valence is truncated;")
    print("depth 1/r_c varies only because the atomic IE does.  Commensurate D = (2i+1)h.\n")
    print("%5s %9s %9s %9s | %10s %8s %8s | %7s %7s | %8s %8s"
          % ("sys", "IE exp", "r_c", "1/r_c", "E(X)", "D_min", "E_b eV",
             "R_e exp", "D_e exp", "bond err", "bind err"))
    for name, ie, Re, De in MEAS:
        t0 = time.time()
        rc = fit_rc(ie, h)
        eX = atom_clamp(rc, L, L, h)
        D, E2, edge = scan(rc, h, L, dlo=max(0.5, 0.45 * Re), dhi=min(2 * L - 4, 2.2 * Re))
        eb = (2 * eX - E2) * HA_EV
        print("%5s %9.4f %9.5f %9.2f | %10.5f %8.3f %8.3f%s | %7.3f %7.2f | %+7.1f%% %+7.1f%%  [%.0fs]"
              % (name, ie, rc, 1.0 / rc, eX, D, eb, "*" if edge else " ", Re, De,
                 100 * (D - Re) / Re, 100 * (eb - De) / De, time.time() - t0))
        sys.stdout.flush()
    print("\n* = minimum at a scan edge, NOT an interior minimum -- do not read it as a bond.")
    print("Read DOWN the bond-error column: a trend with 1/r_c means the clamp degrades with depth.")
