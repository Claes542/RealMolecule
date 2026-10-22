"""
WHERE RealQM's FERMI PRESSURE COMES FROM: the partition, and only the partition.

THE QUESTION.  RealQM replaces antisymmetry with geometry -- N non-overlapping densities.  Does that
actually supply the kinetic pressure that holds a metal up against its own Madelung attraction?

THE TEST, in the cleanest 3D geometry there is.  A simple cubic lattice with cubic electron domains:
the cube IS the exact Wigner-Seitz cell of a simple cubic lattice, so the domains tile space with no
spherical approximation, non-overlap is exact rather than approximate, and the vanishing normal
derivative at the cell faces is a consequence of crystal symmetry rather than an imposed condition.
Nothing is fitted and nothing is assumed.  Then compare ONE electron per cell against TWO.

RESULT 2026-10-22.

      a      1 electron: T        E   |   2 electrons: T        E
    2.40          0.02210   -0.60969  |         1.54519   -0.83500
    3.20          0.03750   -0.47400  |         0.97090   -0.85485   <- bracketed minimum
    4.00          0.05514   -0.39577  |         0.70240   -0.81607
    6.40          0.14035   -0.29785  |         0.47056   -0.68776

ONE electron per cell has NO stable equilibrium: E falls monotonically under compression all the way
down, and T FALLS with it (0.140 -> 0.022).  With Neumann on all six faces -- which symmetry gives
-- a CONSTANT density is admissible and carries no localisation energy at all, so the kinetic term
offers no resistance while U_II = -alpha/r_s diverges as -1/a.  That is CLAUDE.md's house rule
("a uniform density plus a free boundary is unstable") in three dimensions, with no core and no
imposed boundary condition anywhere: the null mode is produced by the crystal symmetry itself.

TWO electrons per cell has a genuine bracketed minimum at a = 3.20 (dE/da turns -0.025 -> +0.049),
and T RISES under compression (0.471 -> 1.545), scaling close to 1/a^2: the ratio from a=3.2 to 2.4
is 1.59 against 1.78 for exact 1/a^2.  That is Fermi pressure, produced by nothing but geometry.

SO NON-OVERLAP DOES SUPPLY FERMI PRESSURE -- BUT IT NEEDS SOMETHING TO PARTITION.  Two densities
sharing a cell must each occupy half of it, and the localisation cost of that holds the lattice up.
With one electron per cell there is no interface, the constant mode is free, and the construction
has no resistance.  Pauli-as-geometry works, and this is the condition under which it works.

THREE CONSEQUENCES.
  * The radial-cell null mode and the cubic-cell collapse are the same thing: both are the
    one-electron case.  Neither is evidence against the construction.
  * metal_alkalis' empty-core term 1.5 r_c^2/r_s^3 is STANDING IN for the Fermi pressure that one
    valence electron per cell cannot supply.  That is a sharper account of r_c than "pseudopotential
    parameter", and it predicts that removing it must collapse the cell -- which it does.
  * The alkalis are close to the worst available test case: one valence electron per cell is exactly
    the degenerate configuration in which RealQM's distinguishing mechanism is switched off.  An
    all-electron treatment partitions the cell with its core electrons and should behave differently.

DISCIPLINE.  sigma is HELD FIXED in absolute units (0.30 a0), not tied to 1.2h.  Letting it scale
with the mesh changes the ion's size as the mesh refines, so it is not a mesh ladder at all -- with
sigma fixed the discretisation error drifts by 0.9% of the signal where it was 49% before, and B
converges to four figures (-5.857/-5.859/-5.860 GPa) across a factor two in h.

usage:  python3 tools/fermi_pressure_partition.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elastic3d_cell import cell, cell2

SIG = 0.30          # absolute, NOT 1.2h -- see DISCIPLINE above
H = 0.20


def scan(ns=(12, 16, 20, 24, 28, 32), h=H, sig=SIG):
    rows = []
    for n in ns:
        s1 = cell(n, h, rc=0.0, sig_cells=sig/h)
        s2 = cell2(n, h, Z=2.0, rc=0.0, sig_cells=sig/h)
        rows.append((s1['a'], s1['rs'], s1['T'], s1['E'], s2['T'], s2['E']))
    return rows


def main():
    print("FERMI PRESSURE FROM THE PARTITION -- simple cubic lattice, exact cubic domains\n")
    print("one electron per cell: nothing to partition, the constant density is free")
    print("two per cell: each confined to half the cube, so T ~ 1/a^2 must beat Madelung ~ -1/a\n")
    print(f"{'a':>6}{'r_s':>7} | {'1e: T':>9}{'E':>11} | {'2e: T':>9}{'E':>11}")
    rows = scan()
    for a, rs, T1, E1, T2, E2 in rows:
        print(f"{a:>6.2f}{rs:>7.3f} | {T1:>9.5f}{E1:>11.5f} | {T2:>9.5f}{E2:>11.5f}", flush=True)
    A = np.array([r[0] for r in rows])
    for nm, col in (("1 electron", 3), ("2 electrons", 5)):
        E = np.array([r[col] for r in rows])
        i = int(np.argmin(E))
        edge = "  <-- SCAN EDGE, no minimum bracketed" if i in (0, len(E)-1) else "  <-- bracketed"
        print(f"\n   {nm}: lowest at a = {A[i]:.2f}, E = {E[i]:.5f}{edge}")
        print("      dE/da: " + ", ".join(f"{(E[j+1]-E[j])/(A[j+1]-A[j]):+.4f}"
                                          for j in range(len(E)-1)))
    print("\n   T under compression is the mechanism: it FALLS for one electron (free constant")
    print("   mode) and RISES as ~1/a^2 for two (localisation cost of sharing the cell).")


if __name__ == '__main__':
    main()
