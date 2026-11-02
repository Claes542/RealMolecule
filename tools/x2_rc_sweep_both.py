"""
X2 SWEPT OVER r_c, IN BOTH CORE MODELS, ON ONE GRID.

Reuses the two validated offline solvers rather than rebuilding anything:
  tools/x2_R_sweep.py   HARD DEPLETION -- density identically 0 for r < R, free (Neumann) at R
  tools/x2_clamp.py     CLAMP          -- V = -Z/max(r, r_c), density admitted inside
Both are axisymmetric, exact for two domains on an axis, and agree to 1e-15 at r_c = 0 where the
core vanishes (x2_clamp.validate).

WHY BOTH. The clamp BEAT hard depletion on the alkali dimers when each r_c was fitted to its
element's measured IE -- mean |bond| error 4.9% against 8.5%, and binding -3..+22% against
+36..+101% (tools/x2_clamp.py). That was at five FITTED radii, one per element. This sweeps r_c
continuously instead, so the two models' bond and binding can be compared as FUNCTIONS of the core
radius rather than at a handful of fitted points.

WHAT TO READ OFF.
  - D_min(r_c): does the bond track r_c, and with what slope? The published hard-depletion series
    has R_e ~ 2.2-2.7 x R across Li->Cs, so a near-linear rise is expected.
  - E_bind(r_c): hard depletion over-binds by ~2x throughout the alkali series and the empty core
    under-binds by ~2x; the clamp sits between. Sweeping shows whether any single r_c makes the
    binding right, and if so whether that r_c is anywhere near the one the IE picks -- i.e. whether
    "fit r_c to the IE" and "fit r_c to the binding" are the same number or two different fits.
  - the r_c = 0 row is H2 itself and is the control: both models must give the same answer there.

COMMENSURATE D ONLY: D = (2i+1)h puts both kernels exactly on cell centres of the half grid.
Otherwise the egg-crate noise is the size of the binding itself -- the error that put 0.05 Ha of
alternating-slope noise in a linear-O3 scan when R was stepped in Angstrom on a 0.1 a0 grid.

Hard depletion additionally requires D > 2R + 2h, since two cores cannot interpenetrate; the clamp
has no such restriction (its potential is just the sum of two clamped kernels), so at large r_c the
hard-depletion rows run out of admissible geometries first and are reported as such.

usage:  python3 tools/x2_rc_sweep_both.py [h] [L]
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x2_R_sweep import atomX, x2R
from x2_clamp import atom_clamp, x2_clamp

HA_EV = 27.211386


def scan_min(efun, atom_e, rc, h, L, dmin_extra=0.0):
    """Minimise E(D) over commensurate D, coarse then refine. Returns (D_min, E2, E_bind, edge)."""
    allD = [(2 * i + 1) * h for i in range(1, 40)]
    allD = [D for D in allD if D > dmin_extra and D <= 12.0]
    if not allD:
        return None
    vals = {}
    for k in range(0, len(allD), 3):
        vals[k] = efun(rc, allD[k], L, L, h)
    kb = min(vals, key=vals.get)
    for k in range(max(0, kb - 3), min(len(allD), kb + 4)):
        if k not in vals:
            vals[k] = efun(rc, allD[k], L, L, h)
    kb = min(vals, key=vals.get)
    edge = kb in (0, len(allD) - 1)
    return allD[kb], vals[kb], 2 * atom_e - vals[kb], edge


if __name__ == '__main__':
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.175
    L = float(sys.argv[2]) if len(sys.argv) > 2 else 14.0
    RCS = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]

    print("X2 SWEPT OVER r_c, BOTH CORE MODELS.  h=%.3f  L=%.0f  commensurate D=(2i+1)h" % (h, L))
    print("hard depletion: density = 0 for r < r_c, free Neumann at r_c; needs D > 2r_c + 2h")
    print("clamp:          V = -Z/max(r, r_c), density admitted inside; no D restriction")
    print("E(X) and E(X2) from the same code on the same mesh, so the difference cancels")
    print("discretisation -- the house rule that differences within one composition cancel.\n")
    print("%6s | %9s %8s %9s | %9s %8s %9s | %s" %
          ("r_c", "E(X) hard", "D_min", "Eb eV", "E(X) clamp", "D_min", "Eb eV", "notes"))

    for rc in RCS:
        t0 = time.time()
        note = []
        # --- hard depletion
        try:
            eXh = atomX(rc, L, L, h)
            rh = scan_min(x2R, eXh, rc, h, L, dmin_extra=2 * rc + 2 * h)
        except Exception as e:
            eXh, rh = None, None
            note.append("hard failed: %s" % str(e)[:30])
        # --- clamp
        eXc = atom_clamp(rc, L, L, h)
        rcl = scan_min(x2_clamp, eXc, rc, h, L)

        def fmt(eX, r):
            if eX is None or r is None:
                return "%9s %8s %9s" % ("--", "no D", "--")
            D, E2, eb, edge = r
            return "%9.5f %8.3f %9.3f" % (eX, D, eb * HA_EV) + ("*" if edge else " ")

        if rh and rh[3]:
            note.append("hard at scan edge")
        if rcl and rcl[3]:
            note.append("clamp at scan edge")
        print("%6.2f | %s | %s | %s  [%.0fs]"
              % (rc, fmt(eXh, rh), fmt(eXc, rcl), "; ".join(note), time.time() - t0))
        sys.stdout.flush()

    print("\n* = minimum at the scan edge, i.e. NOT an interior minimum -- do not read it as a bond.")
    print("r_c = 0 is the CONTROL: both models reduce to bare H2 and must agree there.")
