"""
THE ALKALI DIMERS WITH A CLAMPED KERNEL -- the core model molecule.js ACTUALLY IMPLEMENTS.

tools/x2_R_sweep.py runs the reduced-valence dimers with HARD DEPLETION (density excluded from
r < R) and tools/x2_emptycore.py with the EMPTY CORE (V = 0 inside r_c).  Those two bracket the
measured binding.  Neither is what the browser solver does: molecule.js forms

    r_eff = max(r, r_c)        ->        V = -Z / max(r, r_c)

a potential PLATEAU of depth Z/r_c inside r_c.  Density is admitted there AND still attracted, so
the clamp is a third core model, less severe than the empty core, and the 0.855 conversion between
the other two does not carry over to it.

Why this file exists.  The XO2 scans took r_c = 0.5 off a menu, calibrated to nothing, and found
E(R) monotone through the measured bond.  Calibrating r_c to the measured IE is what the alkali
dimer series did, and it is the one input that is not free.  Before trusting a calibrated r_c on
ClO2, the calibrate-then-predict chain has to be shown to work IN THE CLAMP MODEL on systems with a
known answer.  That is this table.

r_c is re-fitted HERE, on the dimer's own mesh, to each element's measured IE, so that E(X) and
E(X2) come from the same code on the same mesh and the discretisation cancels in
E_bind = 2E(X) - E(X2) -- the house rule that differences within one composition cancel their
errors.  D is scanned only over COMMENSURATE values D = (2i+1)h, which put both kernels exactly on
cell centres of the half grid; otherwise the egg-crate noise is the size of the binding itself.

Unlike hard depletion, the clamp is well defined when the two plateaus overlap (the potential is
just the sum of two clamped kernels), so D is NOT restricted to D > 2 r_c here.

VALIDATION, run by --validate: at r_c = 0 the clamp vanishes and this code must solve the same
problem as x2_R_sweep.py and x2_emptycore.py.  The numpy matmul RuntimeWarnings are the lazy FP
flags the existing published code emits too (benign, pre-existing).

usage:  python3 tools/x2_clamp.py
        python3 tools/x2_clamp.py --validate
"""
import os, sys, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from axisym_realqm import Grid, stiffness
from axisym_hom2 import _poisson_setup

HA_EV = 27.211386


def _clamped_K(g, kernels, rc):
    """Sum of clamped kernel attractions: K = sum_a Z_a / max(|r - r_a|, rc).  V = -K."""
    K = np.zeros(g.N)
    for z0, Z in kernels:
        rr = np.sqrt((g.Z - z0) ** 2 + g.R ** 2).ravel()
        K += Z / np.maximum(rr, rc) if rc > 0 else Z / np.maximum(rr, 1e-300)
    return K


def atom_clamp(rc, Lz=14.0, Lr=14.0, h=0.175):
    """One electron on a +1 clamped kernel.  IE = -E."""
    g = Grid(Lz, Lr, h, zmin=-Lz)
    K = _clamped_K(g, [(0.0, 1.0)], rc)
    w = 1.0 / np.sqrt(g.V); W = sp.diags(w)
    A = (W @ (stiffness(g) + sp.diags(-K * g.V)).tocsr() @ W).tocsr()
    return eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000)[0][0]


def x2_clamp(rc, D, Lz=14.0, Lr=14.0, h=0.175, iters=120, tol=1e-9):
    """Two +1 clamped kernels at +-D/2, one electron each, split at the midplane.

    Identical to x2_emptycore.x2_ec except the core clamps the potential instead of cancelling it.
    """
    gh = Grid(Lz, Lr, h, zmin=0.0)
    gf = Grid(Lz, Lr, h, zmin=-Lz)
    nzh = gh.nz
    assert gf.nz == 2 * nzh
    K = _clamped_K(gh, [(-D / 2, 1.0), (D / 2, 1.0)], rc)
    S = stiffness(gh)
    solveP, bnd, rrf = _poisson_setup(gf)
    w = 1.0 / np.sqrt(gh.V); W = sp.diags(w)
    mir = np.arange(nzh)[::-1]
    P1 = np.zeros(gh.N); u = None; Eold = None
    for it in range(iters):
        A = (W @ (S + sp.diags((-K + P1) * gh.V)).tocsr() @ W).tocsr()
        val, vec = eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000,
                         v0=None if u is None else u / w)
        u = w * vec[:, 0]; u = u / np.sqrt(np.sum(u * u * gh.V))
        densf = np.zeros((gf.nz, gf.nr)); densf[nzh:, :] = (u * u).reshape(nzh, gh.nr)
        b = 4 * np.pi * densf.ravel() * gf.V
        b[bnd] = 1.0 / rrf[bnd]
        P1 = solveP(b).reshape(gf.nz, gf.nr)[mir, :].ravel()
        T = float(u @ (S @ u))
        VK = -float(np.sum(K * u * u * gh.V))
        U12 = float(np.sum(P1 * u * u * gh.V))
        E = 2 * (T + VK) + U12 + 1.0 / D
        if Eold is not None and abs(E - Eold) < tol:
            break
        Eold = E
    return E


def fit_rc(ie, h, lo=0.2, hi=9.0, tol=3e-4):
    """Bisect r_c so the clamped pseudo-atom reproduces the measured IE (eV).

    IE falls as r_c grows (a shallower plateau binds less), so the bracket runs lo=deep/hi=shallow.
    """
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if -atom_clamp(mid, h=h) * HA_EV > ie:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def validate():
    """At r_c = 0 the clamp vanishes, so all three codes solve the same problem."""
    from x2_R_sweep import x2R, atomX
    from x2_emptycore import x2_ec, atom_ec
    print("VALIDATION at r_c = 0 -- clamp vanishes, all three codes solve the same problem\n")
    print("%10s %14s %14s %14s %11s" % ("", "x2_clamp", "x2_emptycore", "x2_R_sweep", "max diff"))
    for D in (3.325, 5.425, 7.875):
        a = x2_clamp(0.0, D, 14.0, 14.0, 0.175)
        b = x2_ec(0.0, D, 14.0, 14.0, 0.175)
        c = x2R(0.0, D, 14.0, 14.0, 0.175)
        print("  D=%6.3f %14.9f %14.9f %14.9f %11.2e" % (D, a, b, c, max(abs(a - b), abs(a - c))))
    a = atom_clamp(0.0, 14.0, 14.0, 0.175)
    b = atom_ec(0.0, 14.0, 14.0, 0.175)
    c = atomX(0.0, 14.0, 14.0, 0.175)
    print("  atom   %14.9f %14.9f %14.9f %11.2e   (exact E(H) = -0.5)"
          % (a, b, c, max(abs(a - b), abs(a - c))))


if __name__ == '__main__':
    if '--validate' in sys.argv:
        validate(); sys.exit(0)
    h, L = 0.175, 14.0
    # measured: atomic IE (eV), D_e (eV), R_e (bohr); then the published hard-depletion and
    # empty-core results for the same systems, same h and L, from the two sibling files.
    MEAS = [("Li", 5.392, 1.050, 5.051, 5.425, 6.125),
            ("Na", 5.139, 0.747, 5.818, 5.425, 6.475),
            ("K",  4.341, 0.551, 7.416, 6.825, 8.225),
            ("Rb", 4.177, 0.495, 7.955, 7.175, 8.575),
            ("Cs", 3.894, 0.452, 8.783, 7.875, 9.275)]
    print("ALKALI DIMERS WITH A CLAMPED KERNEL  V = -Z/max(r, r_c)   (the molecule.js core model)")
    print(f"h={h}, L={L}, commensurate D=(2i+1)h; r_c fitted here to each measured IE")
    print("plateaus may overlap, so D is not restricted to D > 2 r_c\n")
    print("%4s %7s %8s | %8s %8s %8s %8s | %9s %8s" %
          ("el", "r_c", "IE eV", "D clamp", "D hard", "D ec", "R_e exp", "Eb clamp", "De exp"))
    rows = []
    for el, ie, de, re, Dhard, Dec in MEAS:
        t0 = time.time()
        rc = fit_rc(ie, h)
        eX = atom_clamp(rc, L, L, h)
        allD = [(2 * i + 1) * h for i in range(1, 40)]
        allD = [D for D in allD if D <= 12.0]
        vals = {}
        for k in range(0, len(allD), 3):
            vals[k] = x2_clamp(rc, allD[k], L, L, h)
        kb = min(vals, key=vals.get)
        for k in range(max(0, kb - 3), min(len(allD), kb + 4)):
            if k not in vals:
                vals[k] = x2_clamp(rc, allD[k], L, L, h)
        kb = min(vals, key=vals.get)
        Dm, E2 = allD[kb], vals[kb]
        eb = (2 * eX - E2) * HA_EV
        rows.append((el, Dm, re, eb, de, Dhard, Dec))
        edge = " <-EDGE, no interior min" if kb in (0, len(allD) - 1) else ""
        print("%4s %7.3f %8.3f | %8.3f %8.3f %8.3f %8.3f | %9.3f %8.3f  [%.0fs]%s" %
              (el, rc, -eX * HA_EV, Dm, Dhard, Dec, re, eb, de, time.time() - t0, edge),
              flush=True)
    import statistics as st
    print("\n   bond error, mean |.|:  clamp %.1f%%   hard depletion %.1f%%   empty core %.1f%%" % (
        st.mean([abs(r[1] - r[2]) / r[2] * 100 for r in rows]),
        st.mean([abs(r[5] - r[2]) / r[2] * 100 for r in rows]),
        st.mean([abs(r[6] - r[2]) / r[2] * 100 for r in rows])))
    print("   clamp bond:    " + ", ".join("%+.0f%%" % ((r[1] - r[2]) / r[2] * 100) for r in rows))
    print("   clamp binding: " + ", ".join("%+.0f%%" % ((r[3] - r[4]) / r[4] * 100) for r in rows))
    print("\n   commensurate step is 2h = %.3f au, so adjacent species need not be resolved;" % (2 * h))
    print("   that limitation is stated in tools/x2_R_sweep.py and applies here unchanged.")
