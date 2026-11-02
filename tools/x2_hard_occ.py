"""
HARD DEPLETION WITH n ELECTRONS PER DOMAIN -- calibrated second-row dimers, never run before.

WHY.  The clamp was the best core model on the alkali dimers (bonds 4.0% / binding 10% against hard
depletion's 8.3% / 86%), but tools/x2_depth_ladder.py showed it INVERTS at depth: on the halogens,
at fixed monovalent occupancy and complete bonding valence, the bond error changes sign (+12.6% at
0.33 Ha to -57% at 1.3 Ha) and the binding saturates at +167%, mesh-converged.  Above ~1.3 Ha hard
depletion's +86% is therefore the BETTER of the two -- and hard depletion has never been tested on
a dimer above the alkalis at all, because tools/x2_R_sweep.py is MONOVALENT.  This supplies the
missing occupancy, exactly as x2_clamp_occ.py did for the clamp.

THE CONSTRUCTION.  Two kernels of charge Zx at +-D/2, density identically zero inside r = R of
EITHER kernel (free Neumann at the surface, via masked_stiffness), n = Zx electrons per domain
sharing one density, domains split at the midplane.  With U_ab = \\int P[u_a^2] u_b^2,

    E = 2[ n(T + V_K) + (n(n-1)/2) U_self ] + n^2 U_12 + Zx^2 / D

-- n(n-1)/2 intra-domain pairs per domain, n^2 inter-domain pairs, each counted once by the i != j
rule.  Varying u_1 under \\int u_1^2 = 1 and dividing by 2n gives

    -(1/2) Lap u_1 + [ V + (n-1) P[u_1^2] + n P[u_2^2] ] u_1 = eps u_1

whose n = 1 case drops the self term and must reproduce x2_R_sweep exactly -- checked by
`--validate`, not assumed.

GEOMETRIC FEASIBILITY IS A REAL CONSTRAINT HERE, unlike the clamp.  Hard depletion forbids the two
cores from interpenetrating, so D > 2R + 2h.  At the IE-calibrated radii that rules systems OUT:

    O2  2R = 1.094 against R_e 2.282   feasible
    N2  2R = 1.453 against R_e 2.074   feasible
    C2  2R = 2.665 against R_e 2.348   IMPOSSIBLE -- the cores overlap at the real bond

and the same check on organic bonds kills aromatic and double-bonded carbon outright (C=C aromatic
2.627 a0 against sum-of-radii 2.665).  So calibrated hard depletion cannot describe carbon
chemistry at all; that is a geometric fact, independent of any energy this file computes.

usage:  python3 tools/x2_hard_occ.py --validate
        python3 tools/x2_hard_occ.py [h] [L]
"""
import os, sys, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from axisym_realqm import Grid, stiffness, kernel_potential, masked_stiffness
from axisym_hom2 import _poisson_setup

HA_EV = 27.211386


def atom_hard_n(Z, R, n_el, Lz=16.0, Lr=16.0, h=0.14, soft=0.0, iters=200, tol=1e-10):
    """n electrons sharing one density outside r=R on a +Z kernel (free Neumann at R)."""
    g = Grid(Lz, Lr, h, zmin=-Lz)
    K = kernel_potential(g, [(0.0, float(Z))], soft)
    m = np.sqrt(g.Z**2 + g.R**2).ravel() > R
    S = masked_stiffness(g, m) if R > 0 else stiffness(g)
    keep = np.where(m)[0]
    solveP, bnd, rrf = _poisson_setup(g)
    w = 1.0 / np.sqrt(g.V); W = sp.diags(w)
    Pself = np.zeros(g.N); u = None; Eold = None
    for it in range(iters):
        A = (W @ (S + sp.diags((-K + (n_el - 1) * Pself) * g.V)).tocsr() @ W).tocsr()[keep][:, keep]
        val, vec = eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000,
                         v0=None if u is None else (u / w)[keep])
        y = np.zeros(g.N); y[keep] = vec[:, 0]
        u = w * y; u = u / np.sqrt(np.sum(u * u * g.V))
        b = 4 * np.pi * (u * u) * g.V
        b[bnd] = 1.0 / rrf[bnd]
        Pself = solveP(b)
        T = float(u @ (S @ u))
        VK = -float(np.sum(K * u * u * g.V))
        Us = float(np.sum(Pself * u * u * g.V))
        E = n_el * (T + VK) + 0.5 * n_el * (n_el - 1) * Us
        if Eold is not None and abs(E - Eold) < tol:
            break
        Eold = E
    return E


def x2_hard_n(Z, R, n_el, D, Lz=16.0, Lr=16.0, h=0.14, soft=0.0, iters=200, tol=1e-9):
    """Two +Z kernels at +-D/2, cores of radius R, n electrons per domain, split at the midplane."""
    gh = Grid(Lz, Lr, h, zmin=0.0)
    gf = Grid(Lz, Lr, h, zmin=-Lz)
    nzh = gh.nz
    assert gf.nz == 2 * nzh
    K = kernel_potential(gh, [(-D / 2, float(Z)), (D / 2, float(Z))], soft)
    rr_near = np.sqrt((gh.Z - D / 2) ** 2 + gh.R ** 2).ravel()
    rr_far = np.sqrt((gh.Z + D / 2) ** 2 + gh.R ** 2).ravel()
    m = (rr_near > R) & (rr_far > R)
    S = masked_stiffness(gh, m) if R > 0 else stiffness(gh)
    keep = np.where(m)[0]
    solveP, bnd, rrf = _poisson_setup(gf)
    w = 1.0 / np.sqrt(gh.V); W = sp.diags(w)
    mir = np.arange(nzh)[::-1]
    Pself = np.zeros(gh.N); Ppart = np.zeros(gh.N); u = None; Eold = None
    for it in range(iters):
        A = (W @ (S + sp.diags((-K + (n_el - 1) * Pself + n_el * Ppart) * gh.V)).tocsr()
             @ W).tocsr()[keep][:, keep]
        val, vec = eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000,
                         v0=None if u is None else (u / w)[keep])
        y = np.zeros(gh.N); y[keep] = vec[:, 0]
        u = w * y; u = u / np.sqrt(np.sum(u * u * gh.V))
        densf = np.zeros((gf.nz, gf.nr)); densf[nzh:, :] = (u * u).reshape(nzh, gh.nr)
        b = 4 * np.pi * densf.ravel() * gf.V
        b[bnd] = 1.0 / rrf[bnd]
        Pfull = solveP(b).reshape(gf.nz, gf.nr)
        Pself = Pfull[nzh:, :].ravel()
        Ppart = Pfull[mir, :].ravel()
        T = float(u @ (S @ u))
        VK = -float(np.sum(K * u * u * gh.V))
        Us = float(np.sum(Pself * u * u * gh.V))
        U12 = float(np.sum(Ppart * u * u * gh.V))
        E = (2 * (n_el * (T + VK) + 0.5 * n_el * (n_el - 1) * Us)
             + n_el * n_el * U12 + Z * Z / D)
        if Eold is not None and abs(E - Eold) < tol:
            break
        Eold = E
    return E


def scan(Z, R, n_el, h, L, dlo, dhi):
    """Minimise over commensurate D=(2i+1)h with D > 2R + 2h.  Returns (D, E2, edge)."""
    allD = [(2 * i + 1) * h for i in range(1, 80)]
    allD = [D for D in allD if max(dlo, 2 * R + 2 * h) <= D <= dhi]
    if not allD:
        return None
    vals = {}
    step = max(1, len(allD) // 16)
    for k in range(0, len(allD), step):
        vals[k] = x2_hard_n(Z, R, n_el, allD[k], L, L, h)
    kb = min(vals, key=vals.get)
    for k in range(max(0, kb - step), min(len(allD), kb + step + 1)):
        if k not in vals:
            vals[k] = x2_hard_n(Z, R, n_el, allD[k], L, L, h)
    kb = min(vals, key=vals.get)
    return allD[kb], vals[kb], kb in (0, len(allD) - 1)


def validate():
    """n=1 drops the self term, so this must reproduce the monovalent x2_R_sweep exactly."""
    from x2_R_sweep import atomX, x2R
    print("VALIDATION: n = 1 must reduce to x2_R_sweep (hard depletion, monovalent).\n")
    print("%26s %16s %16s %11s" % ("", "x2_hard_occ", "x2_R_sweep", "diff"))
    for R, D in ((0.0, 3.325), (0.8, 3.325), (2.0, 5.425)):
        a = x2_hard_n(1, R, 1, D, 14.0, 14.0, 0.175)
        b = x2R(R, D, 14.0, 14.0, 0.175)
        print("  R=%4.2f D=%6.3f  dimer %16.9f %16.9f %11.2e" % (R, D, a, b, abs(a - b)))
    for R in (0.0, 2.0):
        a = atom_hard_n(1, R, 1, 14.0, 14.0, 0.175)
        b = atomX(R, 14.0, 14.0, 0.175)
        extra = "   (exact E(H) = -0.5)" if R == 0 else ""
        print("  R=%4.2f           atom %16.9f %16.9f %11.2e%s" % (R, a, b, abs(a - b), extra))


if __name__ == '__main__':
    if '--validate' in sys.argv:
        validate(); sys.exit(0)
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.175
    L = float(sys.argv[2]) if len(sys.argv) > 2 else 14.0

    # Zx, IE-calibrated HARD-DEPLETION radius (tools/rc_calibrate_hard.py), R_e (bohr), D_e (eV).
    # Li2 is the shallow control: it must reproduce the published 5.425 / 1.43 eV.
    # C2 is listed to be REFUSED by the feasibility test, not because it might work.
    ROWS = [("Li2", 1, 1.99144, 5.051, 1.05),
            ("O2",  2, 0.54711, 2.282, 5.12),
            ("N2",  3, 0.72670, 2.074, 9.76),
            ("C2",  4, 1.33246, 2.348, 6.21)]

    print("CALIBRATED HARD DEPLETION BEYOND THE ALKALIS.  h=%.3f L=%.0f  commensurate D=(2i+1)h" % (h, L))
    print("density = 0 inside r=R of EITHER kernel, so D > 2R + 2h is a HARD constraint.\n")
    print("%5s %3s %9s %8s | %11s %8s %8s | %7s %8s | %8s %9s"
          % ("sys", "Zx", "R", "2R", "E(X)", "D_min", "E_b eV", "R_e", "D_e", "bond err", "bind err"))
    for name, Z, R, Re, De in ROWS:
        if 2 * R + 2 * h >= Re:
            print("%5s %3d %9.5f %8.3f | %s"
                  % (name, Z, R, 2 * R, "INFEASIBLE: 2R+2h exceeds the measured bond %.3f" % Re))
            sys.stdout.flush()
            continue
        t0 = time.time()
        eX = atom_hard_n(Z, R, Z, L, L, h)
        s = scan(Z, R, Z, h, L, dlo=0.45 * Re, dhi=min(2 * L - 6, 2.4 * Re))
        if s is None:
            print("%5s   no admissible D" % name); continue
        D, E2, edge = s
        eb = (2 * eX - E2) * HA_EV
        print("%5s %3d %9.5f %8.3f | %11.5f %8.3f %8.3f%s | %7.3f %8.2f | %+7.1f%% %+8.1f%%  [%.0fs]"
              % (name, Z, R, 2 * R, eX, D, eb, "*" if edge else " ", Re, De,
                 100 * (D - Re) / Re, 100 * (eb - De) / De, time.time() - t0))
        sys.stdout.flush()
    print("\n* = minimum at a scan edge, NOT an interior minimum -- do not read it as a bond.")
