"""
CLAMPED X2 WITH n ELECTRONS PER DOMAIN -- the axisymmetric check on mol_fast's second-row runs.

WHY THIS EXISTS.  tools/x2_clamp.py measured the clamp against hard depletion on the alkali dimers
and won (bonds 4.9% vs 8.5%, binding -3..+22% vs +36..+101%).  But it is MONOVALENT: one electron
per domain, so Zx = 1 only.  Every alkali plateau is shallow, Zx/r_c = 0.21-0.33 Ha, while the
second row -- the row real molecules are made of -- sits at 2.54 (O) and 2.71 (Cl), ten times
deeper.  Nothing offline could reach that depth, so the clamp's one untested regime was also the
only one that matters for ClO2 and CO2.

This generalises x2_clamp to n electrons sharing ONE density per domain (the 'hom' reading, the same
construction rc_calibrate.py uses for the pseudo-atom and clo2.html uses for its oxygens), which
puts Zx = 2, 3, 4 in reach and gives an INDEPENDENT number for what mol_fast computes in 3D.

THE CONSTRUCTION.  Two kernels of charge Zx at +-D/2, each clamped as V = -Zx/max(r, r_c), each
owning n = Zx valence electrons in a shared normalised density u_m^2, the domains split at the
midplane.  With U_ab = \int P[u_a^2] u_b^2,

    E = 2[ n(T + V_K) + (n(n-1)/2) U_self ] + n^2 U_12 + Zx^2 / D

n(n-1)/2 intra-domain pairs in each of the two domains, n^2 inter-domain pairs, every pair counted
ONCE by the i != j rule -- the house construction, no (Z-1)/Z factor and no kinetic fudge.  Varying
u_1 under \int u_1^2 = 1 and dividing by 2n gives the one-electron equation

    -(1/2) Lap u_1 + [ V + (n-1) P[u_1^2] + n P[u_2^2] ] u_1 = eps u_1

whose n = 1 case drops the self term and reproduces x2_clamp exactly -- so this is a strict
extension, and `--validate` checks it against x2_clamp numerically rather than by inspection.

The atom is solved in the SAME code on the SAME mesh so that E_bind = 2E(X) - E(X2) cancels the
discretisation error, per the house rule that differences within one composition cancel.

D is scanned only over COMMENSURATE values, inherited from x2_clamp: the kernels must sit on cell
centres or the egg-crate noise is the size of the binding itself.

usage:  python3 tools/x2_clamp_occ.py --validate        # n=1 must reproduce x2_clamp
        python3 tools/x2_clamp_occ.py [h] [L]           # the second-row table
"""
import os, sys, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from axisym_realqm import Grid, stiffness
from axisym_hom2 import _poisson_setup
from x2_clamp import _clamped_K

HA_EV = 27.211386


def atom_clamp_n(Z, rc, n_el, Lz=14.0, Lr=14.0, h=0.175, iters=200, tol=1e-10):
    """n electrons sharing one density on a clamped +Z kernel, axisymmetric.

    E = n(T + V_K) + (n(n-1)/2) U_self,  with the one-electron potential V + (n-1) P[u^2].
    """
    g = Grid(Lz, Lr, h, zmin=-Lz)
    K = _clamped_K(g, [(0.0, float(Z))], rc)
    S = stiffness(g)
    solveP, bnd, rrf = _poisson_setup(g)
    w = 1.0 / np.sqrt(g.V); W = sp.diags(w)
    Pself = np.zeros(g.N); u = None; Eold = None
    for it in range(iters):
        A = (W @ (S + sp.diags((-K + (n_el - 1) * Pself) * g.V)).tocsr() @ W).tocsr()
        val, vec = eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000,
                         v0=None if u is None else u / w)
        u = w * vec[:, 0]; u = u / np.sqrt(np.sum(u * u * g.V))
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


def x2_clamp_n(Z, rc, n_el, D, Lz=14.0, Lr=14.0, h=0.175, iters=200, tol=1e-9):
    """Two clamped +Z kernels at +-D/2, n electrons per domain, split at the midplane."""
    gh = Grid(Lz, Lr, h, zmin=0.0)
    gf = Grid(Lz, Lr, h, zmin=-Lz)
    nzh = gh.nz
    assert gf.nz == 2 * nzh
    K = _clamped_K(gh, [(-D / 2, float(Z)), (D / 2, float(Z))], rc)
    S = stiffness(gh)
    solveP, bnd, rrf = _poisson_setup(gf)
    w = 1.0 / np.sqrt(gh.V); W = sp.diags(w)
    mir = np.arange(nzh)[::-1]
    Pself = np.zeros(gh.N); Ppart = np.zeros(gh.N); u = None; Eold = None
    for it in range(iters):
        A = (W @ (S + sp.diags((-K + (n_el - 1) * Pself + n_el * Ppart) * gh.V)).tocsr() @ W).tocsr()
        val, vec = eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000,
                         v0=None if u is None else u / w)
        u = w * vec[:, 0]; u = u / np.sqrt(np.sum(u * u * gh.V))
        densf = np.zeros((gf.nz, gf.nr)); densf[nzh:, :] = (u * u).reshape(nzh, gh.nr)
        b = 4 * np.pi * densf.ravel() * gf.V
        b[bnd] = 1.0 / rrf[bnd]
        Pfull = solveP(b).reshape(gf.nz, gf.nr)
        Pself = Pfull[nzh:, :].ravel()     # own density, own half
        Ppart = Pfull[mir, :].ravel()      # own density at mirrored points = partner's at own
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


def scan(Z, rc, n_el, h, L, dlo=0.7, dhi=6.0):
    """Minimise E over commensurate D = (2i+1)h; coarse then refine.  Returns (D, E2, edge)."""
    allD = [(2 * i + 1) * h for i in range(1, 60)]
    allD = [D for D in allD if dlo <= D <= dhi]
    vals = {}
    for k in range(0, len(allD), 3):
        vals[k] = x2_clamp_n(Z, rc, n_el, allD[k], L, L, h)
    kb = min(vals, key=vals.get)
    for k in range(max(0, kb - 3), min(len(allD), kb + 4)):
        if k not in vals:
            vals[k] = x2_clamp_n(Z, rc, n_el, allD[k], L, L, h)
    kb = min(vals, key=vals.get)
    return allD[kb], vals[kb], kb in (0, len(allD) - 1)


def validate():
    """n = 1 must reproduce x2_clamp bit for bit: the self term is multiplied by zero."""
    from x2_clamp import x2_clamp, atom_clamp
    print("VALIDATION: n = 1 drops the self term, so this must reduce to x2_clamp.\n")
    print("%26s %16s %16s %11s" % ("", "x2_clamp_occ", "x2_clamp", "diff"))
    for rc, D in ((0.0, 3.325), (0.8, 3.325), (2.0, 4.025)):
        a = x2_clamp_n(1, rc, 1, D, 14.0, 14.0, 0.175)
        b = x2_clamp(rc, D, 14.0, 14.0, 0.175)
        print("  rc=%4.2f D=%6.3f  dimer %16.9f %16.9f %11.2e" % (rc, D, a, b, abs(a - b)))
    for rc in (0.0, 0.8):
        a = atom_clamp_n(1, rc, 1, 14.0, 14.0, 0.175)
        b = atom_clamp(rc, 14.0, 14.0, 0.175)
        extra = "   (exact E(H) = -0.5)" if rc == 0 else ""
        print("  rc=%4.2f           atom %16.9f %16.9f %11.2e%s" % (rc, a, b, abs(a - b), extra))


if __name__ == '__main__':
    if '--validate' in sys.argv:
        validate(); sys.exit(0)
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.175
    L = float(sys.argv[2]) if len(sys.argv) > 2 else 14.0

    # Zx, r_c calibrated to the MEASURED IE1 in the clamp model (tools/rc_calibrate.py), then the
    # measured dimer for reference.  R_e in bohr, D_e in eV.  The reduced model keeps only Zx of the
    # element's valence electrons, so these dimers are MODELS of X2, not X2 -- read the bond, and
    # read the binding as an order of magnitude.
    # r_c values are the CLAMP-calibrated ones (tools/rc_calibrate.py, core='clamp'), NOT the
    # hard-depletion radii -- the two are different numbers for the same element and no conversion
    # carries between core models. Earlier placeholders (Li 2.00, C 1.24) were hard-depletion /
    # guessed values and produced two invalid rows.
    ROWS = [("H2",  1, 0.0,     1.401, 4.75),
            ("Li2", 1, 3.03588, 5.051, 1.05),
            ("O2",  2, 0.78683, 2.282, 5.12),
            ("N2",  3, 0.93741, 2.074, 9.76),
            ("C2",  4, 1.50648, 2.348, 6.21)]

    print("CLAMPED X2, n = Zx ELECTRONS PER DOMAIN.  h=%.3f  L=%.0f  commensurate D=(2i+1)h" % (h, L))
    print("V = -Zx/max(r, r_c); r_c calibrated to the measured IE1 in this same core model.")
    print("Zx/r_c is the PLATEAU DEPTH: 0.5 for Li, 2.54 for O -- the regime being tested.\n")
    print("%5s %3s %8s %7s | %11s %8s %8s | %7s %8s | %s"
          % ("sys", "Zx", "r_c", "Zx/r_c", "E(X)", "D_min", "E_b eV", "R_e exp", "D_e exp", "bond err"))
    for name, Z, rc, Re, De in ROWS:
        t0 = time.time()
        eX = atom_clamp_n(Z, rc, Z, L, L, h)
        D, E2, edge = scan(Z, rc, Z, h, L, dlo=max(0.7, 0.4 * Re), dhi=min(8.0, 3.0 * Re))
        eb = (2 * eX - E2) * HA_EV
        print("%5s %3d %8.5f %7.2f | %11.5f %8.3f %8.3f%s | %7.3f %8.2f | %+6.1f%%   [%.0fs]"
              % (name, Z, rc, (Z / rc if rc > 0 else float('inf')), eX, D, eb,
                 "*" if edge else " ", Re, De, 100 * (D - Re) / Re, time.time() - t0))
        sys.stdout.flush()
    print("\n* = minimum at the scan edge, NOT an interior minimum -- do not read it as a bond.")
