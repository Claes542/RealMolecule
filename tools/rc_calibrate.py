"""
CALIBRATE r_c AGAINST THE MEASURED IONIZATION ENERGY, IN THE MODEL molecule.js ACTUALLY RUNS.

Why this file exists.  The alkali dimer series (tools/x2_R_sweep.py) got bond lengths to 7.7% with
R fixed per element by making the pseudo-atom reproduce that element's MEASURED first ionization
energy.  The XO2 runs instead took r_c = 0.5 off a menu, calibrated to nothing.  This computes what
the calibration would give for Cl and O.

TWO CORE MODELS ARE SUPPORTED, because molecule.js turns out to use BOTH -- see `core=`:

    core='clamp'  V(r) = -Zx / max(r, r_c)      a plateau of depth Zx/r_c, density admitted inside
    core='hard'   density identically 0 for r < R, free (Neumann) surface at R

WHICH ONE molecule.js USES DEPENDS ON WHOSE CORE IT IS (verified by reading the three shaders that
carry isInsideRc: updateU_WGSL, chebyshevStepWGSL, applyH_WGSL).

  - a domain's OWN core is HARD DEPLETED: isInsideRc zeroes the field (U = 0, HU = 0) and the
    surviving neighbours are blended flat, which is the free Neumann surface.  The clamp at
    molecule.js:1963 is never sampled there because the density is zero.
  - ANOTHER domain's core, with USER_FULL_RC off (the default), is CLAMPED AND PENETRABLE:
    isInsideRc tests only atoms[lbl].rc, so the cell is not excluded and its density feels
    -Z/max(r, r_c).  molecule.js:297 documents this and offers USER_FULL_RC as the fix, "off by
    default so the change can be MEASURED against the old behaviour rather than assumed".

So the solver is a HYBRID whose core model is selected by which domain owns the cell.  For a SINGLE
CENTRE the distinction collapses -- every cell inside r_c is excluded by the own-atom test -- so
every atomic result, including every IE on record (atom_ie_ladder.html "valence excluded from
r < R", tools/col2n.py, tools/x2_R_sweep.py), is hard depletion whatever the potential says.  The
hybrid exists only from two centres up.

Calibrate in the model you intend to run.  The 0.855 conversion in project_rc_universal_constant
relates hard depletion to Ashcroft's empty core and does NOT carry over to the clamp, so the clamp
is bisected directly here and needs no conversion constant.

NOTE on which model is better, measured not assumed (tools/x2_clamp.py, alkali dimers, each r_c
fitted to that element's measured IE): the CLAMP gives mean |bond| error 4.9% and binding -3..+22%,
against hard depletion 8.5% / +36..+101% and the empty core 11.4% / -47..-68%.  The clamp closes the
factor-two binding error.  That is measured for SHALLOW plateaus only: Zx/r_c is 0.21-0.33 Ha for
the alkalis but 2.54 (O) and 2.71 (Cl), ten times deeper, for the second row.

The pseudo-atom.  Kernel charge Zx (house convention: O = 2, Cl = 3, N = 3, C = 4), with Zx valence
electrons sharing ONE radial density -- the 'hom' reading, which is exactly spherical and so exactly
1D.  n electrons in a shared normalised density u^2 give

    E(n) = n (T + V_K) + [n(n-1)/2] U,     T = (1/2) \int |u'|^2 dV,  V_K = \int V u^2 dV,
                                           U = \int P[u^2] u^2 dV

with the self-consistent one-electron potential V + (n-1) P[u^2].  The n(n-1)/2 is the i != j rule:
n unit charges, each pair counted once, no (Z-1)/Z factor and no kinetic fudge.

    IE1 = E(Zx - 1) - E(Zx)

computed in the SAME construction on the SAME mesh, so the discretisation largely cancels -- the
house rule that differences within one composition cancel their errors.  Bisect r_c on IE1.

Discretisation.  Linear FEM on a radial grid with the measure 4 pi r^2 dr, 3-point Gauss per element,
and a NODE PLACED EXACTLY AT r_c so the kink in V is resolved and not smeared.  Natural (Neumann)
conditions at both ends: free at the origin, free at infinity.  No basis set, no fitted parameter.

Verification built in (run with --check): E(H) = -0.5 to 4.2e-06 and the hydrogenic Hartree integral
5Z/8 to 4.2e-06.  Hydrogen correctly refuses to bracket, since a bare +1 kernel IS hydrogen at
r_c = 0 -- the R = 0 row of the published dimer series.  Cross-validated two further ways: the
clamp radii agree to <=0.2% with x2_clamp.fit_rc, which re-bisects on a 2D axisymmetric mesh
(h=0.175), and the hard-depletion one-electron energies agree with col2n.one_elec, an independent
exact 1D solve at n=60000 out to 160 a0.

Ladder (tools/rc_calibrate_ladder.py): r_c is converged to 5.7e-06 by h=0.01 and unchanged from
Rout = 40 to 80, so the radius itself carries no discretisation error worth propagating.
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

HA_EV = 27.211386

# MEASURED FIRST IONIZATION ENERGIES, eV.  NIST values from model knowledge -- flagged UNVERIFIED,
# they are the one external input here and every calibrated radius below depends on them.
IE1_EV = {
    'H':  13.598434,
    'Li':  5.391715,
    'Na':  5.139077,
    'K':   4.340663,
    'Rb':  4.177128,
    'Cs':  3.893905,
    'C':  11.260288,
    'N':  14.534134,
    'O':  13.618055,
    'Cl': 12.967633,
    'Br': 11.814,
}

# House pseudo-kernel charges (valence left outside the screened core).
ZX = {'H': 1, 'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1,
      'C': 4, 'N': 3, 'O': 2, 'Cl': 3, 'Br': 3}


# ---------------------------------------------------------------- radial mesh and FEM matrices

def radial_grid(rc, Rout, h, core='clamp'):
    """Grid for the chosen core model.

    'clamp' -- [0, Rout] with a node exactly at rc, so V's kink is a node.
    'hard'  -- [rc, Rout] only: the density is EXCLUDED from r < rc, and the natural (no-flux)
               condition at the first node is the FREE Neumann surface at the core.  This is what
               molecule.js does to a domain's own core (isInsideRc zeroes the wavefunction and the
               surviving neighbours are blended flat), and what tools/x2_R_sweep.py and
               tools/col2n.py do via masked_stiffness / dropped cells.
    """
    if core == 'hard':
        if rc <= 0:
            return np.linspace(0.0, Rout, int(round(Rout / h)) + 1)
        return np.linspace(rc, Rout, max(8, int(round((Rout - rc) / h))) + 1)
    if rc <= 0:
        return np.linspace(0.0, Rout, int(round(Rout / h)) + 1)
    n_in = max(4, int(round(rc / h)))
    n_out = max(8, int(round((Rout - rc) / h)))
    return np.concatenate([np.linspace(0.0, rc, n_in + 1),
                           np.linspace(rc, Rout, n_out + 1)[1:]])


# 3-point Gauss-Legendre on [-1, 1]
_GX = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
_GW = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])


def _elements(r):
    """Per-element quadrature data, vectorised over elements.  Returns (dV, phi, he, rq)."""
    a = r[:-1][:, None]
    b = r[1:][:, None]
    he = (b - a)
    rq = 0.5 * (a + b) + 0.5 * he * _GX[None, :]            # ne x nq
    dV = 4.0 * np.pi * rq ** 2 * (0.5 * he * _GW[None, :])
    phi = np.stack([(b - rq) / he, (rq - a) / he])           # 2 x ne x nq
    return dV, phi, he[:, 0], rq


def _scatter(r, blocks):
    """Assemble a tridiagonal from per-element 2x2 blocks blocks[i][j] of shape (ne,)."""
    n = r.size
    ne = n - 1
    e = np.arange(ne)
    rows = np.concatenate([e, e, e + 1, e + 1])
    cols = np.concatenate([e, e + 1, e, e + 1])
    vals = np.concatenate([blocks[0][0], blocks[0][1], blocks[1][0], blocks[1][1]])
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def assemble(r, Z, rc, core='clamp'):
    """Return (T, M, VK) as sparse tridiagonals for the measure dV = 4 pi r^2 dr.

    T   = (1/2) \int phi_i' phi_j' dV
    M   =       \int phi_i  phi_j  dV
    VK  =       \int V phi_i phi_j dV  with V = -Z/max(r, rc)
    """
    dV, phi, he, rq = _elements(r)
    # V * dV is smooth even at r = 0 (V r^2 ~ -Z r), so no singularity in the integrand
    rclamp = np.maximum(rq, rc) if (rc > 0 and core == 'clamp') else rq
    Vq = np.where(rclamp > 0, -Z / np.where(rclamp > 0, rclamp, 1.0), 0.0)
    dphi = np.array([-1.0, 1.0])[:, None] / he[None, :]      # 2 x ne
    sdV = dV.sum(axis=1)
    T = _scatter(r, [[0.5 * dphi[i] * dphi[j] * sdV for j in range(2)] for i in range(2)])
    M = _scatter(r, [[np.sum(phi[i] * phi[j] * dV, axis=1) for j in range(2)] for i in range(2)])
    VK = _scatter(r, [[np.sum(Vq * phi[i] * phi[j] * dV, axis=1) for j in range(2)]
                      for i in range(2)])
    return T, M, VK


def assemble_pot(r, vnode, cache=None):
    """\int v phi_i phi_j dV for a potential given by its NODAL values (linear interpolation)."""
    dV, phi, he, rq = cache if cache is not None else _elements(r)
    vq = vnode[:-1][:, None] * phi[0] + vnode[1:][:, None] * phi[1]
    return _scatter(r, [[np.sum(vq * phi[i] * phi[j] * dV, axis=1) for j in range(2)]
                        for i in range(2)])


def hartree(r, rho):
    """Radial Poisson for a spherical density: P(r) = (1/r)\int_0^r rho dV + \int_r^inf rho dV/r'.

    Exact-in-form solution of the spherical Poisson equation, integrated by trapezoid on the mesh.
    """
    w = 4.0 * np.pi * r ** 2 * rho
    # cumulative \int_0^r rho dV
    q = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(r))])
    g = 4.0 * np.pi * r * rho                                  # = rho dV / r
    tail = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(r))])
    tail = tail[-1] - tail
    with np.errstate(divide='ignore', invalid='ignore'):
        inner = np.where(r > 0, q / np.where(r > 0, r, 1.0), 0.0)
    P = inner + tail
    # at r = 0 the first term vanishes and P(0) = \int rho dV / r', already in tail
    return P


def solve_n(Z, rc, n_el, Rout=40.0, h=0.01, iters=200, tol=1e-12, core='clamp'):
    """n_el electrons sharing one radial density on a +Z pseudo-kernel.  Returns (E, parts)."""
    r = radial_grid(rc, Rout, h, core)
    T, M, VK = assemble(r, Z, rc, core)
    cache = _elements(r)
    Mc = M.tocsc()
    base = (T + VK).tocsr()
    P = np.zeros(r.size)
    E_old = None
    for _ in range(iters):
        A = (base + (n_el - 1) * assemble_pot(r, P, cache)).tocsc() if n_el > 1 else base.tocsc()
        val, vec = eigsh(A, k=1, M=Mc, sigma=-2.0 * Z * Z - 1.0,
                         which='LM', tol=1e-12, maxiter=100000)
        u = vec[:, 0]
        u = u / np.sqrt(float(u @ (M @ u)))
        rho = u * u
        P_new = hartree(r, rho)
        P = P_new if E_old is None else 0.5 * P + 0.5 * P_new
        Tv = float(u @ (T @ u))
        Vv = float(u @ (VK @ u))
        wdV = 4.0 * np.pi * r ** 2 * rho * P                     # U = \int P rho dV, trapezoid
        Uv = float(np.sum(0.5 * (wdV[1:] + wdV[:-1]) * np.diff(r)))
        E = n_el * (Tv + Vv) + 0.5 * n_el * (n_el - 1) * Uv
        if n_el == 1:
            return E, dict(T=Tv, VK=Vv, U=0.0, r=r, u=u)
        if E_old is not None and abs(E - E_old) < tol:
            break
        E_old = E
    return E, dict(T=Tv, VK=Vv, U=Uv, r=r, u=u)


def ie1(Z, rc, n_el, **kw):
    """IE1 = E(n-1) - E(n), same construction, same mesh."""
    if n_el == 1:
        return -solve_n(Z, rc, 1, **kw)[0]
    return solve_n(Z, rc, n_el - 1, **kw)[0] - solve_n(Z, rc, n_el, **kw)[0]


def bisect_rc(Z, n_el, target_ha, lo=0.05, hi=8.0, tol=1e-5, **kw):
    """Find r_c with IE1(r_c) = target.  IE1 DECREASES with r_c (a shallower core binds less)."""
    f_lo = ie1(Z, lo, n_el, **kw) - target_ha
    f_hi = ie1(Z, hi, n_el, **kw) - target_ha
    if f_lo * f_hi > 0:
        return None, (f_lo + target_ha, f_hi + target_ha)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = ie1(Z, mid, n_el, **kw) - target_ha
        if f_lo * f <= 0:
            hi, f_hi = mid, f
        else:
            lo, f_lo = mid, f
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi), None


# ---------------------------------------------------------------- verification

def check(h=0.01, Rout=40.0):
    print("VERIFICATION (house rule: verify against exact values, do not read code)")
    E, p = solve_n(1.0, 0.0, 1, Rout=Rout, h=h)
    print("  H atom, Z=1 n=1 rc=0        E = %.8f   exact -0.5         err %.1e"
          % (E, abs(E + 0.5)))
    # hydrogenic Hartree integral:  for rho = (Z^3/pi) exp(-2Zr),  U = 5Z/8
    for Z in (1.0, 2.0):
        r = radial_grid(0.0, Rout, h)
        rho = (Z ** 3 / np.pi) * np.exp(-2 * Z * r)
        P = hartree(r, rho)
        w = 4 * np.pi * r ** 2 * P * rho
        U = float(np.sum(0.5 * (w[1:] + w[:-1]) * np.diff(r)))
        print("  Hartree on 1s(Z=%.0f)          U = %.8f   exact %.8f   err %.1e"
              % (Z, U, 5 * Z / 8, abs(U - 5 * Z / 8)))
    print()


def main():
    args = sys.argv[1:]
    h = 0.01
    Rout = 40.0
    if '--check' in args:
        check(h, Rout)
    which = [a for a in args if not a.startswith('--')]
    if not which:
        which = ['Li', 'Na', 'K', 'Rb', 'Cs', 'O', 'Cl', 'N', 'C', 'Br']
    # both core models side by side, since the solver uses one for the own core and the other
    # for a partner's core; pass --clamp or --hard to restrict
    cores = []
    if '--hard' in args: cores = ['hard']
    elif '--clamp' in args: cores = ['clamp']
    else: cores = ['clamp', 'hard']
    print("r_c CALIBRATED TO THE MEASURED IE1,  IE1 = E(Zx-1) - E(Zx), same mesh, hom reading")
    print("  clamp: V = -Zx/max(r,r_c), density admitted inside   (a PARTNER's core, FULL_RC off)")
    print("  hard : density = 0 for r < R, free Neumann at R       (a domain's OWN core; all atoms)")
    print("h=%.3f  Rout=%.0f;  IEs are NIST from model knowledge, UNVERIFIED\n" % (h, Rout))
    hdr = "  el  Zx   IE1 exp eV |" + "".join("  r_c(%-5s)  resid eV |" % c for c in cores)
    print(hdr)
    for el in which:
        if el not in IE1_EV:
            print("  %-3s  -- not tabulated" % el); continue
        Z = float(ZX[el]); n_el = int(ZX[el])
        tgt = IE1_EV[el] / HA_EV
        row = "  %-3s %3d   %9.4f |" % (el, n_el, IE1_EV[el])
        for c in cores:
            rc, rng = bisect_rc(Z, n_el, tgt, Rout=Rout, h=h, core=c)
            if rc is None:
                row += "  NO BRACKET %.2f-%.2f eV |" % (rng[1] * HA_EV, rng[0] * HA_EV)
            else:
                got = ie1(Z, rc, n_el, Rout=Rout, h=h, core=c) * HA_EV
                row += "   %8.5f  %+8.1e |" % (rc, got - IE1_EV[el])
        print(row)
        sys.stdout.flush()


if __name__ == '__main__':
    main()
