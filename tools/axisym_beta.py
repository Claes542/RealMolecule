# ROBIN beta ON THE HARD (face-omission) CONSTRUCTION -- axisym, deterministic, no browser.
#
# WHY HERE AND NOT molecule.js. The beta ladder on molecule.js works (E_bind +23.1% -> +3.0% across
# beta 0 -> 0.050, concave throughout, D_e crossed at beta ~ 0.059) but molecule.js has +-0.03 Ha
# RUN-TO-RUN SCATTER and the whole effect being fitted is dE = +0.040. Signal ~ noise, so beta comes
# out as "0.05 +- 0.04" -- a trend, not a value. axisym is a deterministic eigensolve: no scatter,
# and its one-electron limit is EXACT (H to 0.02%), where molecule.js's H is -0.4396 vs -0.5 (12%).
#
# WHAT beta IS. The surface energy (beta/2) INT Psi^2 dS on the interface, whose natural boundary
# condition is d(psi)/dn + beta*psi = 0 -- Robin, interpolating beta=0 (Neumann = the hard
# construction) to beta->inf (Dirichlet). It is the repulsion between EQUIVALENT domains across a
# mirror plane that territorial exclusion cannot supply (the masked and continuous constructions
# agree to 4e-8 there, i.e. exclusion generates NOTHING at a mirror plane), so it acts exactly where
# the over-binding lives. Nothing is imposed: adding the energy is enough.
#
# GEOMETRY. commensurate.solve() puts the domain on gh = Grid(L,L,h,zmin=0.0), the half-space z>0.
# Its no-flux outer face at z=0 IS the hard-Neumann midplane. So the interface Gamma is z=0, the
# adjacent cells are i=0, and face j has area 2*pi*r_j*h.
#
# *** THE FACTOR IS NOT ASSUMED -- IT IS MEASURED. ***
# E = 2(T+VK) + U12 + V_KK counts T,VK per electron (x2) but U12 once; the SCF operator carries the
# partner potential at weight 1. Where a SHARED one-interface surface term lands in that bookkeeping
# is exactly the kind of factor-of-2 that corrupts a fit silently -- molecule.js's own beta
# validation had to check "the face counting and the factor of 1/2" for this reason. So:
#   FIRST-ORDER PERTURBATION THEORY IS THE GATE. For small beta,
#         dE/dbeta  ->  (1/2) INT_Gamma u^2 dS      evaluated on the beta=0 state.
# The script computes both and prints their RATIO. Ratio 1 => factor right. Ratio 2 or 1/2 => the
# placement is off by that factor and the fitted beta would be wrong by it. Do not read the fit
# until the ratio is 1.
#
# ALSO CHECKED: concavity. E(beta) is a minimum over psi of quantities linear in beta, hence below
# its tangent at zero, so dE/dbeta must FALL with beta. A rising slope means the state is not
# relaxing to the surface energy it is charged for -- the exact defect molecule.js's beta had before
# its boundary force was added (slopes rose 3.93/4.01/4.27; after the fix they fall).
import sys, os, numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, factorized
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
os.chdir('/Users/cgjoh/Development/H2O/tools')
from axisym_realqm import Grid, stiffness, laplacian, kernel_potential
np.seterr(all='ignore')

SOFT = 1.0        # eps = h, matching mol_fast's convention (commensurate.py hardcodes 0.2846)
_c = {}

def solve_beta(h, L, D, beta, surf_op_factor=0.25):
    """H2 on the hard half-space construction with a Robin surface energy on the midplane.

    surf_op_factor multiplies A in the SCF OPERATOR; the ENERGY always gets (beta/2)*sum u^2 A.
    Default 0.25 follows from E_surf = (beta/4)INT u1^2 + (beta/4)INT u2^2 and dividing the
    stationarity condition by 2 as the other one-body terms are -- but it is VERIFIED, not trusted:
    see the perturbation-theory gate in the header.
    Returns (E, u, S_int) with S_int = INT_Gamma u^2 dS.
    """
    nzh = int(round(L/h)); L = nzh*h
    gh = Grid(L, L, h, zmin=0.0); gf = Grid(L, L, h, zmin=-L)
    m = (D/2)/h - 0.5
    assert abs(m - round(m)) < 1e-9, "D=%.4f NOT commensurate with h=%.4f (m=%.6f)" % (D, h, m)
    key = (h, L)
    if key not in _c:
        Lp = laplacian(gf).tolil(); bnd = np.zeros(gf.N, bool)
        bnd[gf.idx(0, np.arange(gf.nr))] = True
        bnd[gf.idx(gf.nz-1, np.arange(gf.nr))] = True
        bnd[gf.idx(np.arange(gf.nz), gf.nr-1)] = True
        for k in np.where(bnd)[0]: Lp.rows[k] = [k]; Lp.data[k] = [1.0]
        _c[key] = (stiffness(gh), factorized(Lp.tocsc()), bnd,
                   np.sqrt(gf.Z**2 + gf.R**2).ravel(), gh, gf)
    S, solveP, bnd, rrf, gh, gf = _c[key]
    eps = SOFT*h
    K = kernel_potential(gh, [(-D/2, 1.), (D/2, 1.)], eps)
    # INTERFACE FACE AREAS on Gamma (z=0): cells i=0, face area 2*pi*r_j*h. Zero elsewhere.
    A = np.zeros(gh.N)
    for j in range(gh.nr):
        A[gh.idx(0, j)] = 2*np.pi*gh.r[j]*h
    w = 1.0/np.sqrt(gh.V); W = sp.diags(w); mir = np.arange(gh.nz)[::-1]
    P1 = np.zeros(gh.N); u = None; Eo = None
    for it in range(120):
        diag = (-K + P1)*gh.V + (beta*surf_op_factor)*A
        Aop = (W @ (S + sp.diags(diag)).tocsr() @ W).tocsr()
        val, vec = eigsh(Aop, k=1, which='SA', tol=1e-12, maxiter=400000,
                         v0=None if u is None else u/w)
        u = w*vec[:, 0]; u = u/np.sqrt(np.sum(u*u*gh.V))
        densf = np.zeros((gf.nz, gf.nr)); densf[gh.nz:, :] = (u*u).reshape(gh.nz, gh.nr)
        b = 4*np.pi*densf.ravel()*gf.V; b[bnd] = 1.0/rrf[bnd]
        P1 = solveP(b).reshape(gf.nz, gf.nr)[mir, :].ravel()
        T = float(u @ (S @ u)); VK = -float(np.sum(K*u*u*gh.V))
        U12 = float(np.sum(P1*u*u*gh.V)); S_int = float(np.sum(A*u*u))
        E = 2*(T + VK) + U12 + 1.0/D + 0.5*beta*S_int
        if Eo is not None and abs(E - Eo) < 1e-11: break
        Eo = E
    return E, u, S_int

if __name__ == '__main__':
    h, L, D = 0.10, 10.0, 2.1
    EXACT_H2 = -1.174470      # Kolos-Wolniewicz at R_e = 1.40028 -- a COMPUTATION, not a datum
    print("axisym + Robin beta, hard face-omission construction")
    print("h=%.3f L=%.1f D=%.2f eps=SOFT*h with SOFT=%.3f\n" % (h, L, D, SOFT))

    # --- GATE 1: beta=0 must reproduce the construction's own value -------------------
    E0, u0, S0 = solve_beta(h, L, D, 0.0)
    print("beta=0:  E = %.6f   INT_Gamma u^2 dS = %.6f" % (E0, S0))
    print("         (matched-settings reference measured earlier today: -1.163644)")

    # --- GATE 2: first-order perturbation theory fixes the FACTOR ---------------------
    db = 1e-4
    Ep, _, _ = solve_beta(h, L, D, db)
    slope = (Ep - E0)/db
    print("\nPERTURBATION GATE   dE/dbeta(num) = %.6f   (1/2)INT u^2 dS = %.6f   ratio = %.4f"
          % (slope, 0.5*S0, slope/(0.5*S0)))
    print("   ratio 1 => factor right; 2 or 0.5 => operator placement off by that factor")

    # --- the ladder, with concavity ---------------------------------------------------
    print("\n%-8s %-12s %-11s %-11s" % ('beta', 'E(H2)', 'dE', 'slope'))
    prev = None
    for b in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20):
        E, _, Si = solve_beta(h, L, D, b)
        sl = '' if prev is None else "%.4f" % ((E - prev[1])/(b - prev[0]))
        print("%-8.3f %-12.6f %-11.6f %-11s" % (b, E, E - E0, sl), flush=True)
        prev = (b, E)
    print("\nexact H2 at its own R_e: %.6f" % EXACT_H2)
    print("NOTE: D=2.1 is NOT R_e. Fitting beta on the ENERGY requires E at each beta's OWN")
    print("minimum, so this ladder sets the SCALE of beta; the geometry is then a TEST, not a fit.")
