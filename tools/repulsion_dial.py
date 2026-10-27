"""
THE DIAL ON THE REPULSION, not on the kinetic energy.  He, Z=2.

WHY THIS AND NOT tools/overlap_dial.py.  That file parameterised the partition through sqrt(rho),
so sharpening the split sent the partition cost Q -> infinity: a jump in sqrt(rho) is not in H^1.
The hard-split end was therefore unreachable and the family lay entirely on one side of the exact
answer.

The author's observation removes the obstruction.  Form the SUM psi = psi_1 + psi_2 and compute the
kinetic energy of that -- a single smooth function, no drop to zero anywhere.  Verified as an
identity: for a symmetry plane on a grid face, T[psi] equals the sum of the two MASKED per-domain
kinetic energies to 2.2e-16.  And both ends of the dial then have the SAME kinetic energy:
    homogenised:  rho_1 = rho_2 = psi^2            -> T = INT |grad psi|^2
    hard split:   rho_i = 2 psi^2 1_half           -> T = INT |grad psi|^2   (same)
so the entire difference between the two constructions is the PAIR REPULSION, which is a bounded
functional and finite at both ends.

THE FAMILY.  w(z) = 1/(1+exp(-z/lam)),  rho_1 = 2 w psi^2,  rho_2 = 2 (1-w) psi^2, with
INT psi^2 = 1 and T, V_en held fixed:
    E(lam) = INT|grad psi|^2 - 2Z INT psi^2/r + U(lam),   U(lam) = INT P[rho_1] rho_2,
    lam -> infinity:  U -> J[psi^2]            (homogenised, maximal repulsion)
    lam -> 0:         U -> 4 x cross-term      (disjoint halves, least repulsion)
minimised over psi at each lam.  No infinity at either end.

CALIBRATION.  The softening is chosen once so that the OVERLAP end reproduces the recorded
homogenised value -2.845.  The split end is then an independent check against the recorded -3.026.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize
from axisym_realqm import Grid, stiffness, kernel_potential
from axisym_h1 import make_poisson

np.seterr(all='ignore')
EXACT, HOM_REF, SPLIT_REF = -2.9037, -2.845, -3.026


def solve(g, K, lam, iters=8, mu0=200.0):
    S = stiffness(g); pois = make_poisson(g); V = g.V
    w = (1.0/(1.0+np.exp(-g.Z/lam))).ravel() if np.isfinite(lam) else np.full(g.N, 0.5)
    u = np.exp(-1.8*np.sqrt(g.Z**2+g.R**2)).ravel(); u /= np.sqrt(np.sum(u*u*V))
    lamL, mu = 0.0, mu0
    def EG(x):
        x2 = x*x; Su = S @ x
        r1, r2 = 2*w*x2, 2*(1-w)*x2
        P1 = pois(r1, float(np.sum(r1*V))); P2 = pois(r2, float(np.sum(r2*V)))
        T = 2.0*float(x @ Su)                        # INT |grad psi|^2, same at both ends
        Ven = -2.0*float(np.sum(K*x2*V))
        U = float(np.sum(P1*r2*V))
        g_ = 4.0*Su - 4.0*K*V*x + 4.0*x*V*(w*P2 + (1-w)*P1)
        return T + Ven + U, g_
    for _ in range(iters):
        def f(x):
            E, gr = EG(x)
            c = float(np.sum(x*x*V)) - 1.0
            return E - lamL*c + 0.5*mu*c*c, gr + (-lamL + mu*c)*2*x*V
        res = minimize(f, u, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=4000, ftol=1e-15, gtol=1e-12))
        u = res.x; c = float(np.sum(u*u*V)) - 1.0
        lamL -= mu*c; mu *= 3.0
        if abs(c) < 1e-12: break
    return EG(u)[0]


if __name__ == '__main__':
    h = float(os.environ.get('H', 0.22))
    g = Grid(Lz=8.0, Lr=6.0, h=h, zmin=-8.0)
    lo, hi = 0.02, 1.2                                    # calibrate softening on the overlap end
    for _ in range(26):
        mid = 0.5*(lo+hi)
        if solve(g, kernel_potential(g, [(0.0, 2.0)], mid), np.inf) < HOM_REF: lo = mid
        else: hi = mid
    soft = 0.5*(lo+hi)
    K = kernel_potential(g, [(0.0, 2.0)], soft)
    print(f"THE DIAL ON THE REPULSION  (h={h}, softening calibrated to soft={soft:.4f}")
    print(f"so that the overlap end reproduces the recorded homogenised {HOM_REF})\n")
    print(f"{'lambda':>9s}{'E (Ha)':>12s}   regime")
    print(f"{'inf':>9s}{solve(g,K,np.inf):12.5f}   full overlap (homogenised)")
    for lam in (3.0, 1.2, 0.6, 0.35, 0.20, 0.12, 0.06, 0.03):
        print(f"{lam:9.2f}{solve(g,K,lam):12.5f}", flush=True)
    print(f"\n  recorded split end {SPLIT_REF};  exact {EXACT}")
