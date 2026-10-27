"""
THE OVERLAP DIAL, INTERIOR COMPUTED -- not just its two ends.

The open question of realqm_open_choices.md #1: strict non-overlap is stated in the theory
(RealQM_method:101, "disjoint"), overlap is what the codes do, and He's exact energy lies BETWEEN
the two ends (homogenised -2.845, disjoint hemispheres -3.026, exact -2.9037). Nothing fixes the
position. Nobody has computed the interior.

THE FAMILY.  Two electrons sharing one profile u with position-dependent weights:
    rho_1 = 2 w u^2,   rho_2 = 2 (1-w) u^2,   w(z) = 1/(1+exp(-z/lam)),
so lam -> infinity is full overlap (w = 1/2, the homogenised shell) and shrinking lam moves toward
exclusion.  With INT u^2 = 1 and u even in z, each electron carries unit charge automatically.

WHY THE ALGEBRA IS KIND.  Put s_i = sqrt(2 w_i), so sqrt(rho_i) = s_i u.  Then
    T = SUM_i (1/2) INT |grad(s_i u)|^2
and since SUM_i s_i^2 = 2 identically, SUM_i s_i grad s_i = (1/2) grad(2) = 0 -- the cross terms
cancel exactly.  What is left is
    T = INT |grad u|^2  +  (1/2) INT u^2 Q,      Q = |grad s_1|^2 + |grad s_2|^2,
with Q a FIXED function of lam.  So the whole problem is a standard SCF in u, and Q is literally the
price of the partition: zero at full overlap, divergent as w approaches a step.

WHAT THAT LAST POINT MEANS.  A step in w makes sqrt(rho) discontinuous, so Q diverges -- in H^1 a
jump costs infinite energy.  **So the disjoint-hemisphere end is NOT in this family at all.**  It is
reachable only by a construction that lets the density jump for free, which is what a masked
interface does.  The prediction to test is therefore that the H^1-admissible dial lies entirely on
ONE SIDE of the exact answer, and that the apparent bracketing of -2.9037 was an artefact of
including an inadmissible end.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize
from axisym_realqm import Grid, stiffness, kernel_potential
from axisym_h1 import make_poisson

np.seterr(all='ignore')


def weights(g, lam):
    """w and the partition cost Q = |grad s_1|^2 + |grad s_2|^2, s_i = sqrt(2 w_i)."""
    Z = g.Z
    w = 1.0/(1.0 + np.exp(-Z/lam))
    s1, s2 = np.sqrt(2*w), np.sqrt(2*(1-w))
    h = g.h
    Q = np.zeros_like(w)
    for s in (s1, s2):
        dz = np.gradient(s, h, axis=0)
        dr = np.gradient(s, h, axis=1)
        Q += dz*dz + dr*dr
    return w.ravel(), Q.ravel()


def solve(g, K, lam, Zk=2.0, iters=9, mu0=200.0):
    S = stiffness(g); pois = make_poisson(g); V = g.V
    w, Q = weights(g, lam)
    u = np.exp(-Zk*np.sqrt(g.Z**2 + g.R**2)).ravel()
    u /= np.sqrt(np.sum(u*u*V))
    lamL, mu = 0.0, mu0
    def EG(x):
        x2 = x*x
        Su = S @ x
        r1, r2 = 2*w*x2, 2*(1-w)*x2
        P1, P2 = pois(r1, float(np.sum(r1*V))), pois(r2, float(np.sum(r2*V)))
        T = 2.0*float(x @ Su) + 0.5*float(np.sum(Q*x2*V))      # 2*u^T S u = INT |grad u|^2
        Ven = -float(np.sum(K*(r1+r2)*V))
        Vee = float(np.sum(P1*r2*V))
        E = T + Ven + Vee
        gr = (4.0*Su + (Q*V)*x
              - 2.0*K*V*x*2.0
              + 4.0*x*V*(w*P2 + (1-w)*P1))
        return E, gr
    for _ in range(iters):
        def f(x):
            E, gr = EG(x)
            c = float(np.sum(x*x*V)) - 1.0
            return E - lamL*c + 0.5*mu*c*c, gr + (-lamL + mu*c)*2*x*V
        res = minimize(f, u, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=4000, ftol=1e-15, gtol=1e-12))
        u = res.x
        c = float(np.sum(u*u*V)) - 1.0
        lamL -= mu*c; mu *= 3.0
        if abs(c) < 1e-12: break
    E, _ = EG(u)
    return E, float(np.sum(Q*u*u*V))*0.5, abs(c)


if __name__ == '__main__':
    h = float(os.environ.get('H', 0.25))
    g = Grid(Lz=9.0, Lr=7.0, h=h, zmin=-9.0)
    K = kernel_potential(g, [(0.0, 2.0)], 0.285*h)
    print("HELIUM ACROSS THE OVERLAP DIAL (h = %.3f)" % h)
    print("ends on record: homogenised -2.845 | exact -2.9037 | disjoint hemispheres -3.026\n")
    print(f"{'lambda':>9s}{'E (Ha)':>12s}{'partition cost':>16s}{'|c|':>9s}   regime")
    for lam in (40.0, 8.0, 3.0, 1.5, 0.8, 0.4, 0.2):
        E, Qc, c = solve(g, K, lam)
        reg = 'full overlap' if lam > 20 else ('toward exclusion' if lam < 1 else '')
        print(f"{lam:9.2f}{E:12.5f}{Qc:16.5f}{c:9.0e}   {reg}", flush=True)
