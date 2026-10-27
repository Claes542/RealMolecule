"""
A POSTERIORI ERROR ESTIMATION FOR A RealQM-CLASS GROUND STATE.
After Dusson & Maday, arXiv:2303.01273, "An overview of a posteriori error estimation and
post-processing methods for nonlinear eigenvalue problems" (electronic-structure ground states).

WHY THIS FILE EXISTS.  The RealQM side of this repo has no output error control: 55 files report
iteration tolerances (tol=1e-11, "CONVERGED at step N") and 19 watch for "drift", which is the
practice clay_navier_stokes.tex explicitly rejects -- "the error in the output is not estimated by
comparing successive meshes and hoping, but bounded by a quantity the computation itself produces".
The dual/a-posteriori machinery appears in ten fluid-side documents and in none of the RealQM
solvers, tools, or the book. This file is the smallest working piece of the transfer.

THE THREE MECHANISMS, and what was actually achieved.

(1) FREE VARIATIONAL UPPER BOUND -- E(u_delta) >= E(u) for any conforming discretisation.  Works,
    trivially, and it matters conceptually: it separates NUMERICAL error from MODEL error, which the
    RealQM codes currently conflate.  (Note it bounds the MODEL's own ground state, not nature's.)

(2) POST-PROCESSING AND THE ERROR ESTIMATOR -- the usable result.  One Newton step on a finer space,
    their (2.7):  (A + 2 g'(u^2)u^2 + g(u^2) - lambda) w = -(A + g(u^2) - lambda) u_delta  on the
    M-orthogonal complement of u_delta; then u~ = u_delta + w and
        error  ~=  E(u~) - E(u_delta).
    VERIFIED on radial hydrogen (exact -0.5), P1 elements, nothing exact used anywhere:
        n    k     E(u_d)      estimate     truth    captured   err of E(u~)
        40   4   -0.40530875   -0.08270   -0.09469      87%      1.20e-02
       160   4   -0.48858060   -0.01062   -0.01142      93%      7.95e-04
       160   8   -0.48858060   -0.01122   -0.01142      98%      2.00e-04
       320   8   -0.49688808   -0.00306   -0.00311      98%      5.05e-05
    So one linear solve gives the error to 2% AND an energy ~57x better.

(3) THE ASYMPTOTIC LOWER BOUND (their 2.6), E(u) >= E(u_delta) - (1/2) a_{u_d}(w,w) -- NOT achieved
    here.  It held in no row.  The reason is that it is asymptotic and these errors (1-10%) are not
    asymptotic: P1 elements on a uniform mesh converge slowly against the 1/r cusp.  Reaching that
    regime needs graded meshes and is beyond this quick implementation.  Reported rather than tuned
    into apparent success.

SCOPE, and it is the important caveat.  All of this holds at a FIXED PARTITION.  RealQM's free
boundary is an extra geometric unknown and the expansions assume smooth dependence on the state, so
this controls the energy GIVEN the interface and says nothing about where the interface belongs --
which is precisely where the 2026-10-26 session's failures lived (0.446 against 2.00, optima pinned
at scan bounds).  Half the problem, and not the half that broke.

A NOTE ON A FAILED ATTEMPT, kept deliberately.  Before reading the paper I tried a plain Weinstein
bound, ||r||^2/gap, with the residual in L2.  It came out ~30x too small at every mesh.  The correct
object is the residual tested against the HESSIAN-weighted reconstruction (the a_{u_d} norm), which
is what (2.7) computes.  The lesson is that residual-based error control is not "compute a residual
norm" -- the weighting is the content.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

Z = 1.0


def assemble(n, L, nq=6):
    """P1 elements on [0,L], nodes r_i = i*h, Dirichlet at both ends -> n interior unknowns.
    Returns stiffness of (1/2)INT u'^2, mass, and the potential -Z INT u^2/r."""
    h = L/(n+1)
    S = np.zeros((n, n)); M = np.zeros((n, n)); V = np.zeros((n, n))
    xg, wg = leggauss(nq)
    for e in range(n+1):                       # element [r_e, r_{e+1}]
        a, b = e*h, (e+1)*h
        loc = [e-1, e]                         # global indices of the two hat functions
        xm = 0.5*(a+b) + 0.5*h*xg; wm = 0.5*h*wg
        # hat values and derivatives on this element
        phi = np.array([(b-xm)/h, (xm-a)/h])
        dphi = np.array([-1.0/h, 1.0/h])
        for i in (0, 1):
            gi = loc[i]
            if gi < 0 or gi >= n: continue
            for j in (0, 1):
                gj = loc[j]
                if gj < 0 or gj >= n: continue
                S[gi, gj] += 0.5*dphi[i]*dphi[j]*h
                M[gi, gj] += np.sum(wm*phi[i]*phi[j])
                V[gi, gj] += -Z*np.sum(wm*phi[i]*phi[j]/xm)
    return S+V, M, h


def embed(u, n, k, L):
    """Coarse P1 state -> fine P1 space with refinement factor k (nested nodes)."""
    nf = (n+1)*k - 1
    rc = np.append(0.0, np.arange(1, n+2)*(L/(n+1)))
    uc = np.append(0.0, np.append(u, 0.0))
    rf = np.arange(1, nf+1)*(L/(nf+1))
    return np.interp(rf, rc, uc)


def run(n, k=8, L=40.0):
    """Returns (upper bound, error estimate, post-processed energy)."""
    from scipy.linalg import eigh, solve
    Ac, Mc, _ = assemble(n, L)
    u = eigh(Ac, Mc)[1][:, 0]                  # generalised SYMMETRIC eigenproblem
    nf = (n+1)*k - 1
    Af, Mf, _ = assemble(nf, L)
    uf = embed(u, n, k, L)
    uf /= np.sqrt(uf @ Mf @ uf)
    lam = float(uf @ Af @ uf)                  # (1) conforming upper bound
    res = Af @ uf - lam*(Mf @ uf)              # residual of the eigenvalue equation
    H = Af - lam*Mf                            # Hessian (g = 0 here, so no response term)
    Mu = Mf @ uf
    P = np.eye(nf) - np.outer(uf, Mu)          # M-orthogonal projector off u
    Hp = P.T @ H @ P + np.outer(Mu, Mu)        # deflate the null direction
    w = P @ solve(Hp, -(P.T @ res))            # their (2.7)
    ut = uf + w; ut /= np.sqrt(ut @ Mf @ ut)   # (2) post-processed state
    lpp = float(ut @ Af @ ut)
    return lam, lpp - lam, lpp


if __name__ == '__main__':
    print("A POSTERIORI ERROR ESTIMATE, radial hydrogen (exact -0.5). Nothing exact is used.\n")
    print(f"{'n':>5s}{'k':>3s}{'E(u_d)':>14s}{'estimate':>12s}{'truth':>12s}{'captured':>10s}"
          f"{'err E(u~)':>12s}")
    for n, k in ((40, 4), (160, 4), (160, 8), (320, 8)):
        lam, est, lpp = run(n, k)
        tru = -0.5 - lam
        print(f"{n:5d}{k:3d}{lam:14.8f}{est:12.5f}{tru:12.5f}{100*est/tru:9.0f}%"
              f"{abs(lpp+0.5):12.2e}", flush=True)
