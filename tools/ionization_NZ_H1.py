"""
N(Z) WITH THE CONTINUITY CONDITION ACTUALLY IMPOSED -- the faithful construction.

WHY A SECOND FILE.  tools/ionization_NZ.py solved each radial shell INDEPENDENTLY, with natural
conditions on both of its own faces, so the density was free to jump at an interface.  It collapsed,
and the collapse turned out to be that freedom: the uniform-blob mode needs a jump at R1, and in
H^1 a jump costs infinite energy.  RealQM does not permit it.  The model is explicit --

    "the psi_i being continuous across joint boundaries Gamma_i ^ Gamma_j as imposed by
     Psi in H^1(R^3), and the boundary location determined by the energy minimisation"
                                                        -- RealQM_method.tex:101

-- and the faithful reading, stated in realqm_hopping.tex, is that the wave function is ONE GLOBAL
FIELD partitioned by labels, continuous across every interface.  So the right object is not N
independent shells but a single continuous u on [0,L] with u(0)=0, cut into N intervals by the
interface radii, with EACH INTERVAL CARRYING UNIT CHARGE.  Continuity is then automatic rather than
imposed, and the collapse mode is simply not representable.

THE PROBLEM, precisely.  Minimise over u (one vector, u(0)=0) and over the interface radii

    E[u] = (1/2) int (u' - u/r)^2 dr  -  Z int u^2/r dr  +  sum_{i<j} int int rho_i rho_j / |x-y|

subject to the N constraints   int_{I_i} u^2 dr = 1,  one per domain.  There is therefore one
Lagrange multiplier PER DOMAIN -- the N chemical potentials -- which is why this is not a single
eigenproblem.  It is solved here by an augmented Lagrangian on the N constraints with L-BFGS inside,
the gradient being available in closed form:

    dE/du = 2h [ (M u) + (V_nuc + P_other) u ],

M the tridiagonal localisation form and P_other the Hartree potential of the OTHER domains, which
is piecewise because each point belongs to exactly one domain (the house i!=j rule, no
self-interaction anywhere).

SELF-TESTS, run before any verdict.  One domain must give E(H) = -0.5, which checks the functional
and the optimiser against an exact answer.  And the two-domain run must not reproduce the collapse:
if continuity is what excludes it, the energy must now stay near the physical scale instead of
tracking -1.5 Z / R1.

THE VERDICT IS STILL READ FROM THE BOX LADDER, for the reason given in the companion file: a
confined solver always reports binding, so bound/unbound is decided by whether E settles below
E(N-1) with the outer extent saturating, or drifts to E(N-1) from above with the extent tracking L.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize

np.seterr(all='ignore')


def grid(L, n):
    h = L/(n + 1)
    r = h*np.arange(1, n + 1)
    return r, h


def loc_form(r, h):
    """Tridiagonal M with (1/2) int (u'-u/r)^2 dr = h * u^T M u.  Global: no faces omitted."""
    n = len(r)
    d = np.zeros(n); e = np.zeros(n - 1)
    for j in range(n - 1):
        rm = 0.5*(r[j] + r[j+1])
        A = 1.0/h - 1.0/(2*rm)
        B = -1.0/h - 1.0/(2*rm)
        d[j] += 0.5*B*B; d[j+1] += 0.5*A*A; e[j] += 0.5*A*B
    rm = 0.5*r[0]                                  # the face to the origin, where u(0)=0
    A = 1.0/h - 1.0/(2*rm)
    d[0] += 0.5*A*A
    return d, e


def apply_M(d, e, u):
    out = d*u
    out[:-1] += e*u[1:]
    out[1:] += e*u[:-1]
    return out


def hartree(u2, r, h):
    q_in = np.cumsum(u2*h)
    tail = np.cumsum((u2*h/r)[::-1])[::-1] - u2*h/r
    return q_in/r + tail


def energy_grad(u, d, e, r, h, Z, segs):
    """E and dE/du for the global field u with domains segs = [(a,b), ...]."""
    u2 = u*u
    Mu = apply_M(d, e, u)
    T = h*float(u @ Mu)
    Ven = -Z*h*float(np.sum(u2/r))
    pots = []
    for (a, b) in segs:
        q = np.zeros_like(u); q[a:b] = u2[a:b]
        pots.append(hartree(q, r, h))
    Vee = 0.0
    Pother = np.zeros_like(u)
    for i, (a, b) in enumerate(segs):
        for j, (c, dd) in enumerate(segs):
            if j > i:
                Vee += h*float(np.sum(pots[j][a:b]*u2[a:b]))
        Po = np.zeros_like(u)
        for j in range(len(segs)):
            if j != i:
                Po += pots[j]
        Pother[a:b] = Po[a:b]
    E = T + Ven + Vee
    g = 2*h*(Mu + (-Z/r + Pother)*u)
    return E, g


def solve(Z, N, L=30.0, n=700, segs=None, mu0=50.0, rounds=7, verbose=False):
    """Augmented Lagrangian on the N unit-charge constraints; returns E and the outer extent."""
    r, h = grid(L, n)
    d, e = loc_form(r, h)
    if segs is None:
        raise ValueError
    u = r*np.exp(-r/max(1.0, Z))                    # seed
    for (a, b) in segs:                             # make each segment carry ~1
        s = np.sum(u[a:b]**2*h)
        if s > 0:
            u[a:b] /= np.sqrt(s)
    lam = np.zeros(len(segs)); mu = mu0
    for rd in range(rounds):
        def f(x):
            E, g = energy_grad(x, d, e, r, h, Z, segs)
            for i, (a, b) in enumerate(segs):
                c = np.sum(x[a:b]**2*h) - 1.0
                E += -lam[i]*c + 0.5*mu*c*c
                g[a:b] += (-lam[i] + mu*c)*2*x[a:b]*h
            return E, g
        res = minimize(f, u, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=4000, ftol=1e-14, gtol=1e-10))
        u = res.x
        cs = np.array([np.sum(u[a:b]**2*h) - 1.0 for (a, b) in segs])
        lam = lam - mu*cs
        mu *= 4.0
        if verbose:
            print(f"      round {rd}: |constraint| = {np.max(np.abs(cs)):.2e}")
        if np.max(np.abs(cs)) < 1e-10:
            break
    E, _ = energy_grad(u, d, e, r, h, Z, segs)
    a, b = segs[-1]
    q = np.cumsum(u[a:b]**2*h)
    r90 = float(r[a:b][np.searchsorted(q, 0.90)]) if q[-1] > 0.9 else float(r[b-1])
    return dict(E=E, u=u, r90=r90, cons=float(np.max(np.abs(cs))), r=r, h=h)


def best_over_splits(Z, N, L, n, radii=None):
    """Minimise over the interface positions -- the boundary is determined by the energy.

    BUG FOUND AND FIXED 2026-10-20: the candidates were FRACTIONS OF THE BOX, so as L grew the
    physically correct interface radius fell off the bottom of the list.  Symptom: E(He) drifting
    -3.074 -> -2.872 -> -2.495 at L = 15, 22, 32, when a tightly bound atom's energy must not
    depend on the box at all.  Candidates are now ABSOLUTE RADII, so the same physical interfaces
    are available at every box size.
    """
    r, h = grid(L, n)
    R = radii or (0.15, 0.25, 0.4, 0.6, 0.9, 1.3, 1.9, 2.8, 4.0, 6.0, 9.0)
    idx = sorted({int(np.searchsorted(r, x)) for x in R if x < 0.8*L})
    best = None
    if N == 1:
        cands = [[(0, n)]]
    elif N == 2:
        cands = [[(0, k), (k, n)] for k in idx]
    else:
        cands = [[(0, k1), (k1, k2), (k2, n)]
                 for i, k1 in enumerate(idx) for k2 in idx[i+1:]]
    for segs in cands:
        if min(b - a for a, b in segs) < 8:
            continue
        S = solve(Z, N, L=L, n=n, segs=segs)
        if S['cons'] > 1e-6:
            continue
        if best is None or S['E'] < best['E']:
            best = S; best['segs'] = segs
    return best


if __name__ == '__main__':
    print("SELF-TEST 1: one domain, hydrogen, must give E = -0.5")
    for n in (400, 700, 1200):
        S = best_over_splits(1, 1, 30.0, n)
        print(f"   n={n:5d}  E(H) = {S['E']:12.7f}   err = {S['E']+0.5:+9.2e}   |constraint| = {S['cons']:.1e}",
              flush=True)

    print("\nSELF-TEST 2: two domains -- is the collapse gone?  (it tracked -1.5Z/R1 before)")
    print("      L      n     E(H-)        r90(outer)   interface split")
    for (L, n) in ((15.0, 700), (22.0, 1000), (32.0, 1500)):
        S = best_over_splits(1, 2, L, n)
        print(f"  {L:6.1f} {n:6d} {S['E']:12.7f} {S['r90']:11.2f}    {S['segs'][0][1]}", flush=True)
