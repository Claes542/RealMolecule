"""
ALL-ELECTRON LITHIUM IN A CELL, HOMOGENISED CORE + VALENCE, H^1 CONTINUOUS.

WHY.  The elasticity paper's cell is the REDUCED model: one valence electron on a +1 kernel with an
empty core of radius r_c.  r_c comes from the atom via the conversion 0.855.  The atom side is
confirmed to 0.05% by two independent routes (free interface R=2.00, hard-depletion radius fitted to
the IE R_HD=1.999), yet the cell delivers r_s +18.8% and B -27.7%.  So the suspect is the TRANSFER,
not the radius -- and this file removes the transfer: solve the cell all-electron, with the interface
placed by minimisation, nothing supplied but Z.

tools/allelectron_cell.py already tried it and got R=2.061 at a=5.6 against the atom's 2.00 -- its
"one cross-check that did work" -- but under A[keep][:,keep], i.e. DIRICHLET at both the interface
and the cell face.  Rerun with true Neumann it COLLAPSED.  The cure, identified there and built
elsewhere since, is H^1 continuity: one global field, continuous across the interface, with one
charge constraint per domain.  The null mode that forbids hard depletion in a cell needs EQUAL
domains and does not arise here, the partition being charge 2 inside R against 1 outside.

THE CONSTRUCTION, which is the paper's own atom construction (its \S sec:core) carried into a cell:
the two core electrons are a single homogenised shell, each carrying the same density, which is the
(N_s-1)/N_s self-repulsion reduction stated as pair counting -- ONE core-core pair, TWO core-valence
pairs.  Radial, u = r*psi with u(0)=0, so a density concentrating at the origin is penalised
correctly and the collapse of the density formulation cannot occur:

    E = 2 T_core + T_val  -  Z (2 INT_core + INT_val) u^2/r  +  U_cc + 2 U_cv

with INT u^2 dr = 1 over EACH region (so core charge 2, valence charge 1), free outer boundary at
r_s, and the interface index chosen by minimisation.  Weights per region are the whole content: the
straddling face belongs to neither, so its weight is a parameter (WFACE) and the answer must not
depend on it -- checked, not assumed.

SELF-TEST BEFORE ANY SCAN: at large r_s this must reproduce the free atom's R = 2.00.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize
from ionization_NZ_H1 import grid, hartree

np.seterr(all='ignore')


def loc_form_w(r, h, k, wface):
    """(1/2) INT (u'-u/r)^2 dr as h * u^T M u, each face weighted by its region.

    Faces 0..n-2 join cells j,j+1; the origin face sits left of cell 0 and is in the core.
    Core faces weight 2 (two electrons share the density), valence faces 1, straddling face WFACE.
    """
    n = len(r)
    d = np.zeros(n); e = np.zeros(n - 1)
    for j in range(n - 1):
        if j <= k - 2:      w = 2.0
        elif j >= k:        w = 1.0
        else:               w = wface
        rm = 0.5*(r[j] + r[j+1])
        A = 1.0/h - 1.0/(2*rm)
        B = -1.0/h - 1.0/(2*rm)
        d[j] += w*0.5*B*B; d[j+1] += w*0.5*A*A; e[j] += w*0.5*A*B
    rm = 0.5*r[0]                                  # origin face: core, weight 2
    A = 1.0/h - 1.0/(2*rm)
    d[0] += 2.0*0.5*A*A
    return d, e


def apply_M(d, e, u):
    out = d*u
    out[:-1] += e*u[1:]
    out[1:] += e*u[:-1]
    return out


def energy_grad(u, d, e, r, h, Z, k):
    core = np.zeros(len(u), bool); core[:k] = True
    val = ~core
    wc = np.where(core, 2.0, 1.0)
    u2 = u*u
    Mu = apply_M(d, e, u)
    T = h*float(u @ Mu)
    Ven = -Z*h*float(np.sum(wc*u2/r))
    dc = np.where(core, u2, 0.0)                   # ONE core electron, charge 1
    dv = np.where(val,  u2, 0.0)                   # the valence electron
    Pc = hartree(dc, r, h); Pv = hartree(dv, r, h)
    U_cc = h*float(np.sum(Pc*dc))                  # one core-core pair
    U_cv = 2.0*h*float(np.sum(Pc*dv))              # two core-valence pairs
    E = T + Ven + U_cc + U_cv
    g = 2*h*Mu - 2*Z*h*wc*u/r + 4*h*u*np.where(core, Pc + Pv, Pc)
    return E, g


def solve(k, L, n, Z=3.0, wface=1.5, mu0=300.0, rounds=9, tol=1e-11, u0=None):
    r, h = grid(L, n)
    d, e = loc_form_w(r, h, k, wface)
    segs = [(0, k), (k, n)]
    u = u0.copy() if u0 is not None else r*np.exp(-Z*r/2) + 0.3*r*np.exp(-r/2)
    for a, b in segs:
        s = np.sum(u[a:b]**2*h)
        if s > 0: u[a:b] /= np.sqrt(s)
    lam = np.zeros(2); mu = mu0
    for _ in range(rounds):
        def f(x):
            E, g = energy_grad(x, d, e, r, h, Z, k)
            for i, (a, b) in enumerate(segs):
                c = np.sum(x[a:b]**2*h) - 1.0
                E += -lam[i]*c + 0.5*mu*c*c
                g[a:b] += (-lam[i] + mu*c)*2*x[a:b]*h
            return E, g
        res = minimize(f, u, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=6000, ftol=1e-15, gtol=1e-12))
        u = res.x
        cs = np.array([np.sum(u[a:b]**2*h) - 1.0 for a, b in segs])
        lam = lam - mu*cs; mu *= 3.0
        if np.max(np.abs(cs)) < tol: break
    E, _ = energy_grad(u, d, e, r, h, Z, k)
    return dict(E=E, u=u, R=float(r[k-1]), cons=float(np.max(np.abs(cs))), r=r, h=h)


def best_R(L, n, Rlo=0.8, Rhi=3.6, Z=3.0, wface=1.5):
    """The interface is a FREE boundary: locate it by minimising over the interface index."""
    r, h = grid(L, n)
    ks = [int(np.searchsorted(r, x)) for x in np.arange(Rlo, Rhi + 1e-9, 0.1)]
    best = None
    for k in sorted(set(ks)):
        if k < 8 or n - k < 8: continue
        S = solve(k, L, n, Z=Z, wface=wface)
        if S['cons'] > 1e-7: continue
        if best is None or S['E'] < best['E']: best = S
    return best
