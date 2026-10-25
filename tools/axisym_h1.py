"""
H^1 CONTINUITY IN 2D: one global field partitioned by labels, N unit-charge constraints.

WHY.  tools/axisym_realqm.py (and molecule.js, 18 masked-stencil lines in 3 shaders) assembles the
kinetic form ONLY over faces joining two cells of the SAME domain.  Omitting the straddling faces
makes each domain an independent homogeneous-Neumann problem: the operator is block diagonal and the
field may JUMP across an interface at zero cost.  RealQM does not permit that --

    "the psi_i being continuous across joint boundaries Gamma_i ^ Gamma_j as imposed by
     Psi in H^1(R^3), and the boundary location determined by the energy minimisation"
                                                        -- RealQM_method.tex:101

and tools/ionization_NZ_H1.py showed in 1D that the freedom to jump IS the collapse mode: the
uniform blob needs a jump, and in H^1 a jump costs infinite energy, so continuity does not penalise
the bad mode, it makes it unrepresentable.

WHEN IT MATTERS, AND WHEN IT DOES NOT.  If the interface is a SYMMETRY plane, the continuous
solution is mirror symmetric, so its normal derivative vanishes there and Neumann is exactly right.
He and H2 split on the midplane are therefore NOT tests of this -- the two constructions must agree,
which makes symmetric H2 a cross-check instead.  The constructions separate only at an ASYMMETRIC
interface: unequal kernels, or a core/cell-face pair.

THE PROBLEM.  Minimise over one global u, and over the interface position,

    E[u] = int 0.5|grad u|^2 dV - int K u^2 dV + sum_{i<j} int int rho_i rho_j/|x-y|

subject to int_{Omega_i} u^2 dV = 1, one constraint per domain -- so there is one Lagrange
multiplier (chemical potential) per domain and this is NOT a single eigenproblem.  Augmented
Lagrangian on the constraints, L-BFGS inside, gradient in closed form:

    dE/du = 2 S u + 2 V u (-K + P_other),

P_other being the potential of the OTHER domains only, piecewise by label: the i!=j rule, with no
self-interaction anywhere.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
from scipy.optimize import minimize
from axisym_realqm import Grid, stiffness, masked_stiffness, laplacian, kernel_potential

np.seterr(all='ignore')


def make_poisson(g):
    """Factorise once: -Lap P = 4 pi rho with P = q/|x| on the outer faces."""
    L = laplacian(g).tolil()
    rr = np.sqrt(g.Z**2 + g.R**2).ravel()
    bnd = np.zeros(g.N, bool)
    bnd[g.idx(0, np.arange(g.nr))] = True
    bnd[g.idx(g.nz-1, np.arange(g.nr))] = True
    bnd[g.idx(np.arange(g.nz), g.nr-1)] = True
    for k in np.where(bnd)[0]:
        L.rows[k] = [k]; L.data[k] = [1.0]
    solve = factorized(L.tocsc())
    V, idxb = g.V, np.where(bnd)[0]
    def P(dens, qtot):
        b = 4*np.pi*dens*V
        b[idxb] = qtot/rr[idxb]
        return solve(b)
    return P


def energy_grad(u, g, S, K, labels, ndom, pois):
    """E and dE/du for the global field u, domains given by the label array."""
    V = g.V
    u2 = u*u
    Su = S @ u
    T = float(u @ Su)
    VK = -float(np.sum(K*u2*V))
    masks = [labels == i for i in range(ndom)]
    Ps = []
    for m in masks:
        d = np.where(m, u2, 0.0)
        Ps.append(pois(d, float(np.sum(d*V))))
    Vee = 0.0
    for i in range(ndom):
        for j in range(i+1, ndom):
            Vee += float(np.sum(Ps[i]*np.where(masks[j], u2, 0.0)*V))
    Pother = np.zeros_like(u)
    for i in range(ndom):
        s = np.zeros_like(u)
        for j in range(ndom):
            if j != i:
                s = s + Ps[j]
        Pother = np.where(masks[i], s, Pother)
    E = T + VK + Vee
    grad = 2*Su + 2*V*u*(-K + Pother)
    return E, grad


def solve_h1(g, K, labels, ndom, u0=None, mu0=200.0, rounds=9, tol=1e-11, verbose=False,
             masked=False):
    """Augmented Lagrangian on the ndom unit-charge constraints."""
    # masked=False -> H^1: every face assembled.  masked=True -> the block-diagonal form.
    S = blockdiag_stiffness(g, labels) if masked else stiffness(g)
    pois = make_poisson(g)
    V = g.V
    masks = [labels == i for i in range(ndom)]
    u = u0 if u0 is not None else np.exp(-np.sqrt(g.Z**2 + g.R**2).ravel())
    for m in masks:                       # seed each domain at unit charge
        s = float(np.sum(u[m]**2*V[m]))
        if s > 0:
            u[m] /= np.sqrt(s)
    lam = np.zeros(ndom); mu = mu0
    for rd in range(rounds):
        def f(x):
            E, gr = energy_grad(x, g, S, K, labels, ndom, pois)
            for i, m in enumerate(masks):
                c = float(np.sum(x[m]**2*V[m])) - 1.0
                E += -lam[i]*c + 0.5*mu*c*c
                gr[m] += (-lam[i] + mu*c)*2*x[m]*V[m]
            return E, gr
        res = minimize(f, u, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=6000, ftol=1e-15, gtol=1e-12))
        u = res.x
        cs = np.array([float(np.sum(u[m]**2*V[m])) - 1.0 for m in masks])
        lam = lam - mu*cs; mu *= 3.0
        if verbose:
            print(f"      round {rd}: |c| = {np.max(np.abs(cs)):.2e}  E = {res.fun:.8f}")
        if np.max(np.abs(cs)) < tol:
            break
    E, _ = energy_grad(u, g, S, K, labels, ndom, pois)
    return dict(E=E, u=u, cons=float(np.max(np.abs(cs))))


def blockdiag_stiffness(g, labels):
    """The MASKED form the existing solvers use: assemble a face ONLY when both of its cells carry
    the SAME label.  Dropping the straddling faces is what makes each domain an independent
    Neumann problem and lets the field jump across an interface for free.  Identical to
    stiffness(g) except for that one test, so swapping the two isolates H^1 continuity alone."""
    h, nz, nr = g.h, g.nz, g.nr
    rows, cols, vals = [], [], []
    def add(a, b, c):
        if labels[a] != labels[b]: return
        rows.extend([a, b, a, b]); cols.extend([a, b, b, a]); vals.extend([c, c, -c, -c])
    for j in range(nr):
        c = np.pi*g.r[j]
        for i in range(nz-1): add(g.idx(i, j), g.idx(i+1, j), c)
    for j in range(nr-1):
        c = np.pi*(j+1)*h
        for i in range(nz): add(g.idx(i, j), g.idx(i, j+1), c)
    return sp.coo_matrix((vals, (rows, cols)), shape=(g.N, g.N)).tocsr()


def plane_labels(g, zi):
    """Two domains split by the plane z = zi."""
    return (g.Z.ravel() >= zi).astype(int)
