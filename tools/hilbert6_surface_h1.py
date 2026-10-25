"""
IS gamma AN ARTEFACT OF THE MASKED INTERFACE?  The surface energy recomputed under H^1 continuity.

WHY THIS NUMBER.  tools/hilbert6_surface.py solves each domain of the finite chain as an
INDEPENDENT problem with "free/Neumann at both ends of the domain" (its line 84).  That is exact
wherever the interface is a SYMMETRY plane -- reflect the configuration and it maps to itself, so
the solution is even and its normal derivative vanishes anyway.  Every interface in the PERIODIC
cell calculations is of that kind, which is why they are untouched by this.

The finite chain is not.  The two END domains have vacuum on one side and lattice on the other, so
the field is not even about their inner face and zero normal derivative is simply wrong there.  And
gamma is read off as the INTERCEPT of E(N) = N e_bulk + 2 gamma, where a per-interface error that
grows with N is absorbed into the slope and the END-specific part -- exactly the asymmetric part --
lands in the intercept.

So gamma is the one published number of this family that the H^1 correction can move.  2D measured
the correction to a modest interface asymmetry at ~0.019 Ha against gamma = 0.0416 Ha, the same
order, which is reason to check rather than assume.

THE CONTROLLED COMPARISON.  One global field u on the whole line, N domains cut at the kernel
midpoints, each constrained to unit charge, augmented Lagrangian with one multiplier per domain.
`masked=True` assembles the kinetic form ONLY over adjacent pairs inside one domain -- dropping the
straddling pairs, which is the block-diagonal form -- and `masked=False` assembles all of them.
Nothing else differs between the two runs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize

np.seterr(all='ignore')


def build(N, a, ncell, eps, vac):
    h = a/ncell
    zk = np.arange(N)*a
    nvac = int(round(vac/h))
    ntot = nvac + (N-1)*ncell + nvac
    z = -vac + (np.arange(ntot) + 0.5)*h
    # Face index from the GEOMETRY, not by argmin over cell centres.  Cell k spans
    # [-vac + k h, -vac + (k+1) h], so the face at zm is index (zm + vac)/h.  The argmin+1
    # form used in hilbert6_surface.py picks the nearest cell CENTRE and steps one cell, which
    # is a tie-break when zm lands exactly on a boundary: measured off by a full cell at
    # ncell=40 and by h/2 for every ODD ncell (17, 25, 41 -- the published ladder).
    faces = [0]
    for i in range(N-1):
        faces.append(int(round((zk[i] + 0.5*a + vac)/h)))
    faces.append(ntot)
    labels = np.zeros(ntot, int)
    for i in range(N):
        labels[faces[i]:faces[i+1]] = i
    K = np.zeros(ntot)
    for zb in zk:
        K += 1.0/np.sqrt((z - zb)**2 + eps*eps)
    G = 1.0/np.sqrt((z[:, None] - z[None, :])**2 + eps*eps)*h
    U_KK = 0.0
    for i in range(N):
        for j in range(i+1, N):
            U_KK += 1.0/np.sqrt((zk[i]-zk[j])**2 + eps*eps)
    return dict(h=h, z=z, labels=labels, K=K, G=G, U_KK=U_KK, ntot=ntot, N=N)


def kinetic_pairs(ntot, h, labels, masked):
    """adjacent pairs (k,k+1) with coefficient 0.5/h; masked drops the straddling ones."""
    ks = np.arange(ntot-1)
    if masked:
        ks = ks[labels[ks] == labels[ks+1]]
    return ks, 0.5/h


def energy_grad(u, B, ks, c, masks):
    h, K, G = B['h'], B['K'], B['G']
    u2 = u*u
    d = u[ks+1] - u[ks]
    T = c*float(np.sum(d*d))
    Su = np.zeros_like(u)                       # dT/du = 2 S u
    np.add.at(Su, ks,   -2*c*d)
    np.add.at(Su, ks+1,  2*c*d)
    U_eK = -h*float(np.sum(K*u2))
    Ps = [G @ np.where(m, u2, 0.0) for m in masks]
    U_ee = 0.0
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            U_ee += h*float(np.sum(np.where(masks[i], u2, 0.0)*Ps[j]))
    Pother = np.zeros_like(u)
    for i, m in enumerate(masks):
        s = np.zeros_like(u)
        for j in range(len(masks)):
            if j != i:
                s = s + Ps[j]
        Pother = np.where(m, s, Pother)
    E = T + U_eK + U_ee + B['U_KK']
    grad = Su + 2*h*u*(-K + Pother)
    return E, grad


def solve_chain_h1(N, a, ncell=41, eps=0.35, vac=6.0, masked=False, mu0=400.0, rounds=10, tol=1e-11):
    B = build(N, a, ncell, eps, vac)
    h, labels = B['h'], B['labels']
    masks = [labels == i for i in range(N)]
    ks, c = kinetic_pairs(B['ntot'], h, labels, masked)
    u = np.exp(-np.min(np.abs(B['z'][:, None] - (np.arange(N)*a)[None, :]), axis=1))
    for m in masks:
        s = h*float(np.sum(u[m]**2))
        if s > 0:
            u[m] /= np.sqrt(s)
    lam = np.zeros(N); mu = mu0
    for rd in range(rounds):
        def f(x):
            E, g = energy_grad(x, B, ks, c, masks)
            for i, m in enumerate(masks):
                cc = h*float(np.sum(x[m]**2)) - 1.0
                E += -lam[i]*cc + 0.5*mu*cc*cc
                g[m] += (-lam[i] + mu*cc)*2*x[m]*h
            return E, g
        res = minimize(f, u, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=8000, ftol=1e-15, gtol=1e-12))
        u = res.x
        cs = np.array([h*float(np.sum(u[m]**2)) - 1.0 for m in masks])
        lam = lam - mu*cs; mu *= 3.0
        if np.max(np.abs(cs)) < tol:
            break
    E, _ = energy_grad(u, B, ks, c, masks)
    return dict(E=E, u=u, cons=float(np.max(np.abs(cs))))
