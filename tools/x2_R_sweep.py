"""
X2 WITH A PSEUDO-KERNEL OF RADIUS R: sweep R, get the dimer binding, compare with measurement.

The model.  One valence electron per site on a screened kernel of charge +1, excluded from the
core ball r < R about ITS OWN kernel (and from the far core, which only matters if D < 2R -- those
geometries are skipped, the cores would interpenetrate).  R is the only thing that changes along
the sweep, so R is what distinguishes one X from another.

R is identified physically by the atom it implies: with one electron outside R on a +1 kernel,
IE = -E(X).  So each rung of the sweep carries its own ionization energy and can be matched to an
alkali without any further input.

Both E(X) and E(X2) are computed in THIS code on THE SAME mesh, so the discretisation largely
cancels in E_bind = 2E(X) - E(X2) -- the house rule that differences within one composition cancel
their errors.  D is scanned only over COMMENSURATE values D = (2i+1)h, which put the kernels exactly
on cell centres of the half grid; otherwise the egg-crate noise is the size of the binding itself.
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from axisym_realqm import Grid, stiffness, kernel_potential
from axisym_hom2 import _poisson_setup

HA_EV = 27.211386


def masked_stiffness(g, keep):
    """0.5*int |grad u|^2 assembled ONLY over faces joining two KEPT cells.

    THIS IS THE WHOLE POINT.  Taking a submatrix A[keep][:,keep] of the FULL stiffness keeps the
    diagonal contributions of the faces leading into the removed core, which is a DIRICHLET
    condition (u=0 inside) -- it pushes density off the core edge and UNDER-binds.  Omitting those
    faces at assembly is no-flux, i.e. the homogeneous NEUMANN condition on the core sphere, which
    is the model's own free-boundary condition.  The two differ by 36% on the one-electron IE, and
    only the Neumann version reproduces the 1D radial solve (R=1.6/1.8/2.0: 6.177/5.727/5.385 eV
    here against 6.027/5.642/5.299 radial, ~2%).
    """
    h, nz, nr = g.h, g.nz, g.nr
    rows, cols, vals = [], [], []
    def add(a, b, c):
        if not (keep[a] and keep[b]): return
        rows.extend([a, b, a, b]); cols.extend([a, b, b, a]); vals.extend([c, c, -c, -c])
    for j in range(nr):
        c = np.pi*g.r[j]
        for i in range(nz-1): add(g.idx(i, j), g.idx(i+1, j), c)
    for j in range(nr-1):
        c = np.pi*(j+1)*h
        for i in range(nz): add(g.idx(i, j), g.idx(i, j+1), c)
    return sp.coo_matrix((vals, (rows, cols)), shape=(g.N, g.N)).tocsr()


def atomX(R, Lz=16.0, Lr=16.0, h=0.14, soft=0.0):
    """One electron outside r=R on a +1 kernel.  Returns E(X); IE = -E(X)."""
    g = Grid(Lz, Lr, h, zmin=-Lz)
    K = kernel_potential(g, [(0.0, 1.0)], soft)
    m = np.sqrt(g.Z**2 + g.R**2).ravel() > R
    S = masked_stiffness(g, m) if R > 0 else stiffness(g)
    keep = np.where(m)[0]
    w = 1.0/np.sqrt(g.V); W = sp.diags(w)
    A = (W @ (S + sp.diags(-K*g.V)).tocsr() @ W).tocsr()[keep][:, keep]
    val, _ = eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000)
    return val[0]


def x2R(R, D, Lz=16.0, Lr=16.0, h=0.14, soft=0.0, iters=120, tol=1e-9):
    """Two +1 kernels at +-D/2, one electron each, split at the midplane, cores of radius R."""
    gh = Grid(Lz, Lr, h, zmin=0.0)
    gf = Grid(Lz, Lr, h, zmin=-Lz)
    nzh = gh.nz
    assert gf.nz == 2*nzh
    K = kernel_potential(gh, [(-D/2, 1.0), (D/2, 1.0)], soft)
    rr_near = np.sqrt((gh.Z - D/2)**2 + gh.R**2).ravel()
    rr_far  = np.sqrt((gh.Z + D/2)**2 + gh.R**2).ravel()
    m = (rr_near > R) & (rr_far > R)
    S = masked_stiffness(gh, m) if R > 0 else stiffness(gh)
    keep = np.where(m)[0]          # excluded from BOTH cores
    solveP, bnd, rrf = _poisson_setup(gf)
    w = 1.0/np.sqrt(gh.V); W = sp.diags(w)
    mir = np.arange(nzh)[::-1]
    P1 = np.zeros(gh.N); u = None; Eold = None
    for it in range(iters):
        A = (W @ (S + sp.diags((-K + P1)*gh.V)).tocsr() @ W).tocsr()[keep][:, keep]
        val, vec = eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000,
                         v0=None if u is None else (u/w)[keep])
        y = np.zeros(gh.N); y[keep] = vec[:, 0]
        u = w*y; u = u/np.sqrt(np.sum(u*u*gh.V))
        densf = np.zeros((gf.nz, gf.nr)); densf[nzh:, :] = (u*u).reshape(nzh, gh.nr)
        b = 4*np.pi*densf.ravel()*gf.V
        b[bnd] = 1.0/rrf[bnd]
        P1 = solveP(b).reshape(gf.nz, gf.nr)[mir, :].ravel()
        T  = float(u @ (S @ u))
        VK = -float(np.sum(K*u*u*gh.V))
        U12= float(np.sum(P1*u*u*gh.V))
        E  = 2*(T + VK) + U12 + 1.0/D
        if Eold is not None and abs(E - Eold) < tol: break
        Eold = E
    return E


if __name__ == '__main__':
    import sys, time
    h, L = 0.175, 14.0
    # measured dimers: atomic IE (eV), D_e (eV), R_e (bohr).
    # REFERENCE VALUES FROM MODEL KNOWLEDGE -- verify before any print use.
    MEAS = [("H2", 13.598, 4.748, 1.401), ("Li2",  5.392, 1.050, 5.051),
            ("Na2", 5.139, 0.747, 5.818), ("K2",   4.341, 0.551, 7.416),
            ("Rb2", 4.177, 0.495, 7.955), ("Cs2",  3.894, 0.452, 8.783)]
    print("X2 WITH A PSEUDO-KERNEL OF RADIUS R.  h=%.3f, L=%.0f, commensurate D=(2i+1)h, soft=0" % (h, L))
    print("one valence electron per +1 kernel, excluded from r<R of BOTH cores; D>2R+2h enforced")
    print("E(X) and E(X2) from the same code on the same mesh, so the difference cancels discretisation")
    print("D located by coarse scan then refinement; granularity 2h = %.3f au\n" % (2*h))
    print("measured, for matching (reference knowledge, UNVERIFIED):")
    for n, ie, de, re in MEAS:
        print("   %-4s IE %6.3f eV   D_e %5.3f eV   R_e %5.3f au" % (n, ie, de, re))
    print()
    print("%5s %9s %8s | %7s %7s | %9s %8s %8s | %s" %
          ("R", "E(X)", "IE eV", "D_min", "R_e exp", "E(X2)", "Eb eV", "De exp", "match by IE"))
    for R in [0.0, 0.8, 1.6, 2.0, 2.4]:
        t0 = time.time()
        eX = atomX(R, L, L, h)
        ie = -eX*HA_EV
        name, ie_m, de_m, re_m = min(MEAS, key=lambda m: abs(m[1] - ie))
        allD = [(2*i+1)*h for i in range(1, 40)]
        allD = [D for D in allD if D > 2*R + 2*h and D <= 12.0]
        # coarse scan, stride 3, then refine around the best
        idx = list(range(0, len(allD), 3))
        vals = {}
        for k in idx:
            vals[k] = x2R(R, allD[k], L, L, h)
        kb = min(vals, key=vals.get)
        for k in range(max(0, kb-3), min(len(allD), kb+4)):
            if k not in vals: vals[k] = x2R(R, allD[k], L, L, h)
        kb = min(vals, key=vals.get)
        Dm, E2 = allD[kb], vals[kb]
        eb = 2*eX - E2
        edge = "  <-- at scan edge" if kb in (0, len(allD)-1) else ""
        print("%5.2f %9.5f %8.3f | %7.3f %7.3f | %9.5f %8.3f %8.3f | %s%s  [%.0fs, %d pts]" %
              (R, eX, ie, Dm, re_m, E2, eb*HA_EV, de_m, name, edge, time.time()-t0, len(vals)))
        sys.stdout.flush()
