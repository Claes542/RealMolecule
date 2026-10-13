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


def atomX(R, Lz=16.0, Lr=16.0, h=0.14, soft=0.0):
    """One electron outside r=R on a +1 kernel.  Returns E(X); IE = -E(X)."""
    g = Grid(Lz, Lr, h, zmin=-Lz)
    S = stiffness(g); K = kernel_potential(g, [(0.0, 1.0)], soft)
    keep = np.where(np.sqrt(g.Z**2 + g.R**2).ravel() > R)[0]
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
    S = stiffness(gh)
    K = kernel_potential(gh, [(-D/2, 1.0), (D/2, 1.0)], soft)
    rr_near = np.sqrt((gh.Z - D/2)**2 + gh.R**2).ravel()
    rr_far  = np.sqrt((gh.Z + D/2)**2 + gh.R**2).ravel()
    keep = np.where((rr_near > R) & (rr_far > R))[0]          # excluded from BOTH cores
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
    h, Lz = 0.14, 16.0
    # measured dimers, D_e in eV and R_e in bohr, with the atomic IE in eV, for matching.
    # REFERENCE VALUES FROM MODEL KNOWLEDGE -- verify before any print use.
    MEAS = [("H2", 13.598, 4.748, 1.401), ("Li2",  5.392, 1.050, 5.051),
            ("Na2", 5.139, 0.747, 5.818), ("K2",   4.341, 0.551, 7.416),
            ("Rb2", 4.177, 0.495, 7.955), ("Cs2",  3.894, 0.452, 8.783)]
    print("X2 WITH A PSEUDO-KERNEL OF RADIUS R.  h=%.2f, L=%.0f, commensurate D=(2i+1)h, soft=0" % (h, Lz))
    print("one valence electron per +1 kernel, excluded from r<R of both cores; D>2R enforced")
    print("measured, for matching (reference knowledge, unverified):")
    for n, ie, de, re in MEAS:
        print("   %-4s IE %6.3f eV   D_e %5.3f eV   R_e %5.3f au" % (n, ie, de, re))
    print()
    print("%5s %9s %9s | %7s %9s %9s %8s | %s" %
          ("R", "E(X)", "IE eV", "D_min", "E(X2)", "Ebind eV", "Ebind Ha", "nearest by IE"))
    for R in [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]:
        t0 = time.time()
        eX = atomX(R, Lz, Lz, h)
        ie = -eX*HA_EV
        # commensurate D, wide enough to hold any alkali minimum, and D > 2R
        Ds = [(2*i+1)*h for i in range(2, 60)]
        Ds = [D for D in Ds if D > 2*R + 2*h and D <= 13.0]
        best = (1e9, None)
        curve = []
        for D in Ds:
            E = x2R(R, D, Lz, Lz, h)
            curve.append((D, E))
            if E < best[0]: best = (E, D)
        E2, Dm = best
        eb = 2*eX - E2
        name = min(MEAS, key=lambda m: abs(m[1] - ie))[0]
        print("%5.2f %9.5f %9.3f | %7.3f %9.5f %9.3f %8.4f | %s  [%.0fs]" %
              (R, eX, ie, Dm, E2, eb*HA_EV, eb, name, time.time()-t0))
        sys.stdout.flush()
        np.save('/private/tmp/claude-501/-Users-cgjoh-Development-H2O/b8b3b078-08a8-4d27-be88-172e049921b0/scratchpad/curve_R%.2f.npy' % R, np.array(curve))
