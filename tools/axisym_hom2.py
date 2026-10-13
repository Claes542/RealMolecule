"""
TWO CONSTRUCTIONS FOR A TWO-ELECTRON SYSTEM, ON ONE MESH, TO BRACKET THE EXACT ENERGY.

hom2   -- both electrons carry the SAME density over the whole domain ("on top of each other"),
          each feeling the OTHER's unit charge.  The pair repulsion is counted once:
              E = 2*(T + V_K) + J,      J = int P[u^2] u^2 dV
          Putting a single charge-2 blob down instead and taking its Coulomb self-energy gives
          (1/2)*int (2P)(2u^2) = 2J, so counting the pair once IS the factor-2 reduction of the
          repulsion energy.  Equivalent to eigen-problem (-1/2 Lap - K + P[u^2]) u = eps u with
          E = 2 eps - J.
          CHECK: this functional is Hartree, which for a closed two-electron shell equals
          Hartree-Fock, so the h->0 limits must be He -2.861680 and H2(R=1.4) -1.133630.

split  -- the RealQM construction: non-overlapping domains, here the two halves of the plane
          perpendicular to the axis, free (Neumann) interface, partner by reflection.

Kernel positions are placed on cell FACES at every rung (h chosen so that 0.7/h and Lz/h are
integers), so the ladder is not contaminated by the kernel wandering inside a cell.
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh, factorized
from axisym_realqm import Grid, stiffness, laplacian, kernel_potential


def _poisson_setup(g):
    L = laplacian(g).tolil()
    rr = np.sqrt(g.Z**2 + g.R**2).ravel()
    bnd = np.zeros(g.N, bool)
    bnd[g.idx(0, np.arange(g.nr))] = True
    bnd[g.idx(g.nz-1, np.arange(g.nr))] = True
    bnd[g.idx(np.arange(g.nz), g.nr-1)] = True
    for k in np.where(bnd)[0]:
        L.rows[k] = [k]; L.data[k] = [1.0]
    return factorized(L.tocsc()), bnd, rr


def hom2(kernels, nn=0.0, Lz=12.0, Lr=12.0, h=0.1, soft=0.0, iters=200, tol=1e-9, verbose=False):
    """Both electrons in one density over the full domain; pair repulsion counted once."""
    g = Grid(Lz, Lr, h, zmin=-Lz)
    S = stiffness(g)
    K = kernel_potential(g, kernels, soft)
    solveP, bnd, rr = _poisson_setup(g)
    w = 1.0/np.sqrt(g.V); W = sp.diags(w)
    P = np.zeros(g.N); u = None; Eold = None; mix = 0.5
    for it in range(iters):
        A = (W @ (S + sp.diags((-K + P)*g.V)).tocsr() @ W).tocsr()
        val, vec = eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000,
                         v0=None if u is None else (u/w))
        u = w*vec[:, 0]
        u = u/np.sqrt(np.sum(u*u*g.V))
        b = 4*np.pi*(u*u)*g.V
        b[bnd] = 1.0/rr[bnd]                       # the partner carries unit charge
        Pnew = solveP(b)
        P = P + mix*(Pnew - P) if it else Pnew
        T  = float(u @ (S @ u))
        VK = -float(np.sum(K*u*u*g.V))
        J  = float(np.sum(P*u*u*g.V))              # pair repulsion, counted ONCE
        E  = 2*(T + VK) + J + nn
        if verbose: print("    it%03d E=%.8f T=%.5f VK=%.5f J=%.5f" % (it, E, T, VK, J))
        if Eold is not None and abs(E-Eold) < tol: break
        Eold = E
    return E, dict(T=T, VK=VK, J=J, eps=val[0], iters=it+1)


def split2(kernels, nn=0.0, Lz=12.0, Lr=12.0, h=0.1, soft=0.0, iters=200, tol=1e-9, verbose=False):
    """Non-overlapping halves z>0 / z<0, free interface, partner by reflection."""
    gh = Grid(Lz, Lr, h, zmin=0.0)
    gf = Grid(Lz, Lr, h, zmin=-Lz)
    nzh = gh.nz
    assert gf.nz == 2*nzh, (gf.nz, nzh)
    S = stiffness(gh)
    K = kernel_potential(gh, kernels, soft)
    solveP, bnd, rrf = _poisson_setup(gf)
    w = 1.0/np.sqrt(gh.V); W = sp.diags(w)
    mir = np.arange(nzh)[::-1]
    P1 = np.zeros(gh.N); u = None; Eold = None
    for it in range(iters):
        A = (W @ (S + sp.diags((-K + P1)*gh.V)).tocsr() @ W).tocsr()
        val, vec = eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000,
                         v0=None if u is None else (u/w))
        u = w*vec[:, 0]
        u = u/np.sqrt(np.sum(u*u*gh.V))
        densf = np.zeros((gf.nz, gf.nr)); densf[nzh:, :] = (u*u).reshape(nzh, gh.nr)
        b = 4*np.pi*densf.ravel()*gf.V
        b[bnd] = 1.0/rrf[bnd]
        P1 = solveP(b).reshape(gf.nz, gf.nr)[mir, :].ravel()
        T  = float(u @ (S @ u))
        VK = -float(np.sum(K*u*u*gh.V))
        U12= float(np.sum(P1*u*u*gh.V))
        E  = 2*(T + VK) + U12 + nn
        if verbose: print("    it%03d E=%.8f T=%.5f VK=%.5f U12=%.5f" % (it, E, T, VK, U12))
        if Eold is not None and abs(E-Eold) < tol: break
        Eold = E
    return E, dict(T=T, VK=VK, U12=U12, eps=val[0], iters=it+1)


def one(kernels, Lz, Lr, h, soft):
    """one electron, no partner: the instrument control."""
    g = Grid(Lz, Lr, h, zmin=-Lz)
    S = stiffness(g); K = kernel_potential(g, kernels, soft)
    w = 1.0/np.sqrt(g.V); W = sp.diags(w)
    A = (W @ (S + sp.diags(-K*g.V)).tocsr() @ W).tocsr()
    val, vec = eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000)
    return val[0]


if __name__ == '__main__':
    import sys, time
    D = 1.4
    EX_HE, EX_H2 = -2.903724, -1.174476          # exact non-relativistic (H2 at R=1.4)
    HF_HE, HF_H2 = -2.861680, -1.133630          # Hartree-Fock limits = the hom2 target
    SOFT = 0.285                                 # x h, atom.js calibration: one-electron exact
    print("TWO CONSTRUCTIONS FOR A TWO-ELECTRON SYSTEM.  soft=%.3f*h, kernels on cell faces" % SOFT)
    print("  hom2  = both electrons in ONE density, pair repulsion counted once (factor-2 reduction)")
    print("  split = non-overlapping halves, free interface  (the construction that generalises)")
    print("exact   He %.6f   H2(R=1.4) %.6f" % (EX_HE, EX_H2))
    print("HF lim  He %.6f   H2(R=1.4) %.6f   <- hom2 must converge HERE if the code is right" % (HF_HE, HF_H2))
    print("control H -0.500000  He+ -2.000000   <- one electron, measures the discretisation\n")
    print("%6s | %9s %9s | %10s %7s | %10s %7s | %10s %7s | %10s %7s" %
          ("h", "H", "He+", "He hom2", "err%", "He split", "err%", "H2 hom2", "err%", "H2 split", "err%"))
    for n in [4, 5, 6, 7, 8]:
        h = 0.7/n; soft = SOFT*h
        Lz = round(12.0/h)*h; Lr = Lz
        t0 = time.time()
        eH   = one([(0.0, 1.0)], Lz, Lr, h, soft)
        eHep = one([(0.0, 2.0)], Lz, Lr, h, soft)
        e_hh, _ = hom2  ([(0.0, 2.0)],              0.0,   Lz, Lr, h, soft)
        e_hs, _ = split2([(0.0, 2.0)],              0.0,   Lz, Lr, h, soft)
        e_2h, _ = hom2  ([(-D/2, 1.0), (D/2, 1.0)], 1.0/D, Lz, Lr, h, soft)
        e_2s, _ = split2([(-D/2, 1.0), (D/2, 1.0)], 1.0/D, Lz, Lr, h, soft)
        pc = lambda e, x: 100*(e-x)/abs(x)
        print("%6.4f | %9.6f %9.6f | %10.5f %+7.2f | %10.5f %+7.2f | %10.5f %+7.2f | %10.5f %+7.2f  [%.0fs]" %
              (h, eH, eHep, e_hh, pc(e_hh, EX_HE), e_hs, pc(e_hs, EX_HE),
               e_2h, pc(e_2h, EX_H2), e_2s, pc(e_2s, EX_H2), time.time()-t0))
        sys.stdout.flush()
