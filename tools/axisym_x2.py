"""X2 with one electron per kernel: midplane split, Neumann at the plane, partner by reflection."""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh, factorized
from axisym_realqm import Grid, stiffness, laplacian, kernel_potential

def x2(D, Zk=1.0, Lz=12.0, Lr=12.0, h=0.12, soft=0.0, rc=0.0, iters=40, tol=1e-7, verbose=False):
    """E_tot for two kernels of charge Zk at z=+-D/2 with one electron each.
       rc>0 excludes each electron from r<rc of ITS OWN kernel (pseudo-kernel mode)."""
    gh = Grid(Lz, Lr, h, zmin=0.0)            # electron 2 lives on z>0
    gf = Grid(Lz, Lr, h, zmin=-Lz)            # full domain, for the Poisson solve
    nzh = gh.nz
    assert gf.nz == 2*nzh
    S  = stiffness(gh)
    K  = kernel_potential(gh, [(-D/2, Zk), (D/2, Zk)], soft)
    # optional inner cutoff about the near kernel (support excluded inside rc)
    mask = np.ones(gh.N, bool)
    if rc > 0:
        rr = np.sqrt((gh.Z - D/2)**2 + gh.R**2).ravel()
        mask = rr > rc
    # Poisson matrix on the full grid, Dirichlet P=q/|x| outside -- factorise ONCE
    L = laplacian(gf).tolil()
    rrf = np.sqrt(gf.Z**2 + gf.R**2).ravel()
    bnd = np.zeros(gf.N, bool)
    bnd[gf.idx(0, np.arange(gf.nr))] = True
    bnd[gf.idx(gf.nz-1, np.arange(gf.nr))] = True
    bnd[gf.idx(np.arange(gf.nz), gf.nr-1)] = True
    for k in np.where(bnd)[0]:
        L.rows[k] = [k]; L.data[k] = [1.0]
    solveP = factorized(L.tocsc())
    w = 1.0/np.sqrt(gh.V)
    W = sp.diags(w)
    mir = np.arange(nzh)[::-1]                 # half cell i  <->  full cell nzh-1-i
    P1 = np.zeros(gh.N); u = None; Eold = None
    for it in range(iters):
        Dg = sp.diags((-K + P1)*gh.V)
        A = (W @ (S + Dg).tocsr() @ W).tocsr()
        if not mask.all():                     # hard-exclude the cutoff region
            keep = np.where(mask)[0]
            Asub = A[keep][:, keep]
            val, vec = eigsh(Asub, k=1, which='SA', tol=1e-10, maxiter=100000,
                             v0=None if u is None else (u/w)[keep])
            y = np.zeros(gh.N); y[keep] = vec[:, 0]
        else:
            val, vec = eigsh(A, k=1, which='SA', tol=1e-10, maxiter=100000,
                             v0=None if u is None else (u/w))
            y = vec[:, 0]
        u = w*y
        u = u/np.sqrt(np.sum(u*u*gh.V))
        # partner density = mirror of this one; solve Poisson on the full grid
        dens2 = (u*u).reshape(nzh, gh.nr)
        densf = np.zeros((gf.nz, gf.nr)); densf[nzh:, :] = dens2
        b = 4*np.pi*densf.ravel()*gf.V
        b[bnd] = 1.0/rrf[bnd]                  # one electron of charge 1
        P2 = solveP(b).reshape(gf.nz, gf.nr)
        P1 = P2[mir, :].ravel()                # P of electron 1, evaluated on z>0
        T  = float(u @ (S @ u))
        VK = -float(np.sum(K*u*u*gh.V))
        U12= float(np.sum(P1*u*u*gh.V))
        E  = 2*(T+VK) + U12 + Zk*Zk/D
        if verbose: print("    it%02d  E=%.6f  T=%.4f VK=%.4f U12=%.4f" % (it,E,T,VK,U12))
        if Eold is not None and abs(E-Eold) < tol: break
        Eold = E
    return E, dict(T=T, VK=VK, U12=U12, nn=Zk*Zk/D, iters=it+1)
