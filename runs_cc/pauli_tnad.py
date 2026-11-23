"""
Borrowed Pauli term (subsystem DFT) between RealQM domain densities, H2 2split (axisym, h 0.1, soft 0.285h):
each electron's half-space density, mirrored for the partner, smoothed ACROSS THE WALL (along z) by a Gaussian of
width w (charge-conserving), then the Thomas-Fermi non-additive kinetic energy
    T_nad = C int [ (rA+rB)^(5/3) - rA^(5/3) - rB^(5/3) ],
C = C_F = 2.8712 (unpolarised) or 2^(2/3) C_F (same spin, fully polarised). Target: what RealQM lacks for two
same-spin electrons, E(triplet) - E(2split) = 0.496 / 0.338 / 0.165 / 0.077 Ha at R = 1.4 / 2 / 3 / 4 (gvb_h2 runs).
"""
import sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_realqm import Grid, stiffness, kernel_potential
from axisym_hom2 import _poisson_setup
h = 0.1; L = 12.0; SOFT = 0.285*h; CF = 0.3*(3*np.pi**2)**(2/3)
TARGET = {1.4: 0.496, 2.0: 0.338, 3.0: 0.165, 4.0: 0.077}
def split_density(R, iters=200):
    gh = Grid(L, L, h, zmin=0.0); gf = Grid(L, L, h, zmin=-L); nzh = gh.nz
    S = stiffness(gh); K = kernel_potential(gh, [(-R/2, 1.0), (R/2, 1.0)], SOFT)
    solveP, bnd, rrf = _poisson_setup(gf); mir = np.arange(nzh)[::-1]
    w = 1/np.sqrt(gh.V); W = sp.diags(w); P1 = np.zeros(gh.N); u = None; Eo = None
    for it in range(iters):
        A = (W @ (S + sp.diags((-K + P1)*gh.V)).tocsr() @ W).tocsr()
        val, vec = eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000, v0=None if u is None else u/w)
        u = w*vec[:, 0]; u /= np.sqrt(np.sum(u*u*gh.V))
        d = np.zeros((gf.nz, gf.nr)); d[nzh:, :] = (u*u).reshape(nzh, gh.nr)
        b = 4*np.pi*d.ravel()*gf.V; b[bnd] = 1/rrf[bnd]
        P1 = solveP(b).reshape(gf.nz, gf.nr)[mir, :].ravel()
        E = 2*(u @ (S @ u) - np.sum(K*u*u*gh.V)) + np.sum(P1*u*u*gh.V) + 1/R
        if Eo is not None and abs(E - Eo) < 1e-9: break
        Eo = E
    rA = np.zeros((gf.nz, gf.nr)); rA[nzh:, :] = (u*u).reshape(nzh, gh.nr)
    return gf, rA, rA[::-1, :].copy(), E
for R in (1.4, 2.0, 3.0, 4.0):
    gf, rA, rB, E = split_density(R); V = gf.V.reshape(gf.nz, gf.nr)
    row = []
    for wd in (0.0, 0.1, 0.2, 0.3, 0.5):
        a = gaussian_filter1d(rA, wd/h, axis=0, mode='constant') if wd > 0 else rA
        b = gaussian_filter1d(rB, wd/h, axis=0, mode='constant') if wd > 0 else rB
        t = np.sum(((a + b)**(5/3) - a**(5/3) - b**(5/3))*V)
        row.append(t)
    print("R=%.1f E_split %.5f target %.3f | T_nad(TF, unpol) w=0/0.1/0.2/0.3/0.5: %s | same-spin x%.3f: %s" %
          (R, E, TARGET[R], ' '.join('%.4f' % (CF*t) for t in row), 2**(2/3), ' '.join('%.4f' % (CF*2**(2/3)*t) for t in row)), flush=True)
