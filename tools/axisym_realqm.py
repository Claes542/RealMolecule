"""
Axisymmetric (2D cylindrical) RealQM solver for one- and two-domain systems on an axis.

WHY 2D IS EXACT HERE.  For two electrons split by the plane perpendicular to the internuclear
axis, and for a valence pair on a single central kernel, the domains and the kernel potential are
all invariant under rotation about that axis.  So u = u(z,rho) with rho = sqrt(y^2+z^2), and the
3D problem reduces to 2D with measure 2*pi*rho drho dz -- nothing is discarded.  The mirror
symmetry in the plane halves it again: electron 1 is the reflection of electron 2, so we solve one
electron on z>0 and obtain its partner's potential by reflection.

Finite volume on a rho-cell-centred grid (rho_j = (j+1/2)h), so rho>0 everywhere and the axis
face has zero area -- the 1/rho singularity never appears and no-flux on the axis is automatic.
Omitting a face IS the homogeneous-Neumann (free interface) condition, which is the natural
boundary condition of the energy functional restricted to a domain.  That is the correct RealQM
interface condition, cf. the hard-Neumann midplane in essence_solver.html.

ENERGY, with the pair term counted ONCE (the house i!=j rule):
    E_tot = 2*(T + V_K) + U_12 + Z_a*Z_b/D
T and V_K are for ONE electron on its half; U_12 = int P_1 u_2^2 dV.  Self-repulsion is excluded
by construction: u_2 feels P_1 only.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, spsolve, factorized

class Grid:
    def __init__(self, Lz, Lr, h, zmin=None):
        self.h = h
        self.zmin = -Lz if zmin is None else zmin
        self.nz = int(round((Lz - self.zmin)/h))
        self.nr = int(round(Lr/h))
        self.z = self.zmin + (np.arange(self.nz)+0.5)*h
        self.r = (np.arange(self.nr)+0.5)*h
        self.Z, self.R = np.meshgrid(self.z, self.r, indexing='ij')
        self.V = (2*np.pi*self.R*h*h).ravel()          # cell volumes
        self.N = self.nz*self.nr
    def idx(self, i, j): return i*self.nr + j

def stiffness(g):
    """0.5*int |grad u|^2 dV  as a symmetric sparse matrix (no-flux on every outer face)."""
    h, nz, nr = g.h, g.nz, g.nr
    rows, cols, vals = [], [], []
    def add(a, b, c):
        rows.extend([a, b, a, b]); cols.extend([a, b, b, a])
        vals.extend([c, c, -c, -c])
    # z-faces: area 2*pi*rho_j*h, distance h  -> coeff 0.5*area/h = pi*rho_j
    for j in range(nr):
        c = np.pi*g.r[j]
        for i in range(nz-1):
            add(g.idx(i, j), g.idx(i+1, j), c)
    # rho-faces at rho=(j+1)*h: area 2*pi*(j+1)*h*h -> coeff 0.5*area/h = pi*(j+1)*h
    for j in range(nr-1):
        c = np.pi*(j+1)*h
        for i in range(nz):
            add(g.idx(i, j), g.idx(i, j+1), c)
    return sp.coo_matrix((vals, (rows, cols)), shape=(g.N, g.N)).tocsr()


def masked_stiffness(g, keep):
    """0.5*int |grad u|^2 assembled ONLY over faces joining two KEPT cells -- TRUE NEUMANN.

    Use this, NOT a submatrix of stiffness(), whenever a region is excluded (a pseudo-kernel core,
    an inner cutoff r_c).  Taking A[keep][:,keep] of the FULL stiffness retains the diagonal
    contributions of the faces leading into the removed region, which imposes DIRICHLET (u=0
    inside): density is pushed off the boundary and the state under-binds.  Omitting those faces
    at assembly is no-flux, the model's own free-boundary condition.
    Measured difference on the one-electron IE outside a core of radius R, h=0.175:
        R        1.6     1.8     2.0
        submatrix 3.977  3.697   3.484   eV   <- Dirichlet, WRONG
        masked    6.177  5.727   5.385   eV   <- Neumann, matches the 1D radial solve to ~2%
    This was the cause of the long-unexplained 36% axisym-vs-radial one-electron discrepancy.
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

def laplacian(g):
    """-Laplacian as a sparse matrix in the FLUX (volume-integrated) form, for the Poisson solve."""
    h, nz, nr = g.h, g.nz, g.nr
    rows, cols, vals = [], [], []
    def add(a, b, c):
        rows.extend([a, b, a, b]); cols.extend([a, b, b, a]); vals.extend([c, c, -c, -c])
    for j in range(nr):
        c = 2*np.pi*g.r[j]*h/h
        for i in range(nz-1): add(g.idx(i, j), g.idx(i+1, j), c)
    for j in range(nr-1):
        c = 2*np.pi*(j+1)*h*h/h
        for i in range(nz): add(g.idx(i, j), g.idx(i, j+1), c)
    return sp.coo_matrix((vals, (rows, cols)), shape=(g.N, g.N)).tocsr()

def kernel_potential(g, kernels, soft):
    """sum_k Z_k/|x-x_k|, softened; kernels = [(z_k, Z_k), ...] all on the axis."""
    K = np.zeros_like(g.Z)
    for zk, Zk in kernels:
        K += Zk/np.sqrt((g.Z-zk)**2 + g.R**2 + soft**2)
    return K.ravel()

def solve_one_electron(g, K, P, nev_sigma=None):
    """lowest eigenpair of (-1/2 Lap - K + P) with natural (Neumann) conditions."""
    S = stiffness(g)
    D = sp.diags((-K + P)*g.V)
    H = (S + D).tocsr()
    w = 1.0/np.sqrt(g.V)
    W = sp.diags(w)
    A = (W @ H @ W).tocsr()
    if nev_sigma is None:
        val, vec = eigsh(A, k=1, which='SA', tol=1e-9, maxiter=50000)
    else:
        val, vec = eigsh(A, k=1, sigma=nev_sigma, which='LM', tol=1e-9)
    u = (w*vec[:, 0])
    nrm = np.sqrt(np.sum(u*u*g.V)); u = u/nrm
    return val[0], u

def poisson(g, dens, qtot, solver=None):
    """solve -Lap P = 4*pi*dens with Dirichlet P=qtot/|x| on the outer faces."""
    L = laplacian(g).tolil()
    b = 4*np.pi*dens*g.V
    rr = np.sqrt(g.Z**2 + g.R**2).ravel()
    bnd = np.zeros(g.N, bool)
    bnd[g.idx(0, np.arange(g.nr))] = True
    bnd[g.idx(g.nz-1, np.arange(g.nr))] = True
    bnd[g.idx(np.arange(g.nz), g.nr-1)] = True
    for k in np.where(bnd)[0]:
        L.rows[k] = [k]; L.data[k] = [1.0]; b[k] = qtot/rr[k]
    return spsolve(L.tocsr(), b)
