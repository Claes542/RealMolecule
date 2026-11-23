"""
He2 with each He ONE 2hom PAIR (spherical-type atom, no dipole), the wall between the atoms at the midplane either
FREE (Neumann, the RealQM rule) or COSTLY (Dirichlet, u = 0 on the midplane: the kinetic price stdQM's orthogonality
charges between same-spin electrons of two closed shells). Axisym half grid z > 0 + mirror; nuclei +2 on cell faces.
  E = 2*[ 2(T - int K u^2) + int u^2 P[u^2] ] + 4 int u^2 Pmir[u^2] + 4/R      (u per electron, pair counted once)
Atom reference: one 2hom pair on the full grid, same h, soft, face placement.
usage: python3 he2_dir.py H neumann|dirichlet R1,R2,...   |   python3 he2_dir.py H atom
"""
import sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_realqm import Grid, stiffness, kernel_potential
from axisym_hom2 import _poisson_setup
h = float(sys.argv[1]); WALL = sys.argv[2]; L = 12.0; SOFT = 0.285*h; ZN = 2.0

def ground(g, S, K, Peff, u0):
    w = 1/np.sqrt(g.V); W = sp.diags(w)
    A = (W @ (S + sp.diags((-K + Peff)*g.V)).tocsr() @ W).tocsr()
    val, vec = eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000, v0=None if u0 is None else u0/w)
    u = w*vec[:, 0]; u *= np.sign(u.sum()); return u/np.sqrt(np.sum(u*u*g.V))

if WALL == 'atom':
    g = Grid(L, L, h, zmin=-L); S = stiffness(g); K = kernel_potential(g, [(0.0, ZN)], SOFT)
    solveP, bnd, rrf = _poisson_setup(g); V = g.V
    def P1(rho): b = 4*np.pi*rho*V; b[bnd] = 1/rrf[bnd]; return solveP(b)
    u = None; P = np.zeros(g.N); Eo = None
    for it in range(300):
        u = ground(g, S, K, P, u); Pn = P1(u*u); P = Pn if it == 0 else P + 0.5*(Pn - P)
        E = 2*(u @ (S @ u) - np.sum(K*u*u*V)) + np.sum(u*u*P1(u*u)*V)
        if Eo is not None and abs(E - Eo) < 1e-9: break
        Eo = E
    print("He atom 2hom h=%.3f E=%.6f" % (h, E), flush=True); sys.exit()

gh = Grid(L, L, h, zmin=0.0); gf = Grid(L, L, h, zmin=-L); nzh, nr, V = gh.nz, gh.nr, gh.V
S = stiffness(gh)
if WALL == 'dirichlet':      # u = 0 on the face z = 0, half a cell from the first centres: 0.5*area*(u/(h/2))^2*(h/2)
    d = np.zeros(gh.N); d[gh.idx(0, np.arange(nr))] = 2*np.pi*gh.r; S = (S + sp.diags(d)).tocsr()
solveP, bnd, rrf = _poisson_setup(gf); Zc = gh.Z.ravel(); ZB, RB = gf.Z.ravel()[bnd], gf.R.ravel()[bnd]
def potm(rho):
    full = np.zeros((gf.nz, gf.nr)); full[nzh:, :] = rho.reshape(nzh, nr)
    zc = np.sum(Zc*rho*V)/np.sum(rho*V)
    b = 4*np.pi*full.ravel()*gf.V; b[bnd] = 1.0/np.sqrt((ZB - zc)**2 + RB**2); P = solveP(b).reshape(gf.nz, gf.nr)
    return P[nzh:, :].ravel(), P[:nzh, :][::-1, :].ravel()
for R in [float(x) for x in sys.argv[3].split(',')]:
    K = kernel_potential(gh, [(-R/2, ZN), (R/2, ZN)], SOFT); u = None; P = None; Eo = None
    for it in range(300):
        if u is None: Pe = np.zeros(gh.N)
        u = ground(gh, S, K, Pe, u); Ps, Pm = potm(u*u); Pn = Ps + 2*Pm
        Pe = Pn if it == 0 else Pe + 0.5*(Pn - Pe)
        E = 2*(2*(u @ (S @ u) - np.sum(K*u*u*V)) + np.sum(u*u*Ps*V)) + 4*np.sum(u*u*Pm*V) + ZN*ZN/R
        if Eo is not None and abs(E - Eo) < 1e-9: break
        Eo = E
    mid = np.sum(2*np.pi*gh.r*h*(u.reshape(nzh, nr)[0, :]**2))      # density on the first layer (wall contact)
    print("He2 2hom %s h=%.3f R=%.2f E=%.6f  contact %.2e" % (WALL, h, R, E, mid), flush=True)
