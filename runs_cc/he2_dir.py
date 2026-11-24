"""
He2 with each He ONE 2hom PAIR (spherical-type atom, no dipole), the wall between the atoms at the midplane either
FREE (Neumann, the RealQM rule) or COSTLY (Dirichlet, u = 0 on the midplane: the kinetic price stdQM's orthogonality
charges between same-spin electrons of two closed shells). Axisym half grid z > 0 + mirror; nuclei +2 on cell faces.
  E = 2*[ 2(T - int K u^2) + int u^2 P[u^2] ] + 4 int u^2 Pmir[u^2] + 4/R      (u per electron, pair counted once)
Atom reference: one 2hom pair on the full grid, same h, soft, face placement.
usage: python3 he2_dir.py H neumann|dirichlet|robin<beta>|tf<kappa> R1,R2,...   |   python3 he2_dir.py H atom
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

def atom_u():
    g = Grid(L, L, h, zmin=-L); S_ = stiffness(g); K_ = kernel_potential(g, [(0.0, ZN)], SOFT)
    solveP, bnd, rrf = _poisson_setup(g); V_ = g.V
    def P1(rho): b = 4*np.pi*rho*V_; b[bnd] = 1/rrf[bnd]; return solveP(b)
    u = None; P = np.zeros(g.N); Eo = None
    for it in range(300):
        u = ground(g, S_, K_, P, u); Pn = P1(u*u); P = Pn if it == 0 else P + 0.5*(Pn - P)
        E = 2*(u @ (S_ @ u) - np.sum(K_*u*u*V_)) + np.sum(u*u*P1(u*u)*V_)
        if Eo is not None and abs(E - Eo) < 1e-9: break
        Eo = E
    return g, u, E

if WALL == 'own':
    # Robin wall with beta = the FREE atom's own log-derivative at each wall cell: beta_j = d ln u_atom / dz at
    # (z = -R/2 from the nucleus, r_j) -- the wall continues each atom's isolated tail. No parameter.
    ga, ua, Ea = atom_u(); U = ua.reshape(ga.nz, ga.nr); za = ga.Z[:, 0]
    print("atom E=%.6f" % Ea, flush=True)
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
S = stiffness(gh); S0 = S; Aw0 = np.zeros(gh.N); Aw0[gh.idx(0, np.arange(nr))] = 2*np.pi*gh.r*h
if WALL == 'dirichlet':      # u = 0 on the face z = 0, half a cell from the first centres: 0.5*area*(u/(h/2))^2*(h/2)
    d = np.zeros(gh.N); d[gh.idx(0, np.arange(nr))] = 2*np.pi*gh.r; S = (S + sp.diags(d)).tocsr()
elif WALL.startswith('tf'):     # TF-form wall cost: E_w = kappa * sum_sides int rho_side^(5/3) dS, rho_side = 2u^2 (first layer)
    KAPPA = float(WALL[2:]); Aw = np.zeros(gh.N); Aw[gh.idx(0, np.arange(nr))] = 2*np.pi*gh.r*h
elif WALL.startswith('robin'):  # Robin wall: per electron (beta/2) int u^2 dS on z = 0 (first-layer value, face area 2 pi r h)
    BETA = float(WALL[5:]); d = np.zeros(gh.N); d[gh.idx(0, np.arange(nr))] = 0.5*BETA*2*np.pi*gh.r*h; S = (S + sp.diags(d)).tocsr()
solveP, bnd, rrf = _poisson_setup(gf); Zc = gh.Z.ravel(); ZB, RB = gf.Z.ravel()[bnd], gf.R.ravel()[bnd]
def potm(rho):
    full = np.zeros((gf.nz, gf.nr)); full[nzh:, :] = rho.reshape(nzh, nr)
    zc = np.sum(Zc*rho*V)/np.sum(rho*V)
    b = 4*np.pi*full.ravel()*gf.V; b[bnd] = 1.0/np.sqrt((ZB - zc)**2 + RB**2); P = solveP(b).reshape(gf.nz, gf.nr)
    return P[nzh:, :].ravel(), P[:nzh, :][::-1, :].ravel()
for R in [float(x) for x in sys.argv[3].split(',')]:
    K = kernel_potential(gh, [(-R/2, ZN), (R/2, ZN)], SOFT); u = None; P = None; Eo = None
    if WALL == 'own':   # atom grid cell centres z = (k+1/2)h - L; wall face at -R/2 relative to the atom nucleus
        zw = -R/2; k = int(np.floor((zw + L)/h - 0.5)); t = ((zw + L)/h - 0.5) - k
        lu = np.log(np.maximum(U, 1e-300)); dl = (lu[k + 1, :] - lu[k, :])/h       # d ln u/dz between the bracketing rows
        bet = np.maximum(dl, 0.0); d = np.zeros(gh.N); d[gh.idx(0, np.arange(nr))] = 0.5*bet[:nr]*2*np.pi*gh.r*h
        S = (stiffness(gh) + sp.diags(d)).tocsr(); print("  R=%.2f beta on axis %.3f, at r=1 %.3f" % (R, bet[0], bet[int(1/h)]), flush=True)
    for it in range(300):
        if u is None: Pe = np.zeros(gh.N)
        if WALL == 'local':  # Robin beta_j = sqrt(2 max(V_j - eps, 0)) per wall cell, V_j = -K + Pe there, self-consistent
            if u is None: bl = np.zeros(gh.N)
            else:
                eps = u @ (S @ u) + np.sum((-K + Pe)*u*u*V)
                bl = np.sqrt(2*np.maximum(-K + Pe - eps, 0.0)) * (Aw0 > 0)
            S = (S0 + sp.diags(0.5*bl*Aw0)).tocsr(); bet = float(np.sum(bl*Aw0)/np.sum(Aw0)) if np.any(Aw0) else 0.0
        if WALL == 'eps':    # Robin beta = sqrt(-2 eps), eps the domain eigenvalue (evanescent continuation), self-consistent
            if u is None: bet = 1.0
            else: eps = u @ (S @ u) + np.sum((-K + Pe)*u*u*V); bet = np.sqrt(max(-2*eps, 0.0))
            S = (S0 + sp.diags(0.5*bet*Aw0)).tocsr()
        u = ground(gh, S, K, Pe, u); Ps, Pm = potm(u*u); Pn = Ps + 2*Pm
        if WALL.startswith('tf'): Pn = Pn + KAPPA*Aw*(5/3)*(2*u*u)**(2/3)/V     # dE_w/du = 8 d u (4 electrons x u.S.u)
        Pe = Pn if it == 0 else Pe + 0.5*(Pn - Pe)
        E = 2*(2*(u @ (S @ u) - np.sum(K*u*u*V)) + np.sum(u*u*Ps*V)) + 4*np.sum(u*u*Pm*V) + ZN*ZN/R
        if WALL.startswith('tf'): E += 2*KAPPA*np.sum(Aw*(2*u*u)**(5/3))
        if Eo is not None and abs(E - Eo) < 1e-9: break
        Eo = E
    mid = np.sum(2*np.pi*gh.r*h*(u.reshape(nzh, nr)[0, :]**2))      # density on the first layer (wall contact)
    W = 2*np.sum(2*np.pi*gh.r*h*(2*u.reshape(nzh, nr)[0, :]**2)**(5/3))     # sum_sides int rho^(5/3) dS
    print("He2 2hom %s h=%.3f R=%.2f E=%.6f  contact %.2e  W53 %.4e%s" % (WALL, h, R, E, mid, W, ('  beta %.4f' % bet) if WALL in ('eps', 'local') else ''), flush=True)
