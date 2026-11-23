"""
He / He2 with a RIGID SHIFTED SPLIT (axisym). Each He: +2 nucleus, two one-electron domains split by a plane s bohr
behind the nucleus (s > 0 = away from the partner; for the lone atom the sign is a convention). Free (Neumann) walls,
no wall motion. Pair counting, soft 0.285h.
  atom  : python3 he2_cal.py atom H s1,s2,...       -> E(He) per s   (full axis grid, one nucleus)
  dimer : python3 he2_cal.py dimer H s R1,R2,...    -> E(He2) per R  (half grid z>0 + mirror, as he2_split.py)
Calibration: s* with E(He) = -2.9037 (exact), then He2 at s* is a prediction.
"""
import sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_realqm import Grid, masked_stiffness, kernel_potential
from axisym_hom2 import _poisson_setup
MODE = sys.argv[1]; h = float(sys.argv[2]); L = 12.0; SOFT = 0.285*h; ZN = 2.0

from scipy.ndimage import label as _label
def connect(g, front):
    # a staircase cone leaves cells cut off from their domain; an isolated cell has no kinetic energy and an electron
    # collapses into it. Keep each domain's largest connected piece; stray cells go to the other domain.
    F = front.reshape(g.nz, g.nr).copy()
    for want in (True, False):
        M = F == want; lab, n = _label(M)
        if n > 1:
            big = 1 + np.argmax([np.sum(lab == k) for k in range(1, n + 1)]); F[M & (lab != big)] = not want
    return F.ravel()
def scf(g, K, doms, pot, Eextra, seedc):
    for d in doms:
        d['S'] = masked_stiffness(g, d['mask']); d['idx'] = np.where(d['mask'])[0]
        d['Ssub'] = d['S'][d['idx']][:, d['idx']].tocsc(); d['eps'] = None
        u = np.where(d['mask'], np.exp(-1.7*seedc), 0.0); d['u'] = u/np.sqrt(np.sum(u*u*g.V))
    V = g.V; Eo = None; Pst = None
    for it in range(300):
        Pn = pot([d['u']**2 for d in doms])
        Pst = Pn if Pst is None else [a + 0.5*(b - a) for a, b in zip(Pst, Pn)]
        for d, Pk in zip(doms, Pst):
            ix = d['idx']; A = (d['Ssub'] + sp.diags(((-K + Pk)*V)[ix])).tocsc(); M = sp.diags(V[ix]).tocsc()
            sig = (d['eps'] - 0.3) if d['eps'] is not None else -3.0
            val, vec = eigsh(A, k=1, M=M, sigma=sig, which='LM', v0=d['u'][ix])
            u = np.zeros(g.N); u[ix] = vec[:, 0]; u *= np.sign(u[ix].sum()); d['u'] = u/np.sqrt(np.sum(u*u*V)); d['eps'] = val[0]
        E = Eextra([d['u'] for d in doms], [d['S'] for d in doms])
        if Eo is not None and abs(E - Eo) < 1e-8: break
        Eo = E
    return E

if MODE == 'atom':
    g = Grid(L, L, h, zmin=-L); solveP, bnd, rrf = _poisson_setup(g); V = g.V; Zc = g.Z.ravel()
    K = kernel_potential(g, [(0.0, ZN)], SOFT); dist = np.sqrt(Zc**2 + g.R.ravel()**2)
    ZB, RB = g.Z.ravel()[bnd], g.R.ravel()[bnd]
    def P1(rho):                                   # Dirichlet value: unit charge at the density's own centroid
        zc = np.sum(Zc*rho*V)/np.sum(rho*V)
        b = 4*np.pi*rho*V; b[bnd] = 1.0/np.sqrt((ZB - zc)**2 + RB**2); return solveP(b)
    for sa in sys.argv[3].split(','):
        if sa.startswith('c'):
            cc = float(sa[1:]); s = cc; fr = connect(g, (-Zc/np.maximum(dist, 1e-12)) >= cc)
            doms = [dict(mask=fr), dict(mask=~fr)]
        else:
            s = float(sa); doms = [dict(mask=Zc < s), dict(mask=Zc >= s)]
        pot = lambda rh: [P1(rh[1]), P1(rh[0])]
        def Ef(us, Ss):
            Pa = P1(us[0]**2)
            return sum(u @ (S @ u) - np.sum(K*u*u*V) for u, S in zip(us, Ss)) + np.sum(us[1]**2*Pa*V)
        E = scf(g, K, doms, pot, Ef, dist)
        zf, zb = [np.sum(Zc*d['u']**2*V) for d in doms]
        print("He atom h=%.3f s=%+.3f E=%.6f  centroids front %+.3f back %+.3f  dipole %+.3f au" % (h, s, E, zf, zb, -(zf + zb)), flush=True)
else:
    CONE = sys.argv[3].startswith('c'); s = float(sys.argv[3].lstrip('c'))
    gh = Grid(L, L, h, zmin=0.0); gf = Grid(L, L, h, zmin=-L); nzh = gh.nz; nr = gh.nr; V = gh.V
    solveP, bnd, rrf = _poisson_setup(gf); Zc = gh.Z.ravel(); ZB, RB = gf.Z.ravel()[bnd], gf.R.ravel()[bnd]
    def potm(rho):
        full = np.zeros((gf.nz, gf.nr)); full[nzh:, :] = rho.reshape(nzh, nr)
        zc = np.sum(Zc*rho*V)/np.sum(rho*V)        # unit charge at its centroid (was: at the origin, an R-dependent error)
        b = 4*np.pi*full.ravel()*gf.V; b[bnd] = 1.0/np.sqrt((ZB - zc)**2 + RB**2); P = solveP(b).reshape(gf.nz, gf.nr)
        return P[nzh:, :].ravel(), P[:nzh, :][::-1, :].ravel()     # own-side, mirror image (left atom's copy)
    for R in [float(x) for x in sys.argv[4].split(',')]:
        K = kernel_potential(gh, [(-R/2, ZN), (R/2, ZN)], SOFT); zp = R/2 + s
        dist = np.sqrt((Zc - R/2)**2 + gh.R.ravel()**2)
        if CONE:                                                  # front electron: cone cos >= s about the bond
            fr = connect(gh, (-(Zc - R/2)/np.maximum(dist, 1e-12)) >= s)
            doms = [dict(mask=fr), dict(mask=~fr)]
        else:
            doms = [dict(mask=Zc < zp), dict(mask=Zc >= zp)]      # front (bond side), back
        def pot(rh):
            (Pi, Li), (Po, Lo) = potm(rh[0]), potm(rh[1]); PL = Li + Lo
            return [Po + PL, Pi + PL]
        def Ef(us, Ss):
            (Pi, Li), (Po, Lo) = potm(us[0]**2), potm(us[1]**2)
            return 2*sum(u @ (S @ u) - np.sum(K*u*u*V) for u, S in zip(us, Ss)) + 2*np.sum(us[0]**2*Po*V) \
                + np.sum((us[0]**2 + us[1]**2)*(Li + Lo)*V) + ZN*ZN/R
        E = scf(gh, K, doms, pot, Ef, dist)
        print("He2 h=%.3f %s=%+.3f R=%.2f E=%.6f" % (h, "cone" if CONE else "s", s, R, E), flush=True)
