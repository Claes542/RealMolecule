"""
SPLIT-BOND variant of xh_freecone.py: H keeps its OWN electron. Domains: core (2, r<Rc) | X bond electron (1, cone TH,
z<s) | H electron (1, z>s, proton inside) | back (Z-3, outside the cone, z<s). TWO free walls by continuity: the cone
(X bond | back) and the plane z=s (lower side | H). Nested bisection: TH*(s) by cone continuity, then s by plane
continuity. The plane position is the charge transfer X<-H.
---- original header ----
ALL-ELECTRON XH (OH, HF) on the axisymmetric grid, FREE-WALL construction, no pseudo-kernel, nothing fitted.
X nucleus (Z) at z=0, bare proton at z=+R. Domains:
  core = 1s^2, 2hom, r_X < Rc (Rc = the 1D Atom Simulator's first wall: O 0.525, F 0.446), nucleus inside;
  bond = ONE 2hom pair (X electron + H's electron) in the cone of half-angle TH about +z, r_X > Rc -- contains the proton;
  back = the other Z-3 valence electrons (lone pairs + unpaired, azimuthal average), the rest of space.
Free walls (Neumann) everywhere; the cone wall moved by CONTINUITY (rho_bond = rho_back across it), core wall at Rc.
E = sum_d [T_d - int K rho_d + (1/2) f_d int rho_d P_d] + sum_{d<e} int rho_d P_e + Z/R,  f_d = (N_d - 1)/N_d.
Dipole mu_z = 1*R - int rho z  (positive = H end positive, as in X^- H^+).
usage: python3 xh_freecone.py O|F R1,R2,... [h]
"""
import sys, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_realqm import Grid, masked_stiffness, kernel_potential
from axisym_hom2 import _poisson_setup

EL = sys.argv[1]; ZX, RC = {'O': (8.0, 0.525), 'F': (9.0, 0.446)}[EL]
Rs = [float(x) for x in sys.argv[2].split(',')]; h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
L = 12.0; SOFT = 0.285*h
g = Grid(L, L, h, zmin=-L); V = g.V; nz, nr = g.nz, g.nr
solveP, bnd, rr = _poisson_setup(g)
Z = g.Z.ravel(); dist = np.sqrt(Z**2 + g.R.ravel()**2); cosz = Z/np.maximum(dist, 1e-12)
Az = np.repeat((2*np.pi*g.r*h)[None, :], nz - 1, 0); Ar = np.repeat((2*np.pi*(np.arange(1, nr)*h)*h)[None, :], nz, 0)
def pot(rho, q):
    b = 4*np.pi*rho*V; b[bnd] = q/rr[bnd]; return solveP(b)

def solve(R, TH, sp_=None):
    K = kernel_potential(g, [(0.0, ZX), (R, 1.0)], SOFT)
    mc = dist < RC; up = Z > sp_
    mb = (~mc) & (~up) & (cosz > np.cos(np.radians(TH))); mk = (~mc) & (~up) & (~mb); me = up & (~mc)
    dom = {'c': dict(mask=mc, N=2.0), 'b': dict(mask=mb, N=1.0), 'e': dict(mask=me, N=1.0), 'k': dict(mask=mk, N=ZX - 3.0)}
    seed = dict(c=np.exp(-ZX*dist), b=np.exp(-1.5*np.sqrt(Z**2 + g.R.ravel()**2)) + np.exp(-np.sqrt((Z - R)**2 + g.R.ravel()**2)),
                k=np.exp(-2.0*dist), e=np.exp(-np.sqrt((Z - R)**2 + g.R.ravel()**2)))
    for k, d in dom.items():
        d['f'] = (d['N'] - 1)/d['N']; d['S'] = masked_stiffness(g, d['mask']); d['idx'] = np.where(d['mask'])[0]
        d['Ssub'] = d['S'][d['idx']][:, d['idx']].tocsc(); d['eps'] = None
        d['u'] = np.where(d['mask'], seed[k], 0.0); d['u'] *= np.sqrt(d['N']/np.sum(d['u']**2*V))
    Eold = None; Pst = None
    for it in range(400):
        P = {k: pot(d['u']**2, d['N']) for k, d in dom.items()}
        Pnew = {k: d['f']*P[k] + sum(P[e] for e in dom if e != k) for k, d in dom.items()}
        Pst = Pnew if Pst is None else {k: Pst[k] + 0.5*(Pnew[k] - Pst[k]) for k in Pnew}
        for k, d in dom.items():
            ix = d['idx']; A = (d['Ssub'] + sp.diags(((-K + Pst[k])*V)[ix])).tocsc(); Mm = sp.diags(V[ix]).tocsc()
            sig = (d['eps'] - 0.3) if d['eps'] is not None else (-0.5*ZX*ZX if k == 'c' else -2.0)
            val, vec = eigsh(A, k=1, M=Mm, sigma=sig, which='LM', v0=d['u'][ix])
            u = np.zeros(g.N); u[ix] = vec[:, 0]; u *= np.sign(u[ix].sum())
            d['u'] = u*np.sqrt(d['N']/np.sum(u*u*V)); d['eps'] = val[0]
        P = {k: pot(d['u']**2, d['N']) for k, d in dom.items()}
        E = ZX/R; ks = list(dom)
        for i, k in enumerate(ks):
            d = dom[k]; rho = d['u']**2
            E += d['u'] @ (d['S'] @ d['u']) - np.sum(K*rho*V) + 0.5*d['f']*np.sum(rho*P[k]*V)
            for e in ks[i+1:]: E += np.sum(rho*P[e]*V)
        if Eold is not None and abs(E - Eold) < 1e-7: break
        Eold = E
    B = mb.reshape(nz, nr); Kb = mk.reshape(nz, nr)
    rb = (dom['b']['u']**2).reshape(nz, nr); rk = (dom['k']['u']**2).reshape(nz, nr); num = den = 0.0
    for (sa, sb, A) in [((slice(0, -1), slice(None)), (slice(1, None), slice(None)), Az),
                        ((slice(None), slice(0, -1)), (slice(None), slice(1, None)), Ar)]:
        p1 = B[sa] & Kb[sb]; p2 = Kb[sa] & B[sb]
        num += np.sum(A*p1*(rb[sa] - rk[sb])) + np.sum(A*p2*(rb[sb] - rk[sa])); den += np.sum(A*(p1 | p2))
    # plane mismatch: faces across z = s, lower side (bond or back) minus the H electron
    lowd = (dom['b']['u']**2 + dom['k']['u']**2).reshape(nz, nr); re_ = (dom['e']['u']**2).reshape(nz, nr)
    Up = me.reshape(nz, nr); Lo = (mb | mk).reshape(nz, nr)
    pl = Lo[:-1, :] & Up[1:, :]
    misp = np.sum(Az*pl*(lowd[:-1, :] - re_[1:, :]))/max(np.sum(Az*pl), 1e-30)
    zc = {k: float(np.sum(d['u']**2*V*Z)) for k, d in dom.items()}      # first moments (charge-weighted)
    mu = R - sum(zc.values())
    return E, (num/den, misp), mu, zc

def cone_root(R, sp_):
    a, b = 25.0, 110.0
    _, (ma, _), _, _ = solve(R, a, sp_); _, (mbv, _), _, _ = solve(R, b, sp_)
    while ma*mbv > 0 and b < 175: b += 20; _, (mbv, _), _, _ = solve(R, b, sp_)
    while b - a > 1.0:
        c = 0.5*(a + b); _, (mc_, _), _, _ = solve(R, c, sp_)
        if mc_*ma > 0: a, ma = c, mc_
        else: b, mbv = c, mc_
    c = 0.5*(a + b); E, (m1, m2), mu, zc = solve(R, c, sp_)
    return c, E, m1, m2, mu, zc
for R in Rs:
    t0 = time.time(); lo, hi = RC + 2*h, R - 2*h
    r_lo = cone_root(R, lo); r_hi = cone_root(R, hi)
    print("  bracket s=%.2f misp %+.4f | s=%.2f misp %+.4f" % (lo, r_lo[3], hi, r_hi[3])); sys.stdout.flush()
    while hi - lo > 0.06 and r_lo[3]*r_hi[3] < 0:
        mid = round(0.5*(lo + hi)/h)*h
        if mid <= lo + 1e-9 or mid >= hi - 1e-9: break
        r_m = cone_root(R, mid)
        if r_m[3]*r_lo[3] > 0: lo, r_lo = mid, r_m
        else: hi, r_hi = mid, r_m
    best = r_lo if abs(r_lo[3]) < abs(r_hi[3]) else r_hi; sbest = lo if best is r_lo else hi
    TH, E, m1, m2, mu, zc = best
    print("%sH-split R=%.3f h=%.3f s*=%.3f TH*=%.1f E=%.6f misc %+.5f misp %+.5f mu=%+.4f au (%+.3f D) moments core %+.4f bondX %+.4f Helec %+.4f back %+.4f (%.0fs)"
          % (EL, R, h, sbest, TH, E, m1, m2, mu, mu*2.5417, zc['c'], zc['b'], zc['e'], zc['k'], time.time() - t0)); sys.stdout.flush()
