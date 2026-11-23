"""
FREE CONE: c2_cone.py's directed valence (rigid 1s^2 core | 1 bond e in a cone toward the partner | 3 back e), with the
cone wall a FREE BOUNDARY under RealQM's continuity rule: the half-angle TH is moved until the bond and back densities
match across the cone wall (area-weighted mean of rho_bond - rho_back over the wall faces = 0). Not minimum energy
(min-E over boundaries runs away). Core wall (Rc, from the 1D atom) and midplane as before.
usage: python3 c2_freecone.py scan D [h]            -> mismatch and E vs TH
       python3 c2_freecone.py root D1,D2,... [h]    -> TH* by secant on the mismatch, E(TH*)
"""
import sys, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_realqm import Grid, masked_stiffness, kernel_potential
from axisym_hom2 import _poisson_setup

h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
L = 12.0; RC = 0.700; SOFT = 0.285*h; ZN = 6.0
gh = Grid(L, L, h, zmin=0.0); gf = Grid(L, L, h, zmin=-L); nzh = gh.nz; nr = gh.nr; V = gh.V
solveP, bnd, rrf = _poisson_setup(gf)
pr = np.load('/Users/cgjoh/Development/H2O/runs_cc/c2_core_profile_h%.3f.npy' % h)
# face areas between neighbouring cells (z-faces and r-faces), for the wall average
Az = np.repeat((2*np.pi*gh.r*h)[None, :], nzh - 1, 0)            # between (i,j) and (i+1,j)
Ar = np.repeat((2*np.pi*(np.arange(1, nr)*h)*h)[None, :], nzh, 0) # between (i,j) and (i,j+1)

def pot(rho_half, q):
    full = np.zeros((gf.nz, gf.nr)); full[nzh:, :] = rho_half.reshape(nzh, nr)
    b = 4*np.pi*full.ravel()*gf.V; b[bnd] = q/rrf[bnd]
    P = solveP(b).reshape(gf.nz, gf.nr)
    return P[nzh:, :].ravel(), P[:nzh, :][::-1, :].ravel()

def solve(D, TH, init=None):
    K = kernel_potential(gh, [(-D/2, ZN), (D/2, ZN)], SOFT)
    dz = (gh.Z - D/2).ravel(); dist = np.sqrt(dz**2 + gh.R.ravel()**2); cosp = -dz/np.maximum(dist, 1e-12)
    mc = dist < RC; mb = (~mc) & (cosp > np.cos(np.radians(TH))); mk = (~mc) & (~mb)
    dom = {'c': dict(mask=mc, N=2.0), 'b': dict(mask=mb, N=1.0), 'k': dict(mask=mk, N=3.0)}
    for k, d in dom.items():
        d['f'] = (d['N'] - 1)/d['N']; d['S'] = masked_stiffness(gh, d['mask']); d['idx'] = np.where(d['mask'])[0]
        d['Ssub'] = d['S'][d['idx']][:, d['idx']].tocsc(); d['eps'] = None
    dom['c']['u'] = np.where(mc, np.interp(dist, pr[0], pr[1]), 0.0)
    for k in ('b', 'k'):
        d = dom[k]; d['u'] = np.where(d['mask'], np.exp(-1.6*dist) if init is None else np.maximum(init[k], 1e-6*np.exp(-dist)), 0.0)
    for d in dom.values(): d['u'] *= np.sqrt(d['N']/np.sum(d['u']**2*V))
    def potentials():
        P = {}; PL = np.zeros(gh.N)
        for k, d in dom.items():
            P[k], px = pot(d['u']**2, d['N']); PL += px
        return P, PL
    Eold = None; Pst = None
    for it in range(400):
        P, PL = potentials()
        Pnew = {k: d['f']*P[k] + sum(P[e] for e in dom if e != k) + PL for k, d in dom.items()}
        Pst = Pnew if Pst is None else {k: Pst[k] + 0.5*(Pnew[k] - Pst[k]) for k in Pnew}
        for k in ('b', 'k'):
            d = dom[k]; ix = d['idx']
            A = (d['Ssub'] + sp.diags(((-K + Pst[k])*V)[ix])).tocsc(); Mm = sp.diags(V[ix]).tocsc()
            sig = (d['eps'] - 0.3) if d['eps'] is not None else -2.0
            val, vec = eigsh(A, k=1, M=Mm, sigma=sig, which='LM', v0=d['u'][ix])
            u = np.zeros(gh.N); u[ix] = vec[:, 0]; u *= np.sign(u[ix].sum())
            d['u'] = u*np.sqrt(d['N']/np.sum(u*u*V)); d['eps'] = val[0]
        P, PL = potentials(); E = ZN*ZN/D; ks = list(dom)
        for i, k in enumerate(ks):
            d = dom[k]; rho = d['u']**2
            E += 2*(d['u'] @ (d['S'] @ d['u']) - np.sum(K*rho*V) + 0.5*d['f']*np.sum(rho*P[k]*V))
            for e in ks[i+1:]: E += 2*np.sum(rho*P[e]*V)
            E += np.sum(rho*PL*V)
        if Eold is not None and abs(E - Eold) < 1e-7: break
        Eold = E
    # cone-wall mismatch: faces with bond on one side and back on the other
    B = mb.reshape(nzh, nr); Kb = mk.reshape(nzh, nr)
    rb = (dom['b']['u']**2).reshape(nzh, nr); rk = (dom['k']['u']**2).reshape(nzh, nr)
    num = 0.0; den = 0.0
    for (sa, sb, A) in [((slice(0, -1), slice(None)), (slice(1, None), slice(None)), Az),
                        ((slice(None), slice(0, -1)), (slice(None), slice(1, None)), Ar)]:
        p1 = B[sa] & Kb[sb]; p2 = Kb[sa] & B[sb]
        num += np.sum(A*p1*(rb[sa] - rk[sb])) + np.sum(A*p2*(rb[sb] - rk[sa])); den += np.sum(A*(p1 | p2))
    mis = num/den if den > 0 else float('nan')
    cen = float(np.sum(dom['b']['u']**2*V*gh.Z.ravel()))
    return E, mis, cen, {k: dom[k]['u'] for k in ('b', 'k')}

if sys.argv[1] == 'scan':
    D = float(sys.argv[2])
    for TH in [40, 60, 80, 100, 120, 140]:
        E, mis, cen, _ = solve(D, TH)
        print("scan D=%.2f TH=%5.1f  E=%.6f  mismatch rho_b-rho_k=%+.5f  <z>_bond %.3f" % (D, TH, E, mis, cen)); sys.stdout.flush()
else:
    for D in [float(x) for x in sys.argv[2].split(',')]:
        t0 = time.time(); a, b = 50.0, 90.0
        Ea, ma, _, _ = solve(D, a); Eb, mbv, _, _ = solve(D, b)
        while ma*mbv > 0 and b < 170: b += 20; Eb, mbv, _, _ = solve(D, b)
        while ma*mbv > 0 and a > 15: a -= 10; Ea, ma, _, _ = solve(D, a)
        while b - a > 0.5:
            c = 0.5*(a + b); Ec, mc_, cen, _ = solve(D, c)
            if mc_*ma > 0: a, ma = c, mc_
            else: b, mbv = c, mc_
        c = 0.5*(a + b); Ec, mc_, cen, _ = solve(D, c)
        print("ROOT D=%.2f TH*=%.2f  E=%.6f  mismatch %+.5f  <z>_bond %.3f  (%.0fs)" % (D, c, Ec, mc_, cen, time.time()-t0))
        sys.stdout.flush()
