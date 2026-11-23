"""
ALL-ELECTRON C2 on the axisymmetric grid: each carbon = +6 nucleus, 1s^2 as ONE 2hom core domain (r < Rc about its own
nucleus, nucleus INSIDE), four valence electrons as ONE 4hom domain filling the rest of its half-space. Free (Neumann)
walls everywhere: core|valence at Rc (fixed, from the 1D 2+4 atom: 0.700) and the midplane between the carbons.
No pseudo-kernel, no r_c, no fitted parameter.

Energy (u_d normalised to its electron count N_d, T_d = 0.5 int |grad u_d|^2 = u S u):
  E = 2 sum_d [T_d - int K rho_d] + 2 [ (1/2)(1/2) int rho_c P_c + (1/2)(3/4) int rho_v P_v ] + 2 int rho_c P_v
      + int (rho_c + rho_v) P_L + 36/D
  self-repulsion (N-1)/N, P_L = potential of the mirror carbon's electrons. Each domain solves
  (S + diag((-K + f_d P_d + P_other_same + P_L) V)) u = eps V u  on its own cells (masked stiffness = true Neumann).
usage: python3 c2_allelec.py D [h] [L] [Rc]
"""
import sys, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_realqm import Grid, masked_stiffness, kernel_potential
from axisym_hom2 import _poisson_setup

D = float(sys.argv[1]); h = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04
L = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0; RC = float(sys.argv[4]) if len(sys.argv) > 4 else 0.700
SOFT = 0.285*h; ZN = 6.0
MODEF = sys.argv[5] if len(sys.argv) > 5 else 'live'     # 'live' | 'save' (live, store the core profile) | 'frozen'
PROF = '/Users/cgjoh/Development/H2O/runs_cc/c2_core_profile_h%.3f.npy' % float(sys.argv[2] if len(sys.argv) > 2 else 0.04)
gh = Grid(L, L, h, zmin=0.0); gf = Grid(L, L, h, zmin=-L); nzh = gh.nz; V = gh.V
assert gf.nz == 2*nzh
K = kernel_potential(gh, [(-D/2, ZN), (D/2, ZN)], SOFT)
dist = np.sqrt((gh.Z - D/2)**2 + gh.R**2).ravel()
mc = dist < RC; mv = ~mc
dom = {'c': dict(mask=mc, N=2.0, f=0.5), 'v': dict(mask=mv, N=4.0, f=0.75)}
for d in dom.values():
    d['S'] = masked_stiffness(gh, d['mask']); d['idx'] = np.where(d['mask'])[0]
    d['Ssub'] = d['S'][d['idx']][:, d['idx']].tocsc()
solveP, bnd, rrf = _poisson_setup(gf)
def pot(rho_half, q):
    full = np.zeros((gf.nz, gf.nr)); full[nzh:, :] = rho_half.reshape(nzh, gh.nr)
    b = 4*np.pi*full.ravel()*gf.V; b[bnd] = q/rrf[bnd]
    P = solveP(b).reshape(gf.nz, gf.nr)
    return P[nzh:, :].ravel(), P[:nzh, :][::-1, :].ravel()      # own half, partner (mirror) half
# seed: hydrogenic-like core and valence
dom['c']['u'] = np.where(mc, np.exp(-5.7*dist), 0.0); dom['v']['u'] = np.where(mv, np.exp(-1.6*dist), 0.0)
if MODEF == 'frozen':
    # RIGID core: the separated-atom core (saved by a 'save' run) as a radial profile about its own nucleus,
    # renormalised to 2 on this grid; never re-solved. Same walls and energy expression as the live core.
    pr = np.load(PROF); dom['c']['u'] = np.where(mc, np.interp(dist, pr[0], pr[1]), 0.0)
for d in dom.values(): d['u'] *= np.sqrt(d['N']/np.sum(d['u']**2*V)); d['eps'] = None
Eold = None; t0 = time.time(); mix = 0.5; Pst = None
for it in range(300):
    rc_, rv_ = dom['c']['u']**2, dom['v']['u']**2
    Pc, Pc_x = pot(rc_, 2.0); Pv, Pv_x = pot(rv_, 4.0); PL = Pc_x + Pv_x
    Pnew = dict(c=dom['c']['f']*Pc + Pv + PL, v=dom['v']['f']*Pv + Pc + PL)
    Pst = Pnew if Pst is None else {k: Pst[k] + mix*(Pnew[k] - Pst[k]) for k in Pnew}
    for k, d in dom.items():
        if MODEF == 'frozen' and k == 'c': continue
        ix = d['idx']; A = (d['Ssub'] + sp.diags(((-K + Pst[k])*V)[ix])).tocsc(); Mm = sp.diags(V[ix]).tocsc()
        sig = (d['eps'] - 0.3) if d['eps'] is not None else (-20.0 if k == 'c' else -2.0)
        val, vec = eigsh(A, k=1, M=Mm, sigma=sig, which='LM', v0=d['u'][ix])
        u = np.zeros(gh.N); u[ix] = vec[:, 0]; u *= np.sign(u[ix].sum())
        d['u'] = u*np.sqrt(d['N']/np.sum(u*u*V)); d['eps'] = val[0]
    rc_, rv_ = dom['c']['u']**2, dom['v']['u']**2
    Pc, Pc_x = pot(rc_, 2.0); Pv, Pv_x = pot(rv_, 4.0); PL = Pc_x + Pv_x
    T = sum(d['u'] @ (d['S'] @ d['u']) for d in dom.values())
    VK = -np.sum(K*(rc_ + rv_)*V)
    Uself = 0.5*0.5*np.sum(rc_*Pc*V) + 0.5*0.75*np.sum(rv_*Pv*V)
    Ucv = np.sum(rc_*Pv*V); UL = np.sum((rc_ + rv_)*PL*V)
    E = 2*(T + VK) + 2*Uself + 2*Ucv + UL + ZN*ZN/D
    if Eold is not None and abs(E - Eold) < 1e-7: break
    Eold = E
if MODEF == 'save':
    db = np.round(dist[mc]/h*2)/2*h; uu = dom['c']['u'][mc]; rs = np.unique(db)
    np.save(PROF, np.vstack([rs, [uu[db == x].mean() for x in rs]]))
# diagnostics: density jump at the core wall (along the outward axis), valence charge toward the partner
iz = int(round((D/2)/h)); jr = 0
zline = gh.z; uc_ax = dom['c']['u'].reshape(nzh, gh.nr)[:, 0]; uv_ax = dom['v']['u'].reshape(nzh, gh.nr)[:, 0]
kw = int(np.searchsorted(zline, D/2 + RC)) ; kin = int(np.searchsorted(zline, D/2 - RC))
print(("C2 all-electron [%s] D=" % MODEF) + "%.3f h=%.3f L=%.1f Rc=%.3f  E=%.6f  (it %d, %.0fs)  T=%.4f VK=%.4f Uself=%.4f Ucv=%.4f UL=%.4f  eps c %.3f v %.3f"
      % (D, h, L, RC, E, it+1, time.time()-t0, 2*T, 2*VK, 2*Uself, 2*Ucv, UL, dom['c']['eps'] or 0.0, dom['v']['eps']))
print("   core wall outer side: rho_c %.4f | rho_v %.4f ; inner (toward partner) rho_c %.4f | rho_v %.4f" %
      (uc_ax[kw-1]**2, uv_ax[kw]**2, uc_ax[kin]**2, uv_ax[kin-1]**2 if kin > 0 else float('nan')))
sys.stdout.flush()
