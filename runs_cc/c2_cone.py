"""
All-electron C2 on the axisymmetric grid with DIRECTED valence: each carbon = +6 nucleus, 1s^2 2hom core (r < Rc,
nucleus inside), and its four valence electrons split by a FIXED cone about the bond axis:
  bond  = 1 electron in the cone of half-angle TH pointing at the partner carbon (TH = 60 deg: a quarter of the sphere,
          the tetrahedral share),
  back  = 3 electrons in the rest (the azimuthal average of the three tetrahedral back sectors; self-repulsion 2/3).
All walls free (Neumann): core|valence at Rc, bond|back on the cone, and the midplane where the two bond electrons meet
(2split). Same energy bookkeeping as c2_allelec.py, generalised to any list of domains:
  E = 2 sum_d [T_d - int K rho_d + (1/2) f_d int rho_d P_d] + 2 sum_{d<e} int rho_d P_e + int rho_R P_L + Z^2/D
usage: python3 c2_cone.py D [h] [L] [Rc] [TH_deg] [live|frozen]
"""
import sys, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_realqm import Grid, masked_stiffness, kernel_potential
from axisym_hom2 import _poisson_setup

D = float(sys.argv[1]); h = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
L = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0; RC = float(sys.argv[4]) if len(sys.argv) > 4 else 0.700
TH = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0; MODEF = sys.argv[6] if len(sys.argv) > 6 else 'live'
SOFT = 0.285*h; ZN = 6.0
gh = Grid(L, L, h, zmin=0.0); gf = Grid(L, L, h, zmin=-L); nzh = gh.nz; V = gh.V
K = kernel_potential(gh, [(-D/2, ZN), (D/2, ZN)], SOFT)
dz = (gh.Z - D/2).ravel(); dist = np.sqrt(dz**2 + gh.R.ravel()**2)
cosp = -dz/np.maximum(dist, 1e-12)                       # cosine of the angle to the partner direction (-z)
mc = dist < RC
mb = (~mc) & (cosp > np.cos(np.radians(TH)))
mk = (~mc) & (~mb)
dom = {'c': dict(mask=mc, N=2.0), 'b': dict(mask=mb, N=1.0), 'k': dict(mask=mk, N=3.0)}
for d in dom.values():
    d['f'] = (d['N'] - 1)/d['N']
    d['S'] = masked_stiffness(gh, d['mask']); d['idx'] = np.where(d['mask'])[0]
    d['Ssub'] = d['S'][d['idx']][:, d['idx']].tocsc(); d['eps'] = None
solveP, bnd, rrf = _poisson_setup(gf)
def pot(rho_half, q):
    full = np.zeros((gf.nz, gf.nr)); full[nzh:, :] = rho_half.reshape(nzh, gh.nr)
    b = 4*np.pi*full.ravel()*gf.V; b[bnd] = q/rrf[bnd]
    P = solveP(b).reshape(gf.nz, gf.nr)
    return P[nzh:, :].ravel(), P[:nzh, :][::-1, :].ravel()
seed = dict(c=np.exp(-5.7*dist), b=np.exp(-1.6*dist), k=np.exp(-1.6*dist))
for k, d in dom.items():
    d['u'] = np.where(d['mask'], seed[k], 0.0)
if MODEF == 'frozen':
    pr = np.load('/Users/cgjoh/Development/H2O/runs_cc/c2_core_profile_h%.3f.npy' % h)
    dom['c']['u'] = np.where(mc, np.interp(dist, pr[0], pr[1]), 0.0)
for d in dom.values(): d['u'] *= np.sqrt(d['N']/np.sum(d['u']**2*V))
def potentials():
    P = {}; PL = np.zeros(gh.N)
    for k, d in dom.items():
        P[k], px = pot(d['u']**2, d['N']); PL += px
    return P, PL
Eold = None; t0 = time.time(); mix = 0.5; Pst = None
for it in range(400):
    P, PL = potentials()
    Pnew = {k: d['f']*P[k] + sum(P[e] for e in dom if e != k) + PL for k, d in dom.items()}
    Pst = Pnew if Pst is None else {k: Pst[k] + mix*(Pnew[k] - Pst[k]) for k in Pnew}
    for k, d in dom.items():
        if MODEF == 'frozen' and k == 'c': continue
        ix = d['idx']; A = (d['Ssub'] + sp.diags(((-K + Pst[k])*V)[ix])).tocsc(); Mm = sp.diags(V[ix]).tocsc()
        sig = (d['eps'] - 0.3) if d['eps'] is not None else (-20.0 if k == 'c' else -2.0)
        val, vec = eigsh(A, k=1, M=Mm, sigma=sig, which='LM', v0=d['u'][ix])
        u = np.zeros(gh.N); u[ix] = vec[:, 0]; u *= np.sign(u[ix].sum())
        d['u'] = u*np.sqrt(d['N']/np.sum(u*u*V)); d['eps'] = val[0]
    P, PL = potentials()
    E = ZN*ZN/D
    ks = list(dom)
    for i, k in enumerate(ks):
        d = dom[k]; rho = d['u']**2
        E += 2*(d['u'] @ (d['S'] @ d['u']) - np.sum(K*rho*V) + 0.5*d['f']*np.sum(rho*P[k]*V))
        for e in ks[i+1:]: E += 2*np.sum(rho*P[e]*V)
        E += np.sum(rho*PL*V)
    if Eold is not None and abs(E - Eold) < 1e-7: break
    Eold = E
cen = {k: float(np.sum(d['u']**2*V*gh.Z.ravel())/d['N']) for k, d in dom.items()}
print("C2 cone [%s] D=%.3f h=%.3f Rc=%.2f TH=%.1f  E=%.6f  (it %d, %.0fs)  <z> core %.3f bond %.3f back %.3f  (nucleus %.3f)"
      % (MODEF, D, h, RC, TH, E, it+1, time.time()-t0, cen['c'], cen['b'], cen['k'], D/2))
sys.stdout.flush()
