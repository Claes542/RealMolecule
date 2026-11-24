"""X-H with an s-ONLY NODE (axisym): X = frozen-core potential (+Z nucleus screened by hydrogenic 1s^2, zeta):
K_X(d) = (Z-2)/d + 2 exp(-2 zeta d)(zeta + 1/d). Valence outside the HF 2s node sphere; plane through X (free) splits it
into BACK (n_back, lone pair) and FRONT (bond: n_front X electrons + the H electron, over a bare proton at +R).
Each domain: area-MEAN of u over its own part of the node sphere = 0. Sphere drawn in by h/2. Nothing fitted;
E(X) checked against -(sum of valence IEs).
usage: xh_s.py ELEMENT h R1,R2,...     ELEMENT: B (Z5, back 2, front 1)"""
import os, sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_linear import Linear
import os, realcore
EL = {'B': dict(Z=5, zeta=4.6795, node=0.459, back=2, front=1, target=-2.6233),
      'C': dict(Z=6, zeta=5.6727, node=0.377, back=2, front=1, target=-5.0261)}   # C+ (2s2 2p1): -(IE2+IE3+IE4); CH+ -> C+ + H

def sphere_area(m):
    g = m.g; nz, nr, h = g.nz, g.nr, g.h; core = ~m.ok.reshape(nz, nr); mk = m.ok.reshape(nz, nr)
    rr = g.r[None, :]*np.ones((nz, 1)); jj = np.arange(nr)[None, :]*np.ones((nz, 1)); A = np.zeros((nz, nr))
    A[1:, :] += np.where(mk[1:, :] & core[:-1, :], 2*np.pi*rr[1:, :]*h, 0)
    A[:-1, :] += np.where(mk[:-1, :] & core[1:, :], 2*np.pi*rr[:-1, :]*h, 0)
    A[:, 1:] += np.where(mk[:, 1:] & core[:, :-1], 2*np.pi*jj[:, 1:]*h*h, 0)
    A[:, :-1] += np.where(mk[:, :-1] & core[:, 1:], 2*np.pi*(jj[:, :-1]+1)*h*h, 0)
    return A.ravel()

class XS(Linear):
    def eig(self, S, Veff, keep, u0, lam0=None):
        A = (self.W @ (S + sp.diags(Veff*self.g.V)).tocsr() @ self.W).tocsr()[keep][:, keep]
        c = (sphere_area(self)*self.w)[keep]
        if not np.any(c): return Linear.eig(self, S, Veff, keep, u0, lam0)
        c = c/np.linalg.norm(c); Pj = lambda x: x - c*(c @ x)
        op = LinearOperator(A.shape, matvec=lambda x: Pj(A @ Pj(x)) + 1e3*c*(c @ x), dtype=float)
        v0 = None if u0 is None else Pj((u0/self.w)[keep])
        val, vec = eigsh(op, k=1, which='SA', tol=1e-10, maxiter=400000, v0=v0)
        y = np.zeros(self.g.N); y[keep] = Pj(vec[:, 0]); u = self.w*y; u /= np.sqrt(np.sum(u*u*self.g.V))
        return (-u if np.sum(u) < 0 else u), float(val[0])

def build(p, R, h, mol):
    rcx = p['node'] - 0.5*h
    nuc = [('X', 0.0, p['Z'] - 2, rcx)] + ([('H', R, 1, 0)] if mol else [])
    m = XS(nuc, [('back', p['back']), ('front', p['front'] + (1 if mol else 0))], [(0.0, 'fixed')],
           h=h, L=12.0, hfix=(0.0 if os.environ.get('FIXH') else R), name='xhs')
    d = np.maximum(np.sqrt(m.zc**2 + m.g.R.ravel()**2), 1e-12)
    m.K = m.K + (realcore.screen(p['Z'])(d) if os.environ.get('CORE') == 'real' else 2*np.exp(-2*p['zeta']*d)*(p['zeta'] + 1/d)); return m

import os
el, h = sys.argv[1], float(sys.argv[2]); p = EL[el]
if 'RNODE' in os.environ: p['node'] = float(os.environ['RNODE'])
if sys.argv[3] == 'fit':          # node radius so the s-node atom reproduces -sum(IE); nothing else fitted
    lo, hi = float(sys.argv[4]), float(sys.argv[5])
    for _ in range(9):
        p['node'] = 0.5*(lo+hi); E = build(p, 2.2, h, False).scf()['E']; print('  rnode %.4f E %.6f' % (p['node'], E), flush=True)
        if E < p['target']: lo = p['node']
        else: hi = p['node']
    print('rnode* %.4f' % (0.5*(lo+hi))); sys.exit()
for R in [float(x) for x in sys.argv[3].split(',')]:
    m = build(p, R, h, True); e = m.scf(); ea = build(p, R, m.h, False).scf()['E']
    eH = Linear([('H', R, 1, 0)], [('H', 1)], [], h=m.h, L=12.0, hfix=(0.0 if os.environ.get('FIXH') else R), name='H').scf()['E']
    print('%sH R=%.3f h=%.5f E=%.6f | %s %.6f (target %.4f) | H %.6f | D_e %.5f' % (
        el, R, m.h, e['E'], el, ea, p['target'], eH, ea + eH - e['E']), flush=True)
