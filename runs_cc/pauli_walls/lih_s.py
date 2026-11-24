"""LiH with an s-ONLY NODE (axisym): Li = frozen-core potential (+3 screened by hydrogenic 1s^2, zeta 2.6906); valence
domain = outside the HF 2s node sphere r_node 0.829 (hard exclusion, masked = Neumann). On the sphere ONE constraint:
the area-weighted MEAN of u over the sphere = 0 (the l=0 part vanishes = orthogonality to the 1s; p-like parts free).
mode: s (mean-zero constraint) | dir (u = 0 on the whole sphere) | free (Neumann only). Nothing fitted.
usage: lih_s.py h s|dir|free R1,R2,..."""
import sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_linear import Linear
import os, realcore
import os
ZETA, RNODE_HF = 2.6906, 0.829
RNODE = float(os.environ.get('RNODE', RNODE_HF))

def sphere_area(m):
    g = m.g; nz, nr, h = g.nz, g.nr, g.h; core = ~m.ok.reshape(nz, nr); mk = m.ok.reshape(nz, nr)
    rr = g.r[None, :]*np.ones((nz, 1)); jj = np.arange(nr)[None, :]*np.ones((nz, 1)); A = np.zeros((nz, nr))
    A[1:, :] += np.where(mk[1:, :] & core[:-1, :], 2*np.pi*rr[1:, :]*h, 0)
    A[:-1, :] += np.where(mk[:-1, :] & core[1:, :], 2*np.pi*rr[:-1, :]*h, 0)
    A[:, 1:] += np.where(mk[:, 1:] & core[:, :-1], 2*np.pi*jj[:, 1:]*h*h, 0)
    A[:, :-1] += np.where(mk[:, :-1] & core[:, 1:], 2*np.pi*(jj[:, :-1]+1)*h*h, 0)
    return A.ravel()

class LS(Linear):
    mode = 'free'
    def robin_faces(self, M):
        if self.mode != 'dir': return []
        return [2.0/self.g.h*sphere_area(self)*m for m in M]
    def eig(self, S, Veff, keep, u0, lam0=None):
        if self.mode != 's': return Linear.eig(self, S, Veff, keep, u0, lam0)
        A = (self.W @ (S + sp.diags(Veff*self.g.V)).tocsr() @ self.W).tocsr()[keep][:, keep]
        c = (sphere_area(self)*self.w)[keep]; c = c/np.linalg.norm(c)              # constraint sum A_j u_j = 0, u = w y
        Pj = lambda x: x - c*(c @ x)
        op = LinearOperator(A.shape, matvec=lambda x: Pj(A @ Pj(x)) + 1e3*c*(c @ x), dtype=float)
        v0 = None if u0 is None else Pj((u0/self.w)[keep])
        val, vec = eigsh(op, k=1, which='SA', tol=1e-10, maxiter=400000, v0=v0)
        y = np.zeros(self.g.N); y[keep] = Pj(vec[:, 0]); u = self.w*y; u /= np.sqrt(np.sum(u*u*self.g.V))
        return (-u if np.sum(u) < 0 else u), float(val[0])

def build(R, h, withH, mode):
    zLi = 0.0 if os.environ.get('FIXH') else -R/2; zH = zLi + R; rcx = RNODE - (0.5*h if mode == 's' else 0.0)     # s: constraint acts on cell centres -> sphere in by h/2
    nuc = [('Li', zLi, 1, rcx)] + ([('H', zH, 1, 0)] if withH else [])
    m = LS(nuc, [('val', 2 if withH else 1)], [], h=h, L=16.0, hfix=(0.0 if os.environ.get('FIXH') else R/2), name='lihs'); m.mode = mode
    d = np.maximum(np.sqrt((m.zc - zLi)**2 + m.g.R.ravel()**2), 1e-12)
    m.K = m.K + (realcore.screen(3)(d) if os.environ.get('CORE') == 'real' else 2*np.exp(-2*ZETA*d)*(ZETA + 1/d)); return m

h, mode = float(sys.argv[1]), sys.argv[2]
if mode == 'fit':          # bisect the node radius so the s-node atom reproduces -IE (nothing else fitted)
    T, lo, hi = -0.19814, float(sys.argv[3]), float(sys.argv[4])
    for _ in range(9):
        RNODE = 0.5*(lo+hi); E = (build(3.0, h, False, 's') if 'withH' in build.__code__.co_varnames else build(3.0, h, False, 's')).scf()['E']
        print('  rnode %.4f E %.6f' % (RNODE, E), flush=True)
        if E < T: lo = RNODE
        else: hi = RNODE
    print('rnode* %.4f' % (0.5*(lo+hi))); sys.exit()
import axisym_linear; axisym_linear.SHIFT_INVERT = getattr(axisym_linear, 'SHIFT_INVERT', True)
for R in [float(x) for x in sys.argv[3].split(',')]:
    m = build(R, h, True, mode); e = m.scf(); a = build(R, m.h, False, mode); ea = a.scf()
    eH = Linear([('H', (R if os.environ.get('FIXH') else R/2), 1, 0)], [('H', 1)], [], h=m.h, L=16.0, hfix=(0.0 if os.environ.get('FIXH') else R/2), name='H').scf()['E']
    us = sphere_area(m); mean = float(np.sum(us*m.u[0])/np.sum(us))
    print('%s R=%.3f h=%.5f E=%.6f | Li %.6f (IE -0.19814) | H %.6f | D_e %.5f | <u>sphere %.2e' % (
        mode, R, m.h, e['E'], ea['E'], eH, ea['E'] + eH - e['E'], mean), flush=True)
