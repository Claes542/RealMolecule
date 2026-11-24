"""Be2 (adapted from beh2_s.py): two s-node Be, 2hom 2s^2 each, midplane free (s) or evanescent Robin (sp).
BeH2 with an s-ONLY NODE (axisym): Be = frozen-core potential (+4 screened by hydrogenic 1s^2, zeta 3.6848):
K_Be(d) = 2/d + 2 exp(-2 zeta d)(zeta + 1/d). Valence outside the HF 2s node sphere (0.590), two 2hom Be-H pairs, plane
through Be FREE (Neumann). Each pair: area-MEAN of u over its own part of the node sphere = 0 (its 1s-projection
vanishes; p-like parts free). Sphere drawn in by h/2 so the constrained cell centres sit on the node. Nothing fitted:
E(Be) checked against -(IE1+IE2) = -1.0118.   usage: beh2_s.py h s|dir R1,R2,..."""
import sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_linear import Linear
import os
ZETA, RNODE_HF = 3.6848, 0.590
RNODE = float(os.environ.get('RNODE', RNODE_HF))

def sphere_area(m):
    g = m.g; nz, nr, h = g.nz, g.nr, g.h; core = ~m.ok.reshape(nz, nr); mk = m.ok.reshape(nz, nr)
    rr = g.r[None, :]*np.ones((nz, 1)); jj = np.arange(nr)[None, :]*np.ones((nz, 1)); A = np.zeros((nz, nr))
    A[1:, :] += np.where(mk[1:, :] & core[:-1, :], 2*np.pi*rr[1:, :]*h, 0)
    A[:-1, :] += np.where(mk[:-1, :] & core[1:, :], 2*np.pi*rr[:-1, :]*h, 0)
    A[:, 1:] += np.where(mk[:, 1:] & core[:, :-1], 2*np.pi*jj[:, 1:]*h*h, 0)
    A[:, :-1] += np.where(mk[:, :-1] & core[:, 1:], 2*np.pi*(jj[:, :-1]+1)*h*h, 0)
    return A.ravel()

class BS(Linear):
    mode = 's'
    def robin_faces(self, M):
        if self.mode == 'sp': return Linear.robin_faces(self, M)        # s-node + uniform evanescent beta on the plane
        if self.mode != 'dir': return []
        return [2.0/self.g.h*sphere_area(self)*m for m in M]
    def eig(self, S, Veff, keep, u0, lam0=None):
        if self.mode not in ('s', 'sp'): return Linear.eig(self, S, Veff, keep, u0, lam0)
        A = (self.W @ (S + sp.diags(Veff*self.g.V)).tocsr() @ self.W).tocsr()[keep][:, keep]
        c = (sphere_area(self)*self.w)[keep]; c = c/np.linalg.norm(c); Pj = lambda x: x - c*(c @ x)
        op = LinearOperator(A.shape, matvec=lambda x: Pj(A @ Pj(x)) + 1e3*c*(c @ x), dtype=float)
        v0 = None if u0 is None else Pj((u0/self.w)[keep])
        val, vec = eigsh(op, k=1, which='SA', tol=1e-10, maxiter=400000, v0=v0)
        y = np.zeros(self.g.N); y[keep] = Pj(vec[:, 0]); u = self.w*y; u /= np.sqrt(np.sum(u*u*self.g.V))
        return (-u if np.sum(u) < 0 else u), float(val[0])

def build(R, h, mol, mode):
    rcx = RNODE - (0.5*h if mode in ('s', 'sp') else 0.0)
    if mol: m = BS([('Be', -0.5*R, 2, rcx), ('Be', 0.5*R, 2, rcx)], [('BeL', 2), ('BeR', 2)], [(0.0, 'fixed')],
                   h=h, L=14.0, hfix=0.5*R, name='be2s')
    else:   m = BS([('Be', 0.0, 2, rcx)], [('Be', 2)], [], h=h, L=14.0, hfix=0.0, name='Be')
    m.mode = mode
    for z0 in ([-0.5*R, 0.5*R] if mol else [0.0]):
        d = np.maximum(np.sqrt((m.zc - z0)**2 + m.g.R.ravel()**2), 1e-12)
        m.K = m.K + 2*np.exp(-2*ZETA*d)*(ZETA + 1/d)
    return m

h, mode = float(sys.argv[1]), sys.argv[2]
if mode == 'fit':          # bisect the node radius so the s-node atom reproduces -IE (nothing else fitted)
    T, lo, hi = -1.0118, float(sys.argv[3]), float(sys.argv[4])
    for _ in range(9):
        RNODE = 0.5*(lo+hi); E = (build(2.5, h, False, 's') if 'withH' in build.__code__.co_varnames else build(2.5, h, False, 's')).scf()['E']
        print('  rnode %.4f E %.6f' % (RNODE, E), flush=True)
        if E < T: lo = RNODE
        else: hi = RNODE
    print('rnode* %.4f' % (0.5*(lo+hi))); sys.exit()
def eps_mean(m):
    V = m.g.V; P = [m.P(x) for x in m.u]; r = []
    for k, x in enumerate(m.u):
        Ve = -m.K + (m.domains[k]['n']-1)*P[k] + sum(m.domains[l]['n']*P[l] for l in range(len(m.u)) if l != k)
        Sb = m.S[k] + (sp.diags(0.5*m.bvec[k]) if m.bvec else 0)
        r.append(float(x @ (Sb @ x)) + float(np.sum(Ve*x*x*V)))
    return float(np.mean(r))
def solve_sp(R, h):
    b = 1.0; u = None
    for outer in range(30):
        m = build(R, h, True, 'sp'); m.beta = b; m.beta_thru = b
        if u is not None: m.u = u
        e = m.scf(); u = m.u; bn = np.sqrt(-2*eps_mean(m))
        if abs(bn - b) < 1e-4: break
        b = bn
    return m, e, b
for R in [float(x) for x in sys.argv[3].split(',')]:
    if mode == 'sp': m, e, bsp = solve_sp(R, h)
    else: m = build(R, h, True, mode); e = m.scf(); bsp = 0.0
    ea = build(R, m.h, False, 's').scf()          # atom: no plane, s-node atom, same h
    print('%s R=%.3f h=%.5f E=%.6f beta_mid %.4f | Be %.6f (target -1.0118) | D_e %.5f' % (
        mode, R, m.h, e['E'], bsp, ea['E'], 2*ea['E'] - e['E']), flush=True)
