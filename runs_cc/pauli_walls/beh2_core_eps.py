"""BeH2 2hom pairs with the EVANESCENT Robin rule on EVERY wall between pairs: the plane through Be (pair|pair) AND the Be
core sphere (pair|1s^2 core pair). beta = sqrt(-2 eps) of the domain, self-consistent. Atom ref: Be +2 with one 2hom
valence domain, same core rule.  usage: beh2_core_eps.py h rc R1,R2,...  |  beh2_core_eps.py h fit TARGET lo hi"""
import sys, numpy as np, scipy.sparse as sp; sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_linear import Linear, H_atom

class LC(Linear):
    bcore = 0.0
    def robin_faces(self, M):
        out = Linear.robin_faces(self, M) or [np.zeros(self.g.N) for _ in M]
        g = self.g; nz, nr, h = g.nz, g.nr, g.h; bad = ~self.ok.reshape(nz, nr)
        rr = g.r[None, :] * np.ones((nz, 1)); jj = np.arange(nr)[None, :] * np.ones((nz, 1))
        for k, m in enumerate(M[:self.nplanar]):
            mk = m.reshape(nz, nr); A = np.zeros((nz, nr))
            A[1:, :] += np.where(mk[1:, :] & bad[:-1, :], 2*np.pi*rr[1:, :]*h, 0)          # z-face below is core
            A[:-1, :] += np.where(mk[:-1, :] & bad[1:, :], 2*np.pi*rr[:-1, :]*h, 0)        # z-face above is core
            A[:, 1:] += np.where(mk[:, 1:] & bad[:, :-1], 2*np.pi*jj[:, 1:]*h*h, 0)        # inner r-face
            A[:, :-1] += np.where(mk[:, :-1] & bad[:, 1:], 2*np.pi*(jj[:, :-1]+1)*h*h, 0)  # outer r-face
            out[k] = out[k] + self.bcore*A.ravel()
        return out

def eps(m):
    V = m.g.V; P = [m.P(x) for x in m.u]; r = []
    for k, x in enumerate(m.u):
        Ve = -m.K + (m.domains[k]['n']-1)*P[k] + sum(m.domains[l]['n']*P[l] for l in range(len(m.u)) if l != k)
        r.append(float(x @ ((m.S[k] + sp.diags(0.5*m.bvec[k])) @ x)) + float(np.sum(Ve*x*x*V)))
    return float(np.mean(r))

def solve(mk):
    beta = 1.0; u = None
    for outer in range(40):
        m = mk(beta)
        if u is not None: m.u = u
        e = m.scf(); u = m.u; bn = np.sqrt(-2*eps(m))
        if abs(bn - beta) < 1e-5: break
        beta = bn
    return m, e, beta

def atom(h, rc):
    def mk(b):
        a = LC([('Be', 0.0, 2, rc)], [('Be', 2)], [], h=h, L=12.0, hfix=0.0, name='Be', beta=1e-12, beta_thru=1e-12)
        a.bcore = b; return a
    return solve(mk)

h = float(sys.argv[1])
if sys.argv[2] == 'fit':
    T, lo, hi = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    for _ in range(10):
        mid = 0.5*(lo+hi); m, e, b = atom(h, mid); print('  rc %.4f E %.6f beta %.4f' % (mid, e['E'], b), flush=True)
        if e['E'] < T: lo = mid
        else: hi = mid
    print('rc* %.4f' % (0.5*(lo+hi))); sys.exit()
rc = float(sys.argv[2])
for R in [float(x) for x in sys.argv[3].split(',')]:
    def mk(b):
        m = LC([('H', -R, 1, 0), ('Be', 0.0, 2, rc), ('H', R, 1, 0)], [('BeHL', 2), ('BeHR', 2)], [(0.0, 'fixed')],
               h=h, L=12.0, hfix=R, name='beh2', beta=b, beta_thru=b)
        m.bcore = b; return m
    m, e, b = solve(mk); a = atom(m.h, rc)[1]['E']; eH = H_atom(m.h, 12.0, z0=R).scf()['E']
    print('R=%.3f h=%.5f E=%.6f beta %.4f Esurf %.5f  Be %.6f H %.6f  atomisation %.5f' % (
        R, m.h, e['E'], b, e['Esurf'], a, eH, a + 2*eH - e['E']), flush=True)
