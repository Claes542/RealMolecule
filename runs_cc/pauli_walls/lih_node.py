"""LiH, axisym: pseudo-Li (+1 point charge, sphere r_c) + ONE 2hom valence pair (Li 2s + H 1s) over a bare proton. Only wall:
the core sphere. WALL free (Neumann) | node (valence u = 0 on the sphere, Dirichlet = Robin 2/h).
Li ref: one electron outside the same sphere, same rule. r_c fitted so E(pseudo-Li) = -IE(Li) = TARGET.
usage: lih_node.py h free|node fit TARGET lo hi   |   lih_node.py h free|node rc R1,R2,..."""
import sys, numpy as np; sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_linear import Linear, H_atom

class LN(Linear):
    node = False
    def robin_faces(self, M):
        if not self.node: return []
        g = self.g; nz, nr, h = g.nz, g.nr, g.h; core = ~self.ok.reshape(nz, nr); bD = 2.0/h
        rr = g.r[None, :]*np.ones((nz, 1)); jj = np.arange(nr)[None, :]*np.ones((nz, 1)); out = []
        for m in M:
            mk = m.reshape(nz, nr); A = np.zeros((nz, nr))
            A[1:, :] += np.where(mk[1:, :] & core[:-1, :], 2*np.pi*rr[1:, :]*h, 0)
            A[:-1, :] += np.where(mk[:-1, :] & core[1:, :], 2*np.pi*rr[:-1, :]*h, 0)
            A[:, 1:] += np.where(mk[:, 1:] & core[:, :-1], 2*np.pi*jj[:, 1:]*h*h, 0)
            A[:, :-1] += np.where(mk[:, :-1] & core[:, 1:], 2*np.pi*(jj[:, :-1]+1)*h*h, 0)
            out.append(bD*A.ravel())
        return out

def mk(nuc, dom, h, hfix, node, L):
    m = LN(nuc, dom, [], h=h, L=L, hfix=hfix, name='lih'); m.node = node; return m

h, wall = float(sys.argv[1]), sys.argv[2]; node = wall == 'node'; L = 16.0
def atom(hh, rc): return mk([('Li', 0.0, 1, rc)], [('Li', 1)], hh, 0.0, node, L).scf()['E']
if sys.argv[3] == 'fit':
    T, lo, hi = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
    for _ in range(9):
        mid = 0.5*(lo+hi); E = atom(h, mid); print('  rc %.4f E %.6f' % (mid, E), flush=True)
        if E < T: lo = mid
        else: hi = mid
    print('rc* %.4f' % (0.5*(lo+hi))); sys.exit()
rc = float(sys.argv[3])
for R in [float(x) for x in sys.argv[4].split(',')]:
    m = mk([('Li', -R/2, 1, rc), ('H', R/2, 1, 0)], [('LiH', 2)], h, R/2, node, L); E = m.scf()['E']
    ea = mk([('Li', -R/2, 1, rc)], [('Li', 1)], m.h, R/2, node, L).scf()['E']
    eH = mk([('H', R/2, 1, 0)], [('H', 1)], m.h, R/2, node, L).scf()['E']
    print('%s R=%.3f h=%.5f E=%.6f | Li %.6f | H %.6f | D_e %.5f' % (wall, R, m.h, E, ea, eH, ea + eH - E), flush=True)
