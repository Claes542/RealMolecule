"""2s NODE COST, pseudo-core BeH2: the valence must VANISH on the core sphere (Dirichlet, u = 0 on faces onto r_c cells:
the kinetic price of the node that orthogonality to the 1s forces), all other walls free (plane through Be Neumann).
Discrete Dirichlet on a face half a cell away = Robin with beta = 2/h (0.5*beta*A = A/h).
r_c refit so the pseudo-Be (one 2hom valence domain) keeps E = TARGET.
usage: beh2_node.py h fit TARGET lo hi   |   beh2_node.py h rc R1,R2,..."""
import sys, numpy as np; sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_linear import Linear, H_atom

class ND(Linear):
    def robin_faces(self, M):
        g = self.g; nz, nr, h = g.nz, g.nr, g.h; core = ~self.ok.reshape(nz, nr); bD = 2.0/h
        rr = g.r[None, :]*np.ones((nz, 1)); jj = np.arange(nr)[None, :]*np.ones((nz, 1)); out = []
        for k, m in enumerate(M):
            mk = m.reshape(nz, nr); A = np.zeros((nz, nr))
            A[1:, :] += np.where(mk[1:, :] & core[:-1, :], 2*np.pi*rr[1:, :]*h, 0)
            A[:-1, :] += np.where(mk[:-1, :] & core[1:, :], 2*np.pi*rr[:-1, :]*h, 0)
            A[:, 1:] += np.where(mk[:, 1:] & core[:, :-1], 2*np.pi*jj[:, 1:]*h*h, 0)
            A[:, :-1] += np.where(mk[:, :-1] & core[:, 1:], 2*np.pi*(jj[:, :-1]+1)*h*h, 0)
            out.append(bD*A.ravel())
        return out

def atom(h, rc):
    a = ND([('Be', 0.0, 2, rc)], [('Be', 2)], [], h=h, L=12.0, hfix=0.0, name='Be'); e = a.scf()
    return e['E']                       # the Dirichlet face term (in Esurf) IS the node's kinetic energy: keep it
h = float(sys.argv[1])
if sys.argv[2] == 'fit':
    T, lo, hi = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    for _ in range(9):
        mid = 0.5*(lo+hi); E = atom(h, mid); print('  rc %.4f E %.6f' % (mid, E), flush=True)
        if E < T: lo = mid
        else: hi = mid
    print('rc* %.4f' % (0.5*(lo+hi))); sys.exit()
rc = float(sys.argv[2])
for R in [float(x) for x in sys.argv[3].split(',')]:
    m = ND([('H', -R, 1, 0), ('Be', 0.0, 2, rc), ('H', R, 1, 0)], [('BeHL', 2), ('BeHR', 2)], [(0.0, 'fixed')],
           h=h, L=12.0, hfix=R, name='beh2node'); e = m.scf(); E = e['E']
    ea = atom(m.h, rc); eH = H_atom(m.h, 12.0, z0=R).scf()['E']
    print('R=%.3f h=%.5f E=%.6f | Be %.6f | H %.6f | atomisation %.5f' % (R, m.h, E, ea, eH, ea + 2*eH - E), flush=True)
