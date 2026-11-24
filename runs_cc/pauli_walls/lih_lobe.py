"""LiH with a PHYSICAL 2s NODE and a FIXED INNER LOBE (axisym). Li = frozen-core potential (+3 nucleus screened by a
hydrogenic 1s^2, zeta 2.6906): K_Li(d) = 1/d + 2 exp(-2 zeta d)(zeta + 1/d). Valence split by the HF 2s node sphere
(r_node 0.829, Li atom) into an INNER domain (n = q, the inner-lobe population, around the nucleus) and the OUTER valence
(n = 1 - q in Li; 2 - q as the LiH pair over a bare proton). u = 0 on the node sphere from BOTH sides (Dirichlet =
Robin 2/h). Nothing fitted: E(Li) is a CHECK against -IE = -0.19814.
usage: lih_lobe.py h q R1,R2,...   (q = 0.012 from HF; q ~ 0 = node only)"""
import sys, numpy as np; sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_linear import Linear
ZETA, RNODE = 2.6906, 0.829

class LL(Linear):
    def robin_faces(self, M):
        g = self.g; nz, nr, h = g.nz, g.nr, g.h; bD = 2.0/h
        rr = g.r[None, :]*np.ones((nz, 1)); jj = np.arange(nr)[None, :]*np.ones((nz, 1)); out = []
        for k, m in enumerate(M):
            mk = m.reshape(nz, nr); oth = np.zeros((nz, nr), bool)
            for l, ml in enumerate(M):
                if l != k: oth |= ml.reshape(nz, nr)
            A = np.zeros((nz, nr))
            A[1:, :] += np.where(mk[1:, :] & oth[:-1, :], 2*np.pi*rr[1:, :]*h, 0)
            A[:-1, :] += np.where(mk[:-1, :] & oth[1:, :], 2*np.pi*rr[:-1, :]*h, 0)
            A[:, 1:] += np.where(mk[:, 1:] & oth[:, :-1], 2*np.pi*jj[:, 1:]*h*h, 0)
            A[:, :-1] += np.where(mk[:, :-1] & oth[:, 1:], 2*np.pi*(jj[:, :-1]+1)*h*h, 0)
            out.append(bD*A.ravel())
        return out

def build(R, h, q, with_H, hfix):
    zLi = -R/2
    nuc = [('Li', zLi, 1, 0)] + ([('H', R/2, 1, 0)] if with_H else [])
    nval = (2.0 if with_H else 1.0) - q
    m = LL(nuc, [('val', nval)], [], h=h, L=16.0, hfix=hfix, name='lih', cores=[('lobe', 0, q, RNODE)])
    m.domains[0]['n'] = nval; m.domains[1]['n'] = q; m.cores[0]['n'] = q      # Linear casts n to int: restore fractions
    d = np.sqrt((m.zc - zLi)**2 + m.g.R.ravel()**2); d = np.maximum(d, 1e-12)
    m.K = m.K + 2*np.exp(-2*ZETA*d)*(ZETA + 1/d)                     # frozen 1s^2 screening term (attraction)
    return m

h, q = float(sys.argv[1]), float(sys.argv[2])
for R in [float(x) for x in sys.argv[3].split(',')]:
    m = build(R, h, q, True, R/2); e = m.scf()
    a = build(R, m.h, q, False, R/2); ea = a.scf()
    hm = Linear([('H', R/2, 1, 0)], [('H', 1)], [], h=m.h, L=16.0, hfix=R/2, name='H'); eH = hm.scf()['E']
    print('q=%.3f R=%.3f h=%.5f rnode=%.3f E=%.6f | Li %.6f (IE check -0.19814) | H %.6f | D_e %.5f' % (
        q, R, m.h, m.rcore(0), e['E'], ea['E'], eH, ea['E'] + eH - e['E']), flush=True)
