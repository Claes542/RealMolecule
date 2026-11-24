"""ALL-ELECTRON BeH2 / Be on the axisym grid (tools/axisym_linear.Linear): Be = +4 nucleus with its 1s^2 as an explicit
2hom CORE domain (sphere radius Rc, nucleus inside), two 2hom Be-H valence pairs (half-spaces), bare protons, plane
through Be fixed. No pseudo-kernel, no r_c.
WALL RULE  neumann : every wall free (du/dn = 0)
           eps     : every wall here is BETWEEN PAIRS (valence|valence plane, valence|core sphere), so each domain k
                     carries Robin du/dn + beta_k u = 0 on all its walls, beta_k = sqrt(-2 eps_k), its OWN eigenvalue,
                     iterated to self-consistency.
usage: beh2_allelec.py atomrelax h Rc0                 -> Be atom, core radius by density continuity (Neumann)
       beh2_allelec.py scan h Rc neumann|eps R1,R2,...  -> BeH2 + refs (Be same Rc and rule, H)"""
import sys, numpy as np, scipy.sparse as sp; sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_linear import Linear, H_atom

class LE(Linear):
    betas = None
    def robin_faces(self, M):
        if self.betas is None: return []
        g = self.g; nz, nr, h = g.nz, g.nr, g.h
        rr = g.r[None, :] * np.ones((nz, 1)); jj = np.arange(nr)[None, :] * np.ones((nz, 1)); out = []
        for k, m in enumerate(M):
            mk = m.reshape(nz, nr); other = np.zeros((nz, nr), bool)
            for l, ml in enumerate(M):
                if l != k: other |= ml.reshape(nz, nr)
            A = np.zeros((nz, nr))
            A[1:, :] += np.where(mk[1:, :] & other[:-1, :], 2*np.pi*rr[1:, :]*h, 0)
            A[:-1, :] += np.where(mk[:-1, :] & other[1:, :], 2*np.pi*rr[:-1, :]*h, 0)
            A[:, 1:] += np.where(mk[:, 1:] & other[:, :-1], 2*np.pi*jj[:, 1:]*h*h, 0)
            A[:, :-1] += np.where(mk[:, :-1] & other[:, 1:], 2*np.pi*(jj[:, :-1]+1)*h*h, 0)
            bk = self.betas[k]
            out.append((bk if np.ndim(bk) else float(bk))*A.ravel())      # scalar or per-cell beta
        return out

def epsk(m):
    V = m.g.V; P = [m.P(x) for x in m.u]; r = []
    for k, x in enumerate(m.u):
        Ve = -m.K + (m.domains[k]['n']-1)*P[k] + sum(m.domains[l]['n']*P[l] for l in range(len(m.u)) if l != k)
        Sb = m.S[k] + (sp.diags(0.5*m.bvec[k]) if m.bvec else 0)
        r.append(float(x @ (Sb @ x)) + float(np.sum(Ve*x*x*V)))
    return r

def veff(m):
    P = [m.P(x) for x in m.u]
    return [-m.K + (m.domains[k]['n']-1)*P[k] + sum(m.domains[l]['n']*P[l] for l in range(len(m.u)) if l != k) for k in range(len(m.u))]

def solve(mk, rule):
    m = mk(); e = m.scf()
    if rule == 'neumann': return m, e, None
    if rule == 'local':      # beta_k(x) = sqrt(2 max(V_k(x) - eps_k, 0)) on the domain's own wall cells (local evanescence)
        def field(m): ep = epsk(m); Vs = veff(m); return [np.sqrt(2*np.maximum(V - x, 0.0)) for V, x in zip(Vs, ep)]
        b = field(m); Eo = e['E']
        for outer in range(60):
            u = m.u; m = mk(); m.betas = b; m.u = u; e = m.scf(); bn = field(m)
            if abs(e['E'] - Eo) < 1e-6: break
            b = [0.5*p + 0.5*q for p, q in zip(b, bn)]; Eo = e['E']
        M = m.masks(); A = m.robin_faces(M)        # report the area-weighted mean beta per domain on its walls
        ar = [float(np.sum(a > 0)) for a in A]
        return m, e, [float(np.sum(bb*(a > 0))/max(c, 1)) for bb, a, c in zip(b, A, ar)]
    b = [np.sqrt(max(-2*x, 0.0)) for x in epsk(m)]
    for outer in range(40):
        u = m.u; m = mk(); m.betas = b; m.u = u; e = m.scf()
        bn = [np.sqrt(max(-2*x, 0.0)) for x in epsk(m)]
        if max(abs(p - q) for p, q in zip(b, bn)) < 1e-5: break
        b = bn
    return m, e, b

def atom(h, Rc, rule):
    return solve(lambda: LE([('Be', 0.0, 4, 0)], [], [], h=h, L=12.0, hfix=0.0, name='Be', cores=[('Be1s', 0, 2, Rc)])
                 if False else LE([('Be', 0.0, 4, 0)], [('Bev', 2)], [], h=h, L=12.0, hfix=0.0, name='Be',
                                  cores=[('Be1s', 0, 2, Rc)]), rule)

mode = sys.argv[1]; h = float(sys.argv[2])
if mode == 'atomrelax':
    m = LE([('Be', 0.0, 4, 0)], [('Bev', 2)], [], h=h, L=12.0, hfix=0.0, name='Be', cores=[('Be1s', 0, 2, float(sys.argv[3]))])
    r = m.relax(verbose=False)
    print('Be atom neumann h=%.3f  Rc=%.4f  E=%.6f  (exact -14.6674, HF -14.5730)' % (h, m.rcore(0), r['E'])); sys.exit()
Rc, rule = float(sys.argv[3]), sys.argv[4]
for R in [float(x) for x in sys.argv[5].split(',')]:
    mk = lambda: LE([('H', -R, 1, 0), ('Be', 0.0, 4, 0), ('H', R, 1, 0)], [('BeHL', 2), ('BeHR', 2)], [(0.0, 'fixed')],
                    h=h, L=12.0, hfix=R, name='beh2ae', cores=[('Be1s', 1, 2, Rc)])
    m, e, b = solve(mk, rule); a, ea, ba = atom(m.h, Rc, rule); eH = H_atom(m.h, 12.0, z0=R).scf()['E']
    print('%s R=%.3f h=%.5f Rc=%.3f E=%.6f beta %s Esurf %.5f | Be %.6f beta %s | H %.6f | atomisation %.5f' % (
        rule, R, m.h, m.rcore(0), e['E'], 'none' if b is None else ','.join('%.3f' % x for x in b), e['Esurf'],
        ea['E'], 'none' if ba is None else ','.join('%.3f' % x for x in ba), eH, ea['E'] + 2*eH - e['E']), flush=True)
