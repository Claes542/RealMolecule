"""PSEUDO-core BeH2 (Be +2, hard r_c, two 2hom Be-H pairs, plane through Be) with the LOCAL evanescent wall rule on every
wall of a valence pair: beta(x) = sqrt(2 max(V_k(x) - eps_k, 0)) per wall cell, self-consistent. Walls = the plane (onto
the other pair) and the core sphere (onto r_c cells). Reports the mean beta on each kind separately. Be ref: one 2hom
valence domain, same rule (core sphere only).  usage: beh2_pseudo_local.py h rc R1,R2,..."""
import sys, numpy as np, scipy.sparse as sp; sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_linear import Linear, H_atom

def faces(m, M):
    """per domain: (area onto another domain, area onto core cells), per cell"""
    g = m.g; nz, nr, h = g.nz, g.nr, g.h; core = ~m.ok.reshape(nz, nr)
    rr = g.r[None, :]*np.ones((nz, 1)); jj = np.arange(nr)[None, :]*np.ones((nz, 1)); out = []
    for k, mm in enumerate(M):
        mk = mm.reshape(nz, nr); oth = np.zeros((nz, nr), bool)
        for l, ml in enumerate(M):
            if l != k: oth |= ml.reshape(nz, nr)
        res = []
        for X in (oth, core):
            A = np.zeros((nz, nr))
            A[1:, :] += np.where(mk[1:, :] & X[:-1, :], 2*np.pi*rr[1:, :]*h, 0)
            A[:-1, :] += np.where(mk[:-1, :] & X[1:, :], 2*np.pi*rr[:-1, :]*h, 0)
            A[:, 1:] += np.where(mk[:, 1:] & X[:, :-1], 2*np.pi*jj[:, 1:]*h*h, 0)
            A[:, :-1] += np.where(mk[:, :-1] & X[:, 1:], 2*np.pi*(jj[:, :-1]+1)*h*h, 0)
            res.append(A.ravel())
        out.append(res)
    return out

class PL(Linear):
    bfield = None
    def robin_faces(self, M):
        if self.bfield is None: return []
        return [b*(Ao + Ac) for b, (Ao, Ac) in zip(self.bfield, faces(self, M))]

def local(m):
    V = m.g.V; P = [m.P(x) for x in m.u]; F = []
    for k, x in enumerate(m.u):
        Ve = -m.K + (m.domains[k]['n']-1)*P[k] + sum(m.domains[l]['n']*P[l] for l in range(len(m.u)) if l != k)
        Sb = m.S[k] + (sp.diags(0.5*m.bvec[k]) if m.bvec else 0)
        ep = float(x @ (Sb @ x)) + float(np.sum(Ve*x*x*V)); F.append(np.sqrt(2*np.maximum(Ve - ep, 0.0)))
    return F

def solve(mk):
    m = mk(); e = m.scf(); b = local(m); Eo = e['E']
    for outer in range(60):
        u = m.u; m = mk(); m.bfield = b; m.u = u; e = m.scf(); bn = local(m)
        if abs(e['E'] - Eo) < 1e-6: break
        b = [0.5*p + 0.5*q for p, q in zip(b, bn)]; Eo = e['E']
    fc = faces(m, m.masks()); Ao, Ac = fc[0]
    mo = float(np.sum(b[0]*Ao)/max(np.sum(Ao), 1e-30)); mc = float(np.sum(b[0]*Ac)/max(np.sum(Ac), 1e-30))
    return m, e, mo, mc, float(b[0][Ac > 0].max()) if (Ac > 0).any() else 0.0

h, rc = float(sys.argv[1]), float(sys.argv[2])
for R in [float(x) for x in sys.argv[3].split(',')]:
    m, e, mo, mc, cmax = solve(lambda: PL([('H', -R, 1, 0), ('Be', 0.0, 2, rc), ('H', R, 1, 0)], [('BeHL', 2), ('BeHR', 2)],
                                          [(0.0, 'fixed')], h=h, L=12.0, hfix=R, name='beh2pl'))
    a, ea, _, ac, acmax = solve(lambda: PL([('Be', 0.0, 2, rc)], [('Be', 2)], [], h=m.h, L=12.0, hfix=0.0, name='Be'))
    eH = H_atom(m.h, 12.0, z0=R).scf()['E']
    print('R=%.3f h=%.5f E=%.6f  beta plane %.3f core %.4f (max %.4f) Esurf %.5f | Be %.6f core beta %.4f (max %.4f) | atomisation %.5f'
          % (R, m.h, e['E'], mo, mc, cmax, e['Esurf'], ea['E'], ac, acmax, ea['E'] + 2*eH - e['E']), flush=True)
