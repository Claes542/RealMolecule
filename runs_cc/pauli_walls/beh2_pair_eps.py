"""BeH2 as two 2hom Be-H pairs (axisym Linear), Robin on the plane through Be (the wall BETWEEN the pairs) with
beta = sqrt(-2 eps), eps the pair domain's own eigenvalue, iterated to self-consistency (outer loop, warm start).
usage: python3 beh2_pair_eps.py h rc R1,R2,..."""
import sys, numpy as np; sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_linear import Linear, H_atom
h, rc = float(sys.argv[1]), float(sys.argv[2]); Rs = [float(x) for x in sys.argv[3].split(',')]
def eps(m):
    V = m.g.V; P = [m.P(x) for x in m.u]; out = []
    for k, x in enumerate(m.u):
        Ve = -m.K + (m.domains[k]['n'] - 1)*P[k] + sum(m.domains[l]['n']*P[l] for l in range(len(m.u)) if l != k)
        out.append(float(x @ ((m.S[k] + __import__('scipy.sparse', fromlist=['diags']).diags(0.5*m.bvec[k])) @ x)) + float(np.sum(Ve*x*x*V)))
    return out
for R in Rs:
    beta = 1.0
    for outer in range(30):
        m = Linear([('H', -R, 1, 0), ('Be', 0.0, 2, rc), ('H', R, 1, 0)], [('BeHL', 2), ('BeHR', 2)], [(0.0, 'fixed')],
                   h=h, L=12.0, hfix=R, name='beh2pair', beta=beta, beta_thru=beta)
        if outer: m.u = uprev
        e = m.scf(); uprev = m.u; ep = eps(m); bn = np.sqrt(-2*np.mean(ep))
        if abs(bn - beta) < 1e-5: break
        beta = bn
    a = Linear([('Be', 0.0, 2, rc)], [('Be', 2)], [], h=m.h, L=12.0, hfix=0.0, name='Be').scf()['E']
    eH = H_atom(m.h, 12.0, z0=R).scf()['E']
    print('R=%.3f h=%.5f E=%.6f eps %.4f beta %.4f (outer %d) Esurf %.5f  Be %.6f H %.6f  atomisation %.5f Ha' % (
        R, m.h, e['E'], np.mean(ep), beta, outer, e['Esurf'], a, eH, a + 2*eH - e['E']), flush=True)
