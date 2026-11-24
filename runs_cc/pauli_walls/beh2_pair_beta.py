"""BeH2 as two 2hom Be-H bond pairs (axisym, tools/axisym_linear.Linear): pseudo-Be +2 (rc), bare protons at +-R, one plane
through Be (fixed) = the wall BETWEEN the two pairs, Robin beta_thru on it. Atom refs: Be +2 one 2hom domain (no wall),
H_atom. usage: python3 beh2_pair_beta.py h rc beta R1,R2,..."""
import sys; sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_linear import Linear, H_atom
h, rc, beta = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]); Rs = [float(x) for x in sys.argv[4].split(',')]
for R in Rs:
    m = Linear([('H', -R, 1, 0), ('Be', 0.0, 2, rc), ('H', R, 1, 0)], [('BeHL', 2), ('BeHR', 2)], [(0.0, 'fixed')],
               h=h, L=12.0, hfix=R, name='beh2pair', beta=beta, beta_thru=beta)
    E = m.scf()['E']
    a = Linear([('Be', 0.0, 2, rc)], [('Be', 2)], [], h=m.h, L=12.0, hfix=0.0, name='Be').scf()['E']
    eH = H_atom(m.h, 12.0, z0=R)
    eH = eH if isinstance(eH, float) else eH['E'] if isinstance(eH, dict) else eH.scf()['E']
    print('beta=%.2f R=%.3f h=%.5f E=%.6f  Be %.6f H %.6f  atomisation %.5f Ha' % (beta, R, m.h, E, a, eH, a + 2*eH - E), flush=True)
