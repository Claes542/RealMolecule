"""Is the free-wall energy error a FORCE defect, and where along R? (2026-11-25)
H2 2split (one electron per half-space, midplane fixed by symmetry, beta 0), axisym, h FIXED (no hfix: nuclei anywhere
in a cell, so h does not jump with R). The midplane does not move with R, so -dE/dR IS the Hellmann-Feynman force of
this model. Reference: FCI singlet (2 e, aug-cc-pVQZ, near-exact). Output: E(R)-2E(H), force -dE/dR (central
differences) for both, their difference, and k at each minimum.   usage: python3 h2_force_profile.py h"""
import sys, numpy as np
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_linear import Linear, H_atom
from pyscf import gto, scf, fci
h = float(sys.argv[1]); Rs = np.round(np.arange(1.0, 6.01, 0.2), 3)
def rq(R):
    m = Linear([('H', -R/2, 1, 0), ('H', R/2, 1, 0)], [('A', 1), ('B', 1)], [(0.0, 'fixed')], h=h, L=12.0, hfix=0.0, name='h2')
    e = m.scf()['E']
    eH = sum(H_atom(m.h, m.L, 0.0, z0=round((z - m.g.zmin) % m.h, 9)).scf()['E'] for z in (-R/2, R/2))
    return e - eH
B = 'aug-cc-pvqz'; eH = scf.UHF(gto.M(atom='H 0 0 0', basis=B, spin=1, verbose=0)).run().e_tot
def ex(R):
    m = gto.M(atom='H 0 0 0; H 0 0 %f' % R, basis=B, unit='Bohr', verbose=0); mf = scf.RHF(m).run()
    return fci.FCI(mf).kernel()[0] - 2*eH
E1 = np.array([rq(R) for R in Rs]); E2 = np.array([ex(R) for R in Rs])
F1 = -np.gradient(E1, Rs); F2 = -np.gradient(E2, Rs)
print('h=%.3f   R   E_RealQM   E_exact |  F_RealQM  F_exact  F_R-F_ex' % h)
for R, a, b, f, g in zip(Rs, E1, E2, F1, F2): print('  %.2f %+.5f %+.5f | %+.5f %+.5f %+.5f' % (R, a, b, f, g, f - g))
for lab, E in (('RealQM', E1), ('exact', E2)):
    i = int(np.argmin(E)); s = slice(max(i-2, 0), i+3); c = np.polyfit(Rs[s], E[s], 2)
    print('%-7s R_e %.4f  D_e %.5f  k %.4f' % (lab, -c[1]/(2*c[0]), -np.polyval(c, -c[1]/(2*c[0])), 2*c[0]))
# where the excess binding is accumulated: INT (F_R - F_ex) dR from R to the end, i.e. E_ex(R) - E_R(R) relative to R=6
print('excess binding accumulated beyond R (E_ex - E_R):', ' '.join('%.1f:%+.4f' % (R, b - a) for R, a, b in zip(Rs[::5], E1[::5], E2[::5])))
