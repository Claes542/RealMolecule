"""He2 interaction energy, stdQM reference: RHF and CCSD(T), aug-cc-pVQZ, counterpoise-corrected (Ha)."""
from pyscf import gto, scf, cc
def e(atoms):
    mol = gto.M(atom=atoms, unit='Bohr', basis='aug-cc-pvqz', verbose=0); mf = scf.RHF(mol).run()
    c = cc.CCSD(mf).run(); return mf.e_tot, c.e_tot + c.ccsd_t()
for R in (2.0, 3.0, 4.0, 5.0, 5.6, 6.0, 8.0):
    d = e([('He', (0, 0, 0)), ('He', (0, 0, R))]); a = e([('He', (0, 0, 0)), ('ghost-He', (0, 0, R))])
    print("R=%.1f  RHF %+.6f  CCSD(T) %+.7f" % (R, d[0] - 2*a[0], d[1] - 2*a[1]), flush=True)
