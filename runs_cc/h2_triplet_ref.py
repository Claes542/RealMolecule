"""H2 triplet reference: UHF (= exact for 2 same-spin e at the HF level, ROHF/UHF triplet) aug-cc-pVQZ, E(R) - 2E(H)."""
from pyscf import gto, scf
B = 'aug-cc-pvqz'; eH = scf.UHF(gto.M(atom='H 0 0 0', basis=B, spin=1, verbose=0)).run().e_tot
for R in (1.4, 2.0, 3.0, 4.0, 5.0):
    m = gto.M(atom='H 0 0 0; H 0 0 %f' % R, basis=B, unit='Bohr', spin=2, verbose=0)
    print('R=%.2f UHF triplet %+.5f' % (R, scf.UHF(m).run().e_tot - 2*eH))
