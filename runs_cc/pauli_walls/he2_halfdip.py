# RHF He2: electron count and dipole of the half-space z>0 about its own nucleus at z=+R/2 (midplane partition)
import numpy as np
from pyscf import gto, scf, dft
for R in (2.0, 3.0, 4.0, 6.0, 10.0):
    mol = gto.M(atom=[('He',(0,0,-R/2)),('He',(0,0,R/2))], unit='Bohr', basis='aug-cc-pvqz', verbose=0)
    mf = scf.RHF(mol).run(); g = dft.gen_grid.Grids(mol); g.level = 6; g.build()
    rho = dft.numint.NumInt().get_rho(mol, mf.make_rdm1(), g)
    w = g.weights*rho; z = g.coords[:,2]; m = z > 0
    N = w[m].sum(); shift = (w[m]*(z[m]-R/2)).sum()      # >0 = electrons pushed OUTWARD (away from partner)
    print("R=%5.1f  N(half)=%.5f  sum of electron displacements %+.4f au  (atom dipole %+.4f au)" % (R, N, shift, -shift), flush=True)
