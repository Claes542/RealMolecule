import numpy as np
from pyscf import gto, scf, dft
at = gto.M(atom=[('He',(0,0,0))], unit='Bohr', basis='aug-cc-pvqz', verbose=0); ma = scf.RHF(at).run(); dma = ma.make_rdm1()
for R in (2.0, 3.0, 4.0, 6.0):
    mol = gto.M(atom=[('He',(0,0,-R/2)),('He',(0,0,R/2))], unit='Bohr', basis='aug-cc-pvqz', verbose=0)
    mf = scf.RHF(mol).run(); g = dft.gen_grid.Grids(mol); g.level = 6; g.build(); ni = dft.numint.NumInt()
    rho = ni.get_rho(mol, mf.make_rdm1(), g)
    pro = sum(ni.get_rho(gto.M(atom=[('He',(0,0,c))], unit='Bohr', basis='aug-cc-pvqz', verbose=0), dma, g) for c in (-R/2, R/2))
    z = g.coords[:,2]; m = z > 0
    d = lambda r: (g.weights*r)[m] @ (z[m]-R/2)
    print("R=%4.1f  RHF %+.4f  free atoms superposed %+.4f  distortion %+.4f au" % (R, d(rho), d(pro), d(rho)-d(pro)), flush=True)
