"""
Reference dipoles for the RealQM protein-surface fragments: reads the geometry h2_beta logged ('frag NAME built: ...'),
computes HF and B3LYP / aug-cc-pVTZ dipoles about the CENTRE OF KERNEL CHARGE (valence charges C 4, N 5, O 6, H 1 --
molecule.js's origin for ions; for a neutral molecule the origin does not matter). All-electron about that point
equals the valence dipole, since each 1s^2 core sits on its nucleus. usage: frag_ref.py LOG [LOG ...]
"""
import sys, re, numpy as np
from pyscf import gto, scf, dft
ZV = {'C': 4, 'N': 5, 'O': 6, 'H': 1}
for log in sys.argv[1:]:
    line = [l for l in open(log) if 'built: valence e' in l][0]
    name = re.search(r'frag (\w+) built', line).group(1); net = int(re.search(r'net charge (-?\d+)', line).group(1))
    heavy = re.findall(r'([CNO])\(([-\d.]+),([-\d.]+),([-\d.]+)\)', line.split('; H')[0])
    hs = re.findall(r'\(([-\d.]+),([-\d.]+),([-\d.]+)\)', line.split('; H')[1])
    atoms = [(e, tuple(map(float, xyz))) for e, *xyz in heavy] + [('H', tuple(map(float, xyz))) for xyz in hs]
    q = np.array([ZV[a[0]] for a in atoms], float); P = np.array([a[1] for a in atoms])
    org = (q[:, None]*P).sum(0)/q.sum()
    m = gto.M(atom=atoms, unit='Bohr', basis='aug-cc-pvtz', charge=net, verbose=0)
    for lab, mf in (('HF', scf.RHF(m)), ('B3LYP', dft.RKS(m))):
        if lab == 'B3LYP': mf.xc = 'b3lyp'
        mf.conv_tol = 1e-10; mf.kernel()
        dm = mf.make_rdm1()
        with m.with_common_orig(org):
            el = -np.einsum('xij,ji->x', m.intor('int1e_r'), dm)
        nuc = (m.atom_charges()[:, None]*(m.atom_coords() - org)).sum(0)
        mu = el + nuc
        print("%-15s %-5s mu (au) %+.4f %+.4f %+.4f  |mu| %.3f D  mu_x %.3f D" % (name, lab, *mu, np.linalg.norm(mu)*2.5417, mu[0]*2.5417), flush=True)
