"""
RealQM as an ANALYSIS LAYER on an accurate density (combination 2): H2O B3LYP/cc-pVTZ at RealQM's own geometry, the 5
occupied orbitals localised (Boys), space partitioned by RealQM's wall rule -- each point to the localised orbital whose
density is largest there, so walls lie where neighbouring densities are equal -- and the TRUE total density integrated
in each region: charge (RealQM assumes exactly 2 per pair domain) and charge centre. Compared with RealQM's own converged
domains (dump g1_h2o: O+6, two bond pairs, two lone pairs, r_c 0.71). Lengths bohr, origin O, H at (+-1.43, 1.108, 0).
"""
import json, numpy as np
from pyscf import gto, dft, lo
from pyscf.dft import numint
D = '/private/tmp/claude-501/-Users-cgjoh-Development-H2O/410447df-ea95-4b1a-874b-087cf46a7616/scratchpad/dumps'
# ---- RealQM domains from the dump
m = json.load(open(D + '/g1_h2o.meta')); NN, h = m['NN'], m['h']; S = NN + 1; n = S**3
psi = np.fromfile(D + '/g1_h2o.psi', dtype='<f4').astype(float)[:n].reshape(S, S, S)
lab = np.fromfile(D + '/g1_h2o.lab', dtype='<u4')[:n].reshape(S, S, S)
O = np.array([m['atoms'][0][c] for c in 'ijk']); ii, jj, kk = np.meshgrid(*[np.arange(S)]*3, indexing='ij')
X = (ii - O[0])*h; Y = (jj - O[1])*h; Z = (kk - O[2])*h
Hs = [np.array([(a[c] - O[k])*h for k, c in enumerate('ijk')]) for a in m['atoms'] if a['el'] == 'H+']
print('RealQM H positions', [np.round(p, 3) for p in Hs])
def describe(tag, q, c):
    d = np.linalg.norm(c); kind = ''
    hb = [np.dot(c, p)/np.dot(p, p) for p in Hs]
    print('  %-16s charge %.3f  centre (%+.3f,%+.3f,%+.3f)  |r| %.3f  along O-H %.3f / %.3f' % (tag, q, *c, d, *hb))
print('RealQM own domains (charge = domain normalisation):')
for L, name in enumerate(['bond P', 'bond M', 'lone pair P', 'lone pair M']):
    w = (lab == L)*psi**2; q = w.sum()*h**3
    describe(name, q, np.array([(w*X).sum(), (w*Y).sum(), (w*Z).sum()])*h**3/q)
# ---- DFT density, localised orbitals, max-density partition
mol = gto.M(atom=[('O', (0, 0, 0))] + [('H', tuple(p)) for p in Hs], unit='Bohr', basis='cc-pvtz', verbose=0)
mf = dft.RKS(mol); mf.xc = 'b3lyp'; mf.kernel()
occ = mf.mo_coeff[:, mf.mo_occ > 0]
# Boys from a random start, without PySCF's own initial guess: from the canonical (symmetric) orbitals every localiser
# stays at the symmetric stationary point. Seed 1 gives the tetrahedral minimum (two O-H bonds, two lone pairs above and
# below the plane), the picture RealQM uses; the other minimum (one sigma and one pi lone pair) also exists.
from scipy.stats import ortho_group
mlb = lo.Boys(mol, occ[:, 1:] @ ortho_group.rvs(occ.shape[1] - 1, random_state=1)); mlb.init_guess = None
loc = np.hstack([occ[:, :1], mlb.kernel()])
g = np.arange(-5.0, 5.0001, 0.1); G = np.array(np.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T
dv = 0.1**3; nl = loc.shape[1]; Q = np.zeros(nl); C = np.zeros((nl, 3)); own = np.zeros(nl); tot = 0.0; Qo = np.zeros(nl); Co = np.zeros((nl, 3))
for s0 in range(0, len(G), 200000):
    pts = G[s0:s0 + 200000]; ao = numint.eval_ao(mol, pts); phi = ao @ loc; d2 = 2*phi**2
    rho = d2.sum(1); arg = d2.argmax(1); tot += rho.sum()*dv
    for i in range(nl):
        sel = arg == i; Q[i] += rho[sel].sum()*dv; C[i] += (rho[sel][:, None]*pts[sel]).sum(0)*dv; own[i] += d2[sel, i].sum()*dv
        so = sel & (np.linalg.norm(pts, axis=1) > 0.71); Qo[i] += rho[so].sum()*dv; Co[i] += (rho[so][:, None]*pts[so]).sum(0)*dv
print('B3LYP density partitioned by the wall rule (total %.3f e in box):' % tot)
order = np.argsort([np.linalg.norm(C[i]/Q[i]) for i in range(nl)])
for i in order:
    c = C[i]/Q[i]; tag = 'core' if np.linalg.norm(c) < 0.2 else ('bond' if c[1] > 0.3 and abs(c[0]) > 0.3 else 'lone pair')
    describe('%s (own %.2f)' % (tag, own[i]), Q[i], c)
print('Same partition, density outside r = 0.71 only (as the RealQM valence, kept out of its core sphere):')
for i in order:
    c = Co[i]/Qo[i]; tag = 'core' if np.linalg.norm(C[i]/Q[i]) < 0.2 else ('bond' if c[1] > 0.3 and abs(c[0]) > 0.3 else 'lone pair')
    describe(tag, Qo[i], c)
