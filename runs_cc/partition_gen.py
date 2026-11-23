"""
Analysis layer, general: an XH_n RealQM dump (one heavy atom X, bare protons) against B3LYP/cc-pVTZ at the same
geometry, its valence orbitals Boys-localised (random start, init_guess None), space split by the largest localised-
orbital density (RealQM's wall rule), true density integrated per region; also outside X's core radius r_c as RealQM's
valence is. Reports charge and centre of each region and of each RealQM domain; 'along X-H' = projection of the centre
on the nearest X-H vector as a fraction of the bond. usage: partition_gen.py DUMP ELEMENT
"""
import sys, json, numpy as np
from pyscf import gto, dft, lo
from pyscf.dft import numint
from scipy.stats import ortho_group
D = '/private/tmp/claude-501/-Users-cgjoh-Development-H2O/410447df-ea95-4b1a-874b-087cf46a7616/scratchpad/dumps'
name, EL = sys.argv[1], sys.argv[2]
m = json.load(open('%s/%s.meta' % (D, name))); NN, h = m['NN'], m['h']; S = NN + 1; n = S**3
psi = np.fromfile('%s/%s.psi' % (D, name), dtype='<f4').astype(float)[:n].reshape(S, S, S)
lab = np.fromfile('%s/%s.lab' % (D, name), dtype='<u4')[:n].reshape(S, S, S)
ents = [a for a in m['atoms'] if a['el']]
X0 = [a for a in ents if a['el'] != 'H+'][0]; RC = X0['rc']; O = np.array([X0[c] for c in 'ijk'])
Hs = [np.array([(a[c] - O[k])*h for k, c in enumerate('ijk')]) for a in ents if a['el'] == 'H+']
dom = [L for L, a in enumerate(ents) if a['el'] != 'H+']
ii, jj, kk = np.meshgrid(*[np.arange(S)]*3, indexing='ij'); X = (ii - O[0])*h; Y = (jj - O[1])*h; Z = (kk - O[2])*h
def along(c):
    p = min(Hs, key=lambda q: np.linalg.norm(c - q)); return np.dot(c, p)/np.dot(p, p)
print('%s: %s, r_c %.2f, H %s' % (name, EL, RC, [np.round(p, 3).tolist() for p in Hs]))
print('RealQM domains:')
for L in dom:
    w = (lab == L)*psi**2; q = w.sum()*h**3; c = np.array([(w*X).sum(), (w*Y).sum(), (w*Z).sum()])*h**3/q
    print('  %-8s charge %.3f  |r| %.3f  along X-H %.3f' % (ents[L]['el'], q, np.linalg.norm(c), along(c)))
mol = gto.M(atom=[(EL, (0, 0, 0))] + [('H', tuple(p)) for p in Hs], unit='Bohr', basis='cc-pvtz', verbose=0)
mf = dft.RKS(mol); mf.xc = 'b3lyp'; mf.kernel()
occ = mf.mo_coeff[:, mf.mo_occ > 0]; nv = occ.shape[1] - 1
if nv > 1:
    ml = lo.Boys(mol, occ[:, 1:] @ ortho_group.rvs(nv, random_state=1)); ml.init_guess = None; val = ml.kernel()
else:
    val = occ[:, 1:]
loc = np.hstack([occ[:, :1], val])
L = max(6.0, 1.5 + max(np.linalg.norm(p) for p in Hs)); g = np.arange(-L, L + 1e-6, 0.1)
G = np.array(np.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T; dv = 0.1**3; nl = loc.shape[1]
Q = np.zeros(nl); C = np.zeros((nl, 3)); Qo = np.zeros(nl); Co = np.zeros((nl, 3)); tot = 0.0
for s0 in range(0, len(G), 200000):
    pts = G[s0:s0 + 200000]; phi = numint.eval_ao(mol, pts) @ loc; d2 = 2*phi**2; rho = d2.sum(1); arg = d2.argmax(1); tot += rho.sum()*dv
    out = np.linalg.norm(pts, axis=1) > RC
    for i in range(nl):
        sel = arg == i; Q[i] += rho[sel].sum()*dv; C[i] += (rho[sel][:, None]*pts[sel]).sum(0)*dv
        so = sel & out; Qo[i] += rho[so].sum()*dv; Co[i] += (rho[so][:, None]*pts[so]).sum(0)*dv
print('B3LYP partitioned by the wall rule (total %.3f e in box); region 0 = core orbital:' % tot)
for i in range(nl):
    c = C[i]/Q[i]; co = Co[i]/Qo[i] if Qo[i] > 1e-6 else np.full(3, np.nan)
    print('  region %d  charge %.3f  |r| %.3f  along X-H %.3f   | outside r_c: charge %.3f  |r| %.3f  along X-H %.3f'
          % (i, Q[i], np.linalg.norm(c), along(c), Qo[i], np.linalg.norm(co), along(co) if Qo[i] > 1e-6 else np.nan))
