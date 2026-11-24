"""Surface energy of a dumped molecule.js state: E_surf = (beta/2) sum_cells psi^2 * area, area = faces onto a DIFFERENT
domain label (molecule.js betaW, inter=0), scaled by |g|2/|g|1 (betaArea, ?betaiso=1), times h^2. Core-excluded cells
(r_c) are relabelled as in check_energy.py and their faces are not counted. usage: python3 surf.py NAME DUMPDIR BETA"""
import sys, json, numpy as np
name, D, beta = sys.argv[1], sys.argv[2], float(sys.argv[3])
meta = json.load(open('%s/%s.meta' % (D, name))); NN, h = meta['NN'], meta['h']; S = NN + 1; n = S**3
psi = np.fromfile('%s/%s.psi' % (D, name), dtype='<f4').astype(float)[:n].reshape(S, S, S)
lab = np.fromfile('%s/%s.lab' % (D, name), dtype='<u4')[:n].reshape(S, S, S).astype(np.int64)
ii, jj, kk = np.meshgrid(np.arange(S), np.arange(S), np.arange(S), indexing='ij'); excl = np.zeros((S, S, S), bool)
for a in meta['atoms']:
    if a['rc'] > 0 and a['Znuc'] > 0: excl |= np.sqrt((ii-a['i'])**2 + (jj-a['j'])**2 + (kk-a['k'])**2)*h < a['rc']
lab = np.where(excl, -1, lab)
def face(ax, sh):
    nb = np.roll(lab, sh, axis=ax); f = ((nb != lab) & (nb >= 0) & (lab >= 0)).astype(float)
    sl = [slice(None)]*3; sl[ax] = -1 if sh == -1 else 0; f[tuple(sl)] = 0; return f   # no wrap at the box edge
fp = [face(a, -1) for a in range(3)]; fm = [face(a, 1) for a in range(3)]
nf = sum(fp) + sum(fm); g = [fp[a] - fm[a] for a in range(3)]
g1 = sum(abs(x) for x in g); g2 = np.sqrt(sum(x*x for x in g))
area = np.where(g1 > 0.5, nf*g2/np.maximum(g1, 1e-12), nf)
Es = 0.5*beta*np.sum(psi**2*area)*h*h
print('%s E_surf(beta=%g, iso) = %.5f   solver E %.5f' % (name, beta, Es, meta['E']))
