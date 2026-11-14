"""Independent recomputation of a molecule.js state dumped by h2_beta.html ?dump=NAME (dump_server.py).
Reads NAME.meta (JSON), NAME.psi (f32, S^3), NAME.lab (u32, S^3), NAME.K (f32, S^3). S = NN + 1, cell (i,j,k) at
(i h, j h, k h), index i*S^2 + j*S + k (molecule.js cellIdx: k fastest).
Computes, with nothing taken from the solver but psi and the labels:
  norms per domain; V_ee by zero-padded FFT convolution with 1/r (Hockney), counting each pair of domains once and
  (n-1)/n of each domain's own Hartree energy (MULTI_OCC rule: a pair counted once); V_eK from the solver's K and
  from an analytic kernel; V_KK point sum; T from finite differences inside each domain (Neumann at walls -- the rule
  that reproduces the solver; the 'jumps' variant is a diagnostic only). Cells inside any pseudo-core r_c are excluded
  (USER_FULL_RC). Validated 2026-11-14: H atom to 1e-5, H2 to 0.008, CH4 T to 0.015 Ha.
usage: python3 tools/check_energy.py NAME DUMPDIR     (dumps from h2_beta.html ?dump=NAME, receiver tools/dump_server.py)"""
import sys, json, numpy as np
name = sys.argv[1]; D = sys.argv[2] if len(sys.argv) > 2 else '.'
meta = json.load(open('%s/%s.meta' % (D, name)))
NN, h = meta['NN'], meta['h']; S = NN + 1
psi = np.fromfile('%s/%s.psi' % (D, name), dtype='<f4').astype(np.float64)
lab = np.fromfile('%s/%s.lab' % (D, name), dtype='<u4')
K = np.fromfile('%s/%s.K' % (D, name), dtype='<f4').astype(np.float64)
n = S ** 3
psi, lab, K = psi[:n].reshape(S, S, S), lab[:n].reshape(S, S, S), K[:n].reshape(S, S, S)
atoms = meta['atoms']; dV = h ** 3
dom = [a for a in range(len(atoms)) if atoms[a]['Z'] > 0]
# cells inside any pseudo-core (USER_FULL_RC: every core excludes every domain) are not part of any domain
ii0, jj0, kk0 = np.meshgrid(np.arange(S), np.arange(S), np.arange(S), indexing='ij')
excl = np.zeros((S, S, S), bool)
for a in atoms:
    if a['rc'] > 0 and a['Znuc'] > 0:
        excl |= np.sqrt((ii0 - a['i']) ** 2 + (jj0 - a['j']) ** 2 + (kk0 - a['k']) ** 2) * h < a['rc']
print('core-excluded cells %d, psi^2 there %.4f' % (excl.sum(), (psi[excl] ** 2).sum() * h ** 3))
psi = np.where(excl, 0.0, psi); lab = np.where(excl, np.uint32(10 ** 6), lab)
rho = psi ** 2
print('%s: NN=%d h=%.5f  solver E=%.5f T=%.5f VeK=%.5f Vee=%.5f VKK=%.5f' % (name, NN, h, meta['E'], meta['T'], meta['VeK'], meta['Vee'], meta['VKK']))
norms = {}
for a in dom:
    m = lab == a; norms[a] = rho[m].sum() * dV
print('norms: ' + ' '.join('%s:%.4f(Z=%g)' % (atoms[a]['el'], norms[a], atoms[a]['Z']) for a in dom))
print('cells labelled to a non-domain entry: %d' % sum(int((lab == a).sum()) for a in range(len(atoms)) if atoms[a]['Z'] <= 0))
# --- V_ee: Hockney FFT convolution with 1/r on a doubled grid
M = 2 * S
x = np.arange(M); x = np.where(x < S, x, x - M) * h
X, Y, Zc = np.meshgrid(x, x, x, indexing='ij'); R = np.sqrt(X ** 2 + Y ** 2 + Zc ** 2)
G = np.zeros_like(R); G[R > 0] = 1.0 / R[R > 0]; G[0, 0, 0] = 2.380077 / h      # cube self term: int 1/r over a cell / h^3
Gf = np.fft.rfftn(G); del X, Y, Zc, R, G
def pot(r):
    P = np.zeros((M, M, M)); P[:S, :S, :S] = r
    return np.fft.irfftn(np.fft.rfftn(P) * Gf, s=(M, M, M), axes=(0, 1, 2))[:S, :S, :S] * dV
rho_d = {a: np.where(lab == a, rho, 0.0) for a in dom}
phi = {a: pot(rho_d[a]) for a in dom}
Vcross = 0.0; Vself = 0.0
for ii, a in enumerate(dom):
    for b in dom[ii + 1:]: Vcross += (rho_d[a] * phi[b]).sum() * dV
    na = atoms[a]['Z']
    if na > 1: Vself += (na - 1) / na * 0.5 * (rho_d[a] * phi[a]).sum() * dV
print('V_ee independent: between domains %.5f + within (n-1)/n %.5f = %.5f   (solver %.5f, diff %+.5f)' % (
    Vcross, Vself, Vcross + Vself, meta['Vee'], Vcross + Vself - meta['Vee']))
# --- V_eK from the solver's K and from an analytic kernel
VeK_K = -(rho * K).sum() * dV
ii, jj, kk = np.meshgrid(np.arange(S), np.arange(S), np.arange(S), indexing='ij')
Ka = np.zeros((S, S, S))
for a in atoms:
    if a['Znuc'] <= 0: continue
    r2 = ((ii - a['i']) ** 2 + (jj - a['j']) ** 2 + (kk - a['k']) ** 2) * h * h
    r = np.maximum(np.sqrt(r2 + 0.04 * h * h), a['rc']) if a['rc'] > 0 else np.maximum(np.sqrt(r2), 2 * h)
    Ka += a['Znuc'] / r
print('V_eK: from solver K %.5f, analytic kernel %.5f   (solver %.5f)' % (VeK_K, -(rho * Ka).sum() * dV, meta['VeK']))
# --- V_KK point sum over distinct nuclear positions
pos = {}
for a in atoms:
    key = (round(a['i'], 4), round(a['j'], 4), round(a['k'], 4)); pos[key] = pos.get(key, 0) + a['Znuc']
P = list(pos.items()); VKK = 0.0
for x1 in range(len(P)):
    for x2 in range(x1 + 1, len(P)):
        d = np.sqrt(sum((P[x1][0][c] - P[x2][0][c]) ** 2 for c in range(3))) * h; VKK += P[x1][1] * P[x2][1] / d
print('V_KK point sum %.5f (solver %.5f)' % (VKK, meta['VKK']))
# --- T: 1/2 sum over neighbour pairs inside one domain of (du/h)^2 h^3 (Neumann at every wall), and with the
# difference to a non-domain/excluded neighbour counted as to zero (Dirichlet at walls to excluded cells)
T_neu = 0.0; T_dir = 0.0
for ax in range(3):
    u1 = np.take(psi, range(0, S - 1), axis=ax); u2 = np.take(psi, range(1, S), axis=ax)
    l1 = np.take(lab, range(0, S - 1), axis=ax); l2 = np.take(lab, range(1, S), axis=ax)
    same = l1 == l2
    T_neu += 0.5 * (((u2 - u1) / h) ** 2 * same).sum() * dV
    T_dir += 0.5 * (((u2 - u1) / h) ** 2 * same).sum() * dV + 0.5 * (((u1 / h) ** 2 + (u2 / h) ** 2) * (~same) * ((u1 == 0) | (u2 == 0))).sum() * dV
print('T: Neumann at walls %.5f, + jumps to zero cells %.5f   (solver %.5f)' % (T_neu, T_dir, meta['T']))
E_ind = T_neu - (rho * K).sum() * dV + Vcross + Vself + VKK
print('E independent (T Neumann, solver K) %.5f   (solver %.5f, diff %+.5f)' % (E_ind, meta['E'], E_ind - meta['E']))
