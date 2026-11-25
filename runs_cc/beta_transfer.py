"""Is a first-order wall correction transferable? (2026-11-25)  Every system is a singlet bond of two ONE-electron
domains meeting at a wall (2split): H2, LiH, NaH, Li2, Na2 (pseudo-X +1, r_c fitted to IE1, hard interface).
For each: R scan at beta = 0 -> R_e, D_e(0); at R_e the wall charge W = sum_k n_k (1/2) INT u_k^2 dS over the bond
wall (the first-order dE/dbeta, density NOT relaxed). beta fitted on H2 only: beta = (D0 - Dex)/W; predicted
D = D0 - beta W for the others. Atoms have no wall, so they carry no correction.
usage: python3 beta_transfer.py h"""
import sys, numpy as np
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
import axisym_linear as AL
from axisym_linear import Linear, preset, fit_rc, H_atom
h = float(sys.argv[1]); L = 12.0
# exact D_e (Ha): H2 Kolos-Wolniewicz; others experimental, FROM MEMORY (the preset notes say verify) -- flagged
SYS = [('h2', None, 0.1745), ('lih', ('Li', -0.19814, (1.0, 3.0)), 0.0924), ('nah', ('Na', -0.18886, (1.0, 3.5)), 0.0724),
       ('li2_split', ('Li', -0.19814, (1.0, 3.0)), 0.0381), ('na2_split', ('Na', -0.18886, (1.0, 3.5)), 0.0276)]
def point(name, o, R):
    p = preset(name, dict(o, R=str(R)))
    m = Linear(p['nuclei'], p['domains'], p['planes'], h=h, L=L, hfix=p['hfix'], name=name, hface=p.get('hface', False))
    r = m.relax(verbose=False)
    Es = 0.0
    for f in p['frags']:
        lab, cnt, nuc, dom, pl = f
        a = Linear(nuc, dom, pl, h=m.h, L=m.L, hfix=0.0, name=lab); a.h = m.h; Es += cnt * a.scf()['E']
    Hn = [n for n in m.nuclei if n['Z'] == 1 and n['rc'] == 0]
    for n in Hn: Es += H_atom(m.h, m.L, 0.0, z0=round((n['z'] - m.g.zmin) % m.h, 9)).scf()['E']
    # wall charge at the converged density: robin faces with beta = 1, density unchanged
    m.beta = 1.0; m.beta_thru = 1.0; M = m.masks(); bv = m.robin_faces(M)
    W = sum(d['n'] * 0.5 * float(np.sum(b * u * u)) for d, b, u in zip(m.domains, bv, m.u))
    return Es - r['E'], W, m.h
res = {}
for name, pse, Dex in SYS:
    o = {}
    if pse:
        X, tgt, (lo, hi) = pse; o['rc' + X] = '%.4f' % fit_rc(name, X, tgt, 'hard', h=h, lo=lo, hi=hi, L=L)
    Robs = preset(name, o)['Robs']; pts = {}
    def at(R):
        R = round(R, 4)
        if R not in pts: pts[R] = point(name, o, R)
        return pts[R]
    st = 0.05 * Robs
    for R in (Robs - st, Robs, Robs + st): at(R)
    for _ in range(10):
        ks = sorted(pts); b = max(ks, key=lambda k: pts[k][0])
        if b == ks[0]: at(b - st)
        elif b == ks[-1]: at(b + st)
        else: break
    ks = sorted(pts); b = max(ks, key=lambda k: pts[k][0]); i = ks.index(b); sel = ks[max(i-1, 0): i+2]
    c = np.polyfit(sel, [pts[k][0] for k in sel], 2); Re = -c[1]/(2*c[0]); D0 = float(np.polyval(c, Re))
    W = float(np.polyval(np.polyfit(sel, [pts[k][1] for k in sel], 2), Re))
    res[name] = (Re, Robs, D0, W, Dex)
    print('%-10s %s R_e %.4f (obs %.4f, %+.1f%%)  D0 %.5f  exact %.4f (%+.0f%%)  W %.5f' % (name, ' '.join('%s=%s' % kv for kv in o.items()),
          Re, Robs, 100*(Re/Robs-1), D0, Dex, 100*(D0/Dex-1), W), flush=True)
b = (res['h2'][2] - res['h2'][4]) / res['h2'][3]
print('beta fitted on H2 (first order): %.4f' % b)
for name in res:
    Re, Robs, D0, W, Dex = res[name]; D = D0 - b*W
    print('  %-10s D0 %.5f -> D %.5f  exact %.4f  error %+.0f%% -> %+.0f%%   (beta that would fit it alone: %.3f)' % (
          name, D0, D, Dex, 100*(D0/Dex-1), 100*(D/Dex-1), (D0 - Dex)/W))
