"""
He2 with each He as 2SPLIT (the user's he2_p5 / he2_sweep construction), axisymmetric: four one-electron domains
on the axis, mirror pairs. Right atom at z = +R/2 (half-space z > 0, the left atom its mirror): 'inner' electron
between the FIXED midplane z = 0 and a plane z = R/2 + s, 'outer' electron beyond it. The atom's own plane s is either
fixed through the nucleus (s = 0, as the prototype) or FREE under the continuity rule (densities equal across it,
bisection on s). Free (Neumann) walls everywhere. Nuclei +2, soft 0.285h, h 0.1. Pair counting: one electron per
domain, all interactions between different domains.
E = 2 sum_d [T_d - int K rho_d] + 2 sum_{d<e} int rho_d P_e + sum_d int rho_d P_L + 4/R.
usage: python3 he2_split.py fixed|free R1,R2,...
"""
import sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_realqm import Grid, masked_stiffness, kernel_potential
from axisym_hom2 import _poisson_setup
MODEP = sys.argv[1]; Rs = [float(x) for x in sys.argv[2].split(',')]
h = 0.1; L = 12.0; SOFT = 0.285*h; ZN = 2.0
gh = Grid(L, L, h, zmin=0.0); gf = Grid(L, L, h, zmin=-L); nzh = gh.nz; nr = gh.nr; V = gh.V
solveP, bnd, rrf = _poisson_setup(gf)
Zc = gh.Z.ravel()
Az = (2*np.pi*gh.r*h)
def pot(rho, q):
    full = np.zeros((gf.nz, gf.nr)); full[nzh:, :] = rho.reshape(nzh, nr)
    b = 4*np.pi*full.ravel()*gf.V; b[bnd] = q/rrf[bnd]; P = solveP(b).reshape(gf.nz, gf.nr)
    return P[nzh:, :].ravel(), P[:nzh, :][::-1, :].ravel()
def solve(R, s):
    K = kernel_potential(gh, [(-R/2, ZN), (R/2, ZN)], SOFT)
    zp = R/2 + s; mi = Zc < zp; mo = ~mi
    dom = {'i': dict(mask=mi), 'o': dict(mask=mo)}
    dist = np.sqrt((Zc - R/2)**2 + gh.R.ravel()**2)
    for d in dom.values():
        d['S'] = masked_stiffness(gh, d['mask']); d['idx'] = np.where(d['mask'])[0]
        d['Ssub'] = d['S'][d['idx']][:, d['idx']].tocsc(); d['eps'] = None
        d['u'] = np.where(d['mask'], np.exp(-1.7*dist), 0.0); d['u'] /= np.sqrt(np.sum(d['u']**2*V))
    Eo = None; Pst = None
    for it in range(300):
        P = {}; PL = np.zeros(gh.N)
        for k, d in dom.items(): P[k], px = pot(d['u']**2, 1.0); PL += px
        Pnew = {k: sum(P[e] for e in dom if e != k) + PL for k in dom}
        Pst = Pnew if Pst is None else {k: Pst[k] + 0.5*(Pnew[k] - Pst[k]) for k in Pnew}
        for k, d in dom.items():
            ix = d['idx']; A = (d['Ssub'] + sp.diags(((-K + Pst[k])*V)[ix])).tocsc(); Mm = sp.diags(V[ix]).tocsc()
            sig = (d['eps'] - 0.3) if d['eps'] is not None else -2.0
            val, vec = eigsh(A, k=1, M=Mm, sigma=sig, which='LM', v0=d['u'][ix])
            u = np.zeros(gh.N); u[ix] = vec[:, 0]; u *= np.sign(u[ix].sum()); d['u'] = u/np.sqrt(np.sum(u*u*V)); d['eps'] = val[0]
        P = {}; PL = np.zeros(gh.N)
        for k, d in dom.items(): P[k], px = pot(d['u']**2, 1.0); PL += px
        ri, ro = dom['i']['u']**2, dom['o']['u']**2
        E = 2*sum(d['u'] @ (d['S'] @ d['u']) - np.sum(K*d['u']**2*V) for d in dom.values()) \
            + 2*np.sum(ri*P['o']*V) + np.sum((ri + ro)*PL*V) + ZN*ZN/R
        if Eo is not None and abs(E - Eo) < 1e-8: break
        Eo = E
    k = int(round(zp/h))                       # plane between cells k-1 | k
    RI = ri.reshape(nzh, nr); RO = ro.reshape(nzh, nr)
    mis = np.sum(Az*(RI[k-1, :] - RO[k, :]))/np.sum(Az) if 0 < k < nzh else np.nan
    return E, mis
# isolated atom reference: same construction, atoms far apart (R = 16 in a box of 12 is not possible) -> use R large
for R in Rs:
    if MODEP == 'fixed' or MODEP.startswith('s='):
        s = float(MODEP[2:]) if MODEP.startswith('s=') else 0.0
        E, mis = solve(R, s)
    else:
        a, b = -0.5, 0.5; Ea, ma = solve(R, a); Eb, mb = solve(R, b)
        if ma*mb > 0: print("R=%.2f no continuity root in s=[-0.5,0.5] (mis %+.4f %+.4f)" % (R, ma, mb)); continue
        while b - a > 0.15:
            c = round(0.5*(a + b)/h)*h
            if c <= a + 1e-9 or c >= b - 1e-9: break
            Ec, mc = solve(R, c)
            if mc*ma > 0: a, ma = c, mc
            else: b, mb = c, mc
        s, (E, mis) = (a, (Ea if False else solve(R, a))) if abs(ma) < abs(mb) else (b, solve(R, b))
    print("He2 2split %s R=%.2f s=%+.2f E=%.6f  mismatch %+.5f" % (MODEP, R, s, E, mis), flush=True)
