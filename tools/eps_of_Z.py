import sys, numpy as np, time; sys.path.insert(0,'.')
from axisym_realqm import *
from scipy.optimize import brentq
np.seterr(all='ignore')
def E1(Z,h,eps,L):
    g=Grid(L,L,h); K=kernel_potential(g,[(0.,float(Z))],eps)
    return solve_one_electron(g,K,np.zeros(g.N))[0]
print("TEST A: does the ONE-electron error vanish as h->0?  (no softening)")
print("  Z    h=0.16     h=0.12     h=0.08     h=0.06     exact     err ratio 0.08->0.06")
for Z,L in ((1,12.),(2,12.),(3,8.)):
    es=[E1(Z,h,0.0,L) for h in (0.16,0.12,0.08,0.06)]
    ex=-0.5*Z*Z
    r=(es[3]-ex)/(es[2]-ex)
    print("  %d  %9.5f %10.5f %10.5f %10.5f %9.4f    %.3f  (linear would be 0.750)"
          % (Z,es[0],es[1],es[2],es[3],ex,r), flush=True)
print("\nTEST B: eps(Z) needed to hit the exact -Z^2/2 at FIXED h.  eta = Z*eps should be const")
print("  Z     h=0.12            h=0.08")
print("        eps    eta        eps    eta")
for Z,L in ((1,12.),(2,12.),(3,8.),(4,8.)):
    out=[]
    for h in (0.12,0.08):
        try:
            e=brentq(lambda s: E1(Z,h,s,L)+0.5*Z*Z, 1e-4, 1.5, xtol=1e-5)
            out.append((e,Z*e))
        except Exception: out.append((float('nan'),float('nan')))
    print("  %d   %.4f  %.4f     %.4f  %.4f" % (Z,out[0][0],out[0][1],out[1][0],out[1][1]), flush=True)
