import sys, numpy as np; sys.path.insert(0,'.')
exec(open('robin.py').read().split('print("ROBIN')[0])
from scipy.optimize import brentq
h=0.08; L=12.
g=Grid(L,L,h); eH=solve_one_electron(g,kernel_potential(g,[(0.,1.)],SOFT*h),np.zeros(g.N))[0]
print("E(H) = %.5f   (h=%.2f)\n" % (eH,h))
lam_He=brentq(lambda t: two_elec(h,L,[(0.,2.)],0.0,t)+2.903724, 0.0, 5.0, xtol=1e-6)
f=lambda t: two_elec(h,L,[(-1.401/2,1.),(1.401/2,1.)],1.0/1.401,t) - (-1.174476)
lam_H2=brentq(f, 0.0, 5.0, xtol=1e-6)
print("lam calibrated on He   = %.5f" % lam_He)
print("lam calibrated on H2   = %.5f   at the exact D=1.401, E=-1.174476" % lam_H2)
print("ratio lam(H2)/lam(He)  = %.2f\n" % (lam_H2/lam_He))
print("CROSS-PREDICTIONS (each lambda used on the OTHER system):")
e=two_elec(h,L,[(0.,2.)],0.0,lam_H2)
print("  lam(H2) -> He = %9.5f   exact -2.90372   err %+8.5f" % (e,e+2.903724))
e=two_elec(h,L,[(-1.401/2,1.),(1.401/2,1.)],1.0/1.401,lam_He)
print("  lam(He) -> H2 = %9.5f   exact -1.17448   err %+8.5f   E_bind %.5f"
      % (e,e+1.174476,2*eH-e))
print("\nHe AS THE D->0 LIMIT OF H2:  E_H2(D) - 1/D  should tend to E(He)")
print("  (needs D >> h; at h=%.2f, D=0.4 is only 5 cells, so the small-D end is strained)" % h)
print("   D      E_H2(D)      E - 1/D    [lam=0]        with lam(H2)")
for D in (0.4,0.6,0.8,1.0,1.401):
    e0=two_elec(h,L,[(-D/2,1.),(D/2,1.)],1.0/D,0.0)
    e1=two_elec(h,L,[(-D/2,1.),(D/2,1.)],1.0/D,lam_H2)
    print("  %5.3f  %10.5f  %10.5f               %10.5f" % (D,e0,e0-1.0/D,e1-1.0/D), flush=True)
print("  He direct (one +2 kernel):  lam=0 -> %.5f ;  lam(H2) -> %.5f ;  exact -2.90372"
      % (two_elec(h,L,[(0.,2.)],0.0,0.0), two_elec(h,L,[(0.,2.)],0.0,lam_H2)))
