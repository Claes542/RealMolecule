import sys, numpy as np; sys.path.insert(0,'.')
exec(open('kernel_family.py').read().split('h=0.12; L=12.')[0])
from scipy.optimize import brentq
EX={2:-2.903724,3:-7.279913,4:-13.655566}
print("CAN A FIXED PHYSICAL eps FIX He?   eps_total = sqrt(eps_phys^2 + (0.285h)^2), f = 1 (no")
print("kinetic compensation).  If eps_phys is mesh-independent it is a real radius.\n")
for h in (0.12,0.08):
    L=12.
    ep=brentq(lambda e: two(h,L,[(0.,2.)],0.0,e,1.0)+2.903724, 0.0, 0.5, xtol=1e-6)
    eH=one(h,L,[(0.,1.)],ep,1.0)
    print("h=%.2f   eps_phys = %.5f   He = %.6f (target)   H = %.5f  (err %+0.5f, %.2f%%)"
          % (h,ep,two(h,L,[(0.,2.)],0.0,ep,1.0),eH,eH+0.5,100*abs(eH+0.5)/0.5), flush=True)
    for Z in (3,4):
        LL=10.0
        e=two(h,LL,[(0.,float(Z))],0.0,ep,1.0)
        print("        Z=%d: %10.4f   exact %10.4f   err %+7.4f  %5.2f%%"
              % (Z,e,EX[Z],e-EX[Z],100*abs(e-EX[Z])/abs(EX[Z])), flush=True)
    for D in (1.30,1.401,1.60):
        E=two(h,L,[(-D/2,1.),(D/2,1.)],1.0/D,ep,1.0)
        print("        H2 D=%.3f: %9.5f   E_bind = %8.5f  (exact -1.17448 / 0.17447)"
              % (D,E,2*eH-E), flush=True)
    print(flush=True)
