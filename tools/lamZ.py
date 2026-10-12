import sys, numpy as np; sys.path.insert(0,'.')
exec(open('robin.py').read().split('print("ROBIN')[0])
from scipy.optimize import brentq
C=0.0737   # lam*Z from He and H2
EX={2:-2.903724,3:-7.279913,4:-13.655566}
print("TEST OF  lam = %.4f / Z  ON THE He-LIKE SERIES (two electrons, one kernel, plane split)" % C)
print("lam*Z was fixed by He (0.07448) and H2 (0.07314). Z=3,4 are PREDICTIONS.\n")
print("  Z    lam=C/Z    E(pred)     exact      err      rel      lam(fitted)  lam*Z")
for Z in (2,3,4):
    L = 12.0 if Z==2 else 10.0
    for h in (0.08,):
        lam=C/Z
        E=two_elec(h,L,[(0.,float(Z))],0.0,lam)
        lf=brentq(lambda t: two_elec(h,L,[(0.,float(Z))],0.0,t)-EX[Z], 0.0, 5.0, xtol=1e-6)
        print("  %d   %7.5f  %10.5f  %10.5f  %+8.5f  %6.2f%%    %7.5f    %6.4f"
              % (Z,lam,E,EX[Z],E-EX[Z],100*abs(E-EX[Z]/1)/abs(EX[Z]),lf,lf*Z), flush=True)
