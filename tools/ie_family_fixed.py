import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
HA=27.211386; A2B=1.8897259/100
def ie(R, Z, Rout=200.0, n=80000):
    """ONE electron on a +Z kernel, FREE (Neumann) at R, free at infinity.
       P = r*u ; the kinetic boundary term is 0.5/R -- NO factor of Z."""
    r=np.linspace(R,Rout,n+1); h=r[1]-r[0]
    w=np.full(r.size,h); w[0]=h/2; w[-1]=h/2
    dia=np.full(r.size,1.0/h); dia[0]=1.0/(2*h); dia[-1]=1.0/(2*h)
    dia=dia-Z*w/r
    dia[0]+=0.5/R; dia[-1]+=-0.5/Rout
    return -eigh_tridiagonal(dia/w,(-1.0/(2*h))/np.sqrt(w[:-1]*w[1:]),
                             select='i',select_range=(0,0))[0][0]
# NIST IEs (eV); Shannon CRYSTAL radii of the ion that remains, 6-coord (pm). radii UNVERIFIED.
COLS=[("V=1  alkali IE1 -> r(M+)",1.0,
       {'Li':(5.3917,90),'Na':(5.1391,116),'K':(4.3407,152),'Rb':(4.1771,166),'Cs':(3.8939,181)}),
      ("V=2  alkaline-earth IE2 -> r(M2+)",2.0,
       {'Be':(18.2112,59),'Mg':(15.0353,86),'Ca':(11.8717,114),'Sr':(11.0301,132),'Ba':(10.0039,149)}),
      ("V=3  group-13 IE3 -> r(M3+)",3.0,
       {'Al':(28.4477,67.5),'Ga':(30.7258,76),'In':(28.03,94),'Tl':(29.83,102.5)})]
print("CORRECTED: kinetic boundary term 0.5/R (was 0.5*Z/R -- a bug for Z>1)\n")
for name,Z,tab in COLS:
    print("=== %s ===" % name)
    print("  el     IE(Ha)   R_fit     r_ion    ratio")
    rat=[]
    for el,(ev,pm) in tab.items():
        t=ev/HA
        try: Rf=brentq(lambda R: ie(R,Z)-t, 0.02, 150, xtol=1e-6)
        except Exception as e: print("  %-3s  no root"%el); continue
        ri=pm*A2B; rat.append((el,Rf/ri))
        print("  %-3s  %7.5f  %7.4f  %8.4f  %7.3f" % (el,t,Rf,ri,Rf/ri))
    v=np.array([r for _,r in rat])
    print("  all: mean %.4f  sd %.1f%%   |  excl first: mean %.4f  sd %.1f%%"
          % (v.mean(),100*v.std()/v.mean(),v[1:].mean(),100*v[1:].std()/v[1:].mean()))
    for lbl,k in (("all",v.mean()),("excl first",v[1:].mean())):
        rr=[]
        for el,(ev,pm) in tab.items():
            p=ie(k*pm*A2B,Z); t=ev/HA; rr.append(p/t)
        rr=np.array(rr)
        print("    k=%.4f (%s): ratios %s   max dev %.1f%%"
              % (k,lbl," ".join("%.2f"%x for x in rr),100*max(abs(rr-1))))
    print()
