import numpy as np
# Closed-shell test: two electrons (two RealQM domains), independent spinor labels, field B_z.
# g=2: up (S_z=+1/2) -> mu_z=-muB ; down -> +muB.  E = -mu_z*B. Question: diamagnetic (net 0)?
muB=0.5; B=0.3
def muE(s1,s2):
    mu=sum(-muB*s for s in (s1,s2)); return mu,-mu*B   # s=+1 up,-1 dn
print("Two independent per-domain spinors, B=%.2f:"%B)
best=None
for nm,(a,b) in {"up up":(+1,+1),"up dn":(+1,-1),"dn dn":(-1,-1)}.items():
    mu,E=muE(a,b); print(f"  {nm:>6}: net mu={mu:+.2f}  E={E:+.4f}")
    if best is None or E<best[2]: best=(nm,mu,E)
print(f"GROUND STATE: {best[0]}, net moment {best[1]:+.2f} -> PARAMAGNETIC (should be diamagnetic 0).")
print("Diamagnetism needs antiparallel pairing (exchange J S1.S2), a spin-statistics term")
print("NOT supplied by geometric non-overlap. Open: can a shared-boundary coupling force it geometrically?")
