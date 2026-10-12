import sys, numpy as np; sys.path.insert(0,'.')
exec(open('kernel_family.py').read().split('h=0.12; L=12.')[0])
h=0.12; L=12.
MEM=[(0.00,1.0006,"standard"),(0.10,0.9478,"bond length exact"),(0.30,0.7679,"H2 energy exact")]
EX={2:-2.903724,3:-7.279913,4:-13.655566}
print("TEST 1: the He-like ATOMIC series along the family (every member has E(H) = -0.5).")
print("(R_k, f) were fixed on HYDROGEN ONLY, so Z=2,3,4 are predictions.\n")
print("  R_k   f       He          err    rel     Li+         err    rel     Be2+        err    rel")
for Rk,f,note in MEM:
    out=[]
    for Z in (2,3,4):
        LL=12.0 if Z==2 else 10.0
        e=two(h,LL,[(0.,float(Z))],0.0,Rk,f)
        out.append((e,e-EX[Z],100*abs(e-EX[Z])/abs(EX[Z])))
    print("  %.2f %6.4f  %9.4f %+7.4f %5.2f%%  %9.4f %+7.4f %5.2f%%  %9.4f %+7.4f %5.2f%%   %s"
          % (Rk,f,out[0][0],out[0][1],out[0][2],out[1][0],out[1][1],out[1][2],
             out[2][0],out[2][1],out[2][2],note), flush=True)
print("\nTEST 2: the DISSOCIATION limit.  E(H) = -0.5 exactly, so E_bind = -1.0 - E(H2).")
print("exact E_bind: 0.1745 (1.401) / 0.1381 (2.0) / ~0.057 (3.0) / ~0.0004 (6.0)\n")
print("  R_k   f        D=1.401    D=2.0      D=3.0      D=6.0")
for Rk,f,note in MEM:
    r=[]
    for D in (1.401,2.0,3.0,6.0):
        E=two(h,L,[(-D/2,1.),(D/2,1.)],1.0/D,Rk,f)
        r.append(-1.0-E)
    print("  %.2f %6.4f  %9.5f  %9.5f  %9.5f  %9.5f   %s"
          % (Rk,f,r[0],r[1],r[2],r[3],note), flush=True)
