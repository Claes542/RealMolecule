import sys, numpy as np; sys.path.insert(0,'.')
exec(open('forces.py').read().split('h=0.08; L=12.')[0])
h=0.08; L=12.
print("DOES HELLMANN-FEYNMAN HOLD IN MY OWN SOLVER?")
print("compare F_elec (from the density) with -dE/dD from NARROW differences on my own energies.\n")
print("   D     F_elec     -dE/dD(+-0.02)  -dE/dD(+-0.05)  -dE/dD(+-0.20)")
for D in (1.30,1.40,1.50):
    E0,F,Fn,Fe=solve(h,L,D)
    out=[]
    for d in (0.02,0.05,0.20):
        Em=solve(h,L,D-d)[0]; Ep=solve(h,L,D+d)[0]
        out.append(-(Ep-Em)/(2*d))
    print("  %4.2f  %+9.5f    %+9.5f      %+9.5f      %+9.5f"
          % (D,F,out[0],out[1],out[2]), flush=True)
