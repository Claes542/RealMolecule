import sys, numpy as np; sys.path.insert(0,'.')
exec(open('kernel_family.py').read().split('h=0.12; L=12.')[0])
h=0.12; L=12.
EX={2:-2.903724,3:-7.279913,4:-13.655566}
print("SCALE-COVARIANT VERSION:  R_k = rho/Z  (fixed R_k is not scale-covariant -- at Z=4 the")
print("1s radius is 0.25 au, smaller than R_k=0.30, so the kernel smears the whole orbital).")
print("H (Z=1) and H2 (two Z=1 kernels) are UNCHANGED, so f stays calibrated and H2 stays fixed.\n")
print("  rho    f        He          err    rel      Li+         err    rel      Be2+        err    rel")
for rho,f,note in [(0.00,1.0006,"standard"),(0.10,0.9478,"H2 length"),(0.294,0.7710,"H2 energy")]:
    out=[]
    for Z in (2,3,4):
        LL=12.0 if Z==2 else 10.0
        e=two(h,LL,[(0.,float(Z))],0.0,rho/Z,f)
        out.append((e,e-EX[Z],100*abs(e-EX[Z])/abs(EX[Z])))
    print("  %.3f %6.4f  %9.4f %+7.4f %5.2f%%  %9.4f %+7.4f %5.2f%%  %9.4f %+7.4f %5.2f%%  %s"
          % (rho,f,out[0][0],out[0][1],out[0][2],out[1][0],out[1][1],out[1][2],
             out[2][0],out[2][1],out[2][2],note), flush=True)
print("\n  (for reference, the FIXED-R_k version gave at rho=0.30:  He 17.7%, Li+ 31.8%, Be2+ 41.3%)")
