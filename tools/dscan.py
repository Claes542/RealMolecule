import sys, numpy as np; sys.path.insert(0,'.')
exec(open('virial.py').read().split('print("VIRIAL')[0])
print("H2 BOND LENGTH with eps = 0.285h (atom.js calibration).  exact: D=1.401, E=-1.1745")
print("    D        h=0.12      h=0.08")
rows=[]
for D in (1.05,1.15,1.25,1.30,1.35,1.40,1.50):
    a=two_elec(0.12,12.,[(-D/2,1.),(D/2,1.)],1.0/D)[0]
    b=two_elec(0.08,12.,[(-D/2,1.),(D/2,1.)],1.0/D)[0]
    rows.append((D,a,b)); print("  %5.2f   %10.5f  %10.5f" % (D,a,b), flush=True)
for col,nm in ((1,'h=0.12'),(2,'h=0.08')):
    D=np.array([r[0] for r in rows]); E=np.array([r[col] for r in rows])
    i=int(np.argmin(E))
    if 0<i<len(D)-1:
        c=np.polyfit(D[i-1:i+2],E[i-1:i+2],2); Dm=-c[1]/(2*c[0]); Em=np.polyval(c,Dm)
        print("  %s  ->  D_min = %.3f au,  E_min = %.5f   (exact 1.401, -1.1745)" % (nm,Dm,Em))
    else:
        print("  %s  ->  minimum at or beyond the scan edge (D=%.2f)" % (nm,D[i]))
