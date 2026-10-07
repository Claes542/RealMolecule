import io, re, contextlib, numpy as np
src0 = open('dual_cube_3d.py').read()
def run(N, slip):
    src = (src0.replace('N      = 64', f'N      = {N}')
                .replace('T_FWD  = 6.0', 'T_FWD  = 3.0')
                .replace('T_DUAL = 1.0', 'T_DUAL = 0.5')
                .replace('SLIP   = True', f'SLIP   = {slip}')
                .replace('np.save("dual_cube_3d_phi.npy", np.stack([p, q, r]))','pass'))
    g={'__name__':'__main__'}; buf=io.StringIO()
    with contextlib.redirect_stdout(buf): exec(compile(src,'m','exec'), g)
    o=buf.getvalue()
    return (float(re.search(r'S = \|\|grad phi\|\|_L2\(Q\) = ([0-9.e+-]+)',o).group(1)),
            float(re.search(r'max\|v\|=([0-9.]+)',o).group(1)),
            re.search(r'\|u\| in \[([-0-9.]+)',o).group(1))
print(f"{'N':>4} | {'SLIP':>22} | {'NO-SLIP':>22}")
print(f"{'':>4} | {'S':>8} {'u_min':>7} {'max|v|':>6} | {'S':>8} {'u_min':>7} {'max|v|':>6}")
print("-"*58)
rows={}
for N in (16,24,32,48):
    Ss,vs,us = run(N, True)
    Sn,vn,un = run(N, False)
    rows[N]=(Ss,Sn)
    print(f"{N:>4} | {Ss:8.3f} {us:>7} {vs:6.3f} | {Sn:8.3f} {un:>7} {vn:6.3f}")
h=np.array([1/N for N in rows]); 
for k,i in (("slip",0),("no-slip",1)):
    S=np.array([rows[N][i] for N in rows]); a,_=np.polyfit(np.log(h),np.log(S),1)
    print(f"  {k:>8}:  S ~ h^{a:+.2f}")
