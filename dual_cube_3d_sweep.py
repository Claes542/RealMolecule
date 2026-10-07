import re, importlib.util, sys, io, contextlib
import numpy as np
src0 = open('/Users/cgjoh/Development/H2O/dual_cube_3d.py').read()
print(f"{'N':>4} {'grid':>14} {'dt':>9} {'max|phi|':>9} {'S':>12}")
print("-"*54)
for N in (16, 24, 32, 48):
    src = (src0.replace('N      = 64', f'N      = {N}')
                .replace('T_FWD  = 6.0', 'T_FWD  = 3.0')
                .replace('T_DUAL = 1.0', 'T_DUAL = 0.5'))
    src = src.replace('np.save("dual_cube_3d_phi.npy", np.stack([p, q, r]))','pass')
    g = {'__name__':'__main__'}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, 'm', 'exec'), g)
    out = buf.getvalue()
    dt   = re.search(r'dt=([0-9.]+)', out).group(1)
    mp   = re.search(r'max\|phi\|=([0-9.e+-]+)', out)
    S    = re.search(r'S = \|\|grad phi\|\|_L2\(Q\) = ([0-9.e+infa-]+)', out).group(1)
    grid = re.search(r'grid (\S+)', out).group(1)
    print(f"{N:>4} {grid:>14} {dt:>9} {mp.group(1) if mp else '-':>9} {S:>12}")
