"""
residual_cube_3d.py

Measure ||h R(U_h)|| directly on the computed base flow, and multiply by the
stability factor S from dual_cube_3d.py, so that the output error bound

    |M(u) - M(U_h)|  <=  S * || h R(U_h) ||_L2(Q)

is a computed product rather than two separately estimated factors.

R(U_h) is the residual of the incompressible Navier-Stokes equations evaluated
on the computed solution: R = u_t + (u.grad)u + grad p - nu lap u, together
with div u.  Evaluated on the SAME grid as the solution (so it measures the
truncation the scheme leaves), then weighted by h and integrated over Q.
"""
import numpy as np, importlib.util, sys, io, contextlib, re

src0 = open('dual_cube_3d.py').read()

def run(N, T_fwd=3.0, T_dual=0.5):
    src = (src0.replace('N      = 64', f'N      = {N}')
                .replace('T_FWD  = 6.0', f'T_FWD  = {T_fwd}')
                .replace('T_DUAL = 1.0', f'T_DUAL = {T_dual}')
                .replace('np.save("dual_cube_3d_phi.npy", np.stack([p, q, r]))', 'pass'))
    g = {'__name__': '__main__'}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, 'm', 'exec'), g)
    out = buf.getvalue()
    S = float(re.search(r'S = \|\|grad phi\|\|_L2\(Q\) = ([0-9.e+-]+)', out).group(1))
    return g, S, out

def residual(g):
    """R on the final base flow, using the module's own operators."""
    nx, ny, nz, hx, h, x, y, z = g['grid'](g['N'], g['AR'])
    solid = g['cube_masks'](x, y, z, g['CUBE'], g['XC'])
    ddx, lap, NU = g['ddx'], g['lap'], g['NU']
    # recompute the base flow and keep two time levels for u_t
    U, V, W, dt = g['forward'](nx, ny, nz, hx, h, solid, g['T_FWD'])
    U2, V2, W2 = U.copy(), V.copy(), W.copy()
    kx2, ky2, kz2 = g['wavenumbers'](nx, ny, nz, hx, h)
    cu = U*ddx(U,hx,0) + V*ddx(U,h,1) + W*ddx(U,h,2)
    cv = U*ddx(V,hx,0) + V*ddx(V,h,1) + W*ddx(V,h,2)
    cw = U*ddx(W,hx,0) + V*ddx(W,h,1) + W*ddx(W,h,2)
    U2 = U + dt*(-cu + NU*lap(U,hx,h)); V2 = V + dt*(-cv + NU*lap(V,hx,h)); W2 = W + dt*(-cw + NU*lap(W,hx,h))
    U2,V2,W2 = g['apply_body'](U2,V2,W2,solid,dt)
    U2,V2,W2,P = g['project'](U2,V2,W2,hx,h,kx2,ky2,kz2)
    Ru = (U2-U)/dt + cu + ddx(P,hx,0) - NU*lap(U,hx,h)
    Rv = (V2-V)/dt + cv + ddx(P,h,1)  - NU*lap(V,hx,h)
    Rw = (W2-W)/dt + cw + ddx(P,h,2)  - NU*lap(W,hx,h)
    Rd = ddx(U,hx,0) + ddx(V,h,1) + ddx(W,h,2)
    out = ~solid                                  # measure in the fluid only
    dV = hx*h*h
    R2 = (Ru**2 + Rv**2 + Rw**2 + Rd**2)[out].sum()*dV
    return h, np.sqrt(R2)

print(f"{'N':>4} {'h':>8} {'S':>9} {'||R||':>11} {'||hR||':>11} {'S*||hR||':>11}")
print("-"*62)
rows=[]
for N in (16, 24, 32, 48):
    g, S, _ = run(N)
    h, Rn = residual(g)
    rows.append((N, h, S, Rn, h*Rn, S*h*Rn))
    print(f"{N:>4} {h:8.4f} {S:9.3f} {Rn:11.4g} {h*Rn:11.4g} {S*h*Rn:11.4g}")
a = np.array(rows)
import numpy as np
for name, col in (("||R||",3), ("||hR||",4), ("S*||hR||",5)):
    s,_ = np.polyfit(np.log(a[:,1]), np.log(a[:,col]), 1)
    print(f"  {name:>10} ~ h^{s:+.2f}")
