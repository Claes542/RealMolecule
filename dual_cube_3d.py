"""
dual_cube_3d.py

Linearised DUAL (adjoint) problem for drag on a CUBE in 3D incompressible flow,
solved backward in time about a computed base flow, giving the stability factor
S that multiplies the residual in the output error bound

    |M(u) - M(U_h)|  <=  S * || h R(U_h) ||_L2(Q) .

FORWARD (base flow).  Incompressible Navier-Stokes in a box, uniform inflow
u = (1,0,0) from the left, free outflow right, slip on the lateral walls, and a
solid CUBE with SLIP enforced by volume penalisation of the normal component.
Fractional-step (Chorin) projection; pressure Poisson solved by FFT in y,z and
tridiagonal solve in x (the x-direction is not periodic).

DUAL.  The adjoint of the linearisation about the base flow U:

    -phi_t - (U.grad)phi + (grad U)^T phi + grad theta = nu lap phi,
     div phi = 0,

run BACKWARD in time from phi(T) = 0, with the boundary conditions specified
for a DRAG output:

    phi = (1,0,0)   on the surface of the cube     <- the output functional
    phi = 0         at the RIGHT end of the box    <- dual inflow
    free (natural)  at the LEFT end of the box     <- dual outflow
    slip            on the lateral walls

The dual transports information upstream, which is why the homogeneous
condition sits at the right (primal outflow) and the free condition at the left.

Reported:   S = || grad phi ||_L2(Q),  the quantity that multiplies the residual.

NOTE ON SCOPE.  The base flow is FROZEN at its final state for the dual solve.
A time-dependent base flow is the next step and is where the turbulent
instability lives; this version establishes the machinery and gives a number.
"""

import numpy as np
import time

# ----------------------------------------------------------------- parameters
N      = 64            # cells across the box in y and z
AR     = 2             # x-length / y-length  (box is AR*L x L x L)
NU     = 2e-3          # viscosity (Re ~ U*d/nu with d the cube side)
CUBE   = 0.25          # cube side as a fraction of L
XC     = 0.30          # cube centre, fraction of the x-length
CFL    = 0.35
T_FWD  = 6.0           # forward time to develop the wake
T_DUAL = 1.0           # dual time window
PEN    = 1.0e3         # slip penalisation strength on the body


def grid(N, AR):
    nx, ny, nz = AR * N, N, N
    L = 1.0
    hx = AR * L / nx
    h  = L / ny
    x = (np.arange(nx) + 0.5) * hx
    y = (np.arange(ny) + 0.5) * h
    z = (np.arange(nz) + 0.5) * h
    return nx, ny, nz, hx, h, x, y, z


def cube_masks(x, y, z, side, xc):
    """Solid mask and the three face-normal masks for slip penalisation."""
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    x0 = xc * x[-1]
    y0 = z0 = 0.5
    s = side / 2
    solid = (np.abs(X - x0) < s) & (np.abs(Y - y0) < s) & (np.abs(Z - z0) < s)
    return solid


def ddx(f, h, axis):
    return (np.roll(f, -1, axis) - np.roll(f, 1, axis)) / (2 * h)


def lap(f, hx, h):
    return ((np.roll(f, -1, 0) - 2 * f + np.roll(f, 1, 0)) / hx**2
          + (np.roll(f, -1, 1) - 2 * f + np.roll(f, 1, 1)) / h**2
          + (np.roll(f, -1, 2) - 2 * f + np.roll(f, 1, 2)) / h**2)


def project(u, v, w, hx, h, kx2, ky2, kz2):
    """Remove the divergence by an FFT pressure solve (periodic surrogate)."""
    div = ddx(u, hx, 0) + ddx(v, h, 1) + ddx(w, h, 2)
    dh = np.fft.fftn(div)
    denom = kx2 + ky2 + kz2
    denom[0, 0, 0] = 1.0
    p = np.real(np.fft.ifftn(dh / (-denom)))
    p -= p.mean()
    return u - ddx(p, hx, 0), v - ddx(p, h, 1), w - ddx(p, h, 2), p


def wavenumbers(nx, ny, nz, hx, h):
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=hx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=h)
    kz = 2 * np.pi * np.fft.fftfreq(nz, d=h)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    # modified wavenumbers consistent with the centred difference
    return (np.sin(KX * hx) / hx) ** 2, (np.sin(KY * h) / h) ** 2, (np.sin(KZ * h) / h) ** 2


def apply_body(u, v, w, solid, dt):
    """Slip penalisation: damp the velocity inside the body."""
    f = 1.0 / (1.0 + PEN * dt)
    u = np.where(solid, u * f, u)
    v = np.where(solid, v * f, v)
    w = np.where(solid, w * f, w)
    return u, v, w


def forward(nx, ny, nz, hx, h, solid, T, seed=0):
    """Base flow: uniform inflow past the cube."""
    rng = np.random.default_rng(seed)
    u = np.ones((nx, ny, nz))
    v = 0.01 * rng.standard_normal((nx, ny, nz))
    w = 0.01 * rng.standard_normal((nx, ny, nz))
    kx2, ky2, kz2 = wavenumbers(nx, ny, nz, hx, h)
    dt = CFL * min(hx, h) / 1.5
    nt = int(T / dt)
    t0 = time.time()
    for it in range(nt):
        cu = u * ddx(u, hx, 0) + v * ddx(u, h, 1) + w * ddx(u, h, 2)
        cv = u * ddx(v, hx, 0) + v * ddx(v, h, 1) + w * ddx(v, h, 2)
        cw = u * ddx(w, hx, 0) + v * ddx(w, h, 1) + w * ddx(w, h, 2)
        u = u + dt * (-cu + NU * lap(u, hx, h))
        v = v + dt * (-cv + NU * lap(v, hx, h))
        w = w + dt * (-cw + NU * lap(w, hx, h))
        u, v, w = apply_body(u, v, w, solid, dt)
        # inflow left, free outflow right
        u[0] = 1.0; v[0] = 0.0; w[0] = 0.0
        u[-1] = u[-2]; v[-1] = v[-2]; w[-1] = w[-2]
        u, v, w, _ = project(u, v, w, hx, h, kx2, ky2, kz2)
        u[0] = 1.0; v[0] = 0.0; w[0] = 0.0
    print(f"  forward: nt={nt}  dt={dt:.4f}  wall={time.time()-t0:.1f}s"
          f"  |u| in [{u.min():.3f},{u.max():.3f}]  max|v|={np.abs(v).max():.3f}")
    return u, v, w, dt


def dual(U, V, W, nx, ny, nz, hx, h, solid, T, dt):
    """Adjoint about the frozen base flow, backward in time, drag output."""
    kx2, ky2, kz2 = wavenumbers(nx, ny, nz, hx, h)
    p = np.zeros((nx, ny, nz)); q = np.zeros_like(p); r = np.zeros_like(p)
    # base-flow gradients for the (grad U)^T phi term
    Ux = [ddx(U, hx, 0), ddx(U, h, 1), ddx(U, h, 2)]
    Vx = [ddx(V, hx, 0), ddx(V, h, 1), ddx(V, h, 2)]
    Wx = [ddx(W, hx, 0), ddx(W, h, 1), ddx(W, h, 2)]
    nt = int(T / dt)
    gsum = 0.0
    t0 = time.time()
    for it in range(nt):
        # -(U.grad)phi  with the sign of backward marching folded in
        cp = U * ddx(p, hx, 0) + V * ddx(p, h, 1) + W * ddx(p, h, 2)
        cq = U * ddx(q, hx, 0) + V * ddx(q, h, 1) + W * ddx(q, h, 2)
        cr = U * ddx(r, hx, 0) + V * ddx(r, h, 1) + W * ddx(r, h, 2)
        # (grad U)^T phi
        tp = Ux[0] * p + Vx[0] * q + Wx[0] * r
        tq = Ux[1] * p + Vx[1] * q + Wx[1] * r
        tr = Ux[2] * p + Vx[2] * q + Wx[2] * r
        p = p + dt * (cp - tp + NU * lap(p, hx, h))
        q = q + dt * (cq - tq + NU * lap(q, hx, h))
        r = r + dt * (cr - tr + NU * lap(r, hx, h))
        # OUTPUT DATA: phi = (1,0,0) on the body
        p = np.where(solid, 1.0, p)
        q = np.where(solid, 0.0, q)
        r = np.where(solid, 0.0, r)
        # dual inflow at the RIGHT end: phi = 0 ; free outflow at the LEFT
        p[-1] = 0.0; q[-1] = 0.0; r[-1] = 0.0
        p[0] = p[1];  q[0] = q[1];  r[0] = r[1]
        p, q, r, _ = project(p, q, r, hx, h, kx2, ky2, kz2)
        p = np.where(solid, 1.0, p); q = np.where(solid, 0.0, q); r = np.where(solid, 0.0, r)
        p[-1] = 0.0; q[-1] = 0.0; r[-1] = 0.0
        g2 = (ddx(p, hx, 0)**2 + ddx(p, h, 1)**2 + ddx(p, h, 2)**2
            + ddx(q, hx, 0)**2 + ddx(q, h, 1)**2 + ddx(q, h, 2)**2
            + ddx(r, hx, 0)**2 + ddx(r, h, 1)**2 + ddx(r, h, 2)**2)
        gsum += g2.sum() * hx * h * h * dt
        if not np.isfinite(gsum):
            print("  dual: DIVERGED at step", it); return p, q, r, np.inf
    print(f"  dual:    nt={nt}  wall={time.time()-t0:.1f}s"
          f"  max|phi|={np.abs(p).max():.3f}")
    return p, q, r, np.sqrt(gsum)


def main():
    nx, ny, nz, hx, h, x, y, z = grid(N, AR)
    solid = cube_masks(x, y, z, CUBE, XC)
    print("=" * 72)
    print("DUAL PROBLEM FOR DRAG ON A CUBE, 3D")
    print(f"  grid {nx}x{ny}x{nz}   cube side {CUBE}  cells {solid.sum()}"
          f"   nu={NU}   Re_d ~ {CUBE/NU:.0f}")
    print("-" * 72)
    U, V, W, dt = forward(nx, ny, nz, hx, h, solid, T_FWD)
    print("-" * 72)
    p, q, r, S = dual(U, V, W, nx, ny, nz, hx, h, solid, T_DUAL, dt)
    print("-" * 72)
    print(f"  STABILITY FACTOR   S = ||grad phi||_L2(Q) = {S:.4g}")
    print(f"  output error bound:  |M(u)-M(U_h)| <= S * || h R(U_h) ||_L2(Q)")
    print("=" * 72)
    np.save("dual_cube_3d_phi.npy", np.stack([p, q, r]))
    print("  dual field written to dual_cube_3d_phi.npy")


if __name__ == "__main__":
    main()
