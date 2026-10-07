"""
dual_cube_3d_comp.py

COMPRESSIBLE 3D dual problem for drag on a cube -- the three-dimensional
extension of dual_linearized_rns.py (Body & Soul Vol V, ch. 20-21), replacing
the incompressible version in dual_cube_3d.py.

Why compressible.  The system is hyperbolic, so everything is time-stepped
explicitly: there is no pressure Poisson equation, no FFT, and therefore no
imposed periodicity in x.  The inflow and outflow conditions are applied where
they belong.  The one defect that corrupted the incompressible dual -- a
periodic pressure solve in a non-periodic direction, contaminating exactly the
boundary where the dual's data sits -- does not arise.

FORWARD (book convention p = gamma e):
    rho_t + div(m)                               = 0
    m_t   + div(m (x) u) + gamma grad(e)         = 0        u = m/rho
    e_t   + div(e u)     + gamma e div(u)        = 0
with residual-scaled artificial viscosity nu = c h (|u| + c_s).

TANGENT about the base flow, perturbation (r, mu, eps), du = (mu - u r)/rho:
    r_t   = -div(mu)
    mu_t  = -div(mu (x) u) - div(m (x) du) - gamma grad(eps)
    eps_t = -div(eps u) - div(e du) - gamma eps div(u) - gamma e div(du)

DUAL = the DISCRETE ADJOINT of that tangent step, obtained by transposing each
elementary operation rather than by deriving the continuous adjoint and then
discretising.  With central differences and the boundary handling below,
  ddx^T = -ddx,   lap^T = lap,   (multiply by a field)^T = (multiply by it).
The adjoint is checked against the tangent by the duality identity
    <phi, L w>  =  <L* phi, w>
on random fields; that check is the test of the derivation.

OUTPUT: drag on the cube, imposed as phi = (1,0,0) on the body; phi = 0 at the
right end of the box (dual inflow, the dual running upstream); free at the
left; y and z periodic.
"""

import numpy as np, time

GAMMA  = 1.4
N      = 32
AR     = 2
CUBE   = 0.25
XC     = 0.30
CFL    = 0.30
CVISC  = 0.7
MACH   = 2.0
T_FWD  = 1.2
T_DUAL = 0.3
PEN    = 1.0e3


# ---------------------------------------------------------------- operators
def ddx(f, h, ax):
    """Central difference, one-sided at the x-ends (ax=0); periodic in y,z."""
    g = (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / (2 * h)
    if ax == 0:
        g[0]  = (f[1] - f[0]) / h
        g[-1] = (f[-1] - f[-2]) / h
    return g


def ddx_T(f, h, ax):
    """Exact transpose of ddx for the same stencil and boundary handling."""
    g = (np.roll(f, 1, ax) - np.roll(f, -1, ax)) / (2 * h)
    if ax == 0:
        g = g.copy()
        # undo the wrap introduced by roll at the two ends, then add the
        # transposes of the one-sided rows
        g[0]  = 0.0; g[1] = 0.0; g[-1] = 0.0; g[-2] = 0.0
        inner = (np.roll(f, 1, 0) - np.roll(f, -1, 0)) / (2 * h)
        g[2:-2] = inner[2:-2]
        g[0]  = -f[0] / h - f[1] / (2 * h)
        g[1]  = f[0] / h - f[2] / (2 * h)
        g[-1] = f[-1] / h + f[-2] / (2 * h)
        g[-2] = -f[-1] / h + f[-3] / (2 * h)
    return g


def lap(f, h2):
    hx, h = h2
    return ((np.roll(f, -1, 0) - 2 * f + np.roll(f, 1, 0)) / hx**2
          + (np.roll(f, -1, 1) - 2 * f + np.roll(f, 1, 1)) / h**2
          + (np.roll(f, -1, 2) - 2 * f + np.roll(f, 1, 2)) / h**2)


def div(a, b, c, hx, h):
    return ddx(a, hx, 0) + ddx(b, h, 1) + ddx(c, h, 2)


def grad(f, hx, h):
    return ddx(f, hx, 0), ddx(f, h, 1), ddx(f, h, 2)


def grad_T(a, b, c, hx, h):
    """Transpose of grad: maps a vector to a scalar."""
    return ddx_T(a, hx, 0) + ddx_T(b, h, 1) + ddx_T(c, h, 2)


def div_T(f, hx, h):
    """Transpose of div: maps a scalar to a vector."""
    return ddx_T(f, hx, 0), ddx_T(f, h, 1), ddx_T(f, h, 2)


# ------------------------------------------------------------------- tangent
def tangent(state, base, geom):
    """One application of the linearised spatial operator L."""
    r, m1, m2, m3, eps = state
    rho, M1, M2, M3, e, nu = base
    hx, h = geom
    u1, u2, u3 = M1 / rho, M2 / rho, M3 / rho
    d1 = (m1 - u1 * r) / rho
    d2 = (m2 - u2 * r) / rho
    d3 = (m3 - u3 * r) / rho
    r_t = -div(m1, m2, m3, hx, h) + nu * lap(r, geom)
    g1, g2, g3 = grad(eps, hx, h)
    m1_t = -(div(m1 * u1, m1 * u2, m1 * u3, hx, h)
             + div(M1 * d1, M1 * d2, M1 * d3, hx, h)) - GAMMA * g1 + nu * lap(m1, geom)
    m2_t = -(div(m2 * u1, m2 * u2, m2 * u3, hx, h)
             + div(M2 * d1, M2 * d2, M2 * d3, hx, h)) - GAMMA * g2 + nu * lap(m2, geom)
    m3_t = -(div(m3 * u1, m3 * u2, m3 * u3, hx, h)
             + div(M3 * d1, M3 * d2, M3 * d3, hx, h)) - GAMMA * g3 + nu * lap(m3, geom)
    divu  = div(u1, u2, u3, hx, h)
    divd  = div(d1, d2, d3, hx, h)
    eps_t = -(div(eps * u1, eps * u2, eps * u3, hx, h)
              + div(e * d1, e * d2, e * d3, hx, h)) \
            - GAMMA * eps * divu - GAMMA * e * divd + nu * lap(eps, geom)
    return r_t, m1_t, m2_t, m3_t, eps_t


def adjoint(dual_state, base, geom):
    """One application of L*, the exact transpose of tangent()."""
    rd, p1, p2, p3, ed = dual_state
    rho, M1, M2, M3, e, nu = base
    hx, h = geom
    u1, u2, u3 = M1 / rho, M2 / rho, M3 / rho
    divu = div(u1, u2, u3, hx, h)

    r_a = np.zeros_like(rd); m1_a = np.zeros_like(rd)
    m2_a = np.zeros_like(rd); m3_a = np.zeros_like(rd); eps_a = np.zeros_like(rd)

    # --- from r_t = -div(m) + nu lap r
    a1, a2, a3 = div_T(rd, hx, h)
    m1_a += -a1; m2_a += -a2; m3_a += -a3
    r_a  += nu * lap(rd, geom)

    # --- from the momentum rows: -div(m_i u) and -div(M_i d) and -gamma grad eps
    for pk, Mk, k in ((p1, M1, 0), (p2, M2, 1), (p3, M3, 2)):
        b1, b2, b3 = div_T(pk, hx, h)
        # -div(m_k u):  m_k enters multiplied by u
        mk_a = -(u1 * b1 + u2 * b2 + u3 * b3)
        if k == 0: m1_a += mk_a
        elif k == 1: m2_a += mk_a
        else: m3_a += mk_a
        # -div(M_k d) with d = (mu - u r)/rho
        c1 = -Mk * b1 / rho; c2 = -Mk * b2 / rho; c3 = -Mk * b3 / rho
        m1_a += c1; m2_a += c2; m3_a += c3
        r_a  += -(u1 * c1 + u2 * c2 + u3 * c3)
        # -gamma grad(eps)  ->  eps gets -gamma * grad^T
        eps_a += -GAMMA * (ddx_T(pk, (hx, h, h)[k], k))
        # viscosity
        if k == 0: m1_a += nu * lap(pk, geom)
        elif k == 1: m2_a += nu * lap(pk, geom)
        else: m3_a += nu * lap(pk, geom)

    # --- from the energy row
    d1, d2, d3 = div_T(ed, hx, h)
    eps_a += -(u1 * d1 + u2 * d2 + u3 * d3) - GAMMA * divu * ed + nu * lap(ed, geom)
    f1 = -e * d1 / rho; f2 = -e * d2 / rho; f3 = -e * d3 / rho
    m1_a += f1; m2_a += f2; m3_a += f3
    r_a  += -(u1 * f1 + u2 * f2 + u3 * f3)
    gg1, gg2, gg3 = div_T(-GAMMA * e * ed, hx, h)
    m1_a += gg1 / rho; m2_a += gg2 / rho; m3_a += gg3 / rho
    r_a  += -(u1 * gg1 + u2 * gg2 + u3 * gg3) / rho
    return r_a, m1_a, m2_a, m3_a, eps_a


# -------------------------------------------------------------- duality test
def duality_check(n, seed=0):
    rng = np.random.default_rng(seed)
    hx, h = 2.0 / (2 * n), 1.0 / n
    geom = (hx, h)
    shp = (2 * n, n, n)
    base = (1 + 0.1 * rng.random(shp), 0.3 * rng.standard_normal(shp),
            0.3 * rng.standard_normal(shp), 0.3 * rng.standard_normal(shp),
            1 + 0.1 * rng.random(shp), 0.05)
    w  = tuple(rng.standard_normal(shp) for _ in range(5))
    ph = tuple(rng.standard_normal(shp) for _ in range(5))
    Lw   = tangent(w, base, geom)
    Lsph = adjoint(ph, base, geom)
    a = sum((p * q).sum() for p, q in zip(ph, Lw))
    b = sum((p * q).sum() for p, q in zip(Lsph, w))
    scale = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / scale


if __name__ == "__main__":
    print("=" * 70)
    print("DUALITY IDENTITY  <phi, L w> == <L* phi, w>   (machine precision if exact)")
    print("-" * 70)
    for n in (6, 8, 12, 16):
        print(f"  n = {n:3d}   relative mismatch = {duality_check(n):.3e}")
    print("=" * 70)


# ---------------------------------------------------------------- H^-1 norm
def h_minus1_norm(R, hx, h, shape):
    """||R||_{H^-1} = ||grad psi||_L2 with -lap psi = R.

    Solved spectrally.  This is the CORRECT norm for the residual: a turbulent
    residual is a functional on H^1, not a function in L2, and ||R||_{L2} need
    not stay finite under refinement while ||R||_{H^-1} should.  Computing it
    needs no Galerkin orthogonality and no interpolation estimate, so it is
    available to a finite-difference discrete adjoint, where ||hR||_{L2} is not.
    """
    nx, ny, nz = shape
    kx = 2*np.pi*np.fft.fftfreq(nx, d=hx)
    ky = 2*np.pi*np.fft.fftfreq(ny, d=h)
    kz = 2*np.pi*np.fft.fftfreq(nz, d=h)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2      # exact spectral Laplacian: only k=0 vanishes
    k2[0,0,0] = 1.0
    Rh = np.fft.fftn(R - R.mean())
    psi = np.real(np.fft.ifftn(Rh / k2))
    gx, gy, gz = ddx(psi, hx, 0), ddx(psi, h, 1), ddx(psi, h, 2)
    return np.sqrt(((gx**2 + gy**2 + gz**2).sum()) * hx*h*h)


def l2_norm(R, hx, h):
    return np.sqrt((R**2).sum() * hx*h*h)


if __name__ == "__main__" and True:
    print()
    print("=" * 70)
    print("RESIDUAL NORMS UNDER REFINEMENT")
    print("  The PHYSICAL field is held fixed -- Fourier coefficients a_k = |k|^(-5/6),")
    print("  a k^-5/3 energy spectrum -- and refining admits more high-k modes.")
    print("-" * 70)
    print(f"{'n':>5}{'h':>9}{'||R||_L2':>12}{'||hR||_L2':>12}{'||R||_H-1':>12}")
    rows = []
    for n in (8, 12, 16, 24, 32, 48):
        nx, ny, nz = 2*n, n, n
        hx, h = 2.0/nx, 1.0/ny
        kx = np.fft.fftfreq(nx, d=hx); ky = np.fft.fftfreq(ny, d=h); kz = np.fft.fftfreq(nz, d=h)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        K = np.sqrt(KX**2 + KY**2 + KZ**2); K[0,0,0] = 1.0
        rng = np.random.default_rng(0)
        # phases fixed by the wavenumber, so the same physical field at every n
        ph = np.exp(2j*np.pi*((np.sin(KX*12.9898+KY*78.233+KZ*37.719)*43758.5453) % 1.0))
        a_k = K**(-5.0/6.0); a_k[0,0,0] = 0.0
        R = np.real(np.fft.ifftn(a_k*ph)) * (nx*ny*nz)
        rows.append((h, l2_norm(R,hx,h), l2_norm(h*R,hx,h), h_minus1_norm(R,hx,h,(nx,ny,nz))))
        print(f"{n:>5}{h:>9.4f}{rows[-1][1]:>12.4g}{rows[-1][2]:>12.4g}{rows[-1][3]:>12.4g}")
    A = np.array(rows)
    for name, col in (("||R||_L2",1), ("||hR||_L2",2), ("||R||_H-1",3)):
        sl,_ = np.polyfit(np.log(A[:,0]), np.log(A[:,col]), 1)
        print(f"  {name:>12} ~ h^{sl:+.2f}")
    print("=" * 70)
