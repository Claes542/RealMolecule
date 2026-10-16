"""
HILBERT-6 PROGRAMME, ITEM 1: TWO domains with a MOVING interface.

WHY THIS IS THE ITEM THAT MATTERS.  Phase 1 verified the Madelung identity for ONE domain, where the
pair is equivalent to the Schrodinger equation and so buys no new physics.  Phase 2 verified the
momentum balance at two domains but at v=0, a hydrostatics.  The open structural question, and the
programme's stated failure condition, is whether the free boundary admits DYNAMICS at all: "ill-posed
free boundary (no dynamics past one electron)" would confine the whole programme.

THE TEST, which has an EXACT answer.  In a harmonic trap the centre of mass of any bound system
obeys Xcm'' = -w^2 Xcm exactly, because the trap is quadratic and every internal force --- including
the non-overlap constraint, which translates with the system --- is translation invariant.  So the
exact time-dependent solution starting from the rigidly displaced ground state is the ground state
TRANSLATED,
        X_i(m,t) = X_i^0(m) + A cos(w t),      v_i(m,t) = -A w sin(w t),
with the interface at s(t) = A cos(w t) and NO change of shape.  This is the two-domain analogue of
Phase 1's coherent state: an exact solution of the interacting, two-domain, moving-interface problem,
against which the computed one can be differenced.  A wrong interface condition does not reproduce
it -- momentum leaks at the boundary and the profile distorts.

MASS COORDINATES, which is the trick that makes the interface easy.  Let m = int rho dx be the
charge coordinate inside a domain, m in [0,1] since every domain carries exactly one electron.  Then
        x = X(m,t),   rho = 1/X_m,   v = X_t,   d/dx = rho d/dm,
and the two difficulties of a free boundary disappear by construction:
  * the interface is the parcel m=1 of the left domain and m=0 of the right, so it is MATERIAL
    automatically -- no interface-tracking equation, no numerical mass flux across it;
  * the charge per domain is exactly 1 for all time, since it IS the coordinate.
The momentum equation becomes  v_t = -rho d_m (V + Q),  with
        u = sqrt(rho),   Q = -(1/2) rho d_m( rho d_m u ) / u .
The Hartree term is also easier here than in x: the potential of the other domain is
        P(x) = int rho_j(y)/sqrt((x-y)^2+delta^2) dy = int_0^1 dm' / sqrt((x-X_j(m'))^2+delta^2),
an unweighted integral over m', and its x-derivative is available in closed form, so no numerical
differentiation of a Coulomb integral is needed.

RESULT 2026-10-16: MASS COORDINATES ARE DISQUALIFIED, and the reason is structural.
The mass coordinate looks like the natural choice -- it makes the interface material for free and
conserves the charge per domain exactly -- and it does not work, because dm/dx = rho vanishes
exponentially in the tails.  MEASURED on the static pair below (600 cells, Gaussian-tailed trap
state): rho spans 8.7e-29 .. 8.9e-01, so the per-cell mass spans the same 1.0e28 ratio.  Two ways
to lay out the m-grid and both fail:
  * nodes at uniform x (so m non-uniform): a 5-point Vandermonde stencil in m has condition number
    9e112 in the tail, 1.5e96 at a quarter of the way in, 4e7 even at the interface.  No finite
    difference in m survives that.
  * nodes at uniform m (well conditioned): then X(m) must come from inverting m(x), and Q needs TWO
    further m-derivatives of the result.  Measured error in Q: linear interpolation 9.5 at M=200
    GROWING to 123 at M=400 -- the 1/dx signature of differentiating an interpolant -- and cubic
    spline 1.9 -> 1.24, converging far too slowly to be usable, against a Q of order unity.
So the free interface cannot be handled by making the coordinate material.  The viable route is a
MAPPED EULERIAN (ALE) grid: a fixed computational coordinate xi per domain with x = X(xi,t) whose
endpoint follows the interface, the interface condition imposed as s_t = v(s) on the boundary, and
the density carried as phi = (1/2) ln rho so the tails stay representable -- the variable
tools/madelung_1d.py already introduced for exactly this reason.  What is kept from this file is
static_pair(), which works: the 1D two-domain ground state with a free interface, satisfying
Q + V = lambda to 1.1e-8 (relative 1.2e-9).

THE TAIL, and why Lagrangian truncation looked legitimate (it is not -- see above).  At m->0 the density vanishes and X_m
diverges, the usual Lagrangian vacuum problem.  The parcels are truncated at m = mu, carrying
1 - mu of the charge.  That is consistent rather than approximate for THIS test: the retained
parcels translate rigidly, so the truncation translates with them and the exact solution is still
exact for the retained part.  mu is varied to show the answer does not depend on it.

DISPERSIVE STEP RESTRICTION, inherited from Phase 1 (tools/madelung_1d.py): Q carries a second
derivative of sqrt(rho), so d_x Q carries a third, and explicit stepping is limited by
dt <~ C dx^3 rather than C dx.  That is a property of the formulation, not of a code, and it is why
this item was never a mere implementation task.  RK4 is used and the step is reported.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

np.seterr(all='ignore')


# --------------------------------------------------------------------------- static ground state
def static_pair(n=1200, Lbox=8.0, w=1.0, delta=0.5, iters=400, tol=1e-12):
    """Two domains in a harmonic trap, split by a free interface at x=0 (symmetry).

    The left domain lives on [-Lbox, 0] with the model's free-interface condition at x=0 --
    homogeneous Neumann, u'(0)=0 -- and u(-Lbox)=0.  Its partner is the mirror image, so the
    Hartree potential it feels is the mirror of its own.  Solved by inverse iteration on the
    discrete Hamiltonian, which is the 1D analogue of what axisym_realqm.py does.
    """
    x = -Lbox + (np.arange(n) + 0.5)*(Lbox/n)          # cell centres on [-Lbox, 0]
    hx = Lbox/n
    Vtrap = 0.5*w*w*x*x

    # -(1/2) d^2/dx^2 with u(-Lbox)=0 (Dirichlet, outer) and u'(0)=0 (Neumann, interface)
    main = np.full(n, 1.0/hx**2)
    main[-1] = 1.0/hx**2 - 0.5/hx**2                   # Neumann at the interface face
    off = np.full(n-1, -0.5/hx**2)
    rho = np.exp(-w*(x + 1.0)**2)                      # seed
    rho = rho/np.sum(rho*hx)
    E = None
    for it in range(iters):
        # Hartree potential from the mirror domain, evaluated on this one
        xm, rm = -x[::-1], rho[::-1]                   # partner's cell centres and density
        P = np.array([np.sum(rm*hx/np.sqrt((xi - xm)**2 + delta*delta)) for xi in x])
        H = np.diag(main + Vtrap + P) + np.diag(off, 1) + np.diag(off, -1)
        val, vec = np.linalg.eigh(H)
        u = vec[:, 0]
        if u[np.argmax(np.abs(u))] < 0:
            u = -u
        u = u/np.sqrt(np.sum(u*u*hx))
        rho_new = u*u
        Enew = float(val[0])
        rho = 0.5*rho + 0.5*rho_new                    # damped fixed point
        rho = rho/np.sum(rho*hx)
        if E is not None and abs(Enew - E) < tol:
            break
        E = Enew
    return dict(x=x, hx=hx, rho=rho, u=np.sqrt(rho), lam=Enew, w=w, delta=delta,
                Lbox=Lbox, iters=it+1)


def to_mass_coordinates(S, M=400, mu=1e-3):
    """Invert m(x) = int rho dx to get X(m) on a uniform m-grid in [mu, 1]."""
    x, rho, hx = S['x'], S['rho'], S['hx']
    m = np.concatenate([[0.0], np.cumsum(rho*hx)])     # m at cell faces
    xf = np.concatenate([[x[0]-hx/2], x + hx/2])
    m = m/m[-1]
    mg = np.linspace(mu, 1.0, M)
    X = np.interp(mg, m, xf)
    return mg, X


# --------------------------------------------------------------------------- derivatives in m
def dm1(f, dm):
    """4th-order first derivative in m, one-sided at the ends."""
    g = np.empty_like(f); n = len(f)
    g[2:n-2] = (f[0:n-4] - 8*f[1:n-3] + 8*f[3:n-1] - f[4:n])/(12*dm)
    for i in (0, 1, n-2, n-1):
        j0 = 0 if i < 2 else n-5
        c = _w(5, i-j0, 1)
        g[i] = c @ f[j0:j0+5]/dm
    return g


def _w(npts, p, order):
    A = np.vander(np.arange(npts) - p, npts, increasing=True).T
    b = np.zeros(npts); b[order] = float(math.factorial(order))
    return np.linalg.solve(A, b)


# --------------------------------------------------------------------------- the dynamics
def rhs(XL, vL, mg, dm, w, delta, interface_condition='material'):
    """d/dt of (X, v) for the LEFT domain; the right domain is its mirror at all times.

    The mirror is not an approximation: the initial condition and the trap are symmetric about the
    instantaneous centre of mass, and a rigid translation preserves that, so the right domain is
    always the reflection of the left through the moving interface.  This halves the work and keeps
    the interface exactly where the symmetry puts it -- which is the point being tested, since the
    interface position is then an OUTPUT (the parcel m=1) and not an imposed constraint.
    """
    Xm = dm1(XL, dm)
    rho = 1.0/np.maximum(Xm, 1e-300)
    u = np.sqrt(rho)
    # Q = -(1/2) rho d_m( rho d_m u )/u
    Q = -0.5*rho*dm1(rho*dm1(u, dm), dm)/np.maximum(u, 1e-300)
    # trap + Hartree from the mirror domain (positions -XL, same unit masses)
    Xo = -XL[::-1]
    d = XL[:, None] - Xo[None, :]
    r = np.sqrt(d*d + delta*delta)
    dVdx = w*w*XL - np.sum(d/r**3, axis=1)*dm            # closed form, no numerical Coulomb
    dQdx = rho*dm1(Q, dm)
    return vL.copy(), -(dVdx + dQdx)


def step_rk4(XL, vL, dt, *a):
    k1 = rhs(XL, vL, *a)
    k2 = rhs(XL + 0.5*dt*k1[0], vL + 0.5*dt*k1[1], *a)
    k3 = rhs(XL + 0.5*dt*k2[0], vL + 0.5*dt*k2[1], *a)
    k4 = rhs(XL + dt*k3[0], vL + dt*k3[1], *a)
    return (XL + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]),
            vL + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]))


def run_translation(M=300, mu=1e-3, A=0.2, nper=1.0, dt=None, n=900, Lbox=8.0,
                    w=1.0, delta=0.5, verbose=False):
    """Displace the pair rigidly by A and integrate; the exact answer is a rigid cosine."""
    S = static_pair(n=n, Lbox=Lbox, w=w, delta=delta)
    mg, X0 = to_mass_coordinates(S, M=M, mu=mu)
    dm = mg[1] - mg[0]
    dx_min = float(np.min(np.diff(X0)))
    if dt is None:
        dt = 0.02*dx_min**2                    # dispersive restriction; reported below
    XL = X0 + A
    vL = np.zeros_like(XL)
    T = nper*2*np.pi/w
    nt = int(round(T/dt))
    dt = T/nt
    worst_shape = 0.0
    rows = []
    for k in range(nt+1):
        t = k*dt
        s_exact = A*np.cos(w*t)
        s_num = XL[-1]                         # the interface IS the parcel m=1
        shape = float(np.max(np.abs((XL - s_num) - X0)))   # distortion after removing the shift
        worst_shape = max(worst_shape, shape)
        if verbose and k % max(1, nt//8) == 0:
            rows.append((t, s_num, s_exact + X0[-1], shape))
        if k < nt:
            XL, vL = step_rk4(XL, vL, dt, mg, dm, w, delta)
    s_err = abs((XL[-1] - X0[-1]) - A*np.cos(w*T))
    return dict(dt=dt, nt=nt, dx_min=dx_min, s_err=s_err, shape=worst_shape,
                M=M, mu=mu, A=A, lam=S['lam'], rows=rows,
                v_spread=float(np.max(vL) - np.min(vL)))


if __name__ == '__main__':
    w = 1.0
    print("HILBERT-6 ITEM 1 -- two domains, MOVING interface, against an exact solution")
    print("rigid displacement in a harmonic trap: s(t) = A cos(wt), profile undistorted\n")

    S = static_pair(n=900)
    print(f"static pair: lambda = {S['lam']:.8f}, charge = {np.sum(S['rho']*S['hx']):.10f}, "
          f"u'(0) Neumann imposed at the interface")

    print("\nTEST 1 -- equilibrium is a fixed point (A=0: nothing may move)")
    r = run_translation(M=300, A=0.0, nper=1.0)
    print(f"   after one period: |interface drift| = {r['s_err']:.3e}, "
          f"shape change = {r['shape']:.3e}, velocity spread = {r['v_spread']:.3e}")

    print("\nTEST 2 -- the rigid translation, against the exact cosine")
    print("    M     mu       A      dt        steps   |s(T)-exact|   max shape distortion")
    for (M, mu, A) in ((200, 1e-3, 0.2), (300, 1e-3, 0.2), (400, 1e-3, 0.2),
                       (300, 1e-2, 0.2), (300, 1e-4, 0.2),
                       (300, 1e-3, 0.1), (300, 1e-3, 0.4)):
        r = run_translation(M=M, mu=mu, A=A, nper=1.0)
        print(f" {M:5d} {mu:7.0e} {A:6.2f} {r['dt']:9.2e} {r['nt']:8d}   {r['s_err']:12.3e}"
              f"   {r['shape']:12.3e}", flush=True)
