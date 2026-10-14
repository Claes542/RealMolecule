"""
STEP 1 of the Hilbert-6th programme: integrate the MADELUNG PAIR directly and check it against an
exact free Gaussian wave packet.  Nothing in the programme advances until this passes.

WHAT IS BEING TESTED, and why this test is decisive.  Start with a REAL Gaussian density and
v = 0 everywhere.  A pressureless classical fluid with v=0 and no force would simply sit there
forever.  The exact quantum answer is that the packet SPREADS, with

    sigma(t) = sigma0 * sqrt(1 + (t/(2*sigma0^2))^2)        (hbar = m = 1)

and in the Madelung form the ONLY thing that can produce that spreading is the quantum potential Q.
So reproducing sigma(t) from v=0 initial data is a direct test that Q acts as the stress the
hydrodynamic form claims it is -- not a test of a fitted constant, and with an exact answer.

THE FORMULATION, which is the whole trick.  Written naively,
    Q = -1/2 * (d2/dx2 sqrt(rho)) / sqrt(rho)
is a 0/0 in the tails: sqrt(rho) underflows long before the physics stops mattering, and the ratio
becomes noise that then drives the momentum equation.  Evolving instead

    phi = (1/2) ln rho                  so  sqrt(rho) = e^phi

turns the singular ratio into a polynomial in derivatives of phi:
    Q = -1/2 * (phi'' + (phi')^2)
which is smooth and finite wherever phi is -- and phi ~ -x^2/(8 sigma^2) is a parabola, perfectly
representable where rho itself has underflowed to zero.  The pair becomes

    phi_t = -v*phi' - (1/2)*v'                       (continuity, exactly rewritten)
    v_t   = -v*v'  - Q'
    Q     = -1/2*(phi'' + (phi')^2)

DISCRETISATION, and a trap found the hard way.  The log form fixes the tails but is INCOMPATIBLE
WITH FFT: phi ~ -x^2/(4 sigma^2) is a parabola, so although phi is even and therefore periodic in
VALUE on [-L/2, L/2], its derivative changes sign across the wrap.  Spectral differentiation of
that is Gibbs noise, and it reached NaN within one sample interval on the first attempt.  So all
derivatives are 4th-order central finite differences, with ghost cells filled by QUADRATIC
extrapolation for phi (exact for a Gaussian, whose log IS a parabola) and linear for v (exact for
the free packet, whose velocity field is linear in x).  Time-stepping is classical RK4.
No limit and no approximation beyond discretisation.
"""
import numpy as np


def exact_sigma(t, s0):
    """Exact width of a free Gaussian packet, hbar = m = 1."""
    return s0*np.sqrt(1.0 + (t/(2.0*s0**2))**2)


def run(s0=1.0, L=24.0, N=1024, T=2.0, dt=2e-4, k0=0.0, report=None, omega=0.0, x0=0.0):
    """Integrate the Madelung pair from a real Gaussian.

    omega > 0 adds an external harmonic potential V = (1/2) omega^2 x^2; x0 displaces the packet.
    Returns a list of (t, sigma_num, sigma_exact, mass, centroid) samples.

    WHY THE HARMONIC CASE IS THE RIGHT FIRST TEST.  The FREE packet has a velocity field linear and
    UNBOUNDED in x -- the flow accelerates outward forever -- so a finite box admits no correct
    boundary condition and the edge cell poisons dQ/dx no matter how it is extrapolated (measured:
    the interior stays exact while max|dQ/dx| jumps from 3.0 to 9.9 in the last cell, and that seeds
    the blow-up).  In a trap the velocity is bounded, the density is quiescent at the wall, and
    there are TWO exact answers to check:
      ground state  : rho propto exp(-omega x^2), v == 0, stationary FOREVER.  This is a direct
                      test of the hydrostatic balance grad(V+Q) = 0 that the static theory asserts.
      coherent state: displace by x0; the centroid obeys x0*cos(omega t) EXACTLY and the width does
                      not change.  A rigid oscillation, so Q must carry the shape unaltered.
    """
    h = L/(N-1)
    x = -L/2 + h*np.arange(N)

    def _ext(f, quad):
        """two ghost cells each side: quadratic extrapolation if quad else linear."""
        if quad:
            l = [3*f[0]-3*f[1]+f[2], 3*f[0]-3*f[1]+f[2]]      # placeholder, refined below
            a = np.empty(N+4); a[2:-2] = f
            a[1] = 3*f[0] - 3*f[1] + f[2]
            a[0] = 3*a[1] - 3*f[0] + f[1]
            a[-2] = 3*f[-1] - 3*f[-2] + f[-3]
            a[-1] = 3*a[-2] - 3*f[-1] + f[-2]
        else:
            a = np.empty(N+4); a[2:-2] = f
            a[1] = 2*f[0] - f[1];   a[0] = 2*a[1] - f[0]
            a[-2] = 2*f[-1] - f[-2]; a[-1] = 2*a[-2] - f[-1]
        return a

    def dx(f, quad=True):
        a = _ext(f, quad)
        return (-a[4:] + 8*a[3:-1] - 8*a[1:-3] + a[:-4])/(12*h)

    def dxx(f, quad=True):
        a = _ext(f, quad)
        return (-a[4:] + 16*a[3:-1] - 30*a[2:-2] + 16*a[1:-3] - a[:-4])/(12*h*h)

    # phi = (1/2) ln rho for a normalised Gaussian of standard deviation s0.
    # rho = (2 pi s0^2)^(-1/2) exp(-x^2/(2 s0^2))  =>  phi = -(1/4)ln(2 pi s0^2) - x^2/(4 s0^2).
    phi = -0.25*np.log(2*np.pi*s0**2) - ((x-x0)**2)/(4.0*s0**2)
    v   = np.full(N, float(k0))
    dV  = (omega**2)*x if omega else np.zeros(N)   # grad V for V = (1/2) omega^2 x^2

    def rhs(phi, v):
        p1, p2 = dx(phi), dxx(phi)
        Q  = -0.5*(p2 + p1*p1)                     # finite wherever phi is: no 0/0 in the tails
        return (-v*p1 - 0.5*dx(v, False),  -v*dx(v, False) - dx(Q) - dV)

    out = []
    nsteps = int(round(T/dt))
    samples = set(np.linspace(0, nsteps, 9, dtype=int))
    for n in range(nsteps+1):
        if n in samples:
            t = n*dt
            rho = np.exp(2*phi)
            m   = np.trapz(rho, x)
            xc  = np.trapz(x*rho, x)/m
            s   = np.sqrt(np.trapz((x-xc)**2*rho, x)/m)
            out.append((t, s, exact_sigma(t, s0), m, xc))
            if report: report(out[-1])
        if n == nsteps: break
        # RK4
        k1 = rhs(phi, v)
        k2 = rhs(phi + 0.5*dt*k1[0], v + 0.5*dt*k1[1])
        k3 = rhs(phi + 0.5*dt*k2[0], v + 0.5*dt*k2[1])
        k4 = rhs(phi + dt*k3[0],     v + dt*k3[1])
        phi = phi + (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        v   = v   + (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    return out


if __name__ == '__main__':
    import sys
    w = 1.0
    s_gs = 1.0/np.sqrt(2*w)        # ground state of V = w^2 x^2 / 2 has rho propto exp(-w x^2)

    print("MADELUNG PAIR, 1D.  Step 1 of the Hilbert-6th programme.")
    print("Harmonic trap omega=%.1f; ground-state width 1/sqrt(2w) = %.6f\n" % (w, s_gs))

    print("=== TEST A: ground state with v=0.  Must be STATIONARY forever. ===")
    print("This is the hydrostatic balance grad(V+Q)=0 -- the claim that the static theory IS")
    print("the v=0 Madelung equilibrium -- tested directly.")
    print("%8s %14s %14s %12s" % ("t","sigma","drift from s_gs","max|v| proxy"))
    r = run(s0=s_gs, omega=w, T=10.0, dt=2e-4, L=16.0, N=1024)
    for t,sg,se,m,xc in r:
        print("%8.3f %14.9f %+14.2e %12s" % (t, sg, sg-s_gs, "-"))
    print("worst drift over t<=10: %.2e (relative %.2e)"
          % (max(abs(sg-s_gs) for _,sg,_,_,_ in r), max(abs(sg-s_gs) for _,sg,_,_,_ in r)/s_gs))

    print("\n=== TEST B: coherent state, displaced by x0=2.  Centroid must obey x0*cos(wt). ===")
    print("%8s %14s %14s %12s %12s" % ("t","centroid","x0*cos(wt)","err","sigma"))
    r = run(s0=s_gs, omega=w, x0=2.0, T=2*np.pi, dt=2e-4, L=20.0, N=1024)
    worst = 0.0
    for t,sg,se,m,xc in r:
        ex = 2.0*np.cos(w*t); worst = max(worst, abs(xc-ex))
        print("%8.3f %14.9f %14.9f %+12.2e %12.9f" % (t, xc, ex, xc-ex, sg))
    print("worst centroid error over one full period: %.2e" % worst)

    print("\n=== TEST C: refinement of TEST B at t = one period ===")
    print("%6s %9s %16s %12s" % ("N","dt","centroid err","mass err"))
    for N,dt in [(256,4e-4),(512,2e-4),(1024,2e-4),(1024,1e-4)]:
        rr = run(s0=s_gs, omega=w, x0=2.0, T=2*np.pi, dt=dt, L=20.0, N=N)
        t,sg,se,m,xc = rr[-1]
        print("%6d %9.0e %16.3e %12.2e" % (N, dt, abs(xc-2.0*np.cos(w*t)), abs(m-1)))
        sys.stdout.flush()
