"""
ANGULAR CORRELATION IN THE PAIR TERM -- and it is NOT a geometric construction.

WHAT IT IS.  Keep rho_1 = rho_2 = psi^2 spherically symmetric and modify the pair repulsion by the
leading anti-correlated Legendre component:

    U(a) = INT INT rho rho / |r-r'|   -   a * (1/3) INT INT rho rho  r_< / r_>^2 .

a = 0 is the present homogenised shell.  Radial throughout: u = r*psi, u(0)=0, localisation
INT (u')^2, and NO softening -- which is why this is the most accurate of the day's constructions.

THE RESULT.  Fitting a per element to the exact two-electron energy:
        n=6000      a*        Z*a*      E(a*)-E(a=0)
        Z=2      0.2079     0.4158        -0.0421
        Z=3      0.1342     0.4026        -0.0440
        Z=4      0.1011     0.4044        -0.0457
    a* ~ 1/Z, so **Z*a* = 0.41 constant to +-1.6%** -- ONE universal number -- and the spread
    TIGHTENS under refinement (+-3.4% at n=3000).  The correction it supplies, -0.042 to -0.046 Ha,
    is the known near-constant correlation energy of the two-electron series (He: -0.042), which the
    fit was not told.

WHAT IT IS NOT, and an earlier version of this docstring had this wrong.  It said the correction
"does not touch the geometry" and "leaves every territory alone".  **There is no intra-shell geometry
in this model at all.**  Both electrons carry the same density and overlap completely; the avoidance
is a statistical property of the pair distribution, not a territory.  Between shells the geometry is
indeed untouched -- those interfaces remain determined by matching -- but within a shell RealQM's
geometric Pauli principle is absent and replaced by a correction term carrying a parameter, which is
structurally what a correlation functional does.

AND THE ATTEMPTED RESCUE FAILS.  Expanding the hemisphere indicator, 1_{z>0} = 1/2 + (3/4)P_1 - ...,
suggests a = 9/4 should BE the sharp split.  It does not: a = 2.25 gives -3.343 against the split's
-3.026, because the Legendre series ALTERNATES and truncating at P_1 over-counts the avoidance.  On
this family the sharp split sits at a_eff ~ 0.79, so a* = 0.208 is about **26% of complete hemispheric
avoidance** -- a clean measure of how far the sharp split over-avoids, consistent with its 4.2%
over-binding.

CONSEQUENCE.  The shell capacities 2 and 8 come from ANGULAR PACKING -- four charges as a tetrahedron
with no centre of inversion, two interleaved tetrahedra giving a centrosymmetric eight.  That argument
needs the sectors to exist.  This model has no angular structure, so it cannot supply the periodic
table.  It buys the energies by giving up the structure, and the construction that might do both --
sharp disjoint territories with the pair distribution corrected away from complete avoidance -- is
NOT this and has not been built.

Exact two-electron energies: H- -0.527751, He -2.903724, Li+ -7.279913, Be2+ -13.655566.
H- is untested here and is the obvious stress test, being weakly bound and diffuse.
"""
import os
import numpy as np
from scipy.optimize import minimize

np.seterr(all='ignore')
EXACT = {1: -0.527751, 2: -2.903724, 3: -7.279913, 4: -13.655566}


def pots(d, r, h):
    c  = np.cumsum(d*h)                          # INT_0^r d
    t  = np.cumsum((d*h/r)[::-1])[::-1] - d*h/r  # INT_r^inf d/s
    V0 = c/r + t
    cs = np.cumsum(d*r*h)                                    # INT_0^r s d
    ts = np.cumsum((d*h/(r*r))[::-1])[::-1] - d*h/(r*r)      # INT_r^inf d/s^2
    V1 = (cs/(r*r) + r*ts)/3.0
    return V0, V1


def energy_grad(u, r, h, Z, a):
    d = u*u
    du = np.empty_like(u); du[0] = u[0]/h; du[1:] = (u[1:]-u[:-1])/h
    T = float(np.sum(du*du)*h)
    gT = np.zeros_like(u)                        # d/du of INT (u')^2
    gT += 2*du/h*h
    gT[:-1] -= 2*du[1:]/h*h
    Ven = -2.0*Z*float(np.sum(d/r)*h)
    V0, V1 = pots(d, r, h)
    U0 = float(np.sum(V0*d)*h); U1 = float(np.sum(V1*d)*h)
    E = T + Ven + U0 - a*U1
    g = gT - 4.0*Z*u/r*h + 4.0*u*(V0 - a*V1)*h
    return E, g


def solve(Z, a, L=30.0, n=4000, rounds=9, mu0=400.0):
    h = L/(n+1); r = h*np.arange(1, n+1)
    u = r*np.exp(-Z*r); u /= np.sqrt(np.sum(u*u*h))
    lam, mu = 0.0, mu0
    for _ in range(rounds):
        def f(x):
            E, g = energy_grad(x, r, h, Z, a)
            c = float(np.sum(x*x*h)) - 1.0
            return E - lam*c + 0.5*mu*c*c, g + (-lam + mu*c)*2*x*h
        u = minimize(f, u, jac=True, method='L-BFGS-B',
                     options=dict(maxiter=6000, ftol=1e-15, gtol=1e-12)).x
        c = float(np.sum(u*u*h)) - 1.0
        lam -= mu*c; mu *= 3.0
        if abs(c) < 1e-12: break
    return energy_grad(u, r, h, Z, a)[0]


def fit_a(Z, n, lo=0.0, hi=4.0):
    for _ in range(22):
        m = 0.5*(lo+hi)
        if solve(Z, m, n=n) > EXACT[Z]: lo = m
        else: hi = m
    return 0.5*(lo+hi)


if __name__ == '__main__':
    print("ANGULAR-CORRELATED SHELL -- radial, u = r*psi, NO softening\n")
    for n in (3000, 6000):
        print(f"  --- n = {n} ---")
        print(f"{'Z':>4s}{'E(a=0)':>12s}{'err':>9s}{'exact':>12s}{'a*':>9s}{'E(a*)':>12s}")
        for Z in (2, 3, 4):
            e0 = solve(Z, 0.0, n=n); ex = EXACT[Z]
            a = fit_a(Z, n); ea = solve(Z, a, n=n)
            print(f"{Z:4d}{e0:12.5f}{100*(e0-ex)/abs(ex):+8.2f}%{ex:12.5f}{a:9.4f}{ea:12.5f}",
                  flush=True)
        print()
    print("  a=0 is the present homogenised shell.  a* is fitted per Z only to ask")
    print("  whether it is the SAME number -- which is the whole question.")
