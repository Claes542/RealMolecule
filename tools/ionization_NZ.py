"""
N(Z): HOW MANY ELECTRONS DOES A KERNEL OF CHARGE Z BIND?  The ionization conjecture, in RealQM.

THE PROBLEM.  N(Z) is the largest N for which a nucleus of charge Z binds N electrons, i.e. for
which E(N) < E(N-1).  Empirically no free atom binds more than one extra electron, and the
conjecture N(Z) <= Z+1 is OPEN for the N-electron Schrodinger equation -- Lieb proved N < 2Z+1,
Lieb-Sigal-Simon-Thirring proved N(Z)/Z -> 1, Solovej proved N <= Z+C for Hartree-Fock, and the
sharp statement is the first item on Simon's problem list for this century.

In RealQM the question is not operator-theoretic but direct: put N unit charge densities on
non-overlapping domains around a kernel of charge Z, minimise the Coulomb energy, and ask whether
the outermost domain stays localised or runs off.  That is a computation with a yes/no answer.

THE VARIABLE, and why the obvious one fails.  A previous attempt (tools/hilbert6_li_allelectron.py)
worked with the density on radial shells and COLLAPSED: with free boundaries on both faces of the
innermost shell a CONSTANT density is admissible and carries zero localisation energy, so nothing
opposes contraction onto the nucleus -- the house rule "a uniform density plus a free boundary is
unstable", running inward.  The cure is u = r*psi with u(0)=0, in which the localisation term is
        (1/2) int (u' - u/r)^2 dr
and concentration at the origin is penalised correctly.  This is parameter-free: no softening, no
inner cutoff, no pseudopotential.  Z is the only input.

IT HAS AN EXACT SELF-TEST, which is run before anything else.  For hydrogen psi = e^-r/sqrt(pi),
so u = 2 r e^-r, and then u' - u/r = -u exactly, giving T = 1/2 and V = -int u^2/r = -1, hence
E = -1/2.  Any discretisation that does not return -0.5 here is not to be trusted downstream.

THE DECISIVE METHODOLOGICAL POINT: THE VERDICT MUST BE BOX-INDEPENDENT.  A confined solver always
reports a bound state, because the electron cannot leave, so a single run would "confirm" binding
for every N and prove nothing.  Both the energy and the outermost domain's EXTENT are therefore
laddered in the box size L:
    bound    ->  E converges below E(N-1), and the outer radius SATURATES;
    unbound  ->  E -> E(N-1) FROM ABOVE, and the outer radius grows in proportion to L.
The extent is the primary signal, because it is structural, which is the regime this model is
reliable in; the energy is secondary.

RESULT 2026-10-20: THE PARAMETER-FREE FORM CANNOT BE RUN, AND THE MECHANISM IS NOW EXACT.
The hydrogen self-test PASSES -- E(H) = -0.5000705, -0.5000461, -0.5000211 at (L,n) = (20,800),
(30,1500), (40,3000), converging on -0.5 -- so the variable and the discretisation are right.
But H- and H(2-) collapse.  Measured E(H-) = -5.22, -3.39, -2.36, -1.64, -1.18 at L = 10, 15, 22,
32, 45 with h held fixed: far below the exact -0.5278, tracking the BOX rather than converging, and
with the energy-minimising inner shell always the smallest the candidate list allows.

THE CAUSE, confirmed to three figures.  The inner shell collapses to the UNIFORM-DENSITY MODE.
In this variable that mode is u ~ r, since u = r*psi and psi = const is a uniform density, and it
carries ZERO localisation energy: u' - u/r = 0 identically for u proportional to r.  So nothing
opposes contraction, and the energy is just the collapse energy -1.5 Z / R1 of a uniform blob.
Measured, at L = 10, 15, 22, 32:
    correlation of the converged inner u with the linear function r:  0.9998, 0.9994, 0.9988, 0.9970
    E_total against -1.5 Z / R1:  -5.225/-5.357,  -3.388/-3.409,  -2.365/-2.344,  -1.640/-1.596
So changing to u = r*psi cured the ORIGIN singularity and did nothing about this, because a uniform
density is zero-energy in EVERY variable.  That is the house rule of CLAUDE.md -- "a uniform density
plus a free boundary is unstable" -- and it does not care how the problem is parametrised.

WHAT THIS SETTLES.  A bare kernel meeting a free interface has NO stable minimum in this model, now
confirmed in two independent discretisations (this one and the density form of
hilbert6_li_allelectron.py).  The inner cutoff r_c is therefore not a convenience of the
implementation but a NECESSITY of the construction, and the N(Z) test cannot be run parameter-free.

HOW IT COULD STILL BE RUN, with the parameter conceded: the construction that works elsewhere is
valence-only on a SCREENED kernel with a cutoff r_c.  For H- there is no core to screen, so H- is
two valence domains on a Z=1 kernel with a cutoff, and the honest question becomes whether the
NON-binding of H(2-) is robust as r_c is laddered.  That is weaker than a parameter-free verdict but
still a real test, since the conjecture's content is a sign and not a magnitude.

KNOWN ANSWERS TO HIT: E(H) = -0.5 exactly; E(H-) = -0.5278, so the first extra electron binds by
0.0277 Ha; H(2-) does not exist, so the second must come out unbound.  Note that 0.0277 Ha is only
about 17 kcal/mol, which sits near the recorded step-convergence floor, so hydrogen is the tightest
case in the whole table as well as the sharpest.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import eigh_tridiagonal

np.seterr(all='ignore')


def grid(L, n):
    h = L/(n + 1)
    r = h*np.arange(1, n + 1)          # u(0)=0 is enforced by omitting r=0
    rm = h*(np.arange(1, n + 1) - 0.5)  # midpoints r_{j-1/2}
    return r, rm, h


def kinetic_tridiag(r, rm, h, lo, hi):
    """(1/2) int (u' - u/r)^2 dr on nodes [lo,hi), as a tridiagonal form divided by h.

    On the midpoint between nodes j and j+1:  u' = (u_{j+1}-u_j)/h,  u/r = (u_j+u_{j+1})/(2 r_mid).
    So the midpoint term is A u_{j+1} + B u_j with A = 1/h - 1/(2 r_mid), B = -1/h - 1/(2 r_mid),
    and the form is (h/2) sum (A u_{j+1} + B u_j)^2.  Tridiagonal, and symmetric by construction.
    The faces at lo and hi are simply omitted, which IS the free-interface (natural) condition.
    """
    k = hi - lo
    d = np.zeros(k)
    e = np.zeros(max(k - 1, 0))
    for j in range(lo, hi - 1):
        rmid = 0.5*(r[j] + r[j+1])
        A = 1.0/h - 1.0/(2*rmid)
        B = -1.0/h - 1.0/(2*rmid)
        d[j-lo] += 0.5*B*B
        d[j-lo+1] += 0.5*A*A
        e[j-lo] += 0.5*A*B
    # the inner face of the innermost shell: u(0)=0 pairs node lo with the origin
    if lo == 0:
        rmid = 0.5*r[0]
        A = 1.0/h - 1.0/(2*rmid)
        d[0] += 0.5*A*A
    return d, e


def hartree(u2, r, h):
    """V(r) = (1/r) int_0^r u^2 dr' + int_r^inf u^2/r' dr', for a spherical unit shell."""
    q_in = np.cumsum(u2*h)
    tail = np.cumsum((u2*h/r)[::-1])[::-1] - u2*h/r
    return q_in/r + tail


def solve_atom(Z, N, L=30.0, n=1500, iters=300, tol=1e-11, splits=None, verbose=False):
    """N electron domains as nested radial shells on a kernel Z. Returns E and the shell extents."""
    r, rm, h = grid(L, n)
    Vnuc = -Z/r
    cands = splits if splits is not None else _splits(N, n)
    best = None
    for bnd in cands:
        bounds = list(zip([0] + list(bnd), list(bnd) + [n]))
        if min(b - a for a, b in bounds) < 6:
            continue
        forms = [kinetic_tridiag(r, rm, h, a, b) for a, b in bounds]
        u2 = []
        for a, b in bounds:                       # seed
            v = np.zeros(n); v[a:b] = r[a:b]*np.exp(-r[a:b]/max(1.0, Z))
            v /= np.sqrt(np.sum(v*v*h)); u2.append(v*v)
        E = None
        for it in range(iters):
            new = []
            for s, (a, b) in enumerate(bounds):
                P = np.zeros(n)
                for t in range(N):
                    if t != s:
                        P += hartree(u2[t], r, h)
                d, e = forms[s]
                d = d + (Vnuc + P)[a:b]
                val, vec = eigh_tridiagonal(d, e, select='i', select_range=(0, 0))
                v = np.zeros(n)
                v[a:b] = vec[:, 0]/np.sqrt(np.sum(vec[:, 0]**2*h))
                new.append(v*v)
            u2 = [0.5*o + 0.5*nn for o, nn in zip(u2, new)]
            u2 = [q/np.sum(q*h) for q in u2]
            T = 0.0
            for s, (a, b) in enumerate(bounds):
                d, e = forms[s]
                uu = np.sqrt(u2[s][a:b])
                # the quadratic form carries the measure: T = h * u^T M u.  The eigenproblem
                # does not need it (it cancels in the Rayleigh quotient) but the ENERGY does --
                # omitting it made E(H) come out at +19 instead of -0.5, growing as 0.5/h.
                T += h*float(np.sum(d*uu*uu) + 2*np.sum(e*uu[:-1]*uu[1:]))
            Ven = sum(float(np.sum(q*Vnuc*h)) for q in u2)
            Vee = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    Vee += float(np.sum(hartree(u2[j], r, h)*u2[i]*h))
            Enew = T + Ven + Vee
            if E is not None and abs(Enew - E) < tol:
                break
            E = Enew
        if best is None or Enew < best[0]:
            # outermost domain extent: the radius containing 90% of its charge
            q = np.cumsum(u2[-1]*h)
            r90 = float(r[np.searchsorted(q, 0.90)]) if q[-1] > 0.9 else float(r[-1])
            best = (Enew, bnd, r90, it+1)
    return dict(E=best[0], split=best[1], r90=best[2], iters=best[3], Z=Z, N=N, L=L, n=n)


def _splits(N, n):
    if N == 1:
        return [()]
    if N == 2:
        return [(int(f*n),) for f in (0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5)]
    if N == 3:
        out = []
        for f1 in (0.02, 0.04, 0.07, 0.11):
            for f2 in (0.15, 0.25, 0.4, 0.6):
                if f1 < f2:
                    out.append((int(f1*n), int(f2*n)))
        return out
    raise NotImplementedError


if __name__ == '__main__':
    print("SELF-TEST FIRST: hydrogen must return E = -0.5 exactly (no Hartree term involved)")
    print("      L      n        E(H)         error")
    for (L, n) in ((20.0, 800), (30.0, 1500), (40.0, 3000), (40.0, 6000)):
        S = solve_atom(1, 1, L=L, n=n)
        print(f"  {L:6.1f} {n:6d} {S['E']:12.7f} {S['E']+0.5:+12.2e}", flush=True)

    print("\nBOX LADDER on H / H- / H(2-).  Bound: E settles below E(N-1) and r90 saturates.")
    print("                   Unbound: E -> E(N-1) from above and r90 grows with L.")
    print("      L     N    E(N)         E(N)-E(N-1)      r90 (a0)   split")
    n = int(os.environ.get('N', 1500))
    prev = {}
    for L in [float(x) for x in os.environ.get('LS', '12,20,30,45').split(',')]:
        row = {}
        for N in (1, 2, 3):
            S = solve_atom(1, N, L=L, n=n)
            row[N] = S['E']
            dif = S['E'] - row[N-1] if N-1 in row else float('nan')
            print(f"  {L:6.1f} {N:4d} {S['E']:12.7f} {dif:+15.7f} {S['r90']:11.3f}   {S['split']}",
                  flush=True)
        print()
