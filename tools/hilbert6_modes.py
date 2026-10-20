"""
HILBERT-6 ITEM 1, by normal modes: is the two-domain free boundary WELL POSED, and does it carry
the one mode whose frequency is known exactly?

WHY MODES AND NOT TIME STEPPING.  The Madelung pair is dispersive -- d_x Q carries a third
derivative of the density -- so explicit stepping needs dt <~ C dx^3 (measured in
tools/madelung_1d.py).  One period of the test oscillation at N=200 is then ~5e6 steps, each
carrying an O(N^2) Hartree sum: ~1e11 operations, which is not feasible in numpy.  Linearising instead
costs one dense eigensolve and answers the two questions that matter:

  1. WELL-POSEDNESS, which is the programme's stated failure condition.  "Ill-posed free boundary
     (no dynamics past one electron)" would confine the whole programme.  Linearly, well posed means
     the squared frequencies Omega^2 come out REAL and NON-NEGATIVE.  A negative Omega^2 is an
     exponentially growing mode; a complex one means the initial-value problem is ill posed in the
     Hadamard sense, which is what happens to free boundaries that have the wrong sign of surface
     condition.

  2. AN EXACT FREQUENCY.  In a harmonic trap the centre of mass obeys Xcm'' = -w^2 Xcm exactly,
     because the trap is quadratic and every internal force -- the interaction AND the non-overlap
     constraint, which translates with the system -- is translation invariant.  So among the modes
     there must be one at EXACTLY Omega = w, whose eigenvector is a rigid translation and whose
     interface displacement is non-zero.  That is a right answer to hit, and it is sensitive to the
     interface condition: get the free-boundary condition wrong and the translation is no longer a
     solution, so the mode moves off w.

THE REGULAR FORM, and why the obvious one is unusable.  Written in (d_rho, d_v) the eigenproblem
carries 1/u0 twice, through d_Q = -(1/2)(1/u0)[d_u_xx - (u0_xx/u0) d_u] with d_u = d_rho/(2 u0).
In the tails u0 ~ 1e-14, so the operator amplifies boundary noise by ~1e28 and the computed spectrum
is boundary garbage: MEASURED, the one-domain self-test returned Omega/w = 0.899, 1.866, 2.799
instead of 1, 2, 3, and gave the SAME wrong numbers at n = 200, 400 and 800 -- n-independence being
the signature of a formulation error rather than a discretisation one.
The cure is to use f = d_u as the variable and the ground-state equation u0_xx = 2(V-lambda) u0,
which removes every 1/u0:
        d_Q = (1/u0) (H - lambda) f ,        H = -(1/2) d_xx + V ,
and then, using d/dx( rho0 d/dx (g/u0) ) = -2 u0 (H - lambda) g,
        Omega^2 f = (H - lambda) [ (H - lambda) f + u0 d_V ] .
With no interaction this is Omega^2 = (E_n - lambda)^2, i.e. Omega = E_n - E_0 exactly -- the
harmonic ladder n*w -- which is the self-test.

THE INTERFACE IS NOT AN INDEPENDENT UNKNOWN.  Charge conservation in the domain,
int_{-L}^{s} rho dx = 1 for all time, gives to first order
        d_s = -(1/rho0(0)) int d_rho dx ,
so the interface displacement is a LINEAR FUNCTIONAL of the density perturbation, and the moving
Neumann condition f_x(0) = -u0_xx(0) d_s becomes an inhomogeneous, NONLOCAL boundary condition -- a
rank-one modification of the operator's boundary row.  That is the whole content of the free
boundary at linear order, and it is what makes the problem different from a fixed-wall one.

THE LINEARISATION.  About the static pair (v=0, interface at x=0), with rho = rho0 + d_rho,
v = d_v, interface at x = d_s:
        d_rho_t = -(rho0 d_v)_x
        d_v_t   = -(d_V + d_Q)_x
so  -Omega^2 d_rho = -d/dx( rho0 d/dx ( d_Q + d_V ) ).
With u0 = sqrt(rho0) and d_u = d_rho/(2 u0),
        d_Q = -(1/2)(1/u0)[ d_u_xx - (u0_xx/u0) d_u ].
d_V is the Hartree response of the PARTNER domain.  A rigid translation is ANTISYMMETRIC about the
interface: translating the pair by A gives d_rho_left(x) = -A rho0'(x) and
d_rho_right(x) = -d_rho_left(-x), so the partner's response enters with a minus sign and a
reflection.  The Kohn mode therefore lives in the antisymmetric sector, which is the one solved here.

THE FREE-BOUNDARY CONDITION, linearised -- this is the part being tested.  The model's interface
condition is homogeneous Neumann on u, imposed on the interface wherever it IS: u_x = 0 at
x = s(t).  Expanding about x = 0, where u0_x(0) = 0 already,
        d_u_x(0) + u0_xx(0) d_s = 0 ,
which is how the interface displacement couples into the field problem.  Together with the material
condition s_t = v(0) this closes the system.  Note what it says: the interface does not sit still
and the field does not satisfy a fixed Neumann condition -- the two are tied, and d_s is determined
by the mode rather than prescribed.

SELF-TEST WITH AN EXACT SPECTRUM.  Before trusting any of it, the same machinery is run on ONE
domain in a harmonic trap with no partner and no interface, where the linearised Madelung spectrum
is known exactly: Omega = w, 2w, 3w, ... (the harmonic-oscillator excitation ladder).  If that comes
out, the operators are right and the two-domain answer can be read.

STATUS 2026-10-16 -- ONE RESULT, ONE THING NOT YET RIGHT. Read before quoting any number from here.

VERIFIED, and it is the physics finding: the linearised two-domain free-boundary problem DOES carry
the exact Kohn mode, provided the Hartree response includes the INTERFACE SURFACE TERM.  Checked
independently of the eigensolver, by applying the operator to the known translation f = u0':
  * d_s recovered from charge conservation  = -0.999985  against the exact -1
  * d_V against P':  volume term alone, relative error 2.0 (wrong, including sign);
    volume + surface term, 1.4e-4
  * (H-lam)[(H-lam) f + u0 d_V] = w^2 f  to 2.2e-3 in the interior
So a fixed-domain linearisation gets this qualitatively wrong, and the correction is a surface term
living on the free boundary -- which is exactly the object the dissipation question is about.

NOT YET ESTABLISHED: well-posedness, and the spectrum printed below is NOT the spectrum of the free
boundary problem.  The translation satisfies the interior rows to 2.2e-3 but violates the four
interface rows by up to 4e3, and no computed eigenvalue sits anywhere near w (the lowest is 1.637 w,
unchanged whether the free-boundary terms are on or off).  The "no negative, no complex eigenvalues"
readout therefore proves nothing yet.

PROGRESS 2026-10-20 -- THE OBSTRUCTION IS NOW PRECISE. Three things settled, one still open.

(a) THE INTERFACE'S OWN EQUATION OF MOTION, derived.  Differentiating the material condition
    s_t = v(s) once more gives s_tt = v_t(0) = -d_x(g/u0)|_0, so in the frequency domain
            d_x(g/u0)|_0 = Omega^2 d_s ,
    which is the boundary condition the second (H-lam) factor needs.  It depends on OMEGA, so the
    free-boundary linearisation is a GENERALISED eigenproblem, not a standard one -- which is why
    no correction to a single matrix could ever have worked.

(b) BUT THAT CONDITION IS REDUNDANT, and the redundancy is exact.  Imposing it makes the mass
    matrix M = I + (u0(0)/2h) e w_int^T, whose determinant is 1 + (u0(0)/2h)*w_int[-1], and
            w_int[-1] = -2 h / u0(0)   =>   (u0(0)/2h)*w_int[-1] = -1   exactly,
    so det M = 0 IDENTICALLY, not as a numerical accident.  The acceleration condition is already
    implied by charge conservation (which is the material condition integrated once), so the
    interface is NOT an independent degree of freedom at linear order and its condition cannot be
    imposed as an extra row.  Measured: the generalised solve returns complex eigenvalues and a
    non-converging spectrum, as a singular pencil must.

(c) THE FIRST-ORDER SYSTEM in (d_rho, d_v, d_s) confirms the physics and fails on conditioning.
    Built with each condition applied exactly once, it satisfies CONTINUITY exactly for the known
    Kohn mode (residual 0.0), and the interface SURFACE TERM improves the momentum residual by a
    factor 3.8 -- 0.316 volume-only against 0.084 with it -- which is independent confirmation,
    from a different formulation, that the surface term is required.  But 0.084 is far worse than
    the 2.2e-3 the twice-applied form achieves, and the reason is identified: mu = g/u0
    REINTRODUCES the division by u0 in the tails that the twice-applied form removes through
    d_x(rho0 d_x(g/u0)) = -2 u0 (H-lam) g.

(d) THE STRUCTURAL RESOLUTION, 2026-10-20, and what is left of it.

    Stop trying to settle well-posedness by computing eigenvalues.  The linearised problem is the
    dynamics about a CRITICAL POINT of an energy, with a positive-definite kinetic metric, so it is
    a SYMMETRIC generalised eigenproblem and nothing else is needed:
            A f = 2 Omega^2 B f,     A = Hessian of E,     B f = int W Wtilde / rho0,  W = int u0 f.
    Both forms are symmetric; B is positive definite because it is a kinetic energy.  Therefore
    **Omega^2 is REAL by construction**, and **Omega^2 >= 0 if and only if the second variation of
    the energy is non-negative on the admissible class** -- which the static state satisfies because
    it was obtained by minimisation.  So the programme's failure condition ("ill-posed free
    boundary") is answered structurally: ill-posedness would require the ground state not to be a
    minimum.  No spectrum has to be computed to know the sign.
    Derivation of B, for the record.  From rho0 d_v = -2 i Omega W and i Omega d_v = -d_x(g/u0),
    d_x(g/u0) = -2 Omega^2 W / rho0; testing against Wtilde and integrating by parts gives
            int ftilde g  -  Wtilde(0) g(0)/u0(0)  =  2 Omega^2 int Wtilde W / rho0 ,
    and with Wtilde(0) = -rho0(0) d_stilde / 2 the boundary term is +(u0(0)/2) d_stilde g(0).

    WHAT IS STILL MISSING, and it is one term.  That boundary term is the interface, and symmetry of
    the left-hand side REQUIRES g(0) proportional to d_s -- so the interface condition is not an
    extra row to impose but a CONSEQUENCE of self-adjointness, which is the resolution of the
    tension in (b).  It is not yet in the discrete A.  Measured without it: Omega^2 comes out real
    as predicted, but one eigenvalue is negative and DIVERGES with n (-66, -102, -138 at
    n=200,300,400), the spectrum does not converge, and the known translation gives a Rayleigh
    quotient of 3.72 w instead of w with residual 1.0.  The admissible class also still needs its
    constraint projected out (f proportional to u0 is pure normalisation).  So: the structure is
    settled, the discrete realisation is not, and the missing piece is identified exactly.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hilbert6_moving_interface import static_pair

np.seterr(all='ignore')


# --------------------------------------------------------------------------- operators
def D2_matrix(n, h, left='dirichlet', right='neumann'):
    """d^2/dx^2 on cell centres, with the stated conditions at the two outer faces."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -2.0
        if i > 0:
            A[i, i-1] = 1.0
        if i < n-1:
            A[i, i+1] = 1.0
    if left == 'neumann':
        A[0, 0] += 1.0
    if right == 'neumann':
        A[-1, -1] += 1.0
    return A/h**2


def D1_matrix(n, h):
    """d/dx on cell centres, 2nd order centred, one-sided at the ends."""
    A = np.zeros((n, n))
    for i in range(1, n-1):
        A[i, i-1] = -0.5; A[i, i+1] = 0.5
    A[0, 0] = -1.5; A[0, 1] = 2.0; A[0, 2] = -0.5
    A[-1, -1] = 1.5; A[-1, -2] = -2.0; A[-1, -3] = 0.5
    return A/h


# --------------------------------------------------------------------------- operators, regular form
def hamiltonian(n, h, V, bc_right='neumann'):
    """H = -(1/2) d_xx + V on cell centres; Dirichlet at the outer face, bc_right at the interface."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 1.0/h**2
        if i > 0:
            A[i, i-1] = -0.5/h**2
        if i < n-1:
            A[i, i+1] = -0.5/h**2
    if bc_right == 'neumann':
        A[-1, -1] -= 0.5/h**2
    return A + np.diag(V)


def modes_single(n=400, Lbox=8.0, w=1.0):
    """ONE domain in a trap, no interface: Omega^2 f = (H-lambda)^2 f, so Omega = n w exactly."""
    L = 2*Lbox
    x = -Lbox + (np.arange(n) + 0.5)*(L/n)
    h = L/n
    H = hamiltonian(n, h, 0.5*w*w*x*x, bc_right='dirichlet')
    ev = np.linalg.eigvalsh(H)
    lam = ev[0]
    A = (H - lam*np.eye(n)) @ (H - lam*np.eye(n))
    om2 = np.linalg.eigvalsh(A)
    om = np.sqrt(np.maximum(om2, 0.0))
    return np.sort(om)[1:]          # drop the zero mode (f = u0 itself)


def modes_pair(n=400, Lbox=8.0, w=1.0, delta=0.5, free_boundary=True, with_hartree=True):
    """Antisymmetric linear modes of the two-domain pair, in the regular variable f = d_u.

    free_boundary=True  : the model's moving-interface condition, f_x(0) tied to d_s by charge
                          conservation (the nonlocal rank-one row).
    free_boundary=False : a FIXED wall at the interface (plain Neumann, d_s forced to 0) -- the
                          control, which should NOT reproduce the exact frequency.
    """
    S = static_pair(n=n, Lbox=Lbox, w=w, delta=delta)
    x, h, rho0, u0 = S['x'], S['hx'], S['rho'], S['u']
    xm, rm = -x[::-1], rho0[::-1]
    P = np.array([np.sum(rm*h/np.sqrt((xi - xm)**2 + delta*delta)) for xi in x])
    V = 0.5*w*w*x*x + P
    H = hamiltonian(n, h, V, bc_right='neumann')
    lam = float(u0 @ (H @ u0))/float(u0 @ u0)
    Hl = H - lam*np.eye(n)

    # --- interface displacement, from charge conservation (NOT an independent unknown)
    w_int = -(2.0*u0*h)/rho0[-1]                     # d_s = w_int . f

    # --- Hartree response of the partner, antisymmetric: d_rho_right(y) = -d_rho_left(-y),
    #     PLUS the interface SURFACE term, which a fixed-domain linearisation misses.
    #     When the pair moves, the partner's EDGE moves too, so its charge changes by a sliver
    #     rho0(0) d_s at the interface:   d_V = int d_rho_R K dy  -  rho0(0) d_s K(x).
    #     Measured: with the volume term alone d_V is wrong by a relative 2.0 against P'; with the
    #     surface term it is right to 1.4e-4, and only then is a rigid translation an eigenvector.
    if with_hartree:
        G = -(1.0/np.sqrt((x[:, None] - xm[None, :])**2 + delta*delta)*h)[:, ::-1]
        dVmat = G @ np.diag(2*u0)                    # d_V as a matrix on f, since d_rho = 2 u0 f
        if free_boundary:
            Kvec = 1.0/np.sqrt(x*x + delta*delta)    # K(x-0), the kernel evaluated at the interface
            dVmat = dVmat - rho0[-1]*np.outer(Kvec, w_int)
        coupling = np.diag(u0) @ dVmat
    else:
        coupling = np.zeros((n, n))

    A = Hl @ (Hl + coupling)

    if free_boundary:
        # moving Neumann on the interface: f_x(0) = -u0_xx(0) d_s, with u0_xx(0) = 2(V(0)-lam) u0(0).
        # H was built assuming f_x(0)=0, so a prescribed gradient adds -g/(2h) to the interface row.
        bcrow = -2.0*(V[-1] - lam)*u0[-1]*w_int      # the required f_x(0), as a functional of f
        corr = np.zeros((n, n)); corr[-1, :] = -0.5*bcrow/h
        A = Hl @ (Hl + coupling + corr)

    ev = np.linalg.eigvals(A)
    re, im = ev.real, ev.imag
    bad_cplx = int(np.sum(np.abs(im) > 1e-6*np.maximum(np.abs(re), 1e-12)))
    good = (np.abs(im) <= 1e-6*np.maximum(np.abs(re), 1e-12))
    neg = re[good & (re < -1e-8)]
    om = np.sort(np.sqrt(np.maximum(re[good & (re > 1e-6)], 0.0)))
    return dict(om=om, n_neg=len(neg), n_cplx=bad_cplx,
                worst_neg=float(neg.min()) if len(neg) else 0.0, lam=lam)


if __name__ == '__main__':
    w = 1.0
    print("HILBERT-6 ITEM 1 by normal modes -- well-posedness, and the one exact frequency\n")
    print("SELF-TEST: one domain in a trap, no interface. Exact spectrum is w, 2w, 3w, ...")
    print("     n     first six Omega/w")
    for n in (200, 400, 800):
        om = modes_single(n=n, w=w)
        print(f"  {n:5d}     " + "  ".join(f"{o/w:7.4f}" for o in om[:6]), flush=True)

    print("\nTWO DOMAINS, antisymmetric sector -- the sector a rigid translation lives in.")
    print("The Kohn mode must sit at Omega/w = 1 EXACTLY, whatever the interaction.")
    print("     n   free bdry  hartree   Omega^2<0  complex    three lowest Omega/w")
    for (fb, ha) in ((True, True), (False, True), (True, False)):
        for n in (200, 400):
            r = modes_pair(n=n, w=w, free_boundary=fb, with_hartree=ha)
            print(f"  {n:5d}  {str(fb):>9s} {str(ha):>8s} {r['n_neg']:10d} {r['n_cplx']:9d}    "
                  + "  ".join(f"{o/w:8.5f}" for o in r['om'][:3]), flush=True)
