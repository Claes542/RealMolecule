"""
HILBERT-6: DOES THE STRESS ROUTE AGREE WITH THE ENERGY ROUTE FOR ELASTICITY?

THE QUESTION.  Section "Direct upscaling" of the paper gets elastic coefficients as second
derivatives of the ground-state energy with respect to an imposed macroscopic strain:
C = (1/V) d^2E/de^2.  Nothing is averaged; a scalar is differentiated.  That route is deliberately
blind to how the energy divides between electric and localisation parts.

Phase 2 established the other route at two domains: the static balance div T = -rho grad V, with
T the localisation (quantum) stress.  In a periodic crystal that balance says the TOTAL momentum
flux through any plane is the same number, and thermodynamics says that number is the pressure.
So there are two independent computations of the same quantity and they must agree:

        sigma(z) = T_q(z) + F_coul(z)        must be INDEPENDENT of z   (the balance)
        sigma                                must equal  -dE/da         (the energy route)
        C = a d(sigma)/da                    must equal  a d^2E/da^2    (the modulus)

If they disagree the direct-upscaling claim is in trouble, because the two are the same physics.
If they agree the claim gains what it presently lacks: a check that the number it differentiates is
the number a stress balance would have given.

THE SYSTEM.  A 1D chain, period a, one kernel of charge +1 per cell and one electron per cell.
Each electron occupies its own cell -- RealQM's non-overlap, with the free interfaces sitting at the
cell faces by symmetry, which is exactly the Wigner-Seitz construction the paper's sodium
calculation uses ("one domain per ion filling the cell with the normal derivative vanishing at the
face by symmetry").  So: Neumann at both faces, kernel at the cell centre, and the electron feels
ALL kernels but only the OTHER electrons -- the model's own exclusion rule, no self-interaction.

WHY THE LATTICE SUMS CONVERGE.  Term by term the Coulomb sums diverge logarithmically in 1D.
Summed as NEUTRAL CELLS (kernel plus its electron) they converge fast: each cell is inversion
symmetric about its kernel, so it carries no dipole, and the leading interaction between two cells
is quadrupolar, ~1/d^3.  All sums here are organised that way and the truncation M is laddered.

TWO LESSONS FROM THE EARLIER PHASES, APPLIED HERE
  * eps is held FIXED, not tied to h.  eps = 0.285 h is the atom.js calibration; it sharpens the
    kernel as the mesh refines, so an h-ladder would measure two things at once (Phase 2).
  * the mesh STRAINS WITH THE CELL: n is fixed and h = a/n, which is how a strain is imposed on a
    periodic cell, and it keeps the discretisation commensurate with the kernel position at every a
    rather than letting the kernel drift between cells (the egg-crate trap).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

np.seterr(all='ignore')


def cell_grid(a, n):
    """n cell centres on [-a/2, a/2]; n ODD puts a cell centre exactly on the kernel at z=0."""
    h = a/n
    z = -a/2 + (np.arange(n) + 0.5)*h
    return z, h


def solve_chain(a, n=81, eps=0.35, M=40, iters=200, tol=1e-12):
    """Ground state of the periodic chain at period a. Returns fields and the energy per cell."""
    assert n % 2 == 1, "use odd n so a cell centre sits on the kernel"
    z, h = cell_grid(a, n)
    ms = np.arange(-M, M+1)

    # kernel potential from ALL kernels (the electron is attracted to its own, too)
    K = np.zeros(n)
    for m in ms:
        K += 1.0/np.sqrt((z - m*a)**2 + eps*eps)

    # -(1/2) d^2/dz^2 with Neumann at BOTH faces (the free interfaces, by symmetry)
    main = np.full(n, 1.0/h**2); main[0] -= 0.5/h**2; main[-1] -= 0.5/h**2
    off = np.full(n-1, -0.5/h**2)
    Lap = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

    rho = np.full(n, 1.0/a)
    E = None
    for it in range(iters):
        # Hartree from the OTHER cells only (self-exclusion is the house rule)
        P = np.zeros(n)
        for m in ms:
            if m == 0:
                continue
            P += np.array([np.sum(rho*h/np.sqrt((zi - (z + m*a))**2 + eps*eps)) for zi in z])
        H = Lap + np.diag(-K + P)
        val, vec = np.linalg.eigh(H)
        u = vec[:, 0]
        if u[np.argmax(np.abs(u))] < 0:
            u = -u
        u = u/np.sqrt(np.sum(u*u*h))
        rho_new = u*u
        rho = 0.5*rho + 0.5*rho_new
        rho = rho/np.sum(rho*h)
        T = float(u @ (Lap @ u))*0 + float(np.sum(u*(Lap @ u))*h)/np.sum(u*u*h)
        Enew = float(val[0])
        if E is not None and abs(Enew - E) < tol:
            break
        E = Enew
    u = np.sqrt(rho)

    # ---- energy per cell, pairs counted once, grouped as neutral cells -------------------------
    Tkin = float(np.sum(u*(Lap @ u))*h)
    U_eK_self = -float(np.sum(rho*(1.0/np.sqrt(z*z + eps*eps))*h))
    U_cells = 0.0
    for m in ms:
        if m == 0:
            continue
        eK = -float(np.sum(rho*(1.0/np.sqrt((z - m*a)**2 + eps*eps))*h))      # e_0 with K_m
        ee = float(np.sum(np.array([np.sum(rho*h/np.sqrt((zi - (z + m*a))**2 + eps*eps))
                                    for zi in z])*rho*h))                      # e_0 with e_m
        KK = 1.0/np.sqrt((m*a)**2 + eps*eps)                                   # K_0 with K_m
        U_cells += eK + 0.5*ee + 0.5*KK
    E_cell = Tkin + U_eK_self + U_cells
    return dict(z=z, h=h, a=a, n=n, eps=eps, M=M, u=u, rho=rho, K=K,
                E=E_cell, lam=val[0], iters=it+1)


# ----------------------------------------------------------------------------- the two routes
def d4(f, h):
    """4th-order first derivative, one-sided at the ends."""
    g = np.empty_like(f); m = len(f)
    g[2:m-2] = (f[0:m-4] - 8*f[1:m-3] + 8*f[3:m-1] - f[4:m])/(12*h)
    for i, j0 in ((0, 0), (1, 0), (m-2, m-5), (m-1, m-5)):
        A = np.vander(np.arange(5) - (i - j0), 5, increasing=True).T
        b = np.zeros(5); b[1] = 1.0
        g[i] = (np.linalg.solve(A, b) @ f[j0:j0+5])/h
    return g


def quantum_stress(S):
    """T_zz = -(1/2)(u u'' - (u')^2), with the Neumann (even) continuation at both faces."""
    u, h = S['u'], S['h']
    ue = np.concatenate([u[2::-1], u, u[:-4:-1]])          # even reflection, 3 ghosts each side
    up = d4(ue, h)
    upp = d4(up, h)
    return (-0.5*(ue*upp - up*up))[3:-3]


def coulomb_flux(S, Mf=60):
    """Net electrostatic force exerted by everything LEFT of the cell face on everything RIGHT.

    Taken at the cell face z = +a/2, which is the free interface.  Left = cells ... -1, 0;
    right = cells 1, 2, ...  Each side is grouped into NEUTRAL cells so the sum converges, and the
    model's exclusion rule needs no special care here because no charge interacts with itself.
    """
    z, h, a, eps, rho = S['z'], S['h'], S['a'], S['eps'], S['rho']
    F = 0.0
    for i in range(-Mf, 1):                 # source cells on the left, including cell 0
        for j in range(1, Mf+1):            # target cells on the right
            zi, zj = z + i*a, z + j*a
            Ki, Kj = i*a, j*a
            d = zj[None, :] - zi[:, None]
            r3 = (d*d + eps*eps)**1.5
            # electron i on electron j
            F += float(np.sum((rho[:, None]*h)*(rho[None, :]*h)*d/r3))
            # electron i on kernel j  (kernel charge +1, electron charge -1 -> attraction)
            dk = Kj - zi
            F -= float(np.sum(rho*h*dk/(dk*dk + eps*eps)**1.5))
            # kernel i on electron j
            dk2 = zj - Ki
            F -= float(np.sum(rho*h*dk2/(dk2*dk2 + eps*eps)**1.5))
            # kernel i on kernel j
            dk3 = Kj - Ki
            F += dk3/(dk3*dk3 + eps*eps)**1.5
    return F


if __name__ == '__main__':
    n = int(os.environ.get('N', 81))
    eps = float(os.environ.get('EPS', 0.35))
    M = int(os.environ.get('M', 40))
    Mf = int(os.environ.get('MF', 25))
    a0 = float(os.environ.get('A', 3.0))

    print("HILBERT-6 -- elasticity two ways: stress balance vs energy derivative")
    print(f"1D chain, one kernel (+1) and one electron per cell, Neumann at the faces")
    print(f"n={n} cells, eps={eps} (FIXED, not tied to h), lattice M={M}, flux M={Mf}\n")

    print("ENERGY ROUTE: E(a), and the pressure and modulus from it")
    print("     a        E(a)          -dE/da        a d2E/da2")
    da = 0.05
    As = [a0 - 2*da, a0 - da, a0, a0 + da, a0 + 2*da]
    Es = {}
    for a in As:
        Es[a] = solve_chain(a, n=n, eps=eps, M=M)['E']
    E_m2, E_m1, E_0, E_p1, E_p2 = (Es[a] for a in As)
    dEda = (E_m2 - 8*E_m1 + 8*E_p1 - E_p2)/(12*da)
    d2E = (-E_m2 + 16*E_m1 - 30*E_0 + 16*E_p1 - E_p2)/(12*da*da)
    print(f"  {a0:6.3f} {E_0:13.8f} {-dEda:13.8f} {a0*d2E:13.8f}\n")

    print("STRESS ROUTE at the same a: sigma = T_q(face) + Coulomb flux through the face")
    S = solve_chain(a0, n=n, eps=eps, M=M)
    Tq = quantum_stress(S)
    Tface = 0.5*(Tq[0] + Tq[-1])          # both faces are the same plane of the lattice
    Fc = coulomb_flux(S, Mf=Mf)
    print(f"     T_q at the face = {Tface:13.8f}")
    print(f"     Coulomb flux    = {Fc:13.8f}")
    print(f"     sigma total     = {Tface + Fc:13.8f}")
    print(f"     -dE/da          = {-dEda:13.8f}")
    print(f"     relative difference = {abs((Tface+Fc) - (-dEda))/max(abs(dEda),1e-30):.3e}")
