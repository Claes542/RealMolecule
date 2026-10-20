"""
HILBERT-6: SURFACE ENERGY FROM A FREE BOUNDARY -- the one elastic-body quantity that needs no
Cauchy-Born assumption.

WHY THIS ONE.  The direct route reaches the COEFFICIENTS of continuum mechanics by differentiating
a computed energy.  To get from coefficients to the statics of a BODY one more thing is needed: that
the material at each point behaves like a uniformly strained cell, so that a macroscopic stress
field sigma(x) exists.  That is the Cauchy-Born rule, and the direct route assumes it rather than
proving it -- the "eps << 1 carried as an assumption" of the paper.

Cauchy-Born fails exactly where elasticity is interesting: at surfaces, defects and crack tips.  In
a lattice model that is fatal, because a surface can only be made by truncating the lattice.  Here
it is not, because a surface is a FREE BOUNDARY, which is the object the theory is built on.  So a
surface energy is the natural first elastic-body quantity: it is a difference of computed energies,
it needs no scale separation, and it tests the formulation where it should be strongest.

THE CALCULATION.  Finite chain of N cells, period a, one kernel (+1) and one electron per cell.
Interior interfaces sit at the midpoints between kernels, which is exact in the bulk by symmetry.
The two END domains are NOT closed off: they run out into vacuum and find their own extent, which
is the free boundary being tested.  Then

        E(N) = N e_bulk + 2 gamma + o(1),

so a straight line through E(N) gives the bulk energy per cell as its SLOPE and twice the surface
energy as its INTERCEPT.  Two things are checked by this and not assumed:

  * the slope must agree with e_bulk computed independently from the PERIODIC calculation
    (tools/hilbert6_elastic.py) at the same a, h and eps -- two different codes, one number;
  * the intercept must be independent of N once N is large enough, which is what makes it a
    surface property rather than a fitting artefact.

Same discipline as the earlier phases: eps FIXED (not tied to h), the mesh identical to the
periodic run so the subtraction is meaningful, and every sum grouped so that it converges.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hilbert6_elastic import solve_chain

np.seterr(all='ignore')


def solve_finite_chain(N, a, ncell=41, eps=0.35, vac=6.0, iters=300, tol=1e-11):
    """N kernels at z = 0, a, ..., (N-1)a; N electron domains; the two end domains run free."""
    h = a/ncell
    zk = np.arange(N)*a                                   # kernel positions
    nvac = int(round(vac/h))
    # one global grid, cell centred, covering vacuum + chain + vacuum
    z0 = -vac
    ntot = nvac + (N-1)*ncell + nvac
    z = z0 + (np.arange(ntot) + 0.5)*h

    # domain boundaries: midpoints between kernels; ends are the box edges
    faces = [0]
    for i in range(N-1):
        faces.append(int(np.argmin(np.abs(z - (zk[i] + 0.5*a)))) + 1)
    faces.append(ntot)
    segs = [slice(faces[i], faces[i+1]) for i in range(N)]

    # kernel potential from all kernels, everywhere
    K = np.zeros(ntot)
    for zb in zk:
        K += 1.0/np.sqrt((z - zb)**2 + eps*eps)

    # Coulomb kernel matrix, once
    G = 1.0/np.sqrt((z[:, None] - z[None, :])**2 + eps*eps)*h

    # seed: each domain uniform on its own segment
    rho = np.zeros((N, ntot))
    for i, sg in enumerate(segs):
        rho[i, sg] = 1.0
        rho[i] /= np.sum(rho[i])*h

    E = None
    for it in range(iters):
        Pall = G @ rho.sum(axis=0)                        # potential of ALL electrons
        new = np.zeros_like(rho)
        lam = np.zeros(N)
        for i, sg in enumerate(segs):
            m = np.arange(ntot)[sg]
            k = len(m)
            Pi = Pall[m] - (G[np.ix_(m, m)] @ rho[i][m])  # exclude self: the house rule
            main = np.full(k, 1.0/h**2)
            main[0] -= 0.5/h**2                           # free/Neumann at both ends of the domain
            main[-1] -= 0.5/h**2
            off = np.full(k-1, -0.5/h**2)
            H = np.diag(main - K[m] + Pi) + np.diag(off, 1) + np.diag(off, -1)
            val, vec = np.linalg.eigh(H)
            u = vec[:, 0]
            if u[np.argmax(np.abs(u))] < 0:
                u = -u
            u = u/np.sqrt(np.sum(u*u*h))
            new[i][m] = u*u
            lam[i] = val[0]
        rho = 0.6*rho + 0.4*new
        for i in range(N):
            rho[i] /= np.sum(rho[i])*h

        # ---- total energy, pairs counted once --------------------------------------------------
        Tkin = 0.0
        for i, sg in enumerate(segs):
            m = np.arange(ntot)[sg]
            u = np.sqrt(rho[i][m])
            du = np.diff(u)/h
            Tkin += 0.5*float(np.sum(du*du))*h            # Neumann ends: no boundary term
        U_eK = -float(np.sum(rho.sum(axis=0)*K)*h)
        Pe = G @ rho.sum(axis=0)
        U_ee = 0.5*float(np.sum(rho.sum(axis=0)*Pe)*h)
        for i in range(N):                                # remove the self terms
            U_ee -= 0.5*float(np.sum(rho[i]*(G @ rho[i]))*h)
        U_KK = 0.0
        for i in range(N):
            for j in range(i+1, N):
                U_KK += 1.0/np.sqrt((zk[i]-zk[j])**2 + eps*eps)
        Enew = Tkin + U_eK + U_ee + U_KK
        if E is not None and abs(Enew - E) < tol:
            break
        E = Enew
    return dict(E=Enew, N=N, a=a, h=h, rho=rho, z=z, zk=zk, iters=it+1, lam=lam)


if __name__ == '__main__':
    a = float(os.environ.get('A', 3.0))
    ncell = int(os.environ.get('N', 41))
    eps = float(os.environ.get('EPS', 0.35))
    Ns = [int(x) for x in os.environ.get('NS', '3,4,5,6,7,8').split(',')]

    print("HILBERT-6 -- surface energy from the free boundary (no Cauchy-Born needed)")
    print(f"1D chain, a={a}, {ncell} mesh cells per period, eps={eps} fixed\n")

    print("PERIODIC reference (tools/hilbert6_elastic.py, different code path)")
    Sp = solve_chain(a, n=ncell, eps=eps, M=30)
    print(f"   e_bulk per cell = {Sp['E']:.8f}\n")

    print("FINITE CHAINS: E(N) = N e_bulk + 2 gamma")
    print("     N      E(N)        E(N)/N      E(N)-E(N-1)    2 gamma = E - N e_bulk")
    prev = None
    rows = []
    for N in Ns:
        S = solve_finite_chain(N, a, ncell=ncell, eps=eps)
        inc = (S['E'] - prev) if prev is not None else float('nan')
        rows.append((N, S['E']))
        print(f"  {N:4d} {S['E']:12.7f} {S['E']/N:12.7f} {inc:14.7f} {S['E']-N*Sp['E']:14.7f}",
              flush=True)
        prev = S['E']

    Nn = np.array([r[0] for r in rows], float)
    En = np.array([r[1] for r in rows])
    A = np.vstack([Nn, np.ones_like(Nn)]).T
    slope, icept = np.linalg.lstsq(A, En, rcond=None)[0]
    print(f"\n  least squares over N={Ns}:")
    print(f"     slope (e_bulk from the finite chains) = {slope:.8f}")
    print(f"     periodic e_bulk                       = {Sp['E']:.8f}")
    print(f"     relative difference                   = {abs(slope-Sp['E'])/abs(Sp['E']):.3e}")
    print(f"     intercept = 2 gamma                   = {icept:.8f}")
    print(f"     surface energy gamma                  = {icept/2:.8f}  Ha per end")
