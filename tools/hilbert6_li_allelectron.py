"""
CORRECTION 2026-10-20, read this before the diagnosis below.  THE COLLAPSE IS THIS FILE'S BUG, NOT
THE MODEL'S.  RealQM requires the densities to be CONTINUOUS ACROSS the free boundary, imposed as
Psi in H^1(R^3) -- stated explicitly in RealQM_method.tex:101 and FoundChem.tex:715: "the psi_i
being continuous across joint boundaries Gamma_i ^ Gamma_j as imposed by Psi in H^1(R^3), and the
boundary location determined by the energy minimisation".  This code does NOT impose it: each shell
carries natural conditions on both of its own faces, so the density may JUMP at an interface.  The
collapse mode needs precisely such a jump -- a constant psi inside, dropping to the outer shell's
value at R1 -- and in H^1 that jump costs INFINITE energy.  So the uniform-blob collapse is outside
the model's admissible class and the conclusion drawn from it was wrong.
Consequences: (a) the inner cutoff r_c is NOT shown to be a necessity of the construction; (b)
atom.js avoids the collapse legitimately, by ramping its level-set weights over ~2 cells, which is
the discrete form of the continuity condition; (c) to run this test properly the shells must be
coupled by continuity at the interfaces, not solved independently.  The hydrogen self-test is
unaffected (one domain, no interface).

ALL-ELECTRON LITHIUM IN A WIGNER-SEITZ CELL: the first macroscopic coefficient from Z alone.

WHY.  The alkali table in the Hilbert-6 paper is honest that it is NOT parameter-free: each element
brings an Ashcroft empty-core radius r_c, taken from the literature rather than fitted here, but an
input all the same.  What IS parameter-free is the upscaling itself -- impose a strain, compute the
energy, differentiate -- and the paper can only say so while conceding the input.

Lithium is the element to fix, for two reasons.  It is the row that FAILS: modulus 4.3x too stiff
(49.6 against 11.6 GPa) and a pressure derivative B' with the wrong sign.  And it has three
electrons, so it is the one alkali whose cell can be done all-electron at no cost.  Doing it
separates the two explanations the paper names: is lithium's failure the empty-core input, or the
one-electron-per-cell picture itself?

THE CONSTRUCTION, and what is supplied.  Nucleus Z=3 at the centre of a sphere of radius r_s.
THREE electron domains, one unit charge each, as nested radial shells [0,R1], [R1,R2], [R2,r_s] --
the house construction, n unit charges giving n(n-1)/2 pairs each counted once by the i!=j rule.
Free (homogeneous Neumann) conditions at every interface, which is what omitting a face gives in
the finite-volume assembly, and at the cell face r_s by the Wigner-Seitz symmetry.  The shell
boundaries are not prescribed: they are chosen to minimise the energy.
THE ONLY NUMBER SUPPLIED IS Z = 3.  No pseudopotential, no empty core, no exchange, no
correlation functional, no softening -- the nucleus is a bare point charge and the radial grid never
evaluates 1/r at r=0 because the grid is cell centred.

RESULT 2026-10-20: THIS DOES NOT WORK, AND THE REASON ANSWERS THE PARAMETER-FREE QUESTION.
Measured: E = -40.26, -40.06, -39.97 Ha at n = 150, 250, 400 for r_s = 3.2 -- CONVERGING, so not a
mesh artefact, and 5.3x too negative against lithium's -7.478.  The energy also rises monotonically
with r_s, so there is no equilibrium, and the energy-minimising split always sits at the smallest
innermost shell the candidate list allows.

THE CAUSE is the house rule in CLAUDE.md: "A uniform density plus a free boundary is unstable -- it
expands without bound."  Here it contracts instead.  With homogeneous Neumann at BOTH faces of the
innermost shell, a CONSTANT density is admissible and carries ZERO localisation energy, so nothing
opposes the shell shrinking onto the nucleus: at R1 = 0.128 a0 the constant density gives
<-Z/r> = -3 * (3/2R1) ~ -35 Ha, which is the -40 observed.  The only thing that stopped it was the
arbitrary floor k1 >= 0.04 n in _candidate_splits.

SO THE ALL-ELECTRON ROUTE DOES NOT REMOVE THE PARAMETER, IT RELOCATES IT.  The model's own
prescription for a core region is an INNER CUTOFF r_c ("Molecules impose only the inner cutoff r_c",
CLAUDE.md), and the atom work uses a kernel softening eps instead, calibrated on He+.  Either is a
number put in.  What survives, and is worth claiming, is that the UPSCALING adds no parameter of its
own: strain the cell, compute the energy, differentiate.  The parameters that appear are those of
the microscopic input, and swapping the Ashcroft empty core for a bare nucleus exchanges r_c
(empty core) for r_c (inner cutoff) or eps (softening) rather than eliminating it.

TO FIX, IF IT IS WORTH FIXING: the collapse is a property of the DENSITY formulation with a free
inner boundary, so the cure is the radial variable u = r psi with u(0) = 0, in which the localisation
term is (1/2) int (u' - u/r)^2 dr and a density concentrating at the origin is penalised correctly.
That is parameter-free.  It is a rewrite, not an adjustment.

NO MADELUNG TERM IS ADDED, AND NONE IS NEEDED.  A neutral, spherically symmetric cell produces no
field outside itself, by the shell theorem, so the inter-cell electrostatic energy vanishes
identically in the Wigner-Seitz approximation.  The familiar -0.9/r_s of the standard treatment is
not an inter-cell term at all: it is the intra-cell energy of a POINT ion in a UNIFORM electron
sphere, an idealisation this calculation does not make, since it computes the actual
nucleus-electron and electron-electron energies of the actual density.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

np.seterr(all='ignore')
HA_GPA = 29421.0158


def radial_grid(rs, n):
    h = rs/n
    r = (np.arange(n) + 0.5)*h
    vol = 4*np.pi*r*r*h                      # cell volumes
    area = 4*np.pi*((np.arange(1, n))*h)**2  # areas of the interior faces
    return r, h, vol, area


def shell_laplacian(r, h, vol, area, lo, hi):
    """-(1/2) Laplacian on the radial interval [lo,hi) with NO-FLUX on both its own faces.

    Omitting a face is the homogeneous-Neumann (free interface) condition -- the model's own
    condition, and the reason a submatrix of the full operator would be wrong (it would impose
    Dirichlet and under-bind; see the note in axisym_realqm.masked_stiffness).
    """
    k = hi - lo
    A = np.zeros((k, k))
    for i in range(lo, hi-1):
        c = 0.5*area[i]/h                    # 0.5 * area / h, the stiffness of one face
        a, b = i-lo, i-lo+1
        A[a, a] += c; A[b, b] += c; A[a, b] -= c; A[b, a] -= c
    return A                                 # quadratic form: 0.5*int |grad u|^2 = u^T A u


def hartree_pot(r, h, vol, dens):
    """V(r) = (1/r)*Q(<r) + int_{r'>r} dens*4pi r' dr' ; both pieces by cumulative sums."""
    q_in = np.cumsum(dens*vol)
    term1 = q_in/r
    w = dens*4*np.pi*r*h                             # dens * 4 pi r' dr'
    tail = np.cumsum(w[::-1])[::-1] - w              # strictly outside the cell
    return term1 + tail


def solve_cell(rs, n=600, iters=400, tol=1e-10, Z=3.0, splits=None, verbose=False):
    """Self-consistent all-electron cell; returns the energy and the shell boundaries used."""
    r, h, vol, area = radial_grid(rs, n)
    Vnuc = -Z/r
    best = None
    cand = splits if splits is not None else _candidate_splits(n)
    for (k1, k2) in cand:
        bounds = [(0, k1), (k1, k2), (k2, n)]
        if min(b-a for a, b in bounds) < 4:
            continue
        Alist = [shell_laplacian(r, h, vol, area, a, b) for a, b in bounds]
        dens = []
        for a, b in bounds:                          # seed: uniform on each shell
            d = np.zeros(n); d[a:b] = 1.0
            d /= np.sum(d*vol); dens.append(d)
        E = None
        for it in range(iters):
            new = []
            lam = []
            for s, (a, b) in enumerate(bounds):
                Pother = np.zeros(n)
                for t in range(3):
                    if t != s:
                        Pother += hartree_pot(r, h, vol, dens[t])
                Veff = (Vnuc + Pother)[a:b]
                M = Alist[s] + np.diag(Veff*vol[a:b])
                w = 1.0/np.sqrt(vol[a:b])
                S = (M*w[:, None])*w[None, :]
                S = 0.5*(S + S.T)
                val, vec = np.linalg.eigh(S)
                u = w*vec[:, 0]
                d = np.zeros(n); d[a:b] = u*u
                d /= np.sum(d*vol)
                new.append(d); lam.append(val[0])
            dens = [0.5*o + 0.5*nn for o, nn in zip(dens, new)]
            dens = [d/np.sum(d*vol) for d in dens]
            # energy, pairs once
            T = 0.0
            for s, (a, b) in enumerate(bounds):
                u = np.sqrt(dens[s][a:b])
                T += float(u @ (Alist[s] @ u))
            Ven = sum(float(np.sum(d*Vnuc*vol)) for d in dens)
            Vee = 0.0
            for i in range(3):
                for j in range(i+1, 3):
                    Vee += float(np.sum(hartree_pot(r, h, vol, dens[j])*dens[i]*vol))
            Enew = T + Ven + Vee
            if E is not None and abs(Enew - E) < tol:
                break
            E = Enew
        if best is None or Enew < best[0]:
            best = (Enew, (k1, k2), [d.copy() for d in dens], it+1)
    return dict(E=best[0], split=best[1], dens=best[2], iters=best[3], rs=rs, n=n)


def _candidate_splits(n):
    out = []
    for f1 in (0.04, 0.06, 0.08, 0.11, 0.15, 0.20):
        for f2 in (0.25, 0.35, 0.45, 0.55, 0.70):
            k1, k2 = int(f1*n), int(f2*n)
            if 4 < k1 < k2 < n-4:
                out.append((k1, k2))
    return out


if __name__ == '__main__':
    n = int(os.environ.get('N', 400))
    print("ALL-ELECTRON LITHIUM IN A WIGNER-SEITZ CELL -- only Z = 3 supplied\n")
    print("  three one-electron domains as nested radial shells, free interfaces,")
    print("  Neumann at the cell face, shell boundaries chosen by minimising the energy")
    print(f"  radial cells n = {n}; no pseudopotential, no softening, no Madelung term\n")
    print("     r_s        E(r_s)        split (cells)   SCF its")
    rows = []
    for rs in [float(x) for x in os.environ.get('RS', '2.6,2.9,3.2,3.5,3.8,4.1').split(',')]:
        S = solve_cell(rs, n=n)
        rows.append((rs, S['E']))
        print(f"  {rs:7.3f} {S['E']:14.7f}   {str(S['split']):>14s} {S['iters']:7d}", flush=True)
    rs = np.array([r[0] for r in rows]); E = np.array([r[1] for r in rows])
    c = np.polyfit(rs, E, 3)
    d1 = np.polyder(c); roots = np.roots(d1)
    roots = [x.real for x in roots if abs(x.imag) < 1e-9 and rs.min() < x.real < rs.max()]
    print(f"\n  cubic fit over the scanned range: stationary points in range = {roots}")
    if roots:
        r0 = roots[0]
        d2 = np.polyval(np.polyder(d1), r0)
        V = (4*np.pi/3)*r0**3
        dVdr = 4*np.pi*r0**2
        B = V*d2/dVdr**2*HA_GPA
        print(f"  equilibrium r_s = {r0:.3f} a0   (measured 3.25)")
        print(f"  bulk modulus    = {B:.2f} GPa  (measured 11.6; pseudopotential column gave 49.6)")
