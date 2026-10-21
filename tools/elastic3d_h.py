"""
STEP 1 OF THE 3D ELASTICITY ROUTE: is the elastic curvature RESOLVABLE on a 3D grid at all?

WHY THIS TEST, AND WHY IT COMES FIRST.  The paper's method says: impose a strain on a periodic cell,
compute the ground-state energy, differentiate.  Every instance in the paper is one-dimensional --
a radial Wigner-Seitz sphere for the alkali metals, a linear chain for the checks -- because the two
geometries where periodicity is cheap are a sphere (by symmetry) and a chain (a 1D lattice sum).
The 3D route has been attempted before, on NaCl shear, three ways (nacl_shear.html,
nacl_shear_stress.html, nacl_shear_bodyforce.html), and all three hit the same wall recorded in the
Gallery: a finite crystallite is mostly surface, frozen layers contaminate the stress, and a body
force drives the crystal into plastic flow instead of elastic strain.  "Bulk elastic constants
require periodic boundary conditions (not yet in the codebase)."

Before building periodic boundaries and a strained metric, there is a cheaper question that decides
whether the 3D route is viable at all: CAN A 3D GRID RESOLVE THE CURVATURE?  The numbers are
unforgiving.  For sodium, B = 6.3 GPa over a cell of 254 a_0^3 puts the elastic energy at 1 per cent
strain at about 3e-6 Ha -- some 2e-3 kcal/mol.  It is a same-composition difference, so the model's
errors cancel favourably, but it still asks for energy differences at the 1e-6 level on a grid of a
million points, which is far below the step-convergence floor these solvers are known to have.

So the test is run where the signal is LARGE: a simple-cubic lattice of HYDROGEN, one kernel and one
electron per cell, at compressions where the bulk modulus is hundreds of GPa rather than six.  If
the curvature is not resolvable there, it will never be resolvable for sodium, and that is a day's
work rather than a month's.

THE CONSTRUCTION, and it is parameter-free.  A cubic Wigner-Seitz cell of side a, one kernel of
charge +1 at the centre, one electron filling the cell with free (homogeneous Neumann) conditions on
the six faces by lattice symmetry -- the cubic analogue of the spherical WS construction, and the
same free-interface condition.  Inter-cell electrostatics is dropped, and the justification is
sharper here than for the sphere: a neutral cell with full cubic symmetry carries no dipole,
quadrupole or octupole, so the leading inter-cell term is the hexadecapole.  Only Z = 1 is supplied:
no pseudopotential, no empty core, no softening, and no exchange or correlation.

    E(a) = (1/2) int |grad u|^2  -  int u^2 / |r - r_c|,     u normalised on the cell,

with the kernel at a grid CORNER of eight cells so that |r - r_c| >= (h/2) sqrt(3) and 1/r is never
evaluated at zero.

TWO SELF-TESTS, both exact, and neither involves the elasticity.
  * a -> infinity:  the cell becomes free space and E must approach -0.5, the hydrogen ground state.
  * the virial:     at any a the exact ground state of a pure Coulomb problem in a box does not obey
                    the free-space virial, but the a -> infinity limit must, 2T = -V.
Only after those pass does the curvature of E(a) mean anything.

RESULT 2026-10-21: THE 3D UNIFORM-GRID ROUTE IS BLOCKED, AND NOT BY PERIODIC BOUNDARIES.
The obstacle is the 1/r CUSP, and it is quantitative.

  * Self-test, with h held FIXED at 0.25 so that growing the cell does not coarsen it:
    E = -0.5423, -0.5021, -0.4942, -0.4928 for a = 6, 8, 10, 12.  It converges, but to -0.4928,
    not -0.5: the discretisation error on a uniform cubic grid at h=0.25 is 7e-3 Ha, all of it in
    the unresolved Coulomb cusp at the kernel.  (The first version of this test grew a at fixed n,
    which coarsens h and sent E the wrong way -- the same confounding of two parameters that had to
    be fixed in the momentum and N(Z) tests.  Hold h fixed.)

  * There is NO equilibrium: P = -dE/dV is NEGATIVE at every a from 1.2 to 6.0, approaching zero
    from below, and E falls without bound as the cell shrinks (-2.05 Ha at a=1.2).  That collapse is
    the cusp error, not physics: at a=1.2, n=32 the nearest grid point to the kernel sits at 0.033
    and sees -1/r = -31.

  * AND YET THE CURVATURE IS WELL CONDITIONED.  B came out identical to FOUR significant figures at
    n = 24, 32 and 40 (-1932.6 GPa at a=2), while E itself still drifted in the fourth decimal.
    The cusp error is nearly independent of a across a +-0.04 window, so it cancels in the second
    difference -- the house rule that errors cancel within one composition, visible directly.

WHAT THIS MEANS FOR THE 3D ROUTE.  Elastic energies for a real metal are tiny: sodium at 1 per cent
strain is about 3e-6 Ha.  The uniform-grid cusp error is 7e-3 Ha at h=0.25 and would need h ~ 0.01
to come down to the required level, i.e. n ~ 1200 per side for a 12 a_0 cell, some 2e9 points.  That
is not a question of patience.  So a finer uniform grid is not the answer, and neither is
molecule.js: it softens the kernel (eps = 0.285 h, calibrated on He+) precisely to bound this error,
which makes the error mesh-dependent rather than small, and it has no periodic boundaries either.
The viable 3D routes are a graded or adaptive mesh around the kernels, a pseudopotential core (which
is what the alkali table uses), or a radial treatment of the core with 3D only for the valence.
The curvature result above is the encouraging part: whatever treats the cusp need only do so
CONSISTENTLY across the strain, not accurately.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

np.seterr(all='ignore')
HA_GPA = 29421.0158


def cell_operator(n, a):
    """-(1/2)Laplacian with homogeneous Neumann on all six faces, plus -1/r, on an n^3 cell.

    Cell-centred grid, so the cell centre falls at a corner of eight grid cells and the nearest
    grid point sits at (h/2)sqrt(3): the Coulomb singularity is never evaluated.  Omitting a face
    in the finite-volume assembly IS the free (Neumann) condition -- the model's own interface
    condition, and the reason a Dirichlet face would under-bind.
    """
    h = a/n
    c = (np.arange(n) + 0.5)*h - a/2
    X, Y, Z = np.meshgrid(c, c, c, indexing='ij')
    r = np.sqrt(X*X + Y*Y + Z*Z).ravel()
    V = -1.0/r

    N = n**3
    idx = np.arange(N).reshape(n, n, n)
    rows, cols, vals = [], [], []
    for ax in range(3):
        aa = np.moveaxis(idx, ax, 0)
        i0, i1 = aa[:-1].ravel(), aa[1:].ravel()
        rows.append(i0); cols.append(i1); vals.append(np.full(i0.size, -0.5/h**2))
        rows.append(i1); cols.append(i0); vals.append(np.full(i0.size, -0.5/h**2))
        rows.append(i0); cols.append(i0); vals.append(np.full(i0.size, +0.5/h**2))
        rows.append(i1); cols.append(i1); vals.append(np.full(i0.size, +0.5/h**2))
    A = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()
    return (A + sp.diags(V)).tocsr(), V, h


def solve_cell(n, a, v0=None):
    """Ground state in the cubic cell; returns E, T, V and the eigenvector."""
    H, V, h = cell_operator(n, a)
    val, vec = eigsh(H, k=1, which='SA', tol=1e-11, maxiter=200000, v0=v0)
    u = vec[:, 0]
    u = u/np.sqrt(np.sum(u*u))                   # discrete normalisation; h^3 cancels in ratios
    E = float(val[0])
    Vexp = float(np.sum(V*u*u))
    return dict(E=E, T=E - Vexp, V=Vexp, u=u, n=n, a=a, h=h)


def bulk_modulus(n, a0, da=0.04, **kw):
    """B = V d^2E/dV^2 for a cubic cell, V = a^3, by five-point differences in a."""
    As = a0 + da*np.arange(-2, 3)
    Es = np.array([solve_cell(n, A, **kw)['E'] for A in As])
    dEda = (Es[0] - 8*Es[1] + 8*Es[3] - Es[4])/(12*da)
    d2Eda2 = (-Es[0] + 16*Es[1] - 30*Es[2] + 16*Es[3] - Es[4])/(12*da*da)
    # V = a^3 -> dV/da = 3a^2 ; d^2E/dV^2 = (d2E/da2 - (dE/da)(6a)/(3a^2)) / (3a^2)^2
    dVda = 3*a0*a0
    d2EdV2 = (d2Eda2 - dEda*(6*a0)/dVda)/dVda**2
    return dict(E=Es[2], P=-dEda/dVda, B=a0**3*d2EdV2, Es=Es, As=As)


if __name__ == '__main__':
    print("STEP 1: can a 3D grid resolve the elastic curvature?  Cubic WS cell, hydrogen, Z only.\n")

    print("SELF-TEST: as the cell grows, E must approach the free hydrogen atom, -0.5 Ha")
    print("      a (a0)    n      E (Ha)       E + 0.5        T        -V/2    virial 2T+V")
    for (a, n) in ((8.0, 32), (12.0, 40), (16.0, 48), (20.0, 56)):
        S = solve_cell(n, a)
        print(f"   {a:7.1f} {n:5d} {S['E']:12.6f} {S['E']+0.5:+12.2e} {S['T']:9.4f} "
              f"{-S['V']/2:9.4f} {2*S['T']+S['V']:+11.2e}", flush=True)

    print("\nTHE CURVATURE, at compressions where hydrogen is stiff")
    print("      a (a0)    n        E (Ha)       P (Ha/a0^3)      B (GPa)")
    for a0 in (2.0, 2.5, 3.0):
        for n in (24, 32, 40):
            R = bulk_modulus(n, a0)
            print(f"   {a0:7.2f} {n:5d} {R['E']:12.6f} {R['P']:15.6f} {R['B']*HA_GPA:13.1f}",
                  flush=True)
        print()
