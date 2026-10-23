"""
RETRACTED 2026-10-22, AND THE REDONE VERSION FAILS TOO -- BOTH BOUNDARY CONDITIONS ARE WRONG.

Rerun with masked_laplacian (true Neumann per domain) the core does not sit at R = 2.06; it
COLLAPSES, E falling monotonically as R shrinks with no interior minimum (at a = 5.60: E =
-4.09, -4.17, -4.31, -4.46, -4.68, -5.26, -6.09 for R = 2.8 down to 1.2).  The Dirichlet walls were
holding it up; remove them and nothing does.

NEITHER CONDITION IS THE MODEL'S.  RealQM requires the densities to be CONTINUOUS across a free
boundary -- Psi in H^1(R^3), stated at RealQM_method.tex:101 -- so the object is ONE GLOBAL FIELD
partitioned by labels, not N independent eigenproblems.  This file solves each domain separately,
which permits a JUMP at the interface, and the collapse mode is exactly such a jump: constant inside,
dropping to the outer domain's value at R.  In H^1 that costs infinite energy, so the collapse is
outside the admissible class.  tools/hilbert6_li_allelectron.py records the same error and the same
correction; tools/ionization_NZ_H1.py implements the fix in 1D -- one global u cut into intervals,
one unit-charge constraint per domain, solved by augmented Lagrangian.

SCOPE.  Every MULTI-DOMAIN result from elastic3d_cell.py inherits this: cell2, cell_blob, ae_cell.
The single-domain cell() is unaffected -- one domain filling the cell, no interface, no jump
possible -- which is why the one-electron null mode and the sigma-fixed drift diagnostic stand.

See the note below.

RETRACTION.  ae_cell builds its domain operator as A[keep][:,keep], a submatrix, which imposes
DIRICHLET at the core/valence interface and at the cell face rather than the model's free-interface
condition -- see elastic3d_cell.masked_laplacian, which exists to avoid exactly this and warns that
the submatrix was "briefly mistaken for the cost of non-overlap".

The reported finding -- that compression costs energy at every cell size, so the core electrons
supply the repulsion with no r_c -- therefore cannot stand: the repulsion may be the hard walls.
The companion result in tools/fermi_pressure_partition.py failed exactly this test when redone with
masked_laplacian.  This file must be rerun with masked_laplacian before anything in it is used.

NOT affected: the interface radius R = 2.06 a0 at a = 5.6 against 2.00 from the radial all-electron
code is a comparison of two independently computed geometries, but it too was obtained under the
wrong boundary condition and should be recomputed before being relied on.

ALL-ELECTRON CELL: the core electrons do the partitioning, so the repulsion is the construction's.

THE PREDICTION BEING TESTED (tools/fermi_pressure_partition.py).  Non-overlap supplies Fermi
pressure only when there is something to partition: one electron per cell collapses because the
constant density in a Neumann box is free, two share a cell and the localisation cost of that holds
the lattice up.  A monovalent metal has ONE valence electron per site -- the degenerate case -- which
is why the reduced model needs an empty-core term 1.5 r_c^2/r_s^3 to stand in for the missing
pressure.  In an ALL-ELECTRON cell the core electrons partition the cell with the valence, so the
repulsion should come from the construction and r_c should not be needed at all.

GEOMETRY.  Lithium: Z = 3 at the cell centre, two concentric domains --

    core     r < R,   2 electrons, self-repulsion reduced by (q-1)/q = 1/2
    valence  r > R,   1 electron, filling the rest of the cube

The interface is a SPHERE AROUND the kernel, so unlike cell2's plane-through-the-ion it never cuts
the Gaussian.  That matters: cell2's mesh error carries genuine CURVATURE in a (second difference
+0.0082 against a signal curvature of 0.005, so d2E/da2 flips sign between meshes), which no amount
of fitting repairs.  A linear mesh error is harmless to a second derivative; a curved one is not.

MULTI-OCCUPANCY.  The core holds two charges in one spatial state, so its q(q-1)/2 = 1 internal pair
must be counted once: rho_s = q_s u_s^2 with u_s normalised, and the spurious self term is removed by
subtracting S_s/q_s rather than S_s, which keeps the (q_s-1)/q_s fraction.  For q = 1 that reduces to
removing the whole self-energy, as it must.  This is atom_simulator's Ered = (E_s-1)/E_s.

RESULT 2026-10-22 (Li-like, Z=3, sigma=0.30, h=0.20, R on a 0.2 grid -- see best_R's caveat).

      a      R*       T      U_eI     U_ee     U_II          E
   4.00    1.70   4.5157  -4.2618   0.3538  -3.1920   -2.58418
   4.80    1.90   3.6958  -4.6599   0.4225  -2.6600   -3.20151
   5.60    2.10   3.2125  -4.9537   0.4762  -2.2800   -3.54505
   6.40    2.30   2.9167  -5.1876   0.5200  -1.9950   -3.74591
   7.20    2.50   2.7180  -5.3750   0.5554  -1.7733   -3.87499

TWO HALVES, AND THEY DIFFER.

  * THE REPULSION IS CONFIRMED.  dE/da < 0 throughout, so compressing the cell COSTS energy: the
    lattice resists compression.  The one-electron cell does the opposite and collapses.  So the
    core electrons do supply the repulsion that the partition was predicted to give, and r_c is not
    needed for it.  That was the point of this file and it works.

  * THERE IS NO COHESION.  dE/da = -0.772, -0.429, -0.251, -0.161 never crosses zero, and summing
    the geometric remainder puts E(inf) near -4.1, below every computed point.  The lattice prefers
    to dissociate into atoms.

THREE DEFECTS stand between that and a verdict, and it is NOT yet a verdict.
  1. R tracked the scan grid exactly (one 0.2 step per row, last pinned at the edge), so the R
     values are unusable AND the large-a rows are under-optimised -- correcting that lowers E at
     large a further, strengthening the no-binding reading rather than rescuing it.  best_R is now
     continuous; rerun before quoting anything.
  2. h = 0.20 cannot resolve a Z = 3 core, whose 1s scale is 0.33 a0 -- under two points across it.
  3. sigma = 0.30 truncates that kernel: the atom comes out under-bound by more than half.
Lithium's cohesive energy is ~0.06 Ha while these energies are off by several Ha, which only works
if the error cancels between atom and solid.  It should, since both share the same core -- but
nothing here demonstrates that it does.

ONE CROSS-CHECK THAT DID WORK.  At a = 5.6 a fine R scan puts the free interface at R = 2.061,
against 2.00 measured for the free Li atom by the radial all-electron solver (tools/atom_headless.js)
-- two entirely different codes, 3% apart.  For lithium the core radius does come out of the full
model rather than being supplied to it.

DISCIPLINE.  sigma HELD FIXED in absolute units, never tied to h -- tying it to the mesh changes the
ion's size as the mesh refines and is not a ladder at all.

usage:  python3 tools/allelectron_cell.py
"""
import os, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elastic3d_cell import (grids, phi_periodic, phi_isolated, self_energy_bg,
                            laplacian, ALPHA_SC, HA_GPA)


def ae_cell(n, h, Z=3.0, occ=(2.0, 1.0), R=None, sig=0.30,
            iters=60, tol=1e-12, mix=0.4):
    """Concentric all-electron cell.  occ = (core count, valence count); R = core radius."""
    a, r, G = grids(n, h)
    if R is None:
        R = 0.25*a
    rho_i = Z*np.exp(-r*r/(2*sig*sig))
    rho_i /= rho_i.sum()*h**3/Z                     # total ion charge = Z
    A = laplacian(n, h)
    masks = [r < R, r >= R]
    if masks[0].sum() < 8 or masks[1].sum() < 8:
        return None
    rho = []
    for q, m in zip(occ, masks):
        d = m.astype(float); d *= q/(d.sum()*h**3); rho.append(d)
    E = None
    for it in range(iters):
        rho_tot = rho[0] + rho[1]
        phi_tot = phi_periodic(rho_i - rho_tot, G, h)
        new = []
        for s_, q in enumerate(occ):
            rho_u = np.full_like(rho[s_], q/a**3)
            # all ions, all other electrons, and (q-1)/q of its own -- the i!=j rule
            Vo = -phi_tot - phi_isolated(rho[s_] - rho_u, n, h)/q
            keep = np.where(masks[s_].ravel())[0]
            Asub = A[keep][:, keep] + sp.diags(Vo.ravel()[keep])
            val, vec = eigsh(Asub.tocsr(), k=1, which='SA', tol=1e-11, maxiter=200000)
            u = vec[:, 0]; u = u/np.sqrt(np.sum(u*u)*h**3)
            z = np.zeros(n**3); z[keep] = q*u*u
            new.append(z.reshape(n, n, n))
        rho = [(1-mix)*o + mix*nn for o, nn in zip(rho, new)]
        rho = [d*q/(d.sum()*h**3) for d, q in zip(rho, occ)]
        # ---- energy, pairs counted once ------------------------------------------------------
        rho_tot = rho[0] + rho[1]
        T = 0.0
        for s_, q in enumerate(occ):
            keep = np.where(masks[s_].ravel())[0]
            uu = np.sqrt(rho[s_].ravel()[keep]/q)
            T += q*float(uu @ (A[keep][:, keep] @ uu))*h**3
        U_eI = -float(np.sum(rho_tot*phi_periodic(rho_i, G, h)))*h**3
        U_ee = 0.5*float(np.sum(rho_tot*phi_periodic(rho_tot, G, h)))*h**3
        for s_, q in enumerate(occ):
            U_ee -= self_energy_bg(rho[s_], n, h, a, q=q)/q
        rs_ion = (3.0/(4*np.pi))**(1.0/3.0)*a
        U_II = -ALPHA_SC*Z*Z/rs_ion
        Enew = T + U_eI + U_ee + U_II
        if E is not None and abs(Enew - E) < tol:
            break
        E = Enew
    return dict(E=Enew, T=T, U_eI=U_eI, U_ee=U_ee, U_II=U_II, a=a, R=R, it=it+1)


def best_R(n, h, lo=1.2, hi=3.4, **kw):
    """The interface is a FREE boundary: locate it by minimising the energy over R.

    CONTINUOUS, not a grid.  The first run of this file scanned R on a 0.2-spaced grid and the
    optimum moved exactly one grid step on every row -- 1.70, 1.90, 2.10, 2.30, 2.50 -- with the
    last pinned at the edge.  That is the optimiser walking the grid, not a measurement, and it
    leaves the large-cell rows UNDER-optimised.  Golden-section on a continuous R instead, and the
    bracket is checked so an edge result is reported rather than silently returned.
    """
    from scipy.optimize import minimize_scalar
    f = lambda R: (ae_cell(n, h, R=R, **kw) or {'E': 1e9})['E']
    r = minimize_scalar(f, bounds=(lo, hi), method='bounded',
                        options={'xatol': 1e-3})
    S = ae_cell(n, h, R=float(r.x), **kw)
    if S is not None:
        S['R_at_edge'] = bool(abs(r.x-lo) < 5e-3 or abs(r.x-hi) < 5e-3)
    return S


if __name__ == '__main__':
    print("ALL-ELECTRON CELL, lithium-like: Z=3, core 2e (r<R) + valence 1e (r>R)")
    print("R located by energy minimisation -- it is a free boundary, not an input.")
    print("sigma=0.30 fixed.  Compare: 1e/cell collapses, 2e/cell is stable.\n")
    print(f"{'a':>6}{'R':>7}{'R/a':>7}{'T':>10}{'U_eI':>10}{'U_ee':>10}{'U_II':>10}{'E':>12}")
    h = 0.20
    rows = []
    for n in (16, 20, 24, 28, 32, 36):
        S = best_R(n, h)
        if S is None:
            continue
        rows.append((S['a'], S['E']))
        print(f"{S['a']:>6.2f}{S['R']:>7.3f}{S['R']/S['a']:>7.3f}{S['T']:>10.4f}"
              f"{S['U_eI']:>10.4f}{S['U_ee']:>10.4f}{S['U_II']:>10.4f}{S['E']:>12.5f}", flush=True)
    if len(rows) > 2:
        A = np.array([r[0] for r in rows]); E = np.array([r[1] for r in rows])
        i = int(np.argmin(E))
        edge = "  <-- SCAN EDGE, not bracketed" if i in (0, len(E)-1) else "  <-- BRACKETED"
        print(f"\n   lowest at a = {A[i]:.2f},  E = {E[i]:.5f}{edge}")
        print("   dE/da: " + ", ".join(f"{(E[j+1]-E[j])/(A[j+1]-A[j]):+.4f}"
                                       for j in range(len(E)-1)))
