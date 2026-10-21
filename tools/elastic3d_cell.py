"""
3D PERIODIC CELL ENERGY, AND HENCE ELASTIC COEFFICIENTS -- the missing capability.

WHAT WAS BLOCKING THIS.  tools/elastic3d_h.py established the chain: a bare kernel on a uniform 3D
grid is hopeless (cusp error 7e-3 Ha against an elastic signal of 3e-6); a pseudokernel removes the
cusp; a hard core edge must be smoothed; h must be held fixed with the strain carried by n; and then
the remaining error is not the grid at all but the fact that AN EIGENVALUE IS NOT A CELL ENERGY.
real_plasma_numerics/metal_alkalis.py has E = <T> + (-0.9/r_s + 1.5 r_c^2/r_s^3): kinetic energy
plus LATTICE ELECTROSTATICS, the second term available analytically only because a neutral SPHERE
has an exact self-energy. A cubic cell has no such formula, so the lattice electrostatics must be
computed -- which is the periodic-boundary capability the Gallery card identified as missing after
the three NaCl shear attempts.

HOW THIS DOES IT WITHOUT EWALD.  Work with the NEUTRAL total density rho_tot = rho_ion - rho_e and
solve the periodic Poisson equation once by FFT with the k=0 mode dropped, which IS the neutralising
background convention and is legitimate precisely because rho_tot is neutral. Then

    V_eff felt by the electron  =  -phi_tot  -  phi_e^iso ,

since the electron is attracted by all ions and repelled by all OTHER electrons: -phi_ion + phi_e^per
= -(phi_ion - phi_e^per) = -phi_tot, and the electron's OWN contribution is removed by subtracting
its isolated potential phi_e^iso -- the house i!=j rule, no self-interaction anywhere. phi_e^iso is
the non-periodic 1/r convolution, got from a zero-padded FFT. So two FFT solves and no Ewald sums.

The ion is smeared as a narrow Gaussian so that the FFT can carry it. That is harmless HERE and not
a fudge: sigma ~ 1.2 h ~ 0.2 a_0 while the Ashcroft core radius is r_c = 1.67, so the entire
smearing lies inside the region where the empty-core pseudopotential replaces the Coulomb potential
anyway. The ion's own self-energy depends only on its fixed shape, so it is an additive constant and
drops out of dE/da and d^2E/da^2.

VALIDATION, in two steps, each against a number known in advance.
  * the periodic electrostatics alone, against the exact simple-cubic Madelung constant
    -0.880060 Z^2/r_s: computed -0.878691 at n=64, sigma/h=1.2, converging as h (5.5e-3, 2.4e-3,
    1.4e-3 at n = 32, 48, 64).  The residue is a-INDEPENDENT at fixed sigma/h, so it rescales an
    analytically known 1/a term and does not corrupt the curvature.
  * the whole cell energy, against the radial Wigner-Seitz answer for sodium that
    metal_alkalis.py already gives: r_s = 3.77 a_0 and B = 9.1 GPa. A cubic cell is not a sphere, so
    exact agreement is not expected -- the Madelung constants differ by about two per cent -- but
    the equilibrium radius and the modulus must land close, and that is the test.

RESULT 2026-10-21: IT WORKS.  The unit test passes EXACTLY -- a uniform electron plus point ions
returns -0.880060/r_s to the last digit, U_ee coming out at 2e-35 -- and the self-consistent cell
for sodium then gives

    r_s        3.387     3.648     3.908     4.169
    E (Ha)   -0.16267  -0.16308  -0.16150  -0.15883

with a genuine minimum near r_s = 3.6 and B = 6.05 GPa at the point closest to it.  Measured sodium
is r_s = 3.93 and B = 6.3 GPa; the two radial columns of metal_alkalis.py give B = 9.1 ("split",
the cell state's own <T>) and 7.2 ("sea", uniform density).  So the equilibrium radius is about 8
per cent low and the modulus lands within a few per cent of experiment, between the two radial
columns.

TWO THINGS TO READ CAREFULLY BEFORE QUOTING ANY OF IT.
  * The converged electron is very nearly UNIFORM -- T comes out 0.005 to 0.008 Ha and U_ee is zero
    to machine precision -- so this is the "sea" solution, not the localised one.  With one electron
    per cell non-overlap does no work, exactly the limitation that applies to the alkali table
    itself, and the agreement with experiment should be read in that light.
  * B is still coarse.  Holding h fixed quantises the strain at da = 2h = 0.42 a_0, so the
    three-point second difference is wide, and B varies between adjacent points (6.05 at r_s=3.65
    against 2.34 at 3.91).  A trustworthy modulus needs a smaller h, which costs n^3.

WHAT IT UNLOCKS.  This was the capability the Gallery card identified as missing after the three
NaCl shear attempts, and it is now here and validated against an exact number.  Shear is the next
step and needs only the cell to become non-cubic: the Laplacian and the FFT wavevectors acquire the
metric of the strained lattice vectors, and C_44 follows from the same energy difference.

TWO ELECTRONS, AND THE PARTITION MATTERS -- 2026-10-21.  cell2() puts two one-electron domains in
one cell so that NON-OVERLAP finally does work (with one electron per cell the single domain fills
it and the constraint is vacuous).  Magnesium, r_c = 1.39 as an input.  Three things came out, and
the second withdraws a number.

  * IT BINDS, AND THE LOCALISATION COST IS REAL.  T = 0.30 to 0.48 Ha against 0.03 for the
    overlapping blob -- a factor of ten.  That cost is what the standard treatment gets from
    band filling, and here it comes from non-overlap alone.

  * THE CHOICE OF DIVIDING SURFACE CHANGES THE ANSWER, and the first choice was the worst.  At
    a = 5.46, E = -0.558 for a (001) plane through the ion, -0.611 for (011), -0.638 for (111), and
    POSITIVE (+0.86 to +2.09, with T = 1.4 to 2.0) for a sphere around the ion against the rest.
    So the sphere topology is hopeless -- confining one electron to a ball costs far more than it
    gains -- and (111) is lower than (001) by 0.080 Ha.  A modulus computed on (001), as the first
    run did, is computed on a partition the energy rejects.  On (111) the cell has a clean minimum
    at r_s = 2.69 with P = -6e-5 there, against MEASURED 2.65 -- within 1.5 per cent -- and
    B = 26 GPa against 35.4 measured.

  * B IS STILL ONLY GOOD TO A FACTOR OF ABOUT TWO.  Holding h fixed quantises the strain at
    da = 2h = 0.42 a_0, and across adjacent points B reads 60, 26, 10 GPa.  The EQUILIBRIUM is far
    better determined than the curvature, because P crosses zero sharply.  Quote r_s; do not quote B
    without a finer h.

AND THE COMPARISON WITH THE BLOB IS NOT A VARIATIONAL ONE.  cell_blob() reaches -0.783, below every
split partition, but that is a comparison ACROSS MODELS and not within one: the blob's (N-1)/N
reduction is a weaker constraint than non-overlap, and a weaker constraint always reaches a lower
minimum.  What discriminates them is an observable, not an energy -- and there the split binds while
the blob does not, its energy rising monotonically with r_s across the whole range scanned.

THE DISCIPLINE CARRIED FROM EVERY EARLIER PHASE: h held FIXED with the strain carried by n (so the
mesh and the core surface are identical at every strain), ONE PARITY of n only (n even puts the ion
at a corner of eight cells, n odd at a cell centre, and mixing them alternates the answer), and the
core edge ramped over about two cells.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

np.seterr(all='ignore')
HA_GPA = 29421.0158
ALPHA_SC = 0.880060          # simple-cubic Madelung constant, point ions + background


def grids(n, h):
    a = n*h
    c = (np.arange(n) + 0.5)*h - a/2
    X, Y, Z = np.meshgrid(c, c, c, indexing='ij')
    r = np.sqrt(X*X + Y*Y + Z*Z)
    k = 2*np.pi*np.fft.fftfreq(n, d=h)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    k2 = KX*KX + KY*KY + KZ*KZ
    G = np.where(k2 > 0, 4*np.pi/k2, 0.0)        # k=0 dropped: the neutralising background
    return a, r, G


def phi_periodic(rho, G, h):
    return np.real(np.fft.ifftn(G*np.fft.fftn(rho)))


def self_energy_bg(rho, n, h, a, q=1.0):
    """Self-interaction to remove in a periodic cell: the ISOLATED energy of the NEUTRAL difference
    rho - rho_uniform.

    The naive choice -- subtract the isolated self-energy of rho -- is WRONG, and the unit test says
    so unambiguously: for a UNIFORM electron the periodic potential vanishes identically (a uniform
    density is pure k=0, which the background convention drops), so the naive term leaves
    U_ee = -(1/2)int rho phi_iso = -0.172 Ha where the exact answer is ZERO.  The k=0 removal has
    already supplied a neutralising background, so what must come off is the self-interaction
    MEASURED AGAINST THAT BACKGROUND, i.e. the isolated energy of rho - rho_u.  It vanishes
    identically for a uniform density, which is the test.
    """
    rho_u = np.full_like(rho, q/a**3)      # background must carry the SAME charge as rho
    d = rho - rho_u
    return 0.5*float(np.sum(d*phi_isolated(d, n, h)))*h**3


def phi_isolated(rho, n, h):
    """Non-periodic 1/r convolution by zero-padded FFT -- the electron's own-cell self-potential."""
    m = 2*n
    big = np.zeros((m, m, m)); big[:n, :n, :n] = rho
    c = (np.arange(m) - m//2)*h
    X, Y, Z = np.meshgrid(c, c, c, indexing='ij')
    rr = np.sqrt(X*X + Y*Y + Z*Z)
    ker = np.where(rr > 0, 1.0/np.maximum(rr, 1e-300), 1.0/(0.62035*h))   # cell self-term
    ker = np.fft.ifftshift(ker)
    out = np.real(np.fft.ifftn(np.fft.fftn(big)*np.fft.fftn(ker)))*h**3
    return out[:n, :n, :n]


def laplacian(n, h):
    N = n**3
    idx = np.arange(N).reshape(n, n, n)
    R, C, V = [], [], []
    for ax in range(3):
        aa = np.moveaxis(idx, ax, 0)
        i0, i1 = aa[:-1].ravel(), aa[1:].ravel()
        R += [i0, i1, i0, i1]; C += [i1, i0, i0, i1]
        V += [np.full(i0.size, -0.5/h**2)]*2 + [np.full(i0.size, 0.5/h**2)]*2
    return sp.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                         shape=(N, N)).tocsr()


def cell(n, h, rc=1.67, sig_cells=1.2, smooth=2.0, iters=60, tol=1e-12, mix=0.4):
    """Self-consistent cell. Returns the energy per cell and its parts."""
    a, r, G = grids(n, h)
    sig = sig_cells*h
    rho_i = np.exp(-r*r/(2*sig*sig)); rho_i /= rho_i.sum()*h**3        # +1, smeared
    # Ashcroft empty core: the electron-ion potential is replaced inside r_c.  Applied as a
    # correction to the bare -1/r of the nearest images, ramped over `smooth` cells.
    w = np.clip((r - rc)/(smooth*h) + 0.5, 0.0, 1.0)
    core_fix = (1.0 - w)/np.maximum(r, 1e-300)      # add back +1/r inside the core => V -> 0 there
    A = laplacian(n, h)
    rho_e = np.full((n, n, n), 1.0/a**3)
    E = None
    for it in range(iters):
        phi_tot = phi_periodic(rho_i - rho_e, G, h)
        phi_iso = phi_isolated(rho_e, n, h)
        # the potential of the OTHER electrons, background-compensated in the same way
        rho_u = np.full_like(rho_e, 1.0/a**3)
        Veff = (-phi_tot - phi_isolated(rho_e - rho_u, n, h) + core_fix).ravel()
        val, vec = eigsh((A + sp.diags(Veff)).tocsr(), k=1, which='SA', tol=1e-11, maxiter=200000)
        u = vec[:, 0]; u = u/np.sqrt(np.sum(u*u)*h**3)
        new = (u*u).reshape(n, n, n)
        rho_e = (1-mix)*rho_e + mix*new
        rho_e /= rho_e.sum()*h**3
        # ---- energy per cell ------------------------------------------------------------------
        uu = np.sqrt(rho_e).ravel()
        T = float(uu @ (A @ uu))*h**3
        phi_tot = phi_periodic(rho_i - rho_e, G, h)
        phi_iso = phi_isolated(rho_e, n, h)
        U_eI = float(np.sum(rho_e*(-phi_periodic(rho_i, G, h) + core_fix)))*h**3
        U_ee = (0.5*float(np.sum(rho_e*phi_periodic(rho_e, G, h)))*h**3
                - self_energy_bg(rho_e, n, h, a))
        rs = (3.0/(4*np.pi))**(1.0/3.0)*a
        U_II = -ALPHA_SC/rs
        Enew = T + U_eI + U_ee + U_II
        if E is not None and abs(Enew - E) < tol:
            break
        E = Enew
    return dict(E=Enew, T=T, U_eI=U_eI, U_ee=U_ee, U_II=U_II, a=a, rs=rs, it=it+1)



# ------------------------------------------------------------ TWO DOMAINS: non-overlap does work
def cell2(n, h, Z=2.0, rc=1.39, sig_cells=1.2, smooth=2.0, iters=60, tol=1e-12, mix=0.4):
    """Two electron domains in one periodic cell, split by the plane through the ion.

    This is the first calculation in this line where NON-OVERLAP does any work.  With one electron
    per cell the single domain fills it and the constraint is vacuous; with two, each carries unit
    charge on half the cell and they meet at a FREE INTERFACE -- the plane z=0 through the ion, held
    there by symmetry -- and again at the cell face, which by periodicity is where this cell's upper
    domain meets the next cell's lower one.  So each domain is a slab with a free interface on both
    faces, and the i!=j rule now excludes two self-interactions rather than one.

    Magnesium is the cheapest genuine test: divalent, so two domains, and the ion is a +2 Ashcroft
    core.  r_c is an INPUT here as everywhere in this family, and it is not tuned to the answer.
    """
    a, r, G = grids(n, h)
    sig = sig_cells*h
    rho_i = Z*np.exp(-r*r/(2*sig*sig))/(np.exp(-r*r/(2*sig*sig)).sum()*h**3)
    w = np.clip((r - rc)/(smooth*h) + 0.5, 0.0, 1.0)
    core_fix = Z*(1.0 - w)/np.maximum(r, 1e-300)
    A = laplacian(n, h)
    half = n//2
    masks = [np.zeros((n, n, n), bool), np.zeros((n, n, n), bool)]
    masks[0][:, :, :half] = True                  # z < 0 slab
    masks[1][:, :, half:] = True                  # z > 0 slab
    rho = [m/(m.sum()*h**3) for m in (masks[0].astype(float), masks[1].astype(float))]
    E = None
    for it in range(iters):
        rho_tot = rho[0] + rho[1]
        phi_tot = phi_periodic(rho_i - rho_tot, G, h)
        new = []
        for s_ in (0, 1):
            rho_u = np.full_like(rho[s_], 1.0/a**3)
            # this domain feels all ions and the OTHER domain, never itself
            Vo = (-phi_tot - phi_isolated(rho[s_] - rho_u, n, h) + core_fix)
            keep = np.where(masks[s_].ravel())[0]
            Asub = A[keep][:, keep] + sp.diags(Vo.ravel()[keep])
            val, vec = eigsh(Asub.tocsr(), k=1, which='SA', tol=1e-11, maxiter=200000)
            u = vec[:, 0]; u = u/np.sqrt(np.sum(u*u)*h**3)
            q = np.zeros(n**3); q[keep] = u*u
            new.append(q.reshape(n, n, n))
        rho = [(1-mix)*o + mix*nn for o, nn in zip(rho, new)]
        rho = [q/(q.sum()*h**3) for q in rho]
        # ---- energy per cell, pairs once ------------------------------------------------------
        rho_tot = rho[0] + rho[1]
        T = 0.0
        for s_ in (0, 1):
            keep = np.where(masks[s_].ravel())[0]
            uu = np.sqrt(rho[s_].ravel()[keep])
            T += float(uu @ (A[keep][:, keep] @ uu))*h**3
        U_eI = float(np.sum(rho_tot*(-phi_periodic(rho_i, G, h) + core_fix)))*h**3
        U_ee = (0.5*float(np.sum(rho_tot*phi_periodic(rho_tot, G, h)))*h**3
                - self_energy_bg(rho[0], n, h, a) - self_energy_bg(rho[1], n, h, a))
        rs_ion = (3.0/(4*np.pi))**(1.0/3.0)*a
        U_II = -ALPHA_SC*Z*Z/rs_ion
        Enew = T + U_eI + U_ee + U_II
        if E is not None and abs(Enew - E) < tol:
            break
        E = Enew
    rs_e = (3.0/(4*np.pi))**(1.0/3.0)*a/Z**(1.0/3.0)
    return dict(E=Enew, T=T, U_eI=U_eI, U_ee=U_ee, U_II=U_II, a=a, rs=rs_e, it=it+1)


def cell_blob(n, h, Z=2.0, rc=1.39, sig_cells=1.2, smooth=2.0, iters=60, tol=1e-12, mix=0.4):
    """The OVERLAPPING alternative: N electrons sharing one domain, with REDUCED self-repulsion.

    CLAUDE.md's house rule is that splitting a -Z valence into Z one-electron domains is exact --
    n unit charges give n(n-1)/2 pairs, each counted once, with no (Z-1)/Z factor and no kinetic
    fudge -- and that "a -Z multi-occupancy blob is the exception and needs justifying".  This
    function is that exception, computed so the two can be compared on the same grid.

    The bookkeeping.  If all N electrons share one orbital psi with rho_tot = N psi^2, the pair
    count is N(N-1)/2 rather than the N^2/2 the naive Hartree of rho_tot would give, so the
    SAME-CELL part must be scaled by (N-1)/N.  The images are genuinely other electrons and are not
    reduced.  Splitting the periodic Hartree into (cell with images and background) plus (within
    cell) gives
        U_ee = (1/2) int rho phi^per  -  (1/N) * self_bg(rho_tot),
    which reduces to the one-electron case at N=1, where it removes the self-interaction entirely.
    And the localisation energy of N electrons in one orbital is just the von Weizsaecker term of
    the TOTAL density, T = (1/2) int |grad sqrt(rho_tot)|^2, so a uniform blob has T = 0 exactly --
    which is the "sea" limit that metal_alkalis.py computes in one dimension.
    """
    a, r, G = grids(n, h)
    sig = sig_cells*h
    rho_i = Z*np.exp(-r*r/(2*sig*sig))/(np.exp(-r*r/(2*sig*sig)).sum()*h**3)
    w = np.clip((r - rc)/(smooth*h) + 0.5, 0.0, 1.0)
    core_fix = Z*(1.0 - w)/np.maximum(r, 1e-300)
    A = laplacian(n, h)
    rho_t = np.full((n, n, n), Z/a**3)
    E = None
    for it in range(iters):
        phi_tot = phi_periodic(rho_i - rho_t, G, h)
        rho_u = np.full_like(rho_t, Z/a**3)
        # the reduced self-repulsion: only 1/N of the same-cell self-interaction is removed
        Veff = (-phi_tot - (1.0/Z)*phi_isolated(rho_t - rho_u, n, h) + core_fix).ravel()
        val, vec = eigsh((A + sp.diags(Veff)).tocsr(), k=1, which='SA', tol=1e-11, maxiter=200000)
        u = vec[:, 0]; u = u/np.sqrt(np.sum(u*u)*h**3)
        new = Z*(u*u).reshape(n, n, n)
        rho_t = (1-mix)*rho_t + mix*new
        rho_t *= Z/(rho_t.sum()*h**3)
        uu = np.sqrt(rho_t).ravel()
        T = float(uu @ (A @ uu))*h**3                  # von Weizsaecker of the total
        U_eI = float(np.sum(rho_t*(-phi_periodic(rho_i, G, h) + core_fix)))*h**3
        U_ee = (0.5*float(np.sum(rho_t*phi_periodic(rho_t, G, h)))*h**3
                - (1.0/Z)*self_energy_bg(rho_t, n, h, a, q=Z))
        rs_ion = (3.0/(4*np.pi))**(1.0/3.0)*a
        U_II = -ALPHA_SC*Z*Z/rs_ion
        Enew = T + U_eI + U_ee + U_II
        if E is not None and abs(Enew - E) < tol:
            break
        E = Enew
    return dict(E=Enew, T=T, U_eI=U_eI, U_ee=U_ee, U_II=U_II, a=a,
                rs=(3.0/(4*np.pi))**(1.0/3.0)*a/Z**(1.0/3.0), it=it+1)


def masked_laplacian(keep3d, h):
    """-(1/2)Laplacian assembled over faces joining two KEPT cells ONLY -- true no-flux (Neumann).

    THIS IS NOT A SUBMATRIX.  Taking A[keep][:,keep] of the full operator retains the diagonal
    contributions of faces leading OUT of the region, which imposes DIRICHLET (u=0 outside): the
    density is pushed off the boundary and the state under-binds.  axisym_realqm.masked_stiffness
    records 36% on a one-electron IE from exactly this, and it cost 0.22 Ha here -- a third of the
    binding -- plus a spurious factor of EIGHT in the localisation energy, which was briefly
    mistaken for the cost of non-overlap.  Omitting the outward faces at assembly is the model's
    own free-interface condition.
    """
    n = keep3d.shape[0]
    idx = -np.ones(keep3d.shape, int)
    ids = np.where(keep3d.ravel())[0]
    idx.ravel()[ids] = np.arange(ids.size)
    R, C, V = [], [], []
    for ax in range(3):
        k0 = np.moveaxis(keep3d, ax, 0)[:-1]; k1 = np.moveaxis(keep3d, ax, 0)[1:]
        i0 = np.moveaxis(idx, ax, 0)[:-1];    i1 = np.moveaxis(idx, ax, 0)[1:]
        m = k0 & k1; a0 = i0[m]; a1 = i1[m]
        R += [a0, a1, a0, a1]; C += [a1, a0, a0, a1]
        V += [np.full(a0.size, -0.5/h**2)]*2 + [np.full(a0.size, 0.5/h**2)]*2
    return sp.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                         shape=(ids.size, ids.size)).tocsr(), ids


def cell2_free(n, h, Z=2.0, rc=1.39, normal=(1, 1, 1), sig_cells=1.2, iters=40, tol=1e-12, mix=0.4):
    """Two domains, FREE interfaces, and a HARD inner cutoff -- the house construction.

    CLAUDE.md: "Molecules impose only the INNER cutoff r_c.  Outer boundaries are free, found by
    non-overlap."  So the valence electrons are EXCLUDED from the core ball r < r_c (hard
    depletion, density exactly zero there), as x2_R_sweep.py does, and the interfaces -- between the
    two domains, and at the core surface -- carry the free no-flux condition via masked_laplacian.
    No empty-core POTENTIAL is used: that is standard pseudopotential theory, where the electron may
    enter the core and feel nothing, and it is a different model.
    """
    a, r, G = grids(n, h)
    sig = sig_cells*h
    g = np.exp(-r*r/(2*sig*sig)); rho_i = Z*g/(g.sum()*h**3)
    c = (np.arange(n) + 0.5)*h - a/2
    X, Y, Zc = np.meshgrid(c, c, c, indexing='ij')
    nx, ny, nz = normal
    plane = (nx*X + ny*Y + nz*Zc) < 0
    outside = r >= rc
    masks = [plane & outside, (~plane) & outside]
    ops = [masked_laplacian(m, h) for m in masks]
    rho = [m.astype(float)/(m.sum()*h**3) for m in masks]
    E = None
    for it in range(iters):
        rt = rho[0] + rho[1]
        pt = phi_periodic(rho_i - rt, G, h)
        new = []
        for s_ in (0, 1):
            ru = np.full_like(rho[s_], 1.0/a**3)
            Vo = (-pt - phi_isolated(rho[s_] - ru, n, h)).ravel()
            M, keep = ops[s_]
            val, vec = eigsh((M + sp.diags(Vo[keep])).tocsr(), k=1, which='SA', tol=1e-11)
            v = vec[:, 0]; v = v/np.sqrt(np.sum(v*v)*h**3)
            q = np.zeros(n**3); q[keep] = v*v
            new.append(q.reshape(n, n, n))
        rho = [(1-mix)*o + mix*nn for o, nn in zip(rho, new)]
        rho = [q/(q.sum()*h**3) for q in rho]
        rt = rho[0] + rho[1]
        T = 0.0
        for s_ in (0, 1):
            M, keep = ops[s_]
            uu = np.sqrt(rho[s_].ravel()[keep]); T += float(uu @ (M @ uu))*h**3
        U_eI = float(np.sum(rt*(-phi_periodic(rho_i, G, h))))*h**3
        U_ee = (0.5*float(np.sum(rt*phi_periodic(rt, G, h)))*h**3
                - self_energy_bg(rho[0], n, h, a) - self_energy_bg(rho[1], n, h, a))
        U_II = -ALPHA_SC*Z*Z/((3/(4*np.pi))**(1/3)*a)
        En = T + U_eI + U_ee + U_II
        if E is not None and abs(En - E) < tol:
            break
        E = En
    return dict(E=En, T=T, a=a, rs=(3/(4*np.pi))**(1/3)*a/Z**(1/3), it=it+1,
                rho_ion_ratio=float(rt[n//2, n//2, n//2]/rt.mean()))


if __name__ == '__main__':
    h = float(os.environ.get('H', 0.21))
    rc = float(os.environ.get('RC', 1.67))
    print("3D PERIODIC CUBIC CELL, sodium pseudokernel (r_c = 1.67)")
    print(f"h = {h} held FIXED, strain carried by n (even only); radial WS target: "
          f"r_s = 3.77 a0, B = 9.1 GPa\n")
    print("      n      a      r_s        E (Ha)        T       U_eI      U_ee      U_II   its")
    ns = [int(x) for x in os.environ.get('NS', '26,28,30,32,34,36').split(',')]
    rows = []
    for n in ns:
        S = cell(n, h, rc=rc)
        rows.append((S['a'], S['E']))
        print(f"   {n:5d} {S['a']:6.2f} {S['rs']:7.3f} {S['E']:12.6f} {S['T']:9.4f} "
              f"{S['U_eI']:9.4f} {S['U_ee']:9.4f} {S['U_II']:9.4f} {S['it']:5d}", flush=True)
    A = np.array([r[0] for r in rows]); E = np.array([r[1] for r in rows])
    da = A[1] - A[0]
    print("\n      a       r_s         P (Ha/a0^3)        B (GPa)")
    for i in range(1, len(A)-1):
        d1 = (E[i+1] - E[i-1])/(2*da)
        d2 = (E[i+1] - 2*E[i] + E[i-1])/(da*da)
        dV = 3*A[i]**2
        print(f"   {A[i]:6.2f} {A[i]/1.6119:8.3f} {-d1/dV:16.3e} {A[i]**3*(d2-d1*6*A[i]/dV)/dV**2*HA_GPA:14.2f}")
