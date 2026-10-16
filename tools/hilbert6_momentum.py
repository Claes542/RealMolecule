"""
HILBERT-6 PROGRAMME, ITEM 2: the momentum balance as an IDENTITY, at TWO domains.

WHAT ALREADY EXISTED (checked before writing this): tools/forces.py computes the Hellmann-Feynman
force on a kernel from the density, and tools/hf_check.py compares it with -dE/dD taken from the
solver's own energies.  That pair is the "forces agree" half of item 2 and is NOT rebuilt here.
What did not exist anywhere is the QUANTUM STRESS itself -- no T_ij, no interface flux -- which is
the part that makes this a test of the hydrodynamic form rather than of the variational one.

THE IDENTITY BEING TESTED.  With u=sqrt(rho) and Q = -(1/2) lap(u)/u, the chain rule alone gives

    rho grad Q = div T,        T_ij = -(1/2) [ u d_i d_j u  -  d_i u d_j u ]
                                    = -(1/4) d_i d_j rho + (d_i rho)(d_j rho)/(4 rho)

(no delta_ij term; verified against the 1D form sigma = (1/4)[(rho')^2/rho - rho'']).  The static
Euler-Lagrange equation of the model says Q + V = lambda with V = -K + P_other, so grad(Q+V)=0 and

    div T  =  - rho grad V .

Integrating the z-component over the half space Omega = {z>0} that holds one electron turns that
into a statement about FORCES, which is the testable form:

    -int_{z=0} T_zz dA  +  (outer faces)  =  - int_Omega rho d_z V dV
    \_______________________________/        \_______________________/
     flux of quantum stress through           net electrostatic force on
     the free interface (a THIRD              this electron, from both
     derivative of the density)               kernels and the partner

The left side is a surface integral of a third-derivative quantity on the interface; the right side
is a volume integral of Coulomb forces.  Nothing makes them equal except the identity.  This is new
content at two domains: for ONE domain the Madelung pair is equivalent to the Schrodinger equation,
so there was nothing to learn (Section "Phase 1" of the paper).

GEOMETRY.  Two kernels Z=1 at z=+-D/2 on the axis, one electron each, split at the midplane, the
far electron obtained by reflection -- the construction of tools/axisym_x2.py.  Omega = {z>0} holds
electron 2; its partner's potential P1 is the mirror of its own.

WHY THE MIDPLANE STENCIL IS THE DELICATE PART, and how it is handled.  The grid is cell-centred, so
z=0 is a FACE and not a grid point, and T_zz has to be evaluated there.  The interface condition is
homogeneous Neumann, u_z=0, which means u continues EVENLY across z=0 -- and that even continuation
is numerically the partner's own u, by the symmetry of the problem.  So every z-derivative at the
midplane is taken on the evenly extended array, and the face value of T_zz (which is even, being
built from u and u_zz) comes from the two adjacent cells as T_face = (9 T_0 - T_1)/8, exact for even
quartics.  The stencils are SELF-TESTED against an analytic Gaussian before being used on the
solution; u_z at the face is reported too, since it must vanish and is therefore a free check that
the extension is the right one.

TRAP INHERITED FROM Phase 1 (tools/madelung_identity.py), do not repeat it: u_z^2/u-type quantities
are meaningless where rho underflows, and masking a region and then differentiating SPECTRALLY
pollutes the whole domain.  All derivatives here are LOCAL finite differences and every norm is
taken on the bulk mask ERODED by the stencil half-width, so no stencil straddles the mask edge.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, factorized
from axisym_realqm import Grid, stiffness, laplacian, kernel_potential

np.seterr(all='ignore')
SOFT = np.sqrt(0.081)          # atom.js kernel calibration: eps = 0.285 h
_cache = {}


# ----------------------------------------------------------------------------- derivatives
def d1_4(f, h, axis):
    """4th-order centred first derivative; one-sided 4th-order at both ends."""
    f = np.asarray(f, float)
    g = np.empty_like(f)
    sl = [slice(None)] * f.ndim

    def take(i0, i1):
        s = list(sl); s[axis] = slice(i0, i1); return f[tuple(s)]

    n = f.shape[axis]
    core = (take(0, n-4)*1 - 8*take(1, n-3) + 8*take(3, n-1) - take(4, n))/(12*h)
    s = list(sl); s[axis] = slice(2, n-2); g[tuple(s)] = core
    for k, idx in ((0, 0), (1, 1), (-2, n-2), (-1, n-1)):
        s = list(sl); s[axis] = idx
        # 5-point one-sided (4th order) from the nearest 5 nodes
        j0 = 0 if idx < 2 else n-5
        st = list(sl); st[axis] = slice(j0, j0+5)
        blk = f[tuple(st)]
        p = idx - j0
        c = _onesided(5, p, 1)
        g[tuple(s)] = np.tensordot(c, blk, axes=([0], [axis]))/h
    return g


def d2_4(f, h, axis):
    """4th-order centred second derivative; one-sided 4th-order at both ends."""
    f = np.asarray(f, float)
    g = np.empty_like(f)
    sl = [slice(None)] * f.ndim
    n = f.shape[axis]

    def take(i0, i1):
        s = list(sl); s[axis] = slice(i0, i1); return f[tuple(s)]

    core = (-take(0, n-4) + 16*take(1, n-3) - 30*take(2, n-2)
            + 16*take(3, n-1) - take(4, n))/(12*h*h)
    s = list(sl); s[axis] = slice(2, n-2); g[tuple(s)] = core
    for idx in (0, 1, n-2, n-1):
        j0 = 0 if idx < 2 else n-6
        st = list(sl); st[axis] = slice(j0, j0+6)
        blk = f[tuple(st)]
        c = _onesided(6, idx-j0, 2)
        s = list(sl); s[axis] = idx
        g[tuple(s)] = np.tensordot(c, blk, axes=([0], [axis]))/(h*h)
    return g


def _onesided(npts, p, order):
    """Finite-difference weights for d^order at node p of a uniform npts-node stencil (spacing 1)."""
    A = np.vander(np.arange(npts) - p, npts, increasing=True).T   # A[k,j] = (x_j)^k
    b = np.zeros(npts); b[order] = float(math.factorial(order))
    return np.linalg.solve(A, b)


def even_extend_z(a):
    """Mirror a (nz,nr) half-grid field in z: the Neumann continuation across the midplane."""
    return np.vstack([a[::-1, :], a])


def even_extend_r(a, k=3):
    """Mirror k columns across the axis -- u is even in r, the axis face having zero area."""
    return np.hstack([a[:, k-1::-1], a])


# ----------------------------------------------------------------------------- the ground state
def solve_x2(h, L, D, iters=120, tol=1e-11, verbose=False, eps_fixed=None):
    """X2, one electron per kernel, midplane split. Returns the fields needed for the balance."""
    nzh = int(round(L/h)); L = nzh*h
    gh = Grid(L, L, h, zmin=0.0)       # electron 2 lives here, z>0
    gf = Grid(L, L, h, zmin=-L)        # full grid, for the Poisson solve
    assert gf.nz == 2*gh.nz
    key = (h, L)
    if key not in _cache:
        Lp = laplacian(gf).tolil()
        bnd = np.zeros(gf.N, bool)
        bnd[gf.idx(0, np.arange(gf.nr))] = True
        bnd[gf.idx(gf.nz-1, np.arange(gf.nr))] = True
        bnd[gf.idx(np.arange(gf.nz), gf.nr-1)] = True
        for k in np.where(bnd)[0]:
            Lp.rows[k] = [k]; Lp.data[k] = [1.0]
        _cache[key] = (stiffness(gh), factorized(Lp.tocsc()), bnd,
                       np.sqrt(gf.Z**2 + gf.R**2).ravel())
    S, solveP, bnd, rrf = _cache[key]

    # eps = 0.285 h is the atom.js CALIBRATION, right for energies but wrong for a convergence
    # study: it sharpens the kernel as the mesh refines, so the continuum problem moves with h and
    # an h-ladder measures two things at once.  For an IDENTITY test the softening is held fixed.
    eps = SOFT*h if eps_fixed is None else eps_fixed
    K = kernel_potential(gh, [(-D/2, 1.0), (D/2, 1.0)], eps)
    w = 1.0/np.sqrt(gh.V); W = sp.diags(w)
    mir = np.arange(gh.nz)[::-1]
    P1 = np.zeros(gh.N); u = None; Eo = None; lam = None
    for it in range(iters):
        A = (W @ (S + sp.diags((-K + P1)*gh.V)).tocsr() @ W).tocsr()
        val, vec = eigsh(A, k=1, which='SA', tol=1e-12, maxiter=400000,
                         v0=None if u is None else u/w)
        lam = float(val[0])
        u = w*vec[:, 0]
        if u[np.argmax(np.abs(u))] < 0:
            u = -u
        u = u/np.sqrt(np.sum(u*u*gh.V))
        densf = np.zeros((gf.nz, gf.nr)); densf[gh.nz:, :] = (u*u).reshape(gh.nz, gh.nr)
        b = 4*np.pi*densf.ravel()*gf.V; b[bnd] = 1.0/rrf[bnd]
        Pfull = solveP(b).reshape(gf.nz, gf.nr)      # potential of THIS electron
        P1 = Pfull[mir, :].ravel()                   # partner's potential, on z>0
        T = float(u @ (S @ u))
        VK = -float(np.sum(K*u*u*gh.V))
        U12 = float(np.sum(P1*u*u*gh.V))
        E = 2*(T + VK) + U12 + 1.0/D
        if verbose:
            print(f"      it{it:03d} E={E:.9f} lam={lam:.6f}")
        if Eo is not None and abs(E - Eo) < tol:
            break
        Eo = E
    return dict(E=E, lam=lam, u=u.reshape(gh.nz, gh.nr), K=K.reshape(gh.nz, gh.nr),
                P1=P1.reshape(gh.nz, gh.nr), Pself=Pfull, gh=gh, gf=gf, h=h, D=D,
                eps=eps, iters=it+1, T=T, VK=VK, U12=U12)


# ----------------------------------------------------------------------------- the balance
def momentum_balance(S, zcuts=(0.0, 0.3, 0.7, 1.5)):
    """z-momentum balance on Omega_k = {z > z_k}, for several cut faces z_k.

    k=0 is the free interface itself.  Moving the cut INTO the bulk gives a whole family of
    independent balances off the same solution: the stress flux through the cut must equal the
    electrostatic force on the charge beyond it, at every k.  That is a much stronger test than
    the single interface statement, and it costs nothing.

    NOTE on a diagnostic that was tried and dropped: u_z is ODD about the midplane, so
    interpolating it to z=0 with the even-function formula (9f_0-f_1)/8 returns ~0.375 h u_zz
    rather than zero.  It looked like a violated Neumann condition and was an artefact of the
    formula.  u_z(0)=0 holds identically under the even extension, so there is nothing to test
    there; the no-flux condition itself is exact by construction, the solver's stiffness simply
    omitting that face.
    """
    h, gh = S['h'], S['gh']
    u, K, P1 = S['u'], S['K'], S['P1']
    nzh, nr = u.shape
    r = gh.r

    ue = even_extend_r(even_extend_z(u), k=3)
    uz = d1_4(ue, h, axis=0)
    uzz = d2_4(ue, h, axis=0)
    ur = d1_4(ue, h, axis=1)
    urz = d1_4(uz, h, axis=1)
    cut = slice(3, None)
    ue, uz, uzz, ur, urz = (a[:, cut] for a in (ue, uz, uzz, ur, urz))

    T_zz = -0.5*(ue*uzz - uz*uz)            # even about z=0
    T_rz = -0.5*(ue*urz - ur*uz)            # odd about z=0

    dA_z = 2*np.pi*r*h
    dA_r = 2*np.pi*r[-1]*h
    dV = 2*np.pi*gh.R*h*h
    rho = u*u

    Ke = even_extend_z(K)
    P1e = np.vstack([S['Pself'][:nzh, :], P1])
    Vz_full = d1_4(-Ke + P1e, h, axis=0)
    Vz = Vz_full[nzh:, :]
    Kz = d1_4(Ke, h, axis=0)[nzh:, :]
    Pz = d1_4(P1e, h, axis=0)[nzh:, :]

    out = {}
    for zc in zcuts:
        k = int(round(zc/h))                 # face at the nearest cell boundary to z=zc
        i = nzh + k                          # the face sits between cells i-1 and i
        # T_zz at an interior face: 4th-order symmetric interpolation from two cells each side
        Tf = (9*(T_zz[i, :] + T_zz[i-1, :]) - (T_zz[i+1, :] + T_zz[i-2, :]))/16.0
        if k == 0:                           # at the midplane T_zz is even: use the even form
            Tf = (9*T_zz[i, :] - T_zz[i+1, :])/8.0
        flux_cut = -float(np.sum(Tf*dA_z))   # outward normal of {z>z_k} is -z_hat
        flux_far = float(np.sum(T_zz[-1, :]*dA_z))
        flux_rad = float(np.sum(T_rz[i:, -1]*dA_r))
        lhs = flux_cut + flux_far + flux_rad
        rhs = -float(np.sum((rho*Vz*dV)[k:, :]))
        scale = float(np.sum(np.abs(rho*Vz*dV)[k:, :]))
        out[zc] = dict(z=k*h, lhs=lhs, rhs=rhs, flux_cut=flux_cut, flux_far=flux_far, flux_rad=flux_rad,
                      rel=abs(lhs-rhs)/max(scale, 1e-300), scale=scale,
                      q=float(np.sum((rho*dV)[k:, :])))
    out['F_kern'] = float(np.sum(rho*Kz*dV))
    out['F_part'] = -float(np.sum(rho*Pz*dV))
    out['norm'] = float(np.sum(rho*dV))
    return out


def euler_lagrange_residual(S, tol=1e-6):
    """Q + V = lambda pointwise, with Q from INDEPENDENT local differences (not the solver's own)."""
    h, gh = S['h'], S['gh']
    u, K, P1 = S['u'], S['K'], S['P1']
    nzh, nr = u.shape
    ue = even_extend_r(even_extend_z(u), k=3)
    lap = d2_4(ue, h, axis=0) + d2_4(ue, h, axis=1)
    re = even_extend_r(np.vstack([gh.R[::-1, :], gh.R]), k=3)
    lap = lap + d1_4(ue, h, axis=1)/np.maximum(re, 1e-300)        # cylindrical 1/r d_r
    lap = lap[nzh:, 3:]
    Q = -0.5*lap/np.maximum(u, 1e-300)
    V = -K + P1
    rho = u*u
    m = rho > tol*rho.max()
    m[:4, :] = m[-4:, :] = False
    m[:, -4:] = False
    tot = (Q + V)[m]
    return dict(lam_mean=float(tot.mean()), lam_sd=float(tot.std()),
                spread_Q=float(Q[m].max()-Q[m].min()),
                spread_V=float(V[m].max()-V[m].min()),
                rel=float(tot.std()/max(abs(Q[m]).max(), 1e-300)),
                lam_solver=S['lam'], n=int(m.sum()))



# ----------------------------------------------------------------------------- test C
def hf_force(S):
    """Electrostatic (Hellmann-Feynman) z-force on the kernel at z=+D/2, from the density alone.

    rho_total = u^2(|z|,r) by mirror symmetry, so the integral runs over the FULL domain.
    The softened kernel must be differentiated CONSISTENTLY with the one the solver used, which is
    why eps is carried through rather than reconstructed.
    """
    gh, h, D, eps = S['gh'], S['h'], S['D'], S['eps']
    d2 = S['u']**2
    rho_full = np.vstack([d2[::-1, :], d2])
    zf = np.concatenate([-gh.z[::-1], gh.z])
    Zf, Rf = np.meshgrid(zf, gh.r, indexing='ij')
    dV = 2*np.pi*Rf*h*h
    dz = (D/2 - Zf)
    r3 = (dz*dz + Rf*Rf + eps*eps)**1.5
    F_elec = -float(np.sum(rho_full*dz/r3*dV))
    return F_elec + 1.0/(D*D), F_elec, 1.0/(D*D)


def test_C(h, L, Ds, eps_fixed, step_cells=2):
    """F on the kernel from the density  vs  -dE/dD from the SAME code's energies.

    D is stepped only over COMMENSURATE values D=(2i+1)h, which put both kernels exactly on cell
    centres of the half grid.  Off-commensurate differences are egg-crate noise the size of the
    quantity (recorded for the X2 sweep in tools/x2_R_sweep.py), and hf_check.py differences at
    +-0.02 au on h=0.08 -- a quarter of a cell -- which is inside that noise.
    """
    rows = []
    dD = step_cells*h                                 # keeps both endpoints commensurate
    for D in Ds:
        i = round((D/h - 1)/2)
        D = (2*i + 1)*h                               # snap onto the commensurate lattice
        S0 = solve_x2(h, L, D, eps_fixed=eps_fixed)
        F, Fe, Fn = hf_force(S0)
        Em = solve_x2(h, L, D-dD, eps_fixed=eps_fixed)['E']
        Ep = solve_x2(h, L, D+dD, eps_fixed=eps_fixed)['E']
        dEdD = -(Ep - Em)/(2*dD)
        rows.append((D, S0['E'], Fn, Fe, F, dEdD))
    return rows


# ----------------------------------------------------------------------------- self-tests
def selftest_stencils():
    """The face interpolation and the derivative stencils, against an even analytic function."""
    ok = True
    for h in (0.2, 0.1, 0.05):
        z = (np.arange(1, 400)+0.5)*h - 200*h          # cell centres straddling 0
        g = np.exp(-0.7*z**2)
        i0 = int(np.argmin(np.abs(z - 0.5*h)))
        a = np.exp(-0.7*z**2)[:, None]
        uzz = d2_4(a, h, axis=0)[:, 0]
        exact = ((-1.4 + (1.4*z)**2)*np.exp(-0.7*z**2))
        err = np.max(np.abs(uzz[5:-5] - exact[5:-5]))
        Tf = (9*g[i0] - g[i0+1])/8.0
        ferr = abs(Tf - 1.0)
        print(f"   stencil h={h:5.3f}: max|u_zz err|={err:.3e}  face interp err={ferr:.3e}")
        ok &= err < 2e-3*(h/0.2)**3 or err < 1e-6
    return ok


# ----------------------------------------------------------------------------- driver
if __name__ == '__main__':
    D = float(os.environ.get('D', 1.4))
    L = float(os.environ.get('L', 9.0))
    hs = [float(x) for x in os.environ.get('HS', '0.25,0.20,0.15,0.12').split(',')]
    EPS = os.environ.get('EPS', 'fixed')
    eps_fixed = None if EPS == 'calib' else float(EPS) if EPS != 'fixed' else 0.0855
    ZCUTS = (0.0, 0.3, 0.7, 1.5)

    print(__doc__.split('TRAP INHERITED')[0].strip()[:0] or '', end='')
    print("HILBERT-6 ITEM 2  --  momentum balance as an identity, two domains")
    print(f"X2, Z=1 kernels at z=+-D/2, D={D}, box L={L}, "
          f"kernel softening eps={'0.285h (calibration)' if eps_fixed is None else eps_fixed}\n")
    print("SELF-TEST of the stencils on an analytic even function:")
    selftest_stencils()

    print("\nTEST A -- the Euler-Lagrange equation pointwise:  Q + V = lambda")
    print("   h      lambda(mean)   sd        rel      spread Q   spread V   lam(solver)  cells")
    A = {}
    for h in hs:
        S = solve_x2(h, L, D, eps_fixed=eps_fixed)
        a = euler_lagrange_residual(S)
        A[h] = (S, a)
        print(f" {h:5.3f}  {a['lam_mean']:12.6f} {a['lam_sd']:9.2e} {a['rel']:9.2e}"
              f" {a['spread_Q']:10.3f} {a['spread_V']:10.3f} {a['lam_solver']:11.6f} {a['n']:7d}",
              flush=True)

    print("\nTEST B -- integrated z-momentum balance on Omega = {z > z_cut}")
    print("   stress flux through the cut  ==  electrostatic force on the charge beyond it")
    print("   h     z_cut   charge     flux_cut       LHS          RHS      |L-R|/scale")
    for h in hs:
        S = A[h][0]
        b = momentum_balance(S, zcuts=ZCUTS)
        for zc in ZCUTS:
            d = b[zc]
            tag = "interface" if zc == 0.0 else ""
            print(f" {h:5.3f} {d['z']:6.3f}  {d['q']:8.5f} {d['flux_cut']:+12.6f}"
                  f" {d['lhs']:+12.6f} {d['rhs']:+12.6f} {d['rel']:11.2e}  {tag}", flush=True)
        print(f"       F_kernels={b['F_kern']:+.6f}  F_partner={b['F_part']:+.6f}"
              f"  net={b['F_kern']+b['F_part']:+.6f}  norm={b['norm']:.6f}", flush=True)

    if os.environ.get('TESTC'):
        print("\nTEST C -- force on the kernel two ways (the two are different code paths)")
        print("   from the density:  F = Z^2/D^2 - Z int rho (z_A-z)/|r_A-r|^3 dV")
        print("   from the energy:   F = -dE/dD, differenced on COMMENSURATE D=(2i+1)h")
        print("   h      D        E          F_nuc      F_elec     F(density)   -dE/dD     rel diff")
        for h in [float(x) for x in os.environ['TESTC'].split(',')]:
            for (D, E, Fn, Fe, F, dEdD) in test_C(h, L, (1.3, 1.5), eps_fixed):
                rel = abs(F - dEdD)/max(abs(dEdD), 1e-30)
                print(f" {h:5.3f} {D:6.3f} {E:11.6f} {Fn:+10.5f} {Fe:+10.5f}"
                      f" {F:+11.6f} {dEdD:+11.6f} {rel:10.2e}", flush=True)
