"""
THE ALKALI DIMERS WITH AN EMPTY CORE INSTEAD OF HARD DEPLETION.

tools/x2_R_sweep.py runs the reduced-valence model on Li2..Cs2 with HARD DEPLETION: the electron is
excluded from the ball r < R about either kernel, via masked_stiffness.  It gets bond lengths to
7.7% mean and binding energies +36..+101%, i.e. about a FACTOR 2 over.  That factor 2 is what stands
between this model and useful molecular energies.

This file runs the same systems with the other admissible core model, the EMPTY CORE: the electron
is admitted everywhere and the kernel attraction is cancelled inside r_c.  It is the simpler code --
full stiffness, no mask, no keep-index bookkeeping -- and the question is whether the binding
improves.

The two radii are not interchangeable: r_c = 0.855 R when both are fitted to the same ionization
energy (tools/rc_universal_constant.py).  So r_c is re-fitted HERE, on the dimer's own mesh, to each
element's measured IE -- E(X) and E(X2) must come from the same code on the same mesh for the
discretisation to cancel in E_bind = 2E(X) - E(X2), which is the house rule that makes differences
within one composition meaningful.

D is scanned only over COMMENSURATE values D = (2i+1)h, which put both kernels exactly on cell
centres of the half grid; otherwise the egg-crate noise is the size of the binding itself.

RESULT 2026-10-21: THE TWO CORE MODELS BRACKET THE MEASURED BINDING.

                     bond vs R_e exp              binding vs D_e exp
    hard depletion   +7, -7, -8, -10, -10%   |   +36, +97, +96, +101, +96%
    empty core      +21, +11, +11, +8, +6%   |   -47, -52, -53, -64, -68%

So the factor 2 is NOT a property of hard depletion.  Both core models miss by about a factor 2, in
OPPOSITE directions, and they bracket the measurement on BOTH observables coherently: the empty core
binds too weakly and its bonds are correspondingly too long, hard depletion binds too strongly and
its bonds are mostly too short.  Core severity controls both, and molecular binding energy is acutely
sensitive to it while geometry is not -- 8.5% against 11.4% mean bond error, both usable.  That is
the "geometry right, energies wrong" regime with a sharper statement attached: the energy error is
set by the core model, and the truth lies between the two admissible ones.

An intermediate severity would therefore reproduce the measured binding, but that is a FIT and must
be labelled one; nothing here predicts where between the two it falls.  Lithium is again the outlier
in the empty-core bond error (+21% against +6..+11%).

VALIDATION, run by --validate below.  At r_c = 0 the empty core vanishes and this code and
x2_R_sweep.py must solve the same problem; they agree to 1e-15 on three dimer geometries and on the
atom.  The three RuntimeWarnings this file emits ("divide by zero / overflow / invalid value in
matmul") are numpy reporting hardware FP flags lazily at the next checked operation; every array is
finite at every SCF iteration and the energy converges monotonically to six decimals.  The EXISTING
published code emits the same three, so they are pre-existing and benign rather than introduced here.

usage:  python3 tools/x2_emptycore.py
        python3 tools/x2_emptycore.py --validate
"""
import os, sys, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from axisym_realqm import Grid, stiffness, kernel_potential
from axisym_hom2 import _poisson_setup

HA_EV = 27.211386


def atom_ec(rc, Lz=14.0, Lr=14.0, h=0.175):
    """One electron on a +1 kernel whose attraction is cancelled inside rc.  IE = -E."""
    g = Grid(Lz, Lr, h, zmin=-Lz)
    K = kernel_potential(g, [(0.0, 1.0)], 0.0)
    K = np.where(np.sqrt(g.Z**2 + g.R**2).ravel() < rc, 0.0, K)
    w = 1.0/np.sqrt(g.V); W = sp.diags(w)
    A = (W @ (stiffness(g) + sp.diags(-K*g.V)).tocsr() @ W).tocsr()
    return eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000)[0][0]


def x2_ec(rc, D, Lz=14.0, Lr=14.0, h=0.175, iters=120, tol=1e-9):
    """Two +1 empty-core kernels at +-D/2, one electron each, split at the midplane.

    Identical to x2_R_sweep.x2R except that the core is applied to the POTENTIAL rather than to the
    domain: full stiffness, and K zeroed inside rc of either kernel.
    """
    gh = Grid(Lz, Lr, h, zmin=0.0)
    gf = Grid(Lz, Lr, h, zmin=-Lz)
    nzh = gh.nz
    assert gf.nz == 2*nzh
    K = kernel_potential(gh, [(-D/2, 1.0), (D/2, 1.0)], 0.0)
    rr_near = np.sqrt((gh.Z - D/2)**2 + gh.R**2).ravel()
    rr_far = np.sqrt((gh.Z + D/2)**2 + gh.R**2).ravel()
    K = np.where((rr_near < rc) | (rr_far < rc), 0.0, K)     # the only change
    S = stiffness(gh)
    solveP, bnd, rrf = _poisson_setup(gf)
    w = 1.0/np.sqrt(gh.V); W = sp.diags(w)
    mir = np.arange(nzh)[::-1]
    P1 = np.zeros(gh.N); u = None; Eold = None
    for it in range(iters):
        A = (W @ (S + sp.diags((-K + P1)*gh.V)).tocsr() @ W).tocsr()
        val, vec = eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000,
                         v0=None if u is None else u/w)
        u = w*vec[:, 0]; u = u/np.sqrt(np.sum(u*u*gh.V))
        densf = np.zeros((gf.nz, gf.nr)); densf[nzh:, :] = (u*u).reshape(nzh, gh.nr)
        b = 4*np.pi*densf.ravel()*gf.V
        b[bnd] = 1.0/rrf[bnd]
        P1 = solveP(b).reshape(gf.nz, gf.nr)[mir, :].ravel()
        T = float(u @ (S @ u))
        VK = -float(np.sum(K*u*u*gh.V))
        U12 = float(np.sum(P1*u*u*gh.V))
        E = 2*(T + VK) + U12 + 1.0/D
        if Eold is not None and abs(E - Eold) < tol:
            break
        Eold = E
    return E


def fit_rc(ie, h, lo=0.2, hi=6.0, tol=3e-4):
    for _ in range(30):
        mid = 0.5*(lo + hi)
        if -atom_ec(mid, h=h)*HA_EV > ie:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5*(lo + hi)


def validate():
    """At r_c = 0 the empty core vanishes, so this code and x2_R_sweep must agree exactly."""
    from x2_R_sweep import x2R, atomX
    print("VALIDATION at r_c = 0 -- both codes solve the same problem\n")
    print("%10s %16s %14s %12s" % ("", "x2_emptycore", "x2_R_sweep", "diff"))
    for D in (3.325, 5.425, 7.875):
        a = x2_ec(0.0, D, 14.0, 14.0, 0.175)
        b = x2R(0.0, D, 14.0, 14.0, 0.175)
        print("  D=%6.3f %16.9f %14.9f %12.2e" % (D, a, b, a - b))
    a = atom_ec(0.0, 14.0, 14.0, 0.175)
    b = atomX(0.0, 14.0, 14.0, 0.175)
    print("  atom   %16.9f %14.9f %12.2e   (exact E(H) = -0.5)" % (a, b, a - b))


if __name__ == '__main__':
    if '--validate' in sys.argv:
        validate(); sys.exit(0)
    h, L = 0.175, 14.0
    # measured: atomic IE (eV), D_e (eV), R_e (bohr).  Hard-depletion results for the same systems
    # are from the table in tools/x2_R_sweep.py, same h and L.
    MEAS = [("Li", 5.392, 1.050, 5.051, 5.425, 1.430),
            ("Na", 5.139, 0.747, 5.818, 5.425, 1.470),
            ("K",  4.341, 0.551, 7.416, 6.825, 1.078),
            ("Rb", 4.177, 0.495, 7.955, 7.175, 0.995),
            ("Cs", 3.894, 0.452, 8.783, 7.875, 0.884)]
    print("ALKALI DIMERS: EMPTY CORE vs HARD DEPLETION")
    print(f"h={h}, L={L}, commensurate D=(2i+1)h; r_c re-fitted here to each measured IE\n")
    print("%4s %7s %8s | %8s %8s %7s | %8s %8s %7s" %
          ("el", "r_c", "IE eV", "D emptyc", "D hard", "R_e exp",
           "Eb emptyc", "Eb hard", "De exp"))
    rows = []
    for el, ie, de, re, Dhard, Ebhard in MEAS:
        t0 = time.time()
        rc = fit_rc(ie, h)
        eX = atom_ec(rc, L, L, h)
        allD = [(2*i+1)*h for i in range(1, 40)]
        allD = [D for D in allD if D > 2*rc + 2*h and D <= 12.0]
        vals = {}
        for k in range(0, len(allD), 3):
            vals[k] = x2_ec(rc, allD[k], L, L, h)
        kb = min(vals, key=vals.get)
        for k in range(max(0, kb-3), min(len(allD), kb+4)):
            if k not in vals:
                vals[k] = x2_ec(rc, allD[k], L, L, h)
        kb = min(vals, key=vals.get)
        Dm, E2 = allD[kb], vals[kb]
        eb = (2*eX - E2)*HA_EV
        rows.append((el, Dm, re, eb, de, Dhard, Ebhard))
        edge = " <-EDGE" if kb in (0, len(allD)-1) else ""
        print("%4s %7.3f %8.3f | %8.3f %8.3f %7.3f | %8.3f %8.3f %7.3f  [%.0fs]%s" %
              (el, rc, -eX*HA_EV, Dm, Dhard, re, eb, Ebhard, de, time.time()-t0, edge),
              flush=True)
    import statistics as st
    print("\n   bond error:   empty core %+.1f%% mean|.|, hard depletion %+.1f%%" % (
        st.mean([abs(r[1]-r[2])/r[2]*100 for r in rows]),
        st.mean([abs(r[5]-r[2])/r[2]*100 for r in rows])))
    print("   binding:      empty core " + ", ".join("%+.0f%%" % ((r[3]-r[4])/r[4]*100) for r in rows))
    print("                 hard depl. " + ", ".join("%+.0f%%" % ((r[6]-r[4])/r[4]*100) for r in rows))
    print("\n   NOTE the commensurate step is 2h = %.3f au, so adjacent species need not be resolved;"
          % (2*h))
    print("   that limitation is stated in tools/x2_R_sweep.py and applies here unchanged.")
