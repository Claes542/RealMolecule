"""
C_44 FROM A SHEARED CELL -- the shear constant, and what it actually takes.

WHY THIS FILE.  realqm_elasticity.tex reaches the bulk modulus by dilating a cell; the remaining
item is the full tensor, which needs a deformation the Wigner-Seitz symmetry does not already
respect.  realqm_hilbert6.tex says of it that "no new machinery is needed".  That is wrong, and this
file is where the correction is measured: a dilated cubic cell stays cubic, so the Cartesian grid,
the per-axis Laplacian and the FFT Poisson kernel 4pi/k^2 all survive it.  A SHEARED cell does not.
In fractional coordinates s with x = H s, the metric g = H^T H is no longer diagonal, so

    the kinetic form picks up CROSS terms,   T = (1/2) det(H) int sum_ij (g^-1)_ij d_i u d_j u ds,
    the Poisson kernel needs the reciprocal metric,   |G|^2 = 4 pi^2 m^T g^-1 m,
    and the self-energy subtraction needs true Cartesian distances |H s|, not |s|.

STAGED, because the leading term is computable without any of that.  At FIXED VOLUME and fixed
uniform electron density, a volume-preserving shear leaves the electron gas contributing nothing, so
the leading shear constant is purely the LATTICE ELECTROSTATICS: point ions in a uniform
compensating background, sheared.  That is the classical Fuchs calculation, it needs only an Ewald
sum, and it has exact reference values to validate against.  Electron relaxation -- the RealQM part
-- is a correction on top, and needs the metric machinery above.

So this file computes stage one and says so.  It is not the full RealQM C_44.

VALIDATION.  The Ewald sum is checked against the standard Madelung constants for point ions in a
uniform neutralising background, E = -alpha Z^2 / r_s with (4pi/3) r_s^3 = Omega per ion:
simple cubic 0.880060, bcc 0.895930, fcc 0.895874.  Independence of the Ewald splitting parameter
eta is checked separately, since agreement at one eta can hide a missing term.

usage:  python3 tools/elastic3d_shear.py
"""
import numpy as np
from scipy.special import erfc

HA_GPA = 29421.0158
REF_ALPHA = {'sc': 0.880060, 'bcc': 0.895930, 'fcc': 0.895874}   # reference knowledge; validated below


def ewald_point_in_background(H, eta=None, nreal=6, nrecip=6, Z=1.0):
    """Electrostatic energy per ion, one point ion of charge Z per cell in a uniform -Z background.

    H has the lattice vectors as COLUMNS, so a position with fractional coordinates s is H @ s and
    the cell volume is det(H).  Returns energy in Hartree per ion.
    """
    Om = abs(float(np.linalg.det(H)))
    if eta is None:
        eta = 2.0*np.pi/Om**(1.0/3.0)/2.0          # a safe default; eta-independence is tested
    B = 2*np.pi*np.linalg.inv(H).T                 # reciprocal lattice vectors as columns

    rng = np.arange(-nreal, nreal + 1)
    I, J, K = np.meshgrid(rng, rng, rng, indexing='ij')
    n = np.stack([I.ravel(), J.ravel(), K.ravel()])
    Rv = H @ n
    d = np.linalg.norm(Rv, axis=0)
    keep = d > 1e-12
    real = float(np.sum(erfc(eta*d[keep])/d[keep]))

    rng = np.arange(-nrecip, nrecip + 1)
    I, J, K = np.meshgrid(rng, rng, rng, indexing='ij')
    m = np.stack([I.ravel(), J.ravel(), K.ravel()])
    Gv = B @ m
    G2 = np.sum(Gv*Gv, axis=0)
    keep = G2 > 1e-12
    recip = (4*np.pi/Om)*float(np.sum(np.exp(-G2[keep]/(4*eta*eta))/G2[keep]))

    self_t = -2*eta/np.sqrt(np.pi)                 # removal of the ion's own Gaussian
    bg_t = -np.pi/(eta*eta*Om)                     # the k=0 / uniform background term
    return 0.5*Z*Z*(real + recip + self_t + bg_t)


def alpha_of(H, **kw):
    """Convert the energy per ion to the Madelung constant alpha, E = -alpha Z^2 / r_s."""
    Om = abs(float(np.linalg.det(H)))
    rs = (3*Om/(4*np.pi))**(1.0/3.0)
    return -ewald_point_in_background(H, **kw)*rs


def lattice(kind, a=1.0):
    """Primitive lattice matrices (vectors as columns), one ion per cell."""
    if kind == 'sc':
        return a*np.eye(3)
    if kind == 'bcc':
        return a/2*np.array([[-1., 1., 1.], [1., -1., 1.], [1., 1., -1.]])
    if kind == 'fcc':
        return a/2*np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    raise ValueError(kind)


def shear(H, gamma):
    """Volume-preserving simple shear in the xy plane: x -> x + gamma*y."""
    S = np.array([[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return S @ H


def C44_electrostatic(kind, rs, gammas=(0.00, 0.02, 0.04, 0.06), **kw):
    """Shear constant from the second derivative of the Ewald energy at fixed volume.

    C_44 = (1/Omega) d^2 E / d gamma^2 at gamma = 0, with E the energy per cell.  The strain is
    volume preserving, so no pressure term enters and no equation of state is needed.
    """
    Om = (4*np.pi/3)*rs**3
    a = (Om/abs(float(np.linalg.det(lattice(kind, 1.0)))))**(1.0/3.0)
    H0 = lattice(kind, a)
    g = np.array(gammas, float)
    E = np.array([ewald_point_in_background(shear(H0, x), **kw) for x in g])
    # E is even in gamma, so fit E = E0 + c2 gamma^2 + c4 gamma^4 and take 2*c2
    A = np.vstack([np.ones_like(g), g**2, g**4]).T
    c = np.linalg.lstsq(A, E, rcond=None)[0]
    return 2*c[1]/Om*HA_GPA, E, g


if __name__ == '__main__':
    print("STAGE ONE: the ELECTROSTATIC shear constant.  Not the full RealQM C_44 -- the electron")
    print("density is held uniform, which is exact only to leading order.  See the docstring.\n")

    print("VALIDATION 1: Madelung constants for a point ion in a uniform neutralising background")
    print(f"{'':6}{'alpha (Ewald)':>16}{'reference':>12}{'diff':>12}")
    ok = True
    for kind in ('sc', 'bcc', 'fcc'):
        al = alpha_of(lattice(kind, 1.0), nreal=8, nrecip=8)
        d = al - REF_ALPHA[kind]
        ok &= abs(d) < 2e-6
        print(f"{kind:6}{al:>16.6f}{REF_ALPHA[kind]:>12.6f}{d:>12.2e}")
    print(f"   all three within 2e-6: {ok}\n")

    print("VALIDATION 2: independence of the Ewald splitting parameter eta (bcc)")
    print(f"{'eta':>8}{'alpha':>16}")
    H = lattice('bcc', 1.0)
    for eta in (3.0, 4.0, 5.0, 6.0, 7.0):
        print(f"{eta:>8.1f}{alpha_of(H, eta=eta, nreal=10, nrecip=10):>16.9f}")

    print("\nVALIDATION 3: the sheared energy must be EVEN in gamma (symmetry, not fitted)")
    H = lattice('bcc', 1.0)
    for gm in (0.02, 0.05):
        ep = ewald_point_in_background(shear(H, gm), nreal=8, nrecip=8)
        em = ewald_point_in_background(shear(H, -gm), nreal=8, nrecip=8)
        print(f"   gamma=+-{gm:.2f}:  E(+) = {ep:.10f}   E(-) = {em:.10f}   diff {ep-em:.2e}")

    print("\nSODIUM: bcc, r_s from the elasticity paper's own equilibrium value")
    print(f"{'r_s':>7}{'C44 es (GPa)':>15}{'C44 exp':>10}")
    for rs, lab in ((3.93, 'measured r_s'), (3.75, 'RealQM r_s, one constant')):
        c44, E, g = C44_electrostatic('bcc', rs, nreal=8, nrecip=8)
        print(f"{rs:>7.2f}{c44:>15.2f}{4.2:>10.1f}   ({lab})")
    print("\n   sodium's measured C_44 is about 4.2 GPa -- REFERENCE KNOWLEDGE, verify before use.")
    print("   A positive electrostatic C_44 of the right order is the result of stage one; the")
    print("   electron-relaxation correction needs the metric machinery named in the docstring and")
    print("   is NOT included here.")
