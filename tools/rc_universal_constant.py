"""
THE CORE RADIUS, DETERMINED INSIDE RealQM -- and the diagnosis of lithium.

THE SETTING.  The elasticity route (realqm_elasticity.tex) computes each alkali's equilibrium
density and bulk modulus in a Wigner-Seitz cell whose electron-ion potential is the EMPTY CORE,
V = 0 inside r_c and -Z/r outside.  Published, r_c was one parameter per element taken from the
Ashcroft literature.  It need not be: RealQM fixes it from the free atom.

TWO CORE MODELS, ONE CONSTANT.  A reduced valence calculation needs a core model, and the programme
admits two:

   hard depletion   density identically zero for r < R, Neumann at R -- RealQM's native non-overlap
   empty core       V = 0 for r < r_c, the valence density ADMITTED inside

Fit BOTH to the same measured ionization energy and the ratio is a constant,

        r_c / R  =  0.855 +- 0.006     over Li, Na, K, Rb, Cs,

with no literature input anywhere and LITHIUM INCLUDED.  So the two are not rival models but two
parameterisations of one ion core: denying territory is the stronger constraint, so the exclusion
radius must exceed the empty-core radius to give the same binding, and this constant is the exchange
rate between them.  That it is one number across a factor 1.6 in size says the two descriptions
differ by a pure scale and nothing element-specific.

LITHIUM.  Fitting the empty core to the free atom's IE reproduces Ashcroft's radii at a NEARLY
CONSTANT +11% for Na, K, Rb, Cs -- and +62% at lithium.  Three independent routes therefore agree
that lithium's literature Ashcroft radius is anomalously small: this one, the ratio above (in which
lithium is NOT an outlier, while against Ashcroft's radii it is), and the alkali-dimer work, where
lithium's core radius exceeds its ionic radius by 0.56 a0 against 0.1-0.2 elsewhere.  Lithium's
failure in the published table is the empty-core INPUT, not the one-electron-per-cell picture.  This
settles the open question stated in realqm_hilbert6.tex.

THE ACCURACY HIERARCHY, which is the honest way to report this.  Four choices of input for the same
cell calculation, in increasing dependence on anything external:

  (1) ZERO fitted constants, r_c from the IE alone .......  r_s 11.5%,  B 23%
  (2) ONE constant fitted to ONE solid datum (Na's r_s) ..  r_s  5.1%,  B 13%
  (3) ONE constant fitted to four Ashcroft radii .........  r_s  2.9%,  B 17%
  (4) FIVE literature radii, one per element .............  r_s  6.2%,  B 79%

(1) is one-sided -- every r_s high, every B low -- because the free atom's core radius runs ~11%
larger than the solid wants, which is a physical atom-to-solid difference and not an error.  (2)
absorbs exactly that offset with a single number and is the claim worth making, since it touches no
pseudopotential literature.  (3) has the best numbers but leans on the literature and requires
lithium to be excluded from the mean.  The constant of (2) is 0.8055 and of (3) is 0.7688: two
independent routes agreeing to 4.8%.

WHAT DOES NOT REPAIR.  Lithium's pressure derivative B' goes from -0.714 to -0.212 -- six times
smaller but still the wrong sign -- and the other four barely move.  So the input error explained
B's failure at lithium, while B' fails for the different reason the paper names: the order of the
derivative.  Routine: tools/hilbert6_form.py:alkali_Bprime.

THE CONSTANT IS MONOVALENT.  Carried to divalent magnesium the ratio does not transfer; see the
module docstring of the memory note, and the figures printed by --mg below.

usage:  python3 tools/rc_universal_constant.py
        RECHECK=1 python3 tools/rc_universal_constant.py    # re-fit every radius from scratch (slow)
        LATEX=1   python3 tools/rc_universal_constant.py    # emit the paper's table rows
"""
import os, sys, statistics as st

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'real_plasma_numerics'))
import metal_alkalis as ma

HA_EV = 27.211386

# Measured first ionization energies (eV), the only per-element input.
IE_EV = {'Li': 5.392, 'Na': 5.139, 'K': 4.341, 'Rb': 4.177, 'Cs': 3.894}

# Radii fitted to those IEs, h = 0.125, L = 14.  Both reproduced from scratch 2026-10-21;
# set RECHECK=1 to re-derive.  R_HD is the hard-depletion radius of tools/x2_R_sweep.py.
R_HD = {'Li': 1.999, 'Na': 2.161, 'K': 2.792, 'Rb': 2.972, 'Cs': 3.295}
RC_EC = {'Li': 1.716, 'Na': 1.865, 'K': 2.386, 'Rb': 2.533, 'Cs': 2.791}


def atomX_ec(rc, h=0.125, L=14.0):
    """EMPTY CORE: V = 0 inside rc, -1/r outside, electron admitted everywhere.

    This is tools/x2_R_sweep.py:atomX with the mask removed and the kernel potential zeroed inside
    rc -- the two core models differ by exactly that, which is why the ratio below is meaningful.
    """
    import numpy as np, scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    from axisym_realqm import Grid, stiffness, kernel_potential
    g = Grid(L, L, h, zmin=-L)
    K = kernel_potential(g, [(0.0, 1.0)], 0.0)
    K = np.where(np.sqrt(g.Z**2 + g.R**2).ravel() < rc, 0.0, K)
    w = 1.0/np.sqrt(g.V)
    W = sp.diags(w)
    A = (W @ (stiffness(g) + sp.diags(-K*g.V)).tocsr() @ W).tocsr()
    return eigsh(A, k=1, which='SA', tol=1e-11, maxiter=200000)[0][0]


def fit_rc(ie, lo=0.2, hi=6.0, tol=3e-4):
    """Bisect the empty-core radius so the pseudo-atom reproduces the measured IE."""
    for _ in range(36):
        mid = 0.5*(lo + hi)
        if -atomX_ec(mid)*HA_EV > ie:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5*(lo + hi)


def observables(rc):
    """Equilibrium r_s and bulk modulus from the paper's own Wigner-Seitz driver."""
    f = ma.make_energy(rc)
    rs = ma.minimise(lambda x: f(x)[0])
    return rs, ma.bulk_modulus(lambda x: f(x)[0], rs)


def fit_C_to_sodium():
    """One constant, fixed by ONE solid-state datum: sodium's measured r_s."""
    lo, hi = 0.60, 1.00
    for _ in range(34):
        mid = 0.5*(lo + hi)
        if observables(mid*R_HD['Na'])[0] < ma.ALKALIS['Na'][1]:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo + hi)


def report(tag, rc_of):
    rows, er, eb = [], [], []
    for el, (_, rs_e, B_e) in ma.ALKALIS.items():
        rc = rc_of(el)
        rs, B = observables(rc)
        er.append(100*(rs - rs_e)/rs_e)
        eb.append(100*(B - B_e)/B_e)
        rows.append((el, rc, rs, rs_e, B, B_e, er[-1], eb[-1]))
    mr, mb = st.mean([abs(x) for x in er]), st.mean([abs(x) for x in eb])
    print(f"\n{tag}")
    print(f"{'':4}{'r_c':>7}{'':2}{'r_s':>7}{'exp':>7}{'err':>8}{'':2}{'B':>7}{'exp':>7}{'err':>8}")
    for el, rc, rs, rs_e, B, B_e, e1, e2 in rows:
        print(f"{el:4}{rc:>7.3f}{'':2}{rs:>7.2f}{rs_e:>7.2f}{e1:>+7.1f}%{'':2}"
              f"{B:>7.1f}{B_e:>7.1f}{e2:>+7.0f}%", flush=True)
    print(f"   mean |err|:  r_s {mr:.1f}%   B {mb:.0f}%")
    return rows, mr, mb


def main():
    rc_ec = dict(RC_EC)
    if os.environ.get('RECHECK'):
        print("re-fitting every empty-core radius to its measured IE")
        for el, ie in IE_EV.items():
            rc_ec[el] = fit_rc(ie)
            print(f"   {el:3s} IE {ie:6.3f} eV -> r_c = {rc_ec[el]:.3f}   "
                  f"[{RC_EC[el]:.3f}]   d = {rc_ec[el]-RC_EC[el]:+.3f}", flush=True)

    print("TWO CORE MODELS FITTED TO THE SAME MEASURED IE -- no literature input\n")
    print(f"{'':4}{'R hard depl':>13}{'r_c empty core':>16}{'ratio':>9}{'Ashcroft':>10}{'vs Ashcroft':>13}")
    rat, off = [], []
    for el in IE_EV:
        rat.append(rc_ec[el]/R_HD[el])
        off.append(100*(rc_ec[el] - ma.ALKALIS[el][0])/ma.ALKALIS[el][0])
        print(f"{el:4}{R_HD[el]:>13.3f}{rc_ec[el]:>16.3f}{rat[-1]:>9.4f}"
              f"{ma.ALKALIS[el][0]:>10.2f}{off[-1]:>+12.1f}%")
    print(f"\n   ratio = {st.mean(rat):.4f}   sd {st.stdev(rat):.4f}   spread {max(rat)-min(rat):.4f}"
          f"   -- all five, lithium included")
    print(f"   vs Ashcroft: {off[0]:+.0f}% at lithium against "
          f"{', '.join(f'{v:+.0f}%' for v in off[1:])} elsewhere -- lithium's literature radius is "
          f"the anomaly")

    C_na = fit_C_to_sodium()
    C_lit = st.mean([ma.ALKALIS[e][0]/R_HD[e] for e in ('Na', 'K', 'Rb', 'Cs')])
    print(f"\n   C fitted to sodium's r_s alone : {C_na:.4f}")
    print(f"   C fitted to four Ashcroft radii: {C_lit:.4f}   "
          f"(agree to {abs(C_na-C_lit)/C_lit*100:.1f}%)")

    r1, m1r, m1b = report("(1) ZERO fitted constants -- r_c from the measured IE alone",
                          lambda el: rc_ec[el])
    r2, m2r, m2b = report(f"(2) ONE constant on ONE solid datum -- C = {C_na:.4f} from Na's r_s",
                          lambda el: C_na*R_HD[el])
    r3, m3r, m3b = report(f"(3) ONE constant on four Ashcroft radii -- C = {C_lit:.4f}",
                          lambda el: C_lit*R_HD[el])
    r4, m4r, m4b = report("(4) FIVE literature radii, one per element",
                          lambda el: ma.ALKALIS[el][0])

    print("\nHIERARCHY (mean |err|), increasing dependence on anything external")
    for lab, mr, mb in (("(1) zero constants      ", m1r, m1b),
                        ("(2) one constant, one datum", m2r, m2b),
                        ("(3) one constant, Ashcroft ", m3r, m3b),
                        ("(4) five literature radii  ", m4r, m4b)):
        print(f"   {lab}  r_s {mr:5.1f}%   B {mb:4.0f}%")
    print("   (1) is one-sided: every r_s high, every B low, because the free atom's core radius")
    print("   runs ~11% larger than the solid wants -- physical, not an error.  (2) absorbs it.")

    if os.environ.get('LATEX'):
        print("\n% ---- table rows, (1) then (2) ----")
        for rows in (r1, r2):
            for el, rc, rs, rs_e, B, B_e, _, _ in rows:
                print(f"{el} & ${rc:.3f}$ & ${rs:.2f}$ & ${rs_e:.2f}$ & ${B:.1f}$ & ${B_e:.1f}$ \\\\")
            print("%")
        print(f"% ratio {st.mean(rat):.4f} sd {st.stdev(rat):.4f}; C_na {C_na:.4f} C_lit {C_lit:.4f}")
        print(f"% means: (1) {m1r:.1f}/{m1b:.0f}  (2) {m2r:.1f}/{m2b:.0f}  "
              f"(3) {m3r:.1f}/{m3b:.0f}  (4) {m4r:.1f}/{m4b:.0f}")
        print("% ratios: " + ", ".join(f"{v:.4f}" for v in rat))


if __name__ == '__main__':
    main()
