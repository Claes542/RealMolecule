"""
ONE CONSTANT FOR THE ALKALI SERIES -- and the diagnosis of lithium.

THE CLAIM.  The elasticity route (realqm_elasticity.tex) brings one parameter per element: the
Ashcroft empty-core radius r_c, taken from the literature.  That per-element input is unnecessary.
Across Na, K, Rb, Cs,

        r_c  =  C * R(IE),     C = 0.7688 +- 0.008   (under 1%)

where R is the PSEUDO-KERNEL RADIUS the programme already fixes elsewhere: one electron outside R on
a charge +1 kernel, R bisected so the pseudo-atom reproduces the element's MEASURED IONIZATION
ENERGY (tools/x2_R_sweep.py:atomX).  R therefore comes from an atomic measurement, with no
solid-state datum in it, and C is one number for the whole series.

WHAT THE CONSTANT MEANS.  r_c and R are two conventions for the same ion core.  The empty-core
potential CANCELS the attraction inside r_c but still admits the electron; RealQM's non-overlap
DENIES the territory outright, density zero with Neumann at R.  Denying territory is the stronger
constraint, so the exclusion radius must exceed the empty-core radius to give the same binding --
and r_c < R for all five metals.  C is the exchange rate between the two, and its being a single
number says they differ by a pure scale, not by anything element-specific.

LITHIUM.  C is the mean over Na, K, Rb, Cs ONLY; lithium's own ratio is 0.530, the outlier.  So
r_c = C*R for lithium is a PREDICTION -- it uses lithium's ionization energy and four other
elements' core radii, and no property of lithium's solid.  It moves lithium from the paper's named
failure to agreement:

        r_c = 1.06 (literature) ->  r_s 2.40 vs 3.25,  B 49.6 vs 11.6  (4.3x too stiff)
        r_c = 1.537 (= C*R)     ->  r_s 3.47 vs 3.25,  B 12.5 vs 11.6  (+7%)

This settles the open question stated in realqm_hilbert6.tex: lithium's failure was the empty-core
INPUT, not the one-electron-per-cell picture.

LIMIT.  C is monovalent.  Divalent magnesium on the literally parallel construction (one electron,
+1 kernel, matched to IE1) gives r_c = 0.813 against the literature 1.39, low by 42%.  A
valence-matched pseudo-atom (+2 kernel, two electrons, self-repulsion reduced by 1/2, matched to
IE1+IE2) gives 1.291, low by 7% -- better, but a different construction on which C was never
fitted.  One fit for all holds within a valence column; whether each column has its own constant is
open.

usage:  python3 tools/rc_universal_constant.py
        RECHECK_R=1 python3 tools/rc_universal_constant.py     # also re-bisect R from scratch (slow)
"""
import os, sys, statistics as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'real_plasma_numerics'))
import metal_alkalis as M

# R bisected by tools/x2_R_sweep.py:atomX to each MEASURED ionization energy, h=0.175, L=14.
# Re-derived from scratch 2026-10-21 and reproduces the published table to three decimals.
R_IE = {'Li': 1.999, 'Na': 2.161, 'K': 2.792, 'Rb': 2.972, 'Cs': 3.295}
IE_EV = {'Li': 5.392, 'Na': 5.139, 'K': 4.341, 'Rb': 4.177, 'Cs': 3.894}
FIT = ('Na', 'K', 'Rb', 'Cs')            # lithium deliberately excluded


def observables(rc):
    """Equilibrium r_s and bulk modulus from the paper's own Wigner-Seitz driver."""
    f = M.make_energy(rc)
    rs = M.minimise(lambda x: f(x)[0])
    return rs, M.bulk_modulus(lambda x: f(x)[0], rs)


def recheck_R():
    """Independently re-bisect R from the measured IEs -- the control for the table above."""
    from x2_R_sweep import atomX, HA_EV
    h, L = 0.175, 14.0
    print("CONTROL: re-bisect R so the pseudo-atom reproduces each measured IE")
    for el, ie in IE_EV.items():
        lo, hi = 0.0, 4.5
        for _ in range(40):
            mid = 0.5*(lo + hi)
            if -atomX(mid, L, L, h)*HA_EV > ie:
                lo = mid
            else:
                hi = mid
            if hi - lo < 2e-4:
                break
        R = 0.5*(lo + hi)
        print(f"   {el:3s} IE {ie:6.3f} eV -> R = {R:.3f}   [{R_IE[el]:.3f}]   "
              f"d = {R - R_IE[el]:+.3f}", flush=True)
    print()


def main():
    if os.environ.get('RECHECK_R'):
        recheck_R()

    ratios = {el: M.ALKALIS[el][0]/R_IE[el] for el in R_IE}
    sub = [ratios[el] for el in FIT]
    C = st.mean(sub)
    print("ONE CONSTANT FOR THE ALKALI SERIES\n")
    print("   r_c/R:  " + "   ".join(f"{el} {ratios[el]:.4f}" for el in R_IE))
    print(f"   C = mean over {', '.join(FIT)} = {C:.4f}   sd {st.stdev(sub):.4f}   "
          f"spread {max(sub) - min(sub):.4f}")
    print(f"   lithium's own ratio is {ratios['Li']:.4f} and is NOT in the mean, so its r_c "
          f"= {C*R_IE['Li']:.3f} is a prediction\n")

    print(f"{'':4}{'r_c lit':>8}{'r_c=CR':>8}{'':2}{'r_s lit':>8}{'r_s CR':>8}{'r_s exp':>8}"
          f"{'':2}{'B lit':>7}{'B CR':>7}{'B exp':>7}")
    dev = {k: [] for k in ('rs_l', 'rs_n', 'B_l', 'B_n')}
    for el, (rc_l, rs_e, B_e) in M.ALKALIS.items():
        (rs_l, B_l), (rs_n, B_n) = observables(rc_l), observables(C*R_IE[el])
        dev['rs_l'].append(100*(rs_l - rs_e)/rs_e); dev['rs_n'].append(100*(rs_n - rs_e)/rs_e)
        dev['B_l'].append(100*(B_l - B_e)/B_e);     dev['B_n'].append(100*(B_n - B_e)/B_e)
        print(f"{el:4}{rc_l:>8.2f}{C*R_IE[el]:>8.3f}{'':2}{rs_l:>8.2f}{rs_n:>8.2f}{rs_e:>8.2f}"
              f"{'':2}{B_l:>7.1f}{B_n:>7.1f}{B_e:>7.1f}", flush=True)

    print("\n   deviations %, order Li Na K Rb Cs")
    for k, lab in (('rs_l', 'r_s, literature r_c'), ('rs_n', 'r_s, one constant  '),
                   ('B_l',  'B,   literature r_c'), ('B_n',  'B,   one constant  ')):
        print(f"      {lab}: " + ", ".join(f"{v:+.0f}" for v in dev[k]))
    print(f"\n   mean |deviation|{'all five':>14}{'excl. Li':>12}")
    for k, lab in (('rs_l', 'r_s literature'), ('rs_n', 'r_s one constant'),
                   ('B_l',  'B   literature'), ('B_n',  'B   one constant')):
        a = [abs(v) for v in dev[k]]
        print(f"      {lab:18s}{st.mean(a):8.1f}%{st.mean(a[1:]):11.1f}%")
    print(f"\n   lithium's modulus: {(100 + dev['B_l'][0])/100:.2f}x too stiff  ->  "
          f"{(100 + dev['B_n'][0])/100:.2f}x")
    print("   the four fitted elements are a wash, so the entire gain is lithium\n")

    print("   ROBUSTNESS -- the repair is not a knife-edge in C (Li exp: r_s 3.25, B 11.6)")
    print(f"{'C':>10}{'r_c(Li)':>10}{'r_s':>8}{'B':>8}")
    for Ct in (0.70, 0.74, C, 0.80, 0.84):
        rs, B = observables(Ct*R_IE['Li'])
        print(f"{Ct:>10.4f}{Ct*R_IE['Li']:>10.3f}{rs:>8.2f}{B:>8.1f}", flush=True)


if __name__ == '__main__':
    main()
