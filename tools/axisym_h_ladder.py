# H2 BOND vs MESH, pushed to h = 0.02 -- the question "at what h is the bond correct?"
#
# Reuses tools/commensurate.py's solve()/report() rather than rebuilding: it already carries the
# commensuration assertion (kernels must sit exactly on cell centres, D = (2m+1)h) that was needed
# to get a clean minimum, and the eps = SOFT*h softening. The last two lines of that file run
# reports at import, so we exec only the part above them -- the pattern dscan.py already uses.
#
# WHY 2D. In 3D, h = 0.02 at box 10 is 500^3 = 1.25e8 cells, ~15x the largest grid that has ever
# run here. H2 is exactly a two-domain system on an axis, which is the case this axisymmetric
# reduction is EXACT for, so h = 0.02 is a 400x400 grid -- 1.6e5 points, affordable.
#
# WHAT IT ANSWERS. mol_fast (weak interface, eps = h) extrapolates to a bond of 1.314 at h -> 0,
# crossing the measured 1.401 near h ~ 0.018. This construction has the STRICT interface condition
# (face omission = homogeneous Neumann) and a different softening coefficient. But eps = SOFT*h
# vanishes with h either way, so BOTH converge to the same bare-Coulomb continuum problem, and the
# only surviving difference in the limit is the interface condition. So:
#   strict limit ~= 1.31  -> the interface condition does not matter in the limit, and RealQM's
#                            bond is genuinely a few percent SHORT, which is a property to report.
#   strict limit ~= 1.40  -> the condition DOES matter, and the weak scheme's undershoot is its own.
# Either way the answer is more useful than a mesh at which the error happens to vanish.
import sys, numpy as np
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
import os; os.chdir('/Users/cgjoh/Development/H2O/tools')
src = open('commensurate.py').read()
exec(src.split('report(0.05')[0])

def odd_bracket(h, centre=1.40, n=5):
    """D values that are ODD multiples of h, bracketing `centre`. The assertion in solve()
    requires (D/2)/h - 0.5 integral, i.e. D = (2m+1)h, so an even multiple is rejected --
    1.40 itself is NOT usable at h=0.02 (70h, even) and the bracket must straddle it."""
    m = int(round(centre/h))
    if m % 2 == 0: m += 1                      # force odd
    step = 2                                    # consecutive odd multiples differ by 2h
    ms = [m + step*k for k in range(-(n//2), n//2 + 1)]
    return [round(mm*h, 10) for mm in ms if mm > 0]

print("H2 BOND vs MESH, axisymmetric, STRICT (Neumann) interface, eps = SOFT*h, SOFT = %.4f" % SOFT)
print("exact: D = 1.40028,  k = 0.3314 Ha/bohr^2\n")
res = []
for h in (0.05, 0.04, 0.03, 0.025, 0.02):
    Ds = odd_bracket(h, 1.40, 5)
    # widen the bracket at coarse h, where the minimum sits well above 1.40
    if h >= 0.04: Ds = odd_bracket(h, 1.42, 5)
    try:
        report(h, 8.0, Ds)
    except Exception as e:
        print("   h=%.3f FAILED: %s" % (h, e)); continue
print("\nDone. Compare the D_min sequence against mol_fast's weak-interface extrapolation (1.314).")
