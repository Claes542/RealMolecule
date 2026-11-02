"""IS THE DEEP CLAMPED DIMER A BOND OR A COLLAPSE?  And is r_c even resolved?

tools/x2_depth_ladder.py put the halogen minima at the LOWER EDGE of its scan window (its own edge
flag caught it), so those rows are not bonds.  Two defects have to be separated:

  (1) THE WINDOW.  The window started at 0.45 R_e.  If E keeps falling below that, the dimer is
      CONTRACTING, not bonding -- the clamp admits density inside r_c, so two EQUIVALENT domains
      across the midplane meet no Pauli repulsion between their cores (the mechanism already on
      record as "every RealQM crystal contracts").  Only V_KK = Z^2/D opposes it, and the clamped
      attraction saturates at -2Z/r_c as D -> 0, so a minimum must exist somewhere -- the question
      is whether it is anywhere near R_e or down at contact.  Scanned here from 3h up.

  (2) THE MESH.  Cl's fitted r_c = 0.276 at h = 0.175 is 1.6 CELLS.  A plateau narrower than two
      cells is not represented, and the two codes show it: axisym 0.27587 vs radial 0.23044, 20%
      apart, where Li (r_c = 3.0, 17 cells) agreed to 0.04%.  So each case is run on an h ladder,
      with r_c REFITTED at each h, and cells-per-r_c reported.

Both must be clean before any statement about the clamp at depth.  Shallow Li2 is carried along as
the control: it is resolved and had an interior minimum, so it must stay put.

usage:  python3 tools/x2_contract_check.py
"""
import sys, time
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from x2_clamp import atom_clamp, x2_clamp, fit_rc
HA = 27.211386

CASES = [("Li2", 5.391715, 5.051, 1.05),
         ("I2",  10.45126, 5.038, 1.56),
         ("Cl2", 12.96763, 3.756, 2.51)]
L = 14.0

print("DEEP CLAMPED DIMERS: full D window from 3h, with r_c refitted on an h ladder.  L=%.0f" % L)
print("cells = r_c/h, how well the plateau is resolved.  EDGE-LO means E still falling at contact")
print("-- a CONTRACTION, not a bond.  Li2 is the resolved control.\n")
print("%5s %7s %9s %6s %7s | %9s %8s %9s | %7s %8s"
      % ("sys", "h", "r_c", "cells", "1/r_c", "E(X)", "D_min", "E_b eV", "R_e exp", "verdict"))
# COARSE MESH FOR ALL THREE FIRST: the I2/Cl2 coarse rows are what decide edge-vs-interior and
# cost ~15 s each, while a fine-mesh row costs tens of minutes.  Do not block the answer behind a
# control.
for h in (0.175, 0.0875):
    for name, ie, Re, De in CASES:
        t0 = time.time()
        rc = fit_rc(ie, h)
        eX = atom_clamp(rc, L, L, h)
        allD = [(2 * i + 1) * h for i in range(1, 400)]
        allD = [D for D in allD if 3 * h <= D <= min(2 * L - 6, 2.4 * Re)]
        vals = {}
        step = max(1, len(allD) // 18)
        for k in range(0, len(allD), step):
            vals[k] = x2_clamp(rc, allD[k], L, L, h)
        kb = min(vals, key=vals.get)
        for k in range(max(0, kb - step), min(len(allD), kb + step + 1)):
            if k not in vals:
                vals[k] = x2_clamp(rc, allD[k], L, L, h)
        kb = min(vals, key=vals.get)
        D, E2 = allD[kb], vals[kb]
        eb = (2 * eX - E2) * HA
        verdict = ("EDGE-LO" if kb == 0 else "EDGE-HI" if kb == len(allD) - 1 else
                   "interior")
        print("%5s %7.4f %9.5f %6.1f %7.2f | %9.5f %8.3f %9.3f | %7.3f %8s  [%.0fs]"
              % (name, h, rc, rc / h, 1.0 / rc, eX, D, eb, Re, verdict, time.time() - t0),
              flush=True)
