"""DRIVER for tools/li_cell_hom.py -- the HOM2 spherical core, which has never been run.

WHY THIS IS THE INTERESTING VARIANT.  All-electron lithium with INDEPENDENT nested shells failed
twice: without H^1 continuity it collapsed (E = -40.26 Ha against -7.478, inner shell pinned at
0.128 a0, the collapse mode a jump at the interface that H^1 forbids); with H^1 it was cured and
stable (E = -8.753, interfaces 0.396 and ~0.897 at EVERY r_s from 1.3 to 4.6 -- so r_c is NOT a
necessity, the cutoff was standing in for a missing continuity condition) but left two defects:
the core/valence interface at 0.897 against the atom's 2.00, and totals 17% off against 1.7%.

That run had TWO interfaces, i.e. the 1s PAIR forced to different radii.  Physically 1s^2 is two
electrons in ONE spatial density.  li_cell_hom.py builds exactly that -- a single homogenised core
shell, ONE core-core pair and TWO core-valence pairs -- so it is the variant the recorded failure
points at, and the artificial radial splitting of the pair is the prime suspect for dragging that
interface inward by a factor 2.2.  It was committed as a library with no __main__ and has produced
no numbers.

ORDER OF BUSINESS, controls before results (the file names the first one itself):
  1. SELF-TEST: at large r_s the cell must reproduce the FREE ATOM's R = 2.00.  If hom2 lands near
     2.00 where nested shells gave 0.897, the pair-splitting WAS the defect.
  2. WFACE INDEPENDENCE: the straddling face belongs to neither region, so its weight is a free
     parameter and the answer "must not depend on it -- checked, not assumed" (the file's words).
  3. MESH LADDER in n at fixed L, because no energy difference here is quotable without one.
Only then the r_s scan.

usage:  python3 tools/li_cell_hom_run.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from li_cell_hom import solve, best_R

ATOM_R = 2.00      # free-atom interface, confirmed to 0.05% by two independent routes
LI_EXACT = -7.478  # lithium total

print(__doc__.split('usage:')[0])
print("=" * 78)
print("1. SELF-TEST -- large r_s must reproduce the free atom's R = 2.00")
print("   (independent nested shells gave 0.897 here; hom2 is the question)\n")
print("%8s %6s %12s %10s %10s %12s" % ("L (r_s)", "n", "E (Ha)", "R", "R/2.00", "constraint"))
for L in (12.0, 20.0, 30.0):
    t0 = time.time()
    S = best_R(L, 900, Rlo=0.8, Rhi=3.6)
    if S is None:
        print("%8.1f %6d   no admissible interface" % (L, 900)); continue
    print("%8.1f %6d %12.5f %10.3f %10.3f %12.1e  [%.0fs]"
          % (L, 900, S['E'], S['R'], S['R'] / ATOM_R, S['cons'], time.time() - t0))
    sys.stdout.flush()

print("\n" + "=" * 78)
print("2. WFACE INDEPENDENCE -- the straddling face is in neither region, so its weight")
print("   is a free parameter.  R and E must not move with it.\n")
print("%8s %12s %10s" % ("wface", "E (Ha)", "R"))
for wf in (1.0, 1.5, 2.0):
    S = best_R(20.0, 900, Rlo=0.8, Rhi=3.6, wface=wf)
    if S is None: print("%8.1f   none" % wf); continue
    print("%8.1f %12.5f %10.3f" % (wf, S['E'], S['R']))
    sys.stdout.flush()

print("\n" + "=" * 78)
print("3. MESH LADDER at L = 20 -- nothing is quotable without it\n")
print("%8s %12s %10s" % ("n", "E (Ha)", "R"))
for n in (600, 900, 1400, 2000):
    S = best_R(20.0, n, Rlo=0.8, Rhi=3.6)
    if S is None: print("%8d   none" % n); continue
    print("%8d %12.5f %10.3f" % (n, S['E'], S['R']))
    sys.stdout.flush()
print("\nreference: free-atom interface R = 2.00; lithium total %.3f Ha" % LI_EXACT)
