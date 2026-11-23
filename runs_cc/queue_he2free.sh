#!/bin/bash
# He2 in 3D (molecule.js): each He a +2 kernel with two one-electron halves (x2, ZX=2, bb=1,1, rcX=0), walls RELEASED
# at step 1500 so the split boundary moves and bends freely (density continuity). Start: plane through the nucleus
# (bbshift 0) or 0.4 away from the partner. beta 0, h 0.1, box 14.
R=/Users/cgjoh/Development/H2O/runs_cc; D=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/410447df-ea95-4b1a-874b-087cf46a7616/scratchpad/dumps
C="mode=x2&ZX=2&bb=1,1&rcX=0&el=He&box=14&h=0.1&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset&relbb=1&rel=1500"
for S in 0 0.4; do for RR in 3.0 4.0 6.0 8.0; do n=he2f_s${S}_R$RR
  $R/run.sh "$C&R=$RR$( [ $S != 0 ] && echo "&bbshift=$S" )&dump=$n" $R/$n.log 30000
  echo "== $n solver $(grep -E 'CONVERGED [a-z]' $R/$n.log | grep -o 'E=[-0-9.]*' | head -1) $(grep -c 'SPLIT RELEASED' $R/$n.log) released $(grep -ci error $R/$n.log) err"
  (cd /Users/cgjoh/Development/H2O && python3 -W ignore tools/check_energy.py $n $D 2>&1 | grep -E "E indep" | cut -c1-90)
done; done
echo HE2FDONE
