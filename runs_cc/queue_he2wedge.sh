#!/bin/bash
# He2 in 3D, each He split into two one-electron halves held FIXED (no release): axial split (plane through the nucleus
# perpendicular to the bond) vs wedge split (plane y=0 containing the bond). Free wall between the atoms.
R=/Users/cgjoh/Development/H2O/runs_cc; D=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/410447df-ea95-4b1a-874b-087cf46a7616/scratchpad/dumps
C="mode=x2&ZX=2&bb=1,1&rcX=0&el=He&box=14&h=0.1&inner=1&win=4000&conv=0.00002&blog=120&nmax=300&reset"
for AX in y x; do for RR in 3.0 4.0 6.0 8.0; do n=he2w_${AX}_R$RR
  $R/run.sh "$C&R=$RR$( [ $AX = y ] && echo '&bbax=y' )&dump=$n" $R/$n.log 30000
  echo "== $n solver $(grep -E 'CONVERGED [a-z(]' $R/$n.log | grep -o 'E=[-0-9.]*' | head -1) $(grep -o 'step=[0-9]*' $R/$n.log | tail -1)"
  (cd /Users/cgjoh/Development/H2O && python3 -W ignore tools/check_energy.py $n $D 2>&1 | grep -E "E indep" | cut -c1-90)
done; done
echo HE2WDONE
