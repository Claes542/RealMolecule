#!/bin/bash
# Bare C2 with each carbon ONE 4hom +4 domain, no H = "H2/Li2 scaled to +4" (user's picture). beta 0, r_c 1.0, h 0.1.
R=/Users/cgjoh/Development/H2O/runs_cc; D=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/410447df-ea95-4b1a-874b-087cf46a7616/scratchpad/dumps
until grep -q H4DONE $R/h4.out 2>/dev/null; do sleep 60; done
for CC in 2.901 2.50 3.30; do n=c2h4hom_CC$CC
  $R/run.sh "mode=c2h6&c4hom=1&noH=1&box=14&CC=$CC&rcX=1.0&el=C&h=0.1&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset&dump=$n" $R/$n.log 30000
  echo "== $n solver $(grep -E 'CONVERGED [a-z]' $R/$n.log | grep -o 'E=[-0-9.]*' | head -1) $(grep -ci error $R/$n.log) err"
  (cd /Users/cgjoh/Development/H2O && python3 -W ignore tools/check_energy.py $n $D 2>&1 | grep -E "E indep" | cut -c1-100)
done
echo C2H4HOMDONE
