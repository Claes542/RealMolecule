#!/bin/bash
# DIAGNOSTIC: ethane with each carbon ONE 4hom +4 domain (unsplit valence), H's keep own electrons. beta 0, r_c 1.0,
# h 0.1. 2.901 done earlier: E indep -16.35362. Waits for the bare-C2 queue.
R=/Users/cgjoh/Development/H2O/runs_cc; D=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/410447df-ea95-4b1a-874b-087cf46a7616/scratchpad/dumps
until grep -q C2BAREDONE $R/c2bare.out 2>/dev/null; do sleep 60; done
for CC in 2.50 3.30 2.10; do n=h4_c2h6_CC$CC
  rm -f $R/$n.log
  $R/run.sh "mode=c2h6&c4hom=1&box=14&CC=$CC&rcX=1.0&el=C&h=0.1&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset&dump=$n" $R/$n.log 30000
  echo "== $n solver $(grep -E 'CONVERGED [a-z]' $R/$n.log | grep -o 'E=[-0-9.]*' | head -1) $(grep -ci error $R/$n.log) err"
  (cd /Users/cgjoh/Development/H2O && python3 -W ignore tools/check_energy.py $n $D 2>&1 | grep -E "E indep" | cut -c1-100)
done
echo H4DONE
