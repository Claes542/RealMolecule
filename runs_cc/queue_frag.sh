#!/bin/bash
# Protein-surface polar groups, RealQM dipoles (h2_beta mode=frag), beta 0, h 0.1, box 16 (ions).
R=/Users/cgjoh/Development/H2O/runs_cc; D=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/410447df-ea95-4b1a-874b-087cf46a7616/scratchpad/dumps
for F in methanol methylammonium acetate; do n=fr_$F
  $R/run.sh "mode=frag&frag=$F&box=16&h=0.1&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset&dump=$n" $R/$n.log 30000
  echo "== $n solver $(grep -E 'CONVERGED [a-z]' $R/$n.log | grep -o 'E=[-0-9.]*' | head -1) $(grep -ci error $R/$n.log) err"
  echo "   $(grep -o 'frag [a-z]* built[^"]*' $R/$n.log | cut -c1-120)"
  echo "   dipole $(grep -o 'mu_x=[^ ]* mu_y=[^ ]* mu_z=[^ ]*' $R/$n.log | tail -1)"
done
echo FRAGDONE
