#!/bin/bash
R=/Users/cgjoh/Development/H2O/runs_cc; D=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/410447df-ea95-4b1a-874b-087cf46a7616/scratchpad/dumps
n=fa1
$R/run.sh "mode=formamide&box=14&h=0.1&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset&dump=$n" $R/$n.log 30000
echo "== $n solver $(grep -E 'CONVERGED [a-z]' $R/$n.log | grep -o 'E=[-0-9.]*' | head -1) $(grep -ci error $R/$n.log) err"
echo "   dipole $(grep -o 'mu_x=[^ ]* mu_y=[^ ]* mu_z=[^ ]*' $R/$n.log | tail -1)"
python3 $R/forces.py $R/$n.log 2>&1 | tail -1
(cd /Users/cgjoh/Development/H2O && python3 -W ignore tools/check_energy.py $n $D 2>&1 | grep -E "E indep" | cut -c1-90)
echo FADONE
