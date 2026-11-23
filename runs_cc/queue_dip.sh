#!/bin/bash
# DIPOLE SET additions (after the geometry queue): H2O O+3 wedges (r_c 0.65, the right-sign variant), HF pseudo-F +3
# (2hom back + 1 bond, r_c 0.49 from the X2 series) at 1.733, OH O+3 plane (r_c 0.59) at 1.832. beta 0, h 0.1.
R=/Users/cgjoh/Development/H2O/runs_cc; D=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/410447df-ea95-4b1a-874b-087cf46a7616/scratchpad/dumps
until grep -q G1DONE $R/geom1.out 2>/dev/null; do sleep 60; done
C="h=0.1&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset"
run() { $R/run.sh "$1&$C&dump=$2" $R/$2.log 30000
  echo "== $2 solver $(grep -E 'CONVERGED [a-z]' $R/$2.log | grep -o 'E=[-0-9.]*' | head -1) $(grep -ci error $R/$2.log) err"
  echo "   dipole $(grep -o 'mu_x=[^ ]* mu_y=[^ ]* mu_z=[^ ]*' $R/$2.log | tail -1)"
  python3 $R/forces.py $R/$2.log 2>&1 | tail -1
  (cd /Users/cgjoh/Development/H2O && python3 -W ignore tools/check_energy.py $2 $D 2>&1 | grep -E "E indep" | cut -c1-90); }
run "mode=h2o&o3=1&rcX=0.65&el=O&bpair=1&Hbare=1&box=12&R=1.809" d_h2o_o3
run "mode=xh&anchor=1&ZX=3&bb=2,1&rcX=0.49&el=F&Hbare=1&bpair=1&box=12&R=1.733" d_hf
run "mode=xh&anchor=1&ZX=3&bb=2,1&rcX=0.59&el=O&Hbare=1&bpair=1&box=12&R=1.832" d_oh
echo DIPDONE
