#!/bin/bash
# GEOMETRY SET, step 1: force on each H at the EXPERIMENTAL geometry, first-row hydrides + C2 hydrocarbons, beta 0,
# h 0.1, the settled construction per molecule (2hom X-H pairs over bare protons). Independent energy for each.
R=/Users/cgjoh/Development/H2O/runs_cc; D=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/410447df-ea95-4b1a-874b-087cf46a7616/scratchpad/dumps
C="h=0.1&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset"
run() { $R/run.sh "$1&$C&dump=$2" $R/$2.log 30000
  echo "== $2 solver $(grep -E 'CONVERGED [a-z]' $R/$2.log | grep -o 'E=[-0-9.]*' | head -1) $(grep -ci error $R/$2.log) err"
  python3 $R/forces.py $R/$2.log 2>&1 | tail -1
  (cd /Users/cgjoh/Development/H2O && python3 -W ignore tools/check_energy.py $2 $D 2>&1 | grep -E "E indep" | cut -c1-90); }
run "mode=ch4&rcX=1.0&el=C&Hbare=1&bpair=1&box=12&R=2.054" g1_ch4
run "mode=xh&anchor=1&ZX=3&bb=2,1&rcX=1.08&el=B&Hbare=1&bpair=1&box=12&R=2.329" g1_bh
run "mode=xh2&ZX=2&bb=1,1&rcX=1.37&el=Be&Hbare=1&bpair=1&box=12&R=2.52" g1_beh2
run "mode=xh&anchor=1&ZX=2&ZnX=1&rcX=1.99&el=Li&Hbare=1&box=20&R=3.015" g1_lih
run "mode=nh3&rcX=0.81&el=N&bpair=1&Hbare=1&box=12&R=1.913" g1_nh3_n3
run "mode=nh3&n5=1&rcX=0.82&el=N&bpair=1&Hbare=1&box=12&R=1.913" g1_nh3_n5
run "mode=h2o&c2v=1&half=1&rcX=0.71&el=O&bpair=1&Hbare=1&box=12&R=1.809" g1_h2o
run "mode=c2h6&rcX=1.0&el=C&Hbare=1&bpair=1&box=14" g1_c2h6
run "mode=c2h4&rcX=1.0&el=C&Hbare=1&bpair=1&box=14" g1_c2h4
run "mode=c2h2&rcX=1.0&el=C&Hbare=1&bpair=1&box=14" g1_c2h2
echo G1DONE
