#!/bin/bash
# STIFFNESS SET: forces at displaced geometries (beta 0, h 0.1), reference runs g1_ch4 / g1_h2o exist.
R=/Users/cgjoh/Development/H2O/runs_cc
C="h=0.1&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset"
run() { $R/run.sh "$1&$C&dump=$2" $R/$2.log 30000; echo "== $2"; python3 $R/forces.py $R/$2.log 2>&1 | tail -2; }
run "mode=ch4&rcX=1.0&el=C&Hbare=1&bpair=1&box=12&R=1.954" fq_ch4_R1.954
run "mode=ch4&rcX=1.0&el=C&Hbare=1&bpair=1&box=12&R=2.154" fq_ch4_R2.154
H="mode=h2o&c2v=1&half=1&rcX=0.71&el=O&bpair=1&Hbare=1&box=12"
run "$H&R=1.709" fq_h2o_R1.709
run "$H&R=1.909" fq_h2o_R1.909
run "$H&R=1.809&ang=99.5" fq_h2o_a99.5
run "$H&R=1.809&ang=109.5" fq_h2o_a109.5
echo FQDONE
