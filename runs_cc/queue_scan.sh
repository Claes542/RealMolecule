#!/bin/bash
# GEOMETRY SET, step 2: H forces at R-0.1 and R-0.2 (experimental R already run in queue_geom1 / dip / freq) to
# bracket the force zero = the model's bond length. beta 0, h 0.1, same constructions as step 1.
R=/Users/cgjoh/Development/H2O/runs_cc
C="h=0.1&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset"
run() { $R/run.sh "$1&$C&dump=$2" $R/$2.log 30000; echo "== $2 $(python3 $R/forces.py $R/$2.log 2>&1 | head -1)"; }
sc() { local base="$1" tag="$2"; shift 2; for r in "$@"; do run "$base&R=$r" sc_${tag}_R$r; done; }
sc "mode=xh2&ZX=2&bb=1,1&rcX=1.37&el=Be&Hbare=1&bpair=1&box=12" beh2 2.42 2.32
sc "mode=xh&anchor=1&ZX=2&ZnX=1&rcX=1.99&el=Li&Hbare=1&box=20" lih 2.915 2.815
sc "mode=xh&anchor=1&ZX=3&bb=2,1&rcX=1.08&el=B&Hbare=1&bpair=1&box=12" bh 2.229 2.129
sc "mode=c2h2&rcX=1.0&el=C&Hbare=1&bpair=1&box=14" c2h2 1.904 1.804
sc "mode=c2h4&rcX=1.0&el=C&Hbare=1&bpair=1&box=14" c2h4 1.954 1.854
sc "mode=c2h6&rcX=1.0&el=C&Hbare=1&bpair=1&box=14" c2h6 1.967 1.867
sc "mode=nh3&rcX=0.81&el=N&bpair=1&Hbare=1&box=12" nh3n3 1.813 1.713
sc "mode=nh3&n5=1&rcX=0.82&el=N&bpair=1&Hbare=1&box=12" nh3n5 1.813 1.713
sc "mode=h2o&c2v=1&half=1&rcX=0.71&el=O&bpair=1&Hbare=1&box=12" h2o6 1.609
sc "mode=xh&anchor=1&ZX=3&bb=2,1&rcX=0.49&el=F&Hbare=1&bpair=1&box=12" hf 1.633 1.533
sc "mode=xh&anchor=1&ZX=3&bb=2,1&rcX=0.59&el=O&Hbare=1&bpair=1&box=12" oh 1.732 1.632
echo SCANDONE
