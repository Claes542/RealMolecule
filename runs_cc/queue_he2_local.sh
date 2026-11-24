#!/bin/bash
# He2 with the LOCAL evanescent wall (beta = sqrt(2 max(V-eps,0)) per wall cell), h 0.1 and 0.05; waits for other scipy jobs.
cd /Users/cgjoh/Development/H2O/runs_cc
while pgrep -f "xh_s.py|lih_s.py|beh2_s.py|queue_node_ie" >/dev/null; do sleep 10; done
(python3 he2_dir.py 0.1 local 2,3,4,6,10; python3 he2_dir.py 0.05 local 2,3,4,10) 2>/dev/null > he2_local.out
echo HE2LOCALDONE
