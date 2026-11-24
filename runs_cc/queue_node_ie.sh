#!/bin/bash
# s-NODE with the node radius fitted to the atom's IE (frozen-core potential, Clementi zeta), replacing the HF node radius.
# Waits for the BH / CH+ h0.05 jobs (one scipy job at a time), then: fit r_node, scan LiH and BeH2 at h 0.05.
SP=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/scratchpad
R=/Users/cgjoh/Development/H2O/runs_cc
while pgrep -f "xh_s.py" >/dev/null; do sleep 10; done
python3 $SP/lih_s.py 0.05 fit 0.70 1.00 2>/dev/null > $R/li_node_ie_fit_h005.out
RL=$(grep 'rnode\*' $R/li_node_ie_fit_h005.out | awk '{print $2}')
RNODE=$RL python3 $SP/lih_s.py 0.05 s 2.8,3.0,3.2,3.4 2>/dev/null > $R/lih_snode_ie_h005.out
python3 $SP/beh2_s.py 0.05 fit 0.45 0.75 2>/dev/null > $R/be_node_ie_fit_h005.out
RB=$(grep 'rnode\*' $R/be_node_ie_fit_h005.out | awk '{print $2}')
RNODE=$RB python3 $SP/beh2_s.py 0.05 s 2.35,2.5,2.65,2.8 2>/dev/null > $R/beh2_snode_ie_h005.out
echo NODEIEDONE
