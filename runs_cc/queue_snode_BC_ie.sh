#!/bin/bash
# B and C+ with the node radius fitted to the IE (frozen-core atom over-bound ~6% with the HF node). Waits for PID 6422.
SP=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/scratchpad
cd /Users/cgjoh/Development/H2O/runs_cc
while kill -0 6422 2>/dev/null; do sleep 15; done
python3 $SP/xh_s.py B 0.05 fit 0.40 0.70 2>/dev/null > b_node_ie_fit_h005.out
RNODE=$(grep 'rnode\*' b_node_ie_fit_h005.out | awk '{print $2}') python3 $SP/xh_s.py B 0.05 2.15,2.3,2.45,2.6 2>/dev/null > bh_snode_ie_h005.out
python3 $SP/xh_s.py C 0.05 fit 0.33 0.60 2>/dev/null > c_node_ie_fit_h005.out
RNODE=$(grep 'rnode\*' c_node_ie_fit_h005.out | awk '{print $2}') python3 $SP/xh_s.py C 0.05 1.9,2.05,2.2,2.35 2>/dev/null > chp_snode_ie_h005.out
echo BCIEDONE
