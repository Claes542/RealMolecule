#!/bin/bash
# ONE sequential queue (no pgrep waits: they self-matched and deadlocked 2026-11-24). s-node fine-mesh runs + node-by-IE + He2 local.
SP=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/scratchpad
cd /Users/cgjoh/Development/H2O/runs_cc
python3 $SP/xh_s.py B 0.05 2.0,2.15,2.3,2.45 2>/dev/null > bh_snode_h005.out
python3 $SP/xh_s.py C 0.05 1.9,2.05,2.2,2.35 2>/dev/null > chp_snode_h005.out
python3 $SP/lih_s.py 0.05 fit 0.70 1.00 2>/dev/null > li_node_ie_fit_h005.out
RNODE=$(grep 'rnode\*' li_node_ie_fit_h005.out | awk '{print $2}') python3 $SP/lih_s.py 0.05 s 2.8,3.0,3.2,3.4 2>/dev/null > lih_snode_ie_h005.out
python3 $SP/beh2_s.py 0.05 fit 0.45 0.75 2>/dev/null > be_node_ie_fit_h005.out
RNODE=$(grep 'rnode\*' be_node_ie_fit_h005.out | awk '{print $2}') python3 $SP/beh2_s.py 0.05 s 2.35,2.5,2.65,2.8 2>/dev/null > beh2_snode_ie_h005.out
(python3 he2_dir.py 0.1 local 2,3,4,6,10; python3 he2_dir.py 0.05 local 2,3,4,10) 2>/dev/null > he2_local.out
echo SNODEALLDONE
