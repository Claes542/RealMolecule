#!/bin/bash
# after queue_axi.sh: linear CH2 (C +4, ring 1+1, cones) and the shift-invert recheck of axisym_linear ch2_4. 2026-11-26
R=/Users/cgjoh/Development/H2O/runs_cc; cd $R
until grep -q AXIDONE /private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/tasks/bc4ezz9lu.output 2>/dev/null; do sleep 60; done
P="python3 -W ignore"
$P h2o_linear_axisym.py 0.1 1.0 1.8,1.9,2.0,2.1,2.2 ch2 > ch2_lin_h0.1_free.out 2>&1
$P h2o_linear_axisym.py 0.1 1.0 1.8,1.9,2.0,2.1,2.2 ch2 ownbeta > ch2_lin_h0.1_ownbeta.out 2>&1
cd /Users/cgjoh/Development/H2O
for RR in 2.0372 2.1594; do for SI in 1 0; do
  echo "ch2_4 R $RR SHIFT_INVERT=$SI: $(AXI_SHIFT_INVERT=$SI $P tools/axisym_linear.py ch2_4 R=$RR h=0.1 2>&1 | grep -E '^ *E|E =|E=' | tail -1 | cut -c1-150)"; done; done > $R/ch2_4_shiftinvert_check.out 2>&1
echo AXI2DONE
