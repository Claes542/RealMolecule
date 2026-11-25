#!/bin/bash
# after the triplet ladder (one scipy job at a time)
while ! grep -q T3DONE /private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/tasks/bvi0m55y1.output 2>/dev/null; do sleep 30; done
cd /Users/cgjoh/Development/H2O/runs_cc
python3 -W ignore beta_transfer.py 0.1 > beta_transfer_h01.out 2>&1
python3 -W ignore beta_transfer.py 0.05 > beta_transfer_h005.out 2>&1
echo BTDONE
