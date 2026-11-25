#!/bin/bash
while ! grep -q BTDONE /private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/tasks/b3tjhgo0l.output 2>/dev/null; do sleep 30; done
cd /Users/cgjoh/Development/H2O/runs_cc
python3 -W ignore h2_force_profile.py 0.1 > h2_force_profile_h01.out 2>&1
python3 -W ignore h2_force_profile.py 0.05 > h2_force_profile_h005.out 2>&1
echo HFDONE
