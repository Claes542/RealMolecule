#!/bin/bash
# F2 with Pauli walls (2026-11-25), after queue_betasc.sh. F +7 = bond electron + three 2hom lone pairs (frag=f2/fatom).
# 1) r_c: F atom at rcF 0.55/0.60/0.65, linear interpolation to -(IE1..7) = -24.2132 Ha (NIST IEs FROM MEMORY, verify).
# 2) F2 at RFF 2.3/2.5/2.668/2.9/3.2 with Pauli walls (betainter=1, betasc=1000, bond pair free via USER_PAIR_ID),
#    and BASIC (no walls) at 2.3/2.668 for comparison. D = 2E(F) - E(F2) from solver energies, same box and h.
R=/Users/cgjoh/Development/H2O/runs_cc
# (started directly, in parallel with the water dimer, 2026-11-25)
C="h=0.1&box=12&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset"
PW="beta=1&betainter=1&betaiso=1&betasc=1000"
E() { grep -E 'CONVERGED [a-z]' $1 | grep -o 'E=[-0-9.]*' | head -1 | cut -c3-; }
for rc in 0.55 0.60 0.65; do $R/run.sh "mode=frag&frag=fatom&rcF=$rc&$C&dump=f_atom_rc$rc" $R/f_atom_rc$rc.log 30000; echo "fatom rc $rc E $(E $R/f_atom_rc$rc.log)"; done
# r_c fitted to E INDEPENDENT (check_energy), not the solver E: the solver V_ee is 0.31-0.34 too high in the F atom
# (rc 0.60: indep -24.92889, rc 0.65: -23.89697 -> 0.635; the solver-E fit gave 0.619, 0.35 Ha too deep). 2026-11-25
RC=0.635
echo "rcF fitted $RC"
$R/run.sh "mode=frag&frag=fatom&rcF=$RC&$C&dump=f_atom_fit" $R/f_atom_rc$RC.fit.log 30000; echo "fatom rc $RC E $(E $R/f_atom_rc$RC.fit.log)"
for RF in 2.3 2.5 2.668 2.9 3.2; do N=f2_pw_R$RF; $R/run.sh "mode=frag&frag=f2&RFF=$RF&rcF=$RC&$PW&$C&dump=$N" $R/$N.log 30000
  echo "$N E $(E $R/$N.log)  $(grep -o 'BETASC.*' $R/$N.log | tail -1 | cut -c1-200)"; done
for RF in 2.3 2.668; do N=f2_basic_R$RF; $R/run.sh "mode=frag&frag=f2&RFF=$RF&rcF=$RC&$C&dump=$N" $R/$N.log 30000; echo "$N E $(E $R/$N.log)"; done
DD=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/scratchpad/dumps
for N in f_atom_fit f2_pw_R2.3 f2_pw_R2.5 f2_pw_R2.668 f2_pw_R2.9 f2_pw_R3.2 f2_basic_R2.3 f2_basic_R2.668; do
  echo "$N $(cd /Users/cgjoh/Development/H2O && python3 -W ignore tools/check_energy.py $N $DD 2>&1 | grep 'E independent' | cut -c1-90)"; done
echo F2DONE
