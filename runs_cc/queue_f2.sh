#!/bin/bash
# F2 with Pauli walls (2026-11-25), after queue_betasc.sh. F +7 = bond electron + three 2hom lone pairs (frag=f2/fatom).
# 1) r_c: F atom at rcF 0.55/0.60/0.65, linear interpolation to -(IE1..7) = -24.2132 Ha (NIST IEs FROM MEMORY, verify).
# 2) F2 at RFF 2.3/2.5/2.668/2.9/3.2 with Pauli walls (betainter=1, betasc=1000, bond pair free via USER_PAIR_ID),
#    and BASIC (no walls) at 2.3/2.668 for comparison. D = 2E(F) - E(F2) from solver energies, same box and h.
R=/Users/cgjoh/Development/H2O/runs_cc
while ! grep -q SCDONE /private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/tasks/b0hrp7qv4.output 2>/dev/null; do sleep 60; done
C="h=0.1&box=12&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset"
PW="beta=1&betainter=1&betaiso=1&betasc=1000"
E() { grep -E 'CONVERGED [a-z]' $1 | grep -o 'E=[-0-9.]*' | head -1 | cut -c3-; }
for rc in 0.55 0.60 0.65; do $R/run.sh "mode=frag&frag=fatom&rcF=$rc&$C" $R/f_atom_rc$rc.log 30000; echo "fatom rc $rc E $(E $R/f_atom_rc$rc.log)"; done
RC=$(python3 -c "
import sys; t=-24.2132; d=[(float(r),float(e)) for r,e in [l.split() for l in '''$(for rc in 0.55 0.60 0.65; do echo "$rc $(E $R/f_atom_rc$rc.log)"; done)'''.splitlines()] if e]
for (a,ea),(b,eb) in zip(d,d[1:]):
    if (ea-t)*(eb-t)<=0: print('%.3f'%(a+(t-ea)*(b-a)/(eb-ea))); sys.exit()
print('%.3f'%min(d,key=lambda x:abs(x[1]-t))[0])")
echo "rcF fitted $RC"
$R/run.sh "mode=frag&frag=fatom&rcF=$RC&$C" $R/f_atom_rc$RC.fit.log 30000; echo "fatom rc $RC E $(E $R/f_atom_rc$RC.fit.log)"
for RF in 2.3 2.5 2.668 2.9 3.2; do N=f2_pw_R$RF; $R/run.sh "mode=frag&frag=f2&RFF=$RF&rcF=$RC&$PW&$C" $R/$N.log 30000
  echo "$N E $(E $R/$N.log)  $(grep -o 'BETASC.*' $R/$N.log | tail -1 | cut -c1-200)"; done
for RF in 2.3 2.668; do N=f2_basic_R$RF; $R/run.sh "mode=frag&frag=f2&RFF=$RF&rcF=$RC&$C" $R/$N.log 30000; echo "$N E $(E $R/$N.log)"; done
echo F2DONE
