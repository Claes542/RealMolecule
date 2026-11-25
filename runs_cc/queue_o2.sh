#!/bin/bash
# O2 with Pauli walls (2026-11-25), after the F2 queue (waits for F2DONE). O +6 = sigma electron + unpaired electron +
# two 2hom lone pairs (frag=o2/oatom). r_c fitted to -(IE1..6) of O = -15.9165 Ha (IEs FROM MEMORY: 13.618 35.121
# 54.936 77.414 113.899 138.120 eV, verify) using E INDEPENDENT (check_energy), since the solver V_ee runs high.
R=/Users/cgjoh/Development/H2O/runs_cc; H=/Users/cgjoh/Development/H2O
FQ=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/tasks/bo553yt1r.output
DD=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/scratchpad/dumps
until grep -q F2DONE $FQ 2>/dev/null; do sleep 120; done
C="h=0.1&box=12&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset"
PW="beta=1&betainter=1&betaiso=1&betasc=1000"
EI() { (cd $H && python3 -W ignore tools/check_energy.py $1 $DD 2>&1 | grep 'E independent' | grep -oE '\) -?[0-9.]+' | head -1 | cut -c3-); }
L=""
for rc in 0.65 0.71 0.77; do $R/run.sh "mode=frag&frag=oatom&rcO=$rc&$C&dump=o_atom_rc$rc" $R/o_atom_rc$rc.log 30000
  e=$(EI o_atom_rc$rc); echo "oatom rc $rc E indep $e"; L="$L$rc $e
"; done
RC=$(python3 -c "
import sys; t=-15.9165; d=[(float(a),float(b)) for a,b in [l.split() for l in '''$L'''.strip().splitlines()] if b]
for (a,ea),(b,eb) in zip(d,d[1:]):
    if (ea-t)*(eb-t)<=0: print('%.3f'%(a+(t-ea)*(b-a)/(eb-ea))); sys.exit()
print('%.3f'%min(d,key=lambda x:abs(x[1]-t))[0]); print('rc fit OUT OF RANGE', file=sys.stderr)")
echo "rcO fitted $RC"
$R/run.sh "mode=frag&frag=oatom&rcO=$RC&$C&dump=o_atom_fit" $R/o_atom_rc$RC.fit.log 30000; echo "oatom fit E indep $(EI o_atom_fit)"
for RO in 2.0 2.15 2.282 2.45 2.7; do N=o2_pw_R$RO; $R/run.sh "mode=frag&frag=o2&RO2=$RO&rcO=$RC&$PW&$C&dump=$N" $R/$N.log 30000
  echo "$N E indep $(EI $N)  $(grep -o 'BETASC.*' $R/$N.log | tail -1 | cut -c1-160)"; done
N=o2_singlet_R2.282; $R/run.sh "mode=frag&frag=o2&RO2=2.282&rcO=$RC&o2singlet=1&$PW&$C&dump=$N" $R/$N.log 30000; echo "$N E indep $(EI $N)"
N=o2_basic_R2.282; $R/run.sh "mode=frag&frag=o2&RO2=2.282&rcO=$RC&$C&dump=$N" $R/$N.log 30000; echo "$N E indep $(EI $N)"
echo O2DONE
