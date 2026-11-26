#!/bin/bash
# O2 with Pauli walls (2026-11-25), after the F2 queue (waits for F2DONE). O +6 = sigma electron + unpaired electron +
# two 2hom lone pairs (frag=o2/oatom). r_c fitted to -(IE1..6) of O = -15.9165 Ha (IEs FROM MEMORY: 13.618 35.121
# 54.936 77.414 113.899 138.120 eV, verify) using E INDEPENDENT (check_energy), since the solver V_ee runs high.
R=/Users/cgjoh/Development/H2O/runs_cc; H=/Users/cgjoh/Development/H2O
FQ=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/tasks/bo553yt1r.output
DD=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/scratchpad/dumps
# (2026-11-25: started directly; F2 controls without walls dropped, O2 first)
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
# large-R check FIRST: D(R=5) must be ~0, else the atom vs molecule comparison is inconsistent. O0 sits at the box
# centre, so R=5 needs box 16 (box 12 leaves O1 1 bohr from the edge), with its own atom reference in box 16.
C16=$(echo $C | sed 's/box=12/box=16/')
$R/run.sh "mode=frag&frag=oatom&rcO=$RC&$C16&dump=o_atom_fit_b16" $R/o_atom_rc$RC.fit_b16.log 30000
$R/run.sh "mode=frag&frag=o2&RO2=5.0&rcO=$RC&$PW&$C16&dump=o2_pw_R5.0_b16" $R/o2_pw_R5.0_b16.log 30000
python3 -c "a=float('$(EI o_atom_fit_b16)'); m=float('$(EI o2_pw_R5.0_b16)'); print('LARGE-R CHECK box16: E(O) %.5f E(O2,R5) %.5f D(5) %.4f' % (a, m, 2*a-m))"
# COMMENSURATE R only (2026-11-26): O1 must sit on the same sub-cell offset as O0, else its r_c sphere covers a different
# cell set (E(r_c) has steps: rc 0.71 -15.96105 vs 0.713 -15.72170 indep). R in multiples of h = 0.1; exp 2.282 -> 2.3.
for RO in 2.3 2.0 2.2 2.5 2.7; do N=o2_pw_R$RO; $R/run.sh "mode=frag&frag=o2&RO2=$RO&rcO=$RC&$PW&$C&dump=$N" $R/$N.log 30000
  echo "$N E indep $(EI $N)  $(grep -o 'BETASC.*' $R/$N.log | tail -1 | cut -c1-160)"; done
N=o2_singlet_R2.3; $R/run.sh "mode=frag&frag=o2&RO2=2.3&rcO=$RC&o2singlet=1&$PW&$C&dump=$N" $R/$N.log 30000; echo "$N E indep $(EI $N)"
echo "== triplet-singlet at 2.3 = E(T) - E(S)"
N=o2_basic_R2.3; $R/run.sh "mode=frag&frag=o2&RO2=2.3&rcO=$RC&$C&dump=$N" $R/$N.log 30000; echo "$N E indep $(EI $N)"
echo O2DONE
