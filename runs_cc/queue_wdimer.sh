#!/bin/bash
# Water dimer (2026-11-25), h2_beta mode=frag frag=wdimer (O +6 rc 0.71, four 2hom pairs, bare protons), h0.1 box 18.
# Basic (beta 0) vs Pauli walls between the two waters (beta 1.1 ~ sqrt(-2 eps) for water's pairs -- HF orbital
# energies -0.5..-0.7, borrowed: molecule.js has no per-domain eps yet; ?betainter=1, ?betaiso=1, wall force on).
# Reference: monomer at the same box/h/offset. E indep + E_surf(inter). Ref D_e ~5.0 kcal/mol at R_OO ~5.50 (from memory).
while ! grep -q FXREDONE /private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/tasks/bglo831bf.output 2>/dev/null; do sleep 60; done
R=/Users/cgjoh/Development/H2O/runs_cc; D=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/scratchpad/dumps
C="box=18&h=0.1&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset"
run() { $R/run.sh "$1&$C&dump=$2" $R/$2.log 30000
  echo "== $2 solver $(grep -E 'CONVERGED [a-z]' $R/$2.log | grep -o 'E=[-0-9.]*' | head -1) $(grep -ci error $R/$2.log) err"
  echo "   $(grep -o 'frag [a-z]* built[^"]*' $R/$2.log | cut -c1-160)"
  echo "   $(grep FORCE $R/$2.log | tail -1 | grep -o 'O[^ ]*@([^)]*) F=([^)]*)' | tr '\n' ' ')"
  (cd /Users/cgjoh/Development/H2O && python3 -W ignore tools/check_energy.py $2 $D 2>&1 | grep -E "E indep" | cut -c1-100)
  [ -n "$3" ] && python3 $R/surf.py $2 $D $3 inter; }
run "mode=frag&frag=water" wd_mono
for B in 0 1.1; do for RO in 5.0 5.3 5.5 5.8 6.2 7.0; do
  if [ $B = 0 ]; then run "mode=frag&frag=wdimer&ROO=$RO" wd_b0_R$RO; else run "mode=frag&frag=wdimer&ROO=$RO&beta=$B&betainter=1&betaiso=1" wd_b${B}_R$RO $B; fi
done; done
echo WDDONE
