#!/bin/bash
# Self-contained s-node protocol (2026-11-24): RealQM's own 2hom 1s^2 core (realcore.py, = RHF He-like ion, checked
# He -2.86168, Li+ -7.23641), node radius fitted to the atom's measured IE, FIXED mesh h 0.05. Compare with experiment only.
export CORE=real FIXH=1; cd /private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/scratchpad; O=/Users/cgjoh/Development/H2O/runs_cc/sc
python3 lih_s.py 0.05 fit 0.70 1.00 2>/dev/null > ${O}_li_fit.out; RN=$(grep 'rnode\*' ${O}_li_fit.out | awk '{print $2}')
RNODE=$RN python3 lih_s.py 0.05 s 2.8,2.9,3.0,3.1,3.2 2>/dev/null > ${O}_lih.out
python3 beh2_s.py 0.05 fit 0.45 0.75 2>/dev/null > ${O}_be_fit.out; RN=$(grep 'rnode\*' ${O}_be_fit.out | awk '{print $2}')
RNODE=$RN python3 beh2_s.py 0.05 s 2.35,2.45,2.55,2.65 2>/dev/null > ${O}_beh2.out
python3 xh_s.py B 0.05 fit 0.40 0.60 2>/dev/null > ${O}_b_fit.out; RN=$(grep 'rnode\*' ${O}_b_fit.out | awk '{print $2}')
RNODE=$RN python3 xh_s.py B 0.05 2.15,2.25,2.35,2.45 2>/dev/null > ${O}_bh.out
python3 xh_s.py C 0.05 fit 0.36 0.46 2>/dev/null > ${O}_c_fit.out; RN=$(grep 'rnode\*' ${O}_c_fit.out | awk '{print $2}')
RNODE=$RN python3 xh_s.py C 0.05 2.0,2.1,2.2,2.3 2>/dev/null > ${O}_chp.out
echo SCDONE
