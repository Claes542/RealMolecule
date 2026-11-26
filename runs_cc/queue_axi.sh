#!/bin/bash
# axisymmetric O +6 cases, sequential (one scipy job at a time), 2026-11-26
R=/Users/cgjoh/Development/H2O/runs_cc; cd $R
P="python3 -W ignore"
$P h2o_linear_axisym.py 0.1 0.71 1.6,1.7,1.8,1.9,2.0,2.1 > hoh_lin_h0.1_free.out 2>&1
$P h2o_linear_axisym.py 0.1 0.71 1.6,1.7,1.8,1.9,2.0,2.1 ownbeta > hoh_lin_h0.1_ownbeta.out 2>&1
$P o2_axisym.py scan 0.1 0.71 2.0,2.2,2.4,2.6,2.8,6.0 > o2_axi_h0.1_triplet.out 2>&1
$P o2_axisym.py mol 0.1 0.71 2.2,2.4 singlet > o2_axi_h0.1_singlet.out 2>&1
$P o2_axisym.py mol 0.1 0.71 2.2,2.4 nowall > o2_axi_h0.1_nowall.out 2>&1
$P h2o_linear_axisym.py 0.05 0.71 1.7,1.8,1.9,2.0 > hoh_lin_h0.05_free.out 2>&1
echo AXIDONE
