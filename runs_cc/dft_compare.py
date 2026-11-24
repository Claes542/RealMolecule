"""Reference table for the Pauli-wall RealQM runs: HF, LDA (SVWN), PBE, B3LYP and CCSD(T) in cc-pVTZ for LiH, BeH2, BH, CH+.
R_e from a parabola through a 5-point scan (BeH2: symmetric stretch), D_e (atomisation) against free atoms
(open-shell atoms unrestricted; CH+ -> C+ + H).  Bohr / Hartree.   usage: python3 dft_compare.py > dft_compare.out"""
import numpy as np
from pyscf import gto, scf, dft, cc
B = 'cc-pvtz'
MOL = {'LiH': ('Li', 0, 3.015), 'BeH2': ('Be', 0, 2.507), 'BH': ('B', 0, 2.329), 'CH+': ('C', 1, 2.137)}
ATOM_SPIN = {'H': 1, 'Li': 1, 'Be': 0, 'B': 1, 'C': 1}          # C here is C+ (2P)
METH = ['HF', 'LDA', 'PBE', 'B3LYP', 'CCSD(T)']

def energy(atoms, charge, spin, meth):
    m = gto.M(atom=atoms, basis=B, charge=charge, spin=spin, unit='Bohr', verbose=0)
    if meth == 'HF' or meth == 'CCSD(T)':
        mf = (scf.RHF(m) if spin == 0 else scf.UHF(m)); mf.conv_tol = 1e-10; mf.kernel()
        if meth == 'HF': return mf.e_tot
        c = cc.CCSD(mf); c.kernel(); return c.e_tot + c.ccsd_t()
    mf = (dft.RKS(m) if spin == 0 else dft.UKS(m))
    mf.xc = {'LDA': 'lda,vwn', 'PBE': 'pbe', 'B3LYP': 'b3lyp'}[meth]; mf.conv_tol = 1e-10; mf.kernel(); return mf.e_tot

def geom(name, R):
    X = MOL[name][0]
    if name == 'BeH2': return f'Be 0 0 0; H 0 0 {R}; H 0 0 {-R}'
    return f'{X} 0 0 0; H 0 0 {R}'

for name, (X, q, R0) in MOL.items():
    nH = 2 if name == 'BeH2' else 1
    for meth in METH:
        Rs = R0 * np.array([0.94, 0.97, 1.0, 1.03, 1.06])
        Es = [energy(geom(name, R), q, 0, meth) for R in Rs]
        c = np.polyfit(Rs, Es, 2); Re = -c[1] / (2 * c[0]); Ee = np.polyval(c, Re)
        Ea = energy(f'{X} 0 0 0', q, ATOM_SPIN[X], meth) + nH * energy('H 0 0 0', 0, 1, meth)
        print(f'{name:5s} {meth:8s} R_e {Re:.4f}  D_e {Ea - Ee:.5f}', flush=True)
