"""
Hartree-Fock X-H bond lengths (PySCF, cc-pVTZ) under the SAME constraints as the RealQM scans in the article:
all X-H bonds stretched together, angles and C-C bonds held at RealQM's values (ethene H-C-H 109.47 as in its
tetrahedral frame, ethane staggered tetrahedral, C-C 2.901/2.530/2.273 bohr). Bond length = minimum of a parabola
through five energies R0 +- 0.04, +-0.08 bohr. RHF for closed shells, UHF for OH. Units bohr.
"""
import numpy as np
from pyscf import gto, scf
BAS = 'cc-pvtz'
def unit(v): v = np.array(v, float); return v/np.linalg.norm(v)
def tetra_dirs():
    return [unit(d) for d in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]

def geom(name, R):
    if name == 'LiH': return [('Li', (0, 0, 0)), ('H', (0, 0, R))], 0
    if name == 'BH': return [('B', (0, 0, 0)), ('H', (0, 0, R))], 0
    if name == 'HF': return [('F', (0, 0, 0)), ('H', (0, 0, R))], 0
    if name == 'OH': return [('O', (0, 0, 0)), ('H', (0, 0, R))], 1
    if name == 'BeH2': return [('Be', (0, 0, 0)), ('H', (0, 0, R)), ('H', (0, 0, -R))], 0
    if name == 'CH4': return [('C', (0, 0, 0))] + [('H', tuple(R*d)) for d in tetra_dirs()], 0
    if name in ('NH3', 'H2O'):
        ang = np.radians(106.7 if name == 'NH3' else 104.5)
        if name == 'H2O':
            return [('O', (0, 0, 0)), ('H', (R*np.sin(ang/2), 0, R*np.cos(ang/2))), ('H', (-R*np.sin(ang/2), 0, R*np.cos(ang/2)))], 0
        # C3v: angle between bonds = ang; bonds at polar angle t from -z with cos(ang) = 1.5 cos^2 t - 0.5 ... solve
        st = np.sqrt((1 - np.cos(ang))/1.5); ct = np.sqrt(1 - st*st)
        return [('N', (0, 0, 0))] + [('H', (R*st*np.cos(p), R*st*np.sin(p), -R*ct)) for p in (0, 2*np.pi/3, 4*np.pi/3)], 0
    if name == 'C2H2':
        CC = 2.273; return [('C', (0, 0, -CC/2)), ('C', (0, 0, CC/2)), ('H', (0, 0, -CC/2 - R)), ('H', (0, 0, CC/2 + R))], 0
    if name == 'C2H4':
        CC = 2.530; hl = np.radians(109.4712/2); at = []
        for s in (-1, 1):
            at.append(('C', (0, 0, s*CC/2)))
            for t in (1, -1): at.append(('H', (t*R*np.sin(hl), 0, s*(CC/2 + R*np.cos(hl)))))
        return at, 0
    if name == 'C2H6':
        CC = 2.901; ct = 1/3; st = np.sqrt(8/9); at = []
        for s, rot in ((-1, 0.0), (1, np.pi/3)):
            at.append(('C', (0, 0, s*CC/2)))
            for k in range(3):
                p = rot + 2*np.pi*k/3; at.append(('H', (R*st*np.cos(p), R*st*np.sin(p), s*(CC/2 + R*ct))))
        return at, 0

def energy(name, R):
    atoms, spin = geom(name, R)
    m = gto.M(atom=atoms, unit='Bohr', basis=BAS, spin=spin, verbose=0)
    mf = (scf.UHF(m) if spin else scf.RHF(m)); mf.conv_tol = 1e-10
    return mf.kernel()

EXP = {'BeH2': 2.520, 'C2H2': 2.004, 'BH': 2.329, 'OH': 1.832, 'C2H6': 2.067, 'C2H4': 2.054, 'CH4': 2.054,
       'LiH': 3.015, 'HF': 1.733, 'NH3': 1.913, 'H2O': 1.809}
for name, R0 in EXP.items():
    xs = R0 + np.array([-0.08, -0.04, 0.0, 0.04, 0.08]) - 0.03
    es = [energy(name, x) for x in xs]
    a, b, c = np.polyfit(xs, es, 2); Re = -b/(2*a)
    print("%-5s HF R_e %.4f  exp %.3f  err %+.2f%%" % (name, Re, R0, 100*(Re - R0)/R0), flush=True)
