"""
AXISYMMETRIC O2 with Pauli walls (2026-11-26). O +6 (O nucleus + 1s2 frozen inside r_c, hard exclusion, Neumann on
the sphere); the six outer electrons of each O in four axisymmetric domains bounded by CONES about the nucleus,
polar angle theta measured from the direction toward the partner:
    sigma   theta < t1         1 e    (the bond electron; sigma|sigma at the midplane = 2split singlet pair, FREE wall)
    unp     t1 < theta < t2    1 e    (unpaired; unp|unp at the midplane: PAULI wall (triplet) or FREE (singlet=1))
    belt    t2 < theta < t3    2hom
    cap     theta > t3         2hom   (the 2s pair, behind the atom on the axis)
Walls within one O (the cones): FREE, angles moved to DENSITY CONTINUITY (area-weighted density jump = 0), not min E.
Pauli wall: Robin du/dn + beta u = 0, beta_k = sqrt(-2 eps_k) self-consistent, with its surface energy
(beta/2) int u^2 dS in E (reported separately as Esurf). By mirror symmetry the only walls between the atoms are at
z = 0 and join domains of the SAME type, so the Pauli walls are: unp|unp (triplet), belt|belt, cap|cap (+ any
sigma-side contacts are sigma|sigma, free).
Atom reference: one O at the SAME z offset (z = +R/2) on the same grid -> identical core cell set.
  E = sum n_k (T_k + V_k + Esurf_k) + sum n_k(n_k-1)/2 U_kk + sum_{k<l} n_k n_l U_kl + V_nn
usage: python3 o2_axisym.py atom  h rc R            -> O atom (nucleus at z=R/2)
       python3 o2_axisym.py mol   h rc R [singlet] [nowall]
       python3 o2_axisym.py scan  h rc R1,R2,... [singlet] [nowall]   -> atom + molecule + D per R
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ.setdefault(_v, '4')
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_realqm import Grid, masked_stiffness
from axisym_hom2 import _poisson_setup

L = 12.0; ZN = 6.0
NAMES = ['sigma', 'unp', 'belt', 'cap']; NEL = [1, 1, 2, 2]
T0 = [np.degrees(np.arccos(2/3)), np.degrees(np.arccos(1/3)), np.degrees(np.arccos(-1/3))]   # equal solid angle / e


class O2:
    def __init__(self, h, rc, R, mol=True, singlet=False, nowall=False):
        self.h, self.rc, self.R, self.mol = h, rc, R, mol
        self.g = g = Grid(L, L, h, zmin=-L); self.a = a = R / 2
        self.Z = g.Z.ravel(); self.r = g.R.ravel(); self.V = g.V; N = g.N
        nuc = [a, -a] if mol else [a]
        self.K = sum(ZN / np.sqrt((self.Z - z) ** 2 + self.r ** 2) for z in nuc)
        self.ok = np.ones(N, bool)
        for z in nuc: self.ok &= np.sqrt((self.Z - z) ** 2 + self.r ** 2) > rc
        self.Vnn = ZN * ZN / R if mol else 0.0
        d = np.sqrt((self.Z - a) ** 2 + self.r ** 2)
        self.cth = -(self.Z - a) / np.maximum(d, 1e-12)          # cos(theta), theta from the -z direction (partner)
        self.half = (self.Z > 0) if mol else np.ones(N, bool)
        self.solveP, self.bnd, self.rrf = _poisson_setup(g)
        self.w = 1 / np.sqrt(self.V); self.W = sp.diags(self.w)
        self.nz, self.nr = g.nz, g.nr
        self.flip = np.arange(N).reshape(self.nz, self.nr)[::-1, :].ravel()     # z -> -z
        # Pauli at the midplane (molecule only): every type except sigma; unp only for the triplet
        self.pauli = [] if (not mol or nowall) else [k for k in (1, 2, 3) if not (k == 1 and singlet)]
        # z = 0 face cells (right side, just above the face) and their face area 2 pi r h
        self.mid = np.where(np.abs(self.Z - h / 2) < 1e-9)[0]
        self.Amid = 2 * np.pi * self.r[self.mid] * h
        self.th = list(T0); self.u = [None] * 4; self.lam = [None] * 4; self.beta = [0.0] * 4

    def masks(self):
        t = [np.cos(np.radians(x)) for x in self.th]
        c = self.cth; base = self.ok & self.half
        return [base & (c > t[0]), base & (c <= t[0]) & (c > t[1]), base & (c <= t[1]) & (c > t[2]), base & (c <= t[2])]

    def P(self, rho):
        # Dirichlet boundary = potential of the unit charge placed at the density's own centroid on the axis
        # (monopole + dipole exact). The library default 1/|x| puts the charge at the origin: wrong off-centre.
        zc = float(np.sum(rho * self.V * self.Z) / np.sum(rho * self.V))
        b = 4 * np.pi * rho * self.V; bb = self.bnd
        b[bb] = 1.0 / np.sqrt((self.Z[bb] - zc) ** 2 + self.r[bb] ** 2); return self.solveP(b)

    def eig(self, S, Veff, keep, u0, lam0):
        A = (self.W @ (S + sp.diags(Veff * self.V)).tocsr() @ self.W).tocsr()[keep][:, keep]
        v0 = None if u0 is None else (u0 / self.w)[keep]
        if v0 is not None and not np.any(v0): v0 = None
        if True:   # 2026-11-26: plain Lanczos ALWAYS. Shift-invert about lam0-0.3-0.2|lam0| picked a wrong state when eps moved
            val, vec = eigsh(A, k=1, which='SA', tol=1e-10, maxiter=200000, v0=v0)
        else: val, vec = eigsh(A, k=1, sigma=lam0 - 0.3 - 0.2 * abs(lam0), which='LM', tol=1e-10, maxiter=200000, v0=v0)
        y = np.zeros(self.g.N); y[keep] = vec[:, 0]; u = self.w * y; u /= np.sqrt(np.sum(u * u * self.V))
        return (-u if u.sum() < 0 else u), float(val[0])

    def scf(self, iters=300, tol=1e-8, mix=0.5):
        M = self.masks(); N = self.g.N
        for k, m in enumerate(M):
            if not m.any(): raise RuntimeError('domain %s empty at %s' % (NAMES[k], self.th))
        S = [masked_stiffness(self.g, m) for m in M]; keep = [np.where(m)[0] for m in M]
        u = [np.where(M[k], self.u[k] if self.u[k] is not None else np.exp(-1.5 * np.sqrt((self.Z - self.a) ** 2 + self.r ** 2)), 0.0)
             for k in range(4)]
        u = [x / np.sqrt(np.sum(x * x * self.V)) for x in u]
        P = None; Eold = None; lam = list(self.lam)
        for it in range(iters):
            b = []
            for k in range(4):
                bk = np.zeros(N)
                if k in self.pauli: bk[self.mid] = self.beta[k] * self.Amid * M[k][self.mid]
                b.append(bk)
            for k in range(4):
                Veff = -self.K.copy()
                if P is not None: Veff += (NEL[k] - 1) * P[k] + sum(NEL[l] * P[l] for l in range(4) if l != k) + PL
                u[k], lam[k] = self.eig(S[k] + sp.diags(0.5 * b[k]), Veff, keep[k], u[k], lam[k])
            nP = [self.P(x * x) for x in u]
            nPL = sum(NEL[l] * nP[l][self.flip] for l in range(4)) if self.mol else np.zeros(N)
            if P is None: P, PL = nP, nPL
            else: P = [(1 - mix) * p + mix * q for p, q in zip(P, nP)]; PL = (1 - mix) * PL + mix * nPL
            for k in self.pauli: self.beta[k] = 0.5 * self.beta[k] + 0.5 * (np.sqrt(-2 * lam[k]) if lam[k] < 0 else 0.0)
            e = self.energy(u, S, b)
            if Eold is not None and abs(e['E'] - Eold) < tol and it > 5: break
            Eold = e['E']
        self.u, self.lam, self.S, self.M = u, lam, S, M
        e['iters'] = it + 1; e['eps'] = lam; e['beta'] = list(self.beta); return e

    def energy(self, u, S, b):
        V = self.V; P = [self.P(x * x) for x in u]
        T = [float(x @ (s @ x)) for x, s in zip(u, S)]
        Vk = [-float(np.sum(self.K * x * x * V)) for x in u]
        Es = [0.5 * float(np.sum(bb * x * x)) for bb, x in zip(b, u)]
        Uself = sum(NEL[k] * (NEL[k] - 1) / 2 * float(np.sum(P[k] * u[k] ** 2 * V)) for k in range(4))
        Ucross = sum(NEL[k] * NEL[l] * float(np.sum(P[l] * u[k] ** 2 * V)) for k in range(4) for l in range(k + 1, 4))
        E1 = sum(NEL[k] * (T[k] + Vk[k] + Es[k]) for k in range(4)) + Uself + Ucross   # one atom's domains
        if not self.mol:
            self.parts = dict(T=sum(n * t for n, t in zip(NEL, T)), VK=sum(n * v for n, v in zip(NEL, Vk)), Uin=Uself + Ucross, Tdom=T, Vdom=Vk)
            return dict(E=E1, T=sum(n * t for n, t in zip(NEL, T)), Esurf=0.0)
        PLk = [P[l][self.flip] for l in range(4)]                                          # mirror (left) potentials
        Ulr = sum(NEL[k] * NEL[l] * float(np.sum(PLk[l] * u[k] ** 2 * V)) for k in range(4) for l in range(4))
        E = 2 * E1 + Ulr + self.Vnn
        self.parts = dict(T=2 * sum(n * t for n, t in zip(NEL, T)), VK=2 * sum(n * v for n, v in zip(NEL, Vk)),
                          Uin=2 * (Uself + Ucross), Ulr=Ulr, Vnn=self.Vnn, Tdom=T, Vdom=Vk)
        # the attraction of the RIGHT domains to the LEFT nucleus is already in K (both nuclei): 2*E1 counts both sides
        return dict(E=E, T=2 * sum(n * t for n, t in zip(NEL, T)), Esurf=2 * sum(n * s for n, s in zip(NEL, Es)))

    def jumps(self):
        """area-weighted density jump (inner - outer) across each cone wall, faces between domain i and i+1"""
        nz, nr, h = self.nz, self.nr, self.h
        lab = -np.ones(self.g.N, int)
        for k, m in enumerate(self.M): lab[m] = k
        L2 = lab.reshape(nz, nr); rho = np.zeros(self.g.N)
        for k in range(4): rho[self.M[k]] = NEL[k] * self.u[k][self.M[k]] ** 2
        R2 = rho.reshape(nz, nr); r = self.g.r
        Az = np.repeat((2 * np.pi * r * h)[None, :], nz - 1, 0); Ar = np.repeat((2 * np.pi * (np.arange(1, nr) * h) * h)[None, :], nz, 0)
        out = []
        for i in range(3):
            num = den = 0.0
            for (sa, sb, A) in [((slice(0, -1), slice(None)), (slice(1, None), slice(None)), Az),
                                ((slice(None), slice(0, -1)), (slice(None), slice(1, None)), Ar)]:
                p1 = (L2[sa] == i) & (L2[sb] == i + 1); p2 = (L2[sa] == i + 1) & (L2[sb] == i)
                num += np.sum(A * p1 * (R2[sa] - R2[sb])) + np.sum(A * p2 * (R2[sb] - R2[sa])); den += np.sum(A * (p1 | p2))
            out.append(num / den if den > 0 else 0.0)
        return out

    def relax(self, outer=40, verbose=False):
        """cones to density continuity: inner domain denser at its wall -> it grows (theta_i up)."""
        e = self.scf(); step = [4.0, 4.0, 4.0]; last = [None] * 3
        for it in range(outer):
            J = self.jumps(); rho0 = max(abs(x) for x in J) + 1e-12
            moved = False
            for i in range(3):
                if last[i] is not None and np.sign(J[i]) != np.sign(last[i]): step[i] *= 0.5
                last[i] = J[i]
                if step[i] > 0.25:
                    self.th[i] = float(np.clip(self.th[i] + step[i] * np.sign(J[i]), (self.th[i - 1] + 2) if i else 5, (self.th[i + 1] - 2) if i < 2 else 175))
                    moved = True
            if verbose: print('   cones %s  J %s  E %.6f' % (' '.join('%.2f' % t for t in self.th), ' '.join('%+.4f' % j for j in J), e['E']), flush=True)
            if not moved: break
            e = self.scf()
        e['th'] = list(self.th); e['J'] = self.jumps(); return e


def run(kind, h, rc, R, singlet=False, nowall=False, verbose=False):
    m = O2(h, rc, R, mol=(kind == 'mol'), singlet=singlet, nowall=nowall)
    t0 = time.time(); e = m.relax(verbose=verbose); e['sec'] = time.time() - t0
    if 'parts' in sys.argv: print('   parts', {k: (round(v, 5) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in m.parts.items()}, flush=True)
    return e


if __name__ == '__main__':
    kind = sys.argv[1]; h = float(sys.argv[2]); rc = float(sys.argv[3])
    flags = sys.argv[5:]; sing = 'singlet' in flags; now = 'nowall' in flags
    fmt = lambda e: 'E %.6f  T %.4f  Esurf %.5f  cones %s  eps %s  beta %s  (%d it, %.0fs)' % (
        e['E'], e['T'], e['Esurf'], '/'.join('%.1f' % t for t in e['th']), ','.join('%.3f' % x for x in e['eps']),
        ','.join('%.3f' % x for x in e['beta']), e['iters'], e['sec'])
    tag = ('singlet' if sing else 'triplet') + (' NOWALL' if now else '')
    for R in [float(x) for x in sys.argv[4].split(',')]:
        if kind in ('atom', 'scan'):
            ea = run('atom', h, rc, R, verbose='v' in flags); print('atom  h %.3f rc %.3f zO %.3f  %s' % (h, rc, R / 2, fmt(ea)), flush=True)
        if kind in ('mol', 'scan'):
            em = run('mol', h, rc, R, sing, now, verbose='v' in flags); print('O2 %s h %.3f rc %.3f R %.3f  %s' % (tag, h, rc, R, fmt(em)), flush=True)
        if kind == 'scan': print('   D(R=%.3f) = 2E(O) - E(O2) = %.5f' % (R, 2 * ea['E'] - em['E']), flush=True)
