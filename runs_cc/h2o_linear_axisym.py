"""
LINEAR H-O-H in the axisymmetric solver (2026-11-26): the top of water's bending barrier, and the base case of the
O +6 model in 2D (mesh ladder possible). O +6 at z = 0 (1s2 frozen inside r_c, hard exclusion, Neumann on the sphere),
bare protons at z = +-R. Domains, all 2hom (O's six electrons + one from each H):
    bond   theta < t1 about +z from O      the O-H pair over the right proton      (mirror: the left one)
    ring   theta > t1, z > 0               half of the lone-pair ring                (mirror: the other half)
Walls within O: the bond|ring cone (t1 by DENSITY CONTINUITY) and the ring|ring midplane, FREE by default;
?ownbeta: Robin du/dn + beta u = 0 on both, beta_k = sqrt(-2 eps_k) self-consistent, surface energy in E.
Mirror symmetry z -> -z: two domains solved, the other two by reflection. Poisson boundary = unit charge at the
density's centroid (monopole + dipole exact). Plain Lanczos eigensolver (shift-invert picked wrong states, O2 test).
  E = 2 [sum n_k (T_k + V_k + Esurf_k) + U_within] + U_left-right + V_nn
usage: python3 h2o_linear_axisym.py h rc R1,R2,... [ownbeta] [ch2]
  ch2: linear H-C-H with C +4 (1s2 frozen): two C-H 2hom bond cones + the ring as two SINGLE electrons (1+1),
       triplet 3CH2 (parallel spins; wall between them free like every wall within one atom unless ownbeta)
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ.setdefault(_v, '4')
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_realqm import Grid, masked_stiffness
from axisym_hom2 import _poisson_setup

L = 12.0; ZO = 6.0; NEL = [2, 2]; NAMES = ['bond', 'ring']


class HOH:
    def __init__(self, h, rc, R, ownbeta=False, ch2=False):
        self.h, self.R, self.ownbeta = h, R, ownbeta
        global NEL
        ZO = 4.0 if ch2 else 6.0; NEL = [2, 1] if ch2 else [2, 2]
        self.g = g = Grid(L, L, h, zmin=-L); N = g.N
        self.Z = g.Z.ravel(); self.r = g.R.ravel(); self.V = g.V
        dO = np.sqrt(self.Z ** 2 + self.r ** 2)
        self.K = ZO / np.maximum(dO, 1e-12) + 1 / np.sqrt((self.Z - R) ** 2 + self.r ** 2) + 1 / np.sqrt((self.Z + R) ** 2 + self.r ** 2)
        self.ok = dO > rc
        self.Vnn = 2 * ZO / R + 1 / (2 * R)
        self.cth = self.Z / np.maximum(dO, 1e-12)
        self.half = self.Z > 0
        self.solveP, self.bnd, self.rrf = _poisson_setup(g)
        self.w = 1 / np.sqrt(self.V); self.W = sp.diags(self.w)
        self.nz, self.nr = g.nz, g.nr
        self.flip = np.arange(N).reshape(self.nz, self.nr)[::-1, :].ravel()
        self.mid = np.where(np.abs(self.Z - h / 2) < 1e-9)[0]; self.Amid = 2 * np.pi * self.r[self.mid] * h
        self.t1 = 54.74; self.u = [None, None]; self.beta = [0.0, 0.0]

    def masks(self):
        c = np.cos(np.radians(self.t1)); b = self.ok & self.half
        return [b & (self.cth > c), b & (self.cth <= c)]

    def P(self, rho):
        zc = float(np.sum(rho * self.V * self.Z) / np.sum(rho * self.V))
        b = 4 * np.pi * rho * self.V; bb = self.bnd
        b[bb] = 1.0 / np.sqrt((self.Z[bb] - zc) ** 2 + self.r[bb] ** 2); return self.solveP(b)

    def eig(self, S, Veff, keep, u0):
        A = (self.W @ (S + sp.diags(Veff * self.V)).tocsr() @ self.W).tocsr()[keep][:, keep]
        v0 = None if u0 is None else (u0 / self.w)[keep]
        if v0 is not None and not np.any(v0): v0 = None
        val, vec = eigsh(A, k=1, which='SA', tol=1e-10, maxiter=200000, v0=v0)
        y = np.zeros(self.g.N); y[keep] = vec[:, 0]; u = self.w * y; u /= np.sqrt(np.sum(u * u * self.V))
        return (-u if u.sum() < 0 else u), float(val[0])

    def wallfaces(self, M):
        """Robin vectors for ownbeta: per domain, beta * face area on its cells at the cone wall and (ring) the midplane."""
        if not self.ownbeta: return [np.zeros(self.g.N), np.zeros(self.g.N)]
        nz, nr, h = self.nz, self.nr, self.h; r = self.g.r
        lab = -np.ones(self.g.N, int); lab[M[0]] = 0; lab[M[1]] = 1; L2 = lab.reshape(nz, nr)
        Az = np.repeat((2 * np.pi * r * h)[None, :], nz - 1, 0); Ar = np.repeat((2 * np.pi * (np.arange(1, nr) * h) * h)[None, :], nz, 0)
        out = []
        for k in (0, 1):
            a = np.zeros((nz, nr)); o = 1 - k
            a[:-1][(L2[:-1] == k) & (L2[1:] == o)] += Az[(L2[:-1] == k) & (L2[1:] == o)]
            a[1:][(L2[1:] == k) & (L2[:-1] == o)] += Az[(L2[1:] == k) & (L2[:-1] == o)]
            a[:, :-1][(L2[:, :-1] == k) & (L2[:, 1:] == o)] += Ar[(L2[:, :-1] == k) & (L2[:, 1:] == o)]
            a[:, 1:][(L2[:, 1:] == k) & (L2[:, :-1] == o)] += Ar[(L2[:, 1:] == k) & (L2[:, :-1] == o)]
            a = a.ravel()
            if k == 1: a[self.mid] += self.Amid * M[1][self.mid]       # ring|ring (mirror) at z = 0
            out.append(self.beta[k] * a)
        return out

    def scf(self, iters=300, tol=1e-8, mix=0.5):
        M = self.masks(); N = self.g.N
        S = [masked_stiffness(self.g, m) for m in M]; keep = [np.where(m)[0] for m in M]
        u = [np.where(M[k], self.u[k] if self.u[k] is not None else np.exp(-1.2 * np.sqrt(self.Z ** 2 + self.r ** 2)), 0.0) for k in (0, 1)]
        u = [x / np.sqrt(np.sum(x * x * self.V)) for x in u]
        P = None; PL = None; Eold = None; lam = [0.0, 0.0]
        for it in range(iters):
            b = self.wallfaces(M)
            for k in (0, 1):
                Veff = -self.K.copy()
                if P is not None: Veff += (NEL[k] - 1) * P[k] + NEL[1 - k] * P[1 - k] + PL
                u[k], lam[k] = self.eig(S[k] + sp.diags(0.5 * b[k]), Veff, keep[k], u[k])
            nP = [self.P(x * x) for x in u]; nPL = sum(NEL[l] * nP[l][self.flip] for l in (0, 1))
            if P is None: P, PL = nP, nPL
            else: P = [(1 - mix) * p + mix * q for p, q in zip(P, nP)]; PL = (1 - mix) * PL + mix * nPL
            if self.ownbeta: self.beta = [0.5 * bt + 0.5 * (np.sqrt(-2 * l) if l < 0 else 0.0) for bt, l in zip(self.beta, lam)]
            e = self.energy(u, S, b)
            if Eold is not None and abs(e['E'] - Eold) < tol and it > 5: break
            Eold = e['E']
        self.u, self.M = u, M; e['iters'] = it + 1; e['eps'] = lam; e['beta'] = list(self.beta); return e

    def energy(self, u, S, b):
        V = self.V; P = [self.P(x * x) for x in u]
        T = [float(x @ (s @ x)) for x, s in zip(u, S)]; Vk = [-float(np.sum(self.K * x * x * V)) for x in u]
        Es = [0.5 * float(np.sum(bb * x * x)) for bb, x in zip(b, u)]
        Uin = sum(NEL[k] * (NEL[k] - 1) / 2 * float(np.sum(P[k] * u[k] ** 2 * V)) for k in (0, 1)) + NEL[0] * NEL[1] * float(np.sum(P[1] * u[0] ** 2 * V))
        E1 = sum(NEL[k] * (T[k] + Vk[k] + Es[k]) for k in (0, 1)) + Uin
        Ulr = sum(NEL[k] * NEL[l] * float(np.sum(P[l][self.flip] * u[k] ** 2 * V)) for k in (0, 1) for l in (0, 1))
        return dict(E=2 * E1 + Ulr + self.Vnn, T=2 * sum(n * t for n, t in zip(NEL, T)), Esurf=2 * sum(n * s for n, s in zip(NEL, Es)))

    def jump(self):
        nz, nr, h = self.nz, self.nr, self.h; r = self.g.r
        lab = -np.ones(self.g.N, int); lab[self.M[0]] = 0; lab[self.M[1]] = 1; L2 = lab.reshape(nz, nr)
        rho = np.zeros(self.g.N)
        for k in (0, 1): rho[self.M[k]] = NEL[k] * self.u[k][self.M[k]] ** 2
        R2 = rho.reshape(nz, nr)
        Az = np.repeat((2 * np.pi * r * h)[None, :], nz - 1, 0); Ar = np.repeat((2 * np.pi * (np.arange(1, nr) * h) * h)[None, :], nz, 0)
        num = den = 0.0
        for (sa, sb, A) in [((slice(0, -1), slice(None)), (slice(1, None), slice(None)), Az),
                            ((slice(None), slice(0, -1)), (slice(None), slice(1, None)), Ar)]:
            p1 = (L2[sa] == 0) & (L2[sb] == 1); p2 = (L2[sa] == 1) & (L2[sb] == 0)
            num += np.sum(A * p1 * (R2[sa] - R2[sb])) + np.sum(A * p2 * (R2[sb] - R2[sa])); den += np.sum(A * (p1 | p2))
        return num / den if den > 0 else 0.0

    def relax(self, outer=40):
        e = self.scf(); step = 4.0; last = None
        for it in range(outer):
            J = self.jump()
            if last is not None and np.sign(J) != np.sign(last): step *= 0.5
            last = J
            if step < 0.25: break
            self.t1 = float(np.clip(self.t1 + step * np.sign(J), 10, 120)); e = self.scf()
        e['t1'] = self.t1; e['J'] = self.jump(); return e


if __name__ == '__main__':
    h = float(sys.argv[1]); rc = float(sys.argv[2]); ob = 'ownbeta' in sys.argv[4:]; c2 = 'ch2' in sys.argv[4:]
    print('linear ' + ('CH2 (C+4, ring 1+1)' if c2 else 'H2O') + '  h %.3f  rc %.3f  %s' % (h, rc, 'beta on O\'s own walls' if ob else 'free walls'), flush=True)
    for R in [float(x) for x in sys.argv[3].split(',')]:
        t0 = time.time(); m = HOH(h, rc, R, ob, c2); e = m.relax()
        print('  R %.3f  E %.6f  T %.4f  Esurf %.5f  cone %.1f  J %+.5f  eps %s  beta %s  (%.0fs)' % (
            R, e['E'], e['T'], e['Esurf'], e['t1'], e['J'], ','.join('%.3f' % x for x in e['eps']),
            ','.join('%.3f' % x for x in e['beta']), time.time() - t0), flush=True)
