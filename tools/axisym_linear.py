"""
LINEAR MOLECULES in the axisymmetric solver: any number of nuclei on the axis, any number of
domains separated by PLANES z = p_k, each domain holding n = 1 or 2 electrons (2 = one shape,
pair counted once).  Generalises axisym_hetero.Diatomic (2 domains) and axisym_xh2.SymXH2 (H-X-H).

  nuclei  : (label, z, Z, rc)    bare softened kernel Z/sqrt(d^2+soft^2); hard core: density zero
                                 inside rc for EVERY domain, Neumann on the sphere (masked stiffness)
  domains : (label, n)           ordered along z; domain k lives between planes k-1 and k
  planes  : (z0, mode)           mode 'free' -> placed by DENSITY CONTINUITY (bisection on the face
                                 index, bracket between the neighbouring cores); 'fixed' -> stays;
                                 'mirror:j' -> z = -z_j (symmetric molecules)

  E = sum_k n_k (T_k + V_k) + sum_k n_k(n_k-1)/2 U_kk + sum_{k<l} n_k n_l U_kl + V_nn
  potential of domain k: -K + (n_k - 1) P_k + sum_{l != k} n_l P_l

Free planes are relaxed by Gauss-Seidel sweeps (each free plane bisected with the others held) until
no face index changes; the reported E is the last SCF at the final faces, plus a per-plane linear
interpolation to J = 0 (exact for one independent plane, as in axisym_xh2).

Grid: z = 0 is a cell face; h is adjusted so that the reference length `hfix` (default: the largest
|z| among the nuclei) sits at a cell CENTRE, (k+1/2)h -- the relation of the H atom reference.
Realised nucleus positions are printed (they are never rounded: the kernel is analytic).

Output: JSON for axisym_view.html (grid, nuclei, planes, per-domain log density, every solve).

usage:  python3 tools/axisym_linear.py PRESET [key=value ...]
        e.g. python3 tools/axisym_linear.py ch2_hom rcC=1.0 h=0.1
        python3 tools/axisym_linear.py list
"""
import os, sys, time, json, base64
# Sparse solves on ~1e4-1e5 cells gain nothing from 10 BLAS threads and the oversubscription slows them.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from axisym_realqm import Grid, masked_stiffness
from axisym_hom2 import _poisson_setup

KCAL, EV = 627.509, 27.2114
SHIFT_INVERT = os.environ.get('AXI_SHIFT_INVERT', '1') == '1'
PLANE_MEMO = {}     # (name, iface, plane) -> plane z / R from the previous run: warm start for an R scan
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(HERE, 'axisym_runs')


def rel_den(m, i):
    """per rho column: the boundary density (max of the two sides) relative to its peak over columns"""
    g = m.g; nL = g.nz // 2; s = m.Sb[i]; d = np.zeros(g.nr)
    U0 = m.u[i].reshape(g.nz, g.nr); U1 = m.u[i + 1].reshape(g.nz, g.nr)
    for j in range(g.nr):
        ib, ia = nL + s[j] - 1, nL + s[j]
        if 0 <= ib and ia < g.nz: d[j] = max(m.domains[i]['n'] * U0[ib, j] ** 2, m.domains[i + 1]['n'] * U1[ia, j] ** 2)
    return d / max(d.max(), 1e-300)


class Linear:
    def __init__(self, nuclei, domains, planes, h=0.1, L=12.0, soft=0.0, hfix=None, name='mol', iface='hard', cores=None,
                 hface=False, beta=0.0, beta_thru=None, curved=False, cfree=None):
        self.name = name; self.iface = iface
        # CURVED free boundaries: after the plane optimum, every free plane becomes a per-cell LABEL boundary (any
        # surface of revolution) moved cell by cell by local density continuity -- molecule.js's rule in (z, rho).
        # cfree: plane indices to curve (default: every 'free' plane).  lab = None means plane masks.
        self.curved = bool(curved); self.cfree = cfree; self.lab = None
        # ROBIN surface energy (beta/2) int_Gamma u^2 dS for EVERY electron on every plane bounding its domain
        # (sharp interface only).  Natural condition du/dn + beta u = 0: beta = 0 is the hard Neumann
        # construction.  beta_thru: separate value for planes THROUGH a nucleus (the He-like interface,
        # whose lambda was measured at about half the between-nuclei value); default = beta.
        self.beta = float(beta); self.beta_thru = self.beta if beta_thru is None else float(beta_thru)
        zs = [abs(n[1]) for n in nuclei]
        ref = hfix if hfix is not None else max(zs)
        if ref > 0:
            if hface:                                  # the reference length on a cell FACE: ref = k h
                k = max(1, int(round(ref / h))); h = ref / k
            else:                                      # ... on a cell CENTRE: ref = (k + 1/2) h
                k = int(round(ref / h - 0.5)); h = ref / (k + 0.5)
        self.h = h
        nL = int(round(L / h)); L = nL * h; self.L = L
        self.g = g = Grid(L, L, h, zmin=-L)
        self.zc = g.Z.ravel(); rr = g.R.ravel()
        self.nuclei = [dict(label=a, z=float(z), Z=float(Z), rc=float(rc)) for a, z, Z, rc in nuclei]
        self.domains = [dict(label=a, n=int(n)) for a, n in domains]
        assert len(planes) == len(domains) - 1
        # LIVE CORES: a sphere about nucleus `nuc` holding n electrons in ONE shape (pair counted once); its
        # radius is a free boundary placed by density continuity, like the planes.  Every planar domain
        # excludes the sphere.  Core domains are appended after the planar ones.
        self.nplanar = len(domains)
        self.cores = [dict(label=a, nuc=int(j), n=int(n)) for a, j, n, r0 in (cores or [])]
        self.cq = [max(2, int(round(2 * float(r0) / h))) for a, j, n, r0 in (cores or [])]   # radius = cq*h/2
        self.domains += [dict(label=c['label'], n=c['n'], core=True) for c in self.cores]
        K = np.zeros(g.N); ok = np.ones(g.N, bool)
        for nu in self.nuclei:
            d = np.sqrt((self.zc - nu['z']) ** 2 + rr ** 2)
            K += nu['Z'] / np.sqrt(d ** 2 + soft ** 2)
            if nu['rc'] > 0: ok &= d > nu['rc']
        self.K, self.ok, self.soft = K, ok, soft
        self.dcore = [np.sqrt((self.zc - self.nuclei[c['nuc']]['z']) ** 2 + rr ** 2) for c in self.cores]
        self.Vnn = sum(a['Z'] * b['Z'] / abs(a['z'] - b['z'])
                       for i, a in enumerate(self.nuclei) for b in self.nuclei[i + 1:])
        # planes as integer FACE indices measured from z = 0
        self.pmode = [m for _, m in planes]
        self.pk = [int(round(z / h)) for z, _ in planes]
        self._tie()
        self.solveP, self.bnd, self.rrf = _poisson_setup(g)
        self.w = 1.0 / np.sqrt(g.V); self.W = sp.diags(self.w)
        self.u = [None] * len(self.domains); self.frames = []

    def _tie(self):
        for i, m in enumerate(self.pmode):
            if m.startswith('mirror:'): self.pk[i] = -self.pk[int(m.split(':')[1])]

    def robin_faces(self, M):
        """per domain: beta x (face area 2 pi r h) on each owned cell touching a plane that bounds the domain;
        [] when no Robin term is in use.  Area = V / h."""
        if self.iface != 'hard' or (self.beta == 0.0 and self.beta_thru == 0.0): return []
        h = self.h; A = self.g.V / h; out = []
        nz = [n['z'] for n in self.nuclei]
        for k in range(len(M)):
            b = np.zeros(self.g.N)
            if k < self.nplanar:
                for i, side in ((k - 1, +1), (k, -1)):         # left plane (cells just above it), right plane (just below)
                    if i < 0 or i >= len(self.pk): continue
                    z = self.pk[i] * h
                    bt = self.beta_thru if any(abs(z - q) < 1e-9 for q in nz) else self.beta
                    cells = M[k] & (np.abs(self.zc - (z + side * h / 2)) < 1e-9)
                    b[cells] += bt * A[cells]
            out.append(b)
        return out

    def rcore(self, c):
        return self.cq[c] * self.h / 2

    def masks(self):
        if self.lab is not None:
            return [self.lab == k for k in range(len(self.domains))]
        edges = [-np.inf] + [k * self.h for k in self.pk] + [np.inf]
        inC = [d < self.rcore(c) for c, d in enumerate(self.dcore)]
        anyC = np.zeros(self.g.N, bool)
        for m in inC: anyC |= m
        M = [(self.zc > edges[k]) & (self.zc < edges[k + 1]) & self.ok & ~anyC for k in range(self.nplanar)]
        return M + [m & self.ok for m in inC]

    def core_jump(self, c, M=None):
        """volume-weighted density jump across the core sphere: (core, shell just inside) - (owners, shell just
        outside).  J > 0: the core is denser at its edge -> the radius grows (same sign rule as the planes)."""
        M = self.masks() if M is None else M
        r = self.rcore(c); d = self.dcore[c]; V = self.g.V; k = self.nplanar + c
        inn = M[k] & (d >= r - self.h)
        rho_in = self.domains[k]['n'] * self.u[k] ** 2
        rho_out = np.zeros(self.g.N); own = np.zeros(self.g.N, bool)
        for j in range(len(M)):
            if j == k: continue
            rho_out[M[j]] = self.domains[j]['n'] * self.u[j][M[j]] ** 2; own |= M[j]
        out = own & ~M[k] & (d < r + self.h)
        if not inn.any() or not out.any(): return 0.0
        return float(np.sum(rho_in[inn] * V[inn]) / np.sum(V[inn]) - np.sum(rho_out[out] * V[out]) / np.sum(V[out]))

    def wstiff(self, w):
        """1/2 int w|grad u|^2 with FACE-AVERAGED w, mol_fast's u-update stencil (FW = (wa+wb)/2).
        Faces touching a hard core cell are omitted (true Neumann there, as in hard mode)."""
        g = self.g; nz, nr, h = g.nz, g.nr, g.h
        W2 = w.reshape(nz, nr); ok2 = self.ok.reshape(nz, nr); I = np.arange(g.N).reshape(nz, nr)
        cz = np.pi * g.r[None, :] * 0.5 * (W2[1:] + W2[:-1]) * (ok2[1:] & ok2[:-1])
        cr = np.pi * (np.arange(1, nr) * h)[None, :] * 0.5 * (W2[:, 1:] + W2[:, :-1]) * (ok2[:, 1:] & ok2[:, :-1])
        a = np.concatenate([I[:-1].ravel(), I[:, :-1].ravel()]); b = np.concatenate([I[1:].ravel(), I[:, 1:].ravel()])
        c = np.concatenate([cz.ravel(), cr.ravel()]); m = c > 0; a, b, c = a[m], b[m], c[m]
        S = sp.coo_matrix((np.concatenate([c, c, -c, -c]), (np.concatenate([a, b, a, b]), np.concatenate([a, b, b, a]))),
                          shape=(g.N, g.N)).tocsr()
        keep = np.zeros(g.N, bool); keep[a] = True; keep[b] = True; keep |= (w > 0)
        return S, np.where(keep & self.ok)[0]

    def P(self, u):
        b = 4 * np.pi * (u * u) * self.g.V
        b[self.bnd] = 1.0 / self.rrf[self.bnd]
        return self.solveP(b)

    def eig(self, S, Veff, keep, u0, lam0=None):
        """lowest eigenpair.  With a previous eigenvalue lam0 it uses SHIFT-INVERT about a point safely below
        it (one sparse LU per call, then a few cheap solves); plain Lanczos 'SA' converges as ~1/h^2 slower."""
        A = (self.W @ (S + sp.diags(Veff * self.g.V)).tocsr() @ self.W).tocsr()[keep][:, keep]
        v0 = None if u0 is None else (u0 / self.w)[keep]
        if v0 is not None and not np.any(v0): v0 = None
        if lam0 is None or not SHIFT_INVERT:
            val, vec = eigsh(A, k=1, which='SA', tol=1e-10, maxiter=200000, v0=v0)
        else:
            sig = lam0 - 0.3 - 0.2 * abs(lam0)
            val, vec = eigsh(A, k=1, sigma=sig, which='LM', tol=1e-10, maxiter=200000, v0=v0)
        y = np.zeros(self.g.N); y[keep] = vec[:, 0]
        u = self.w * y; u /= np.sqrt(np.sum(u * u * self.g.V))
        return (-u if np.sum(u) < 0 else u), float(val[0])

    def scf(self, iters=200, tol=1e-9, mix=0.5):
        V = self.g.V; M = self.masks()
        for k, m in enumerate(M):
            if not m.any(): raise RuntimeError('domain %s empty at planes %s' % (self.domains[k]['label'], self.pk))
        if self.iface == 'weak':
            Wd = [m.astype(float) for m in M]
            SK = [self.wstiff(w) for w in Wd]; S = [x[0] for x in SK]; keep = [x[1] for x in SK]
        else:
            Wd = [np.ones(self.g.N)] * len(M)
            S = [masked_stiffness(self.g, m) for m in M]; keep = [np.where(m)[0] for m in M]
        self.Wd = Wd
        self.bvec = self.robin_faces(M)
        Seig = [Sk + sp.diags(0.5 * b) for Sk, b in zip(S, self.bvec)] if self.bvec else S
        n = [d['n'] for d in self.domains]
        u = [None if self.u[k] is None else np.where(np.isin(np.arange(self.g.N), keep[k]), self.u[k], 0.0)
             for k in range(len(M))]
        P = None; Eold = None; lam = [None] * len(M)
        for it in range(iters):
            for k in range(len(M)):
                Veff = -self.K.copy()
                if P is not None:
                    Veff += (n[k] - 1) * P[k] + sum(n[l] * P[l] for l in range(len(M)) if l != k)
                u[k], lam[k] = self.eig(Seig[k], Veff * Wd[k], keep[k], u[k], lam[k])   # potential gated by w (mol_fast: *wc)
            nP = [self.P(x) for x in u]
            P = nP if P is None else [(1 - mix) * a + mix * b for a, b in zip(P, nP)]
            e = self.energy(u, S)
            if Eold is not None and abs(e['E'] - Eold) < tol: break
            Eold = e['E']
        self.u = u; self.S = S
        e = self.energy(u, S); e['iters'] = it + 1; e['planes'] = [k * self.h for k in self.pk]
        Mj = self.masks()
        e['J'] = [self.jump(i, Mj) for i in range(len(self.pk))]
        e['Jc'] = [self.core_jump(c, Mj) for c in range(len(self.cores))]; e['rcore'] = [self.rcore(c) for c in range(len(self.cores))]
        self.frames.append(self.frame(e))
        return e

    def energy(self, u, S):
        """E = the functional the solver minimises.  weak mode: potential terms carry the domain's w
        (as the u-update does), and E_read is mol_fast's READOUT, which counts every cell (wd = 1)."""
        V = self.g.V; n = [d['n'] for d in self.domains]; Wd = getattr(self, 'Wd', [np.ones(self.g.N)] * len(u))
        P = [self.P(x) for x in u]
        T = [float(x @ (s @ x)) for x, s in zip(u, S)]
        def parts(wt):
            Vk = [-float(np.sum(self.K * x * x * w * V)) for x, w in zip(u, wt)]
            Vee = 0.0
            for k in range(len(u)):
                Vee += n[k] * (n[k] - 1) / 2 * float(np.sum(P[k] * u[k] ** 2 * wt[k] * V))
                for l in range(len(u)):
                    if l != k: Vee += 0.5 * n[k] * n[l] * float(np.sum(P[l] * u[k] ** 2 * wt[k] * V))
            return Vk, Vee
        Vk, Vee = parts(Wd)
        Ttot = sum(a * b for a, b in zip(n, T)); VK = sum(a * b for a, b in zip(n, Vk))
        bv = getattr(self, 'bvec', None)
        Esurf = sum(n[k] * 0.5 * float(np.sum(bv[k] * u[k] ** 2)) for k in range(len(u))) if bv else 0.0
        out = dict(E=Ttot + VK + Vee + self.Vnn + Esurf, T=Ttot, VK=VK, Vee=Vee, Vnn=self.Vnn, Tdom=T, Vdom=Vk, Esurf=Esurf)
        if self.iface == 'weak':
            Vk1, Vee1 = parts([np.ones(self.g.N)] * len(u))
            out['E_read'] = Ttot + sum(a * b for a, b in zip(n, Vk1)) + Vee1 + self.Vnn
            out['overhang'] = [float(np.sum(x * x * (1 - w) * V)) for x, w in zip(u, Wd)]
        return out

    def jump(self, i, M=None):
        """area-weighted density jump (left domain i) - (right domain i+1) across plane i"""
        g = self.g; s = self.pk[i] * self.h; M = self.masks() if M is None else M
        a = np.where(np.abs(self.zc - (s - g.h / 2)) < 1e-9)[0]
        b = a + g.nr                                          # the cell across the face, same rho
        keep = M[i][a] & M[i + 1][b]                          # both cells owned by the two planar domains
        a, b = a[keep], b[keep]
        w = g.R.ravel()[a]
        if w.sum() == 0: return 0.0
        nl, nr = self.domains[i]['n'], self.domains[i + 1]['n']
        d = nl * self.u[i][a] ** 2 - nr * self.u[i + 1][b] ** 2
        return float(np.sum(d * w) / np.sum(w))

    def bracket(self, i):
        """face-index range for plane i: between the cores of the nuclei on either side"""
        h = self.h; zl = -np.inf if i == 0 else self.pk[i - 1] * h
        zr = np.inf if i == len(self.pk) - 1 else self.pk[i + 1] * h
        z0 = self.pk[i] * h
        lo, hi = -np.inf, np.inf
        for j, nu in enumerate(self.nuclei):
            rc = max([nu['rc']] + [self.rcore(c) + self.h for c, cc in enumerate(self.cores) if cc['nuc'] == j])
            if nu['z'] <= z0: lo = max(lo, nu['z'] + rc)
            else: hi = min(hi, nu['z'] - rc)
        lo = max(lo, zl); hi = min(hi, zr)
        return int(np.floor(lo / h)) + 1, int(np.ceil(hi / h)) - 1

    def relax(self, verbose=True, sweeps=6):
        free = [('p', i) for i, m in enumerate(self.pmode) if m == 'free'] + [('c', c) for c in range(len(self.cores))]
        res = self.scf()
        if verbose: self.report(res)
        if not free: return dict(res, interp=dict(res))
        state = lambda: tuple(self.pk) + tuple(self.cq)
        last = {}; cache = {state(): res}                   # ONE cache for every sweep, keyed by ALL boundaries
        for sw in range(sweeps):
            changed = False
            for kind, i in free:
                if kind == 'p':
                    lo0, hi0 = self.bracket(i)
                    def put(k, i=i): self.pk[i] = k; self._tie()
                    def get(i=i): return self.pk[i]
                    def J(r, i=i): return r['J'][i]
                else:
                    nz = self.nuclei[self.cores[i]['nuc']]['z']
                    room = min([abs(p * self.h - nz) for p in self.pk] + [self.L - abs(nz)])
                    lo0, hi0 = 2, max(2, int((room - self.h) / (self.h / 2)))
                    def put(k, i=i): self.cq[i] = k
                    def get(i=i): return self.cq[i]
                    def J(r, i=i): return r['Jc'][i]
                def f(k):
                    put(k); key = state()
                    if key not in cache:
                        r = self.scf(); cache[key] = r
                        if verbose: self.report(r)
                    return J(cache[key])
                def at(k):
                    put(k); return cache[state()]
                old = get()
                # LOCAL search from the current plane: gallop (1, 2, 4.. faces) toward J = 0, then bisect.
                # Same bracket as a global bisection when J is monotone; far fewer solves when the start is close.
                k0 = min(max(old, lo0), hi0)
                if f(k0) > 0:
                    lo, s_ = k0, 1
                    while True:
                        k = min(lo + s_, hi0)
                        if f(k) > 0:
                            lo = k
                            if k == hi0: hi = hi0; break
                            s_ *= 2
                        else: hi = k; break
                else:
                    hi, s_ = k0, 1
                    while True:
                        k = max(hi - s_, lo0)
                        if f(k) <= 0:
                            hi = k
                            if k == lo0: lo = lo0; break
                            s_ *= 2
                        else: lo = k; break
                if lo != hi:
                    while hi - lo > 1:
                        m = (lo + hi) // 2
                        if f(m) > 0: lo = m
                        else: hi = m
                a, b = at(lo), at(hi)
                last[(kind, i)] = (a, b)
                put(lo if abs(J(a)) < abs(J(b)) else hi)
                if kind == 'p':
                    PLANE_MEMO[(self.name, self.iface, i)] = self.pk[i] * self.h / max([abs(n['z']) for n in self.nuclei] + [1e-9])
                else:
                    PLANE_MEMO[(self.name, self.iface, 'core', i)] = self.rcore(i)
                if get() != old: changed = True
            if not changed: break
        res = self.scf()
        # per-plane linear interpolation to J = 0 between the bracketing faces
        dE = 0.0; s_int = list(res['planes']); r_int = list(res['rcore'])
        for (kind, i), (a, b) in last.items():
            ja, jb = (a['J'][i], b['J'][i]) if kind == 'p' else (a['Jc'][i], b['Jc'][i])
            if a is b or ja == jb: continue
            t = ja / (ja - jb)
            Ei = a['E'] + t * (b['E'] - a['E'])
            dE += Ei - res['E']
            if kind == 'p': s_int[i] = a['planes'][i] + t * (b['planes'][i] - a['planes'][i])
            else: r_int[i] = a['rcore'][i] + t * (b['rcore'][i] - a['rcore'][i])
        for i, m in enumerate(self.pmode):
            if m.startswith('mirror:'): s_int[i] = -s_int[int(m.split(':')[1])]
        res['interp'] = dict(E=res['E'] + dE, planes=s_int, rcore=r_int)
        return res

    # ---------- curved free boundaries: a HEIGHT FUNCTION z = s(rho) per free plane ----------
    # Each curved plane i is one face index per rho-column, s_i[j].  Every column moves by the plane rule applied
    # to ITS OWN density jump J_j = n_i u_i^2 (cell below) - n_{i+1} u_{i+1}^2 (cell above): J > 0 -> up one face.
    # A column freezes on the better of the two faces where J changes sign (the discrete J = 0), as a plane does.
    # (A first per-cell flip rule, molecule.js-like, oscillated without settling: in the hard model a ragged
    # boundary LOWERS E, so nothing but continuity holds it, and unconstrained simultaneous flips lose it.)
    # Bowls only: closed shells and rings are not representable.  Mirror planes follow their master column-wise.
    def _profiles(self):
        nr = self.g.nr; S = [np.full(nr, k) for k in self.pk]
        for i, v in self.Sb.items(): S[i] = v.copy()
        for i, m in enumerate(self.pmode):
            if m.startswith('mirror:'): S[i] = -S[int(m.split(':')[1])]
        return S

    def _labels(self):
        g = self.g; Z2 = self.zc.reshape(g.nz, g.nr)
        lab = np.zeros((g.nz, g.nr), int)
        for Sp in self._profiles(): lab += (Z2 > (Sp * self.h)[None, :])
        lab = lab.ravel(); lab[~self.ok] = -1
        return lab

    def _colJ(self, i):
        """J per rho column across curved plane i (nan where a side is not owned: core, box edge)"""
        g = self.g; nL = g.nz // 2; s = self.Sb[i]; J = np.full(g.nr, np.nan); rel = np.full(g.nr, np.nan)
        lab = self.lab.reshape(g.nz, g.nr); nl, nr_ = self.domains[i]['n'], self.domains[i + 1]['n']
        U0 = self.u[i].reshape(g.nz, g.nr); U1 = self.u[i + 1].reshape(g.nz, g.nr)
        for j in range(g.nr):
            ib, ia = nL + s[j] - 1, nL + s[j]
            if ib < 0 or ia >= g.nz or lab[ib, j] != i or lab[ia, j] != i + 1: continue
            a, b = nl * U0[ib, j] ** 2, nr_ * U1[ia, j] ** 2
            J[j] = a - b; rel[j] = abs(a - b) / max(0.5 * (a + b), 1e-300)
        return J, rel

    def boundary_profile(self):
        return {i: self.zmin_face(self.Sb[i]) for i in self.Sb}

    def zmin_face(self, s):
        return s * self.h

    def relax_curved(self, verbose=True, maxit=200):
        res0 = self.relax(verbose=verbose)
        if self.cfree is None: self.cfree = [i for i, m in enumerate(self.pmode) if m == 'free']
        g = self.g; self.Sb = {i: np.full(g.nr, self.pk[i]) for i in self.cfree}
        self.lab = self._labels()
        prev = {i: [None] * g.nr for i in self.cfree}; frozen = {i: np.zeros(g.nr, bool) for i in self.cfree}
        seen = {}; hist = []; states = []
        for it in range(maxit):
            r = self.scf()
            moved = 0; rels = []
            for i in self.cfree:
                J, rel = self._colJ(i); s = self.Sb[i]; rels.append(rel)
                # bounds: strictly between the neighbouring planes' profiles
                P = self._profiles(); lo = P[i - 1] + 1 if i > 0 else np.full(g.nr, -g.nz // 2 + 1)
                hi = P[i + 1] - 1 if i + 1 < len(P) else np.full(g.nr, g.nz // 2 - 1)
                new = s.copy()
                for j in range(g.nr):
                    if np.isnan(J[j]) or frozen[i][j]: continue
                    pv = prev[i][j]
                    if pv is not None and abs(pv[0] - s[j]) == 1 and np.sign(pv[1]) != np.sign(J[j]):
                        new[j] = pv[0] if abs(pv[1]) < abs(J[j]) else s[j]; frozen[i][j] = True
                    else:
                        prev[i][j] = (s[j], J[j]); new[j] = s[j] + (1 if J[j] > 0 else -1)
                    new[j] = min(max(new[j], lo[j]), hi[j])
                # columns with no defined J, or negligible density (< 1e-6 of the boundary peak), follow the
                # column nearer the axis; and the surface stays CONNECTED: |s_j - s_(j-1)| <= 1 face, the axis leads
                dens = np.abs(J) + 0.0; dens[np.isnan(J)] = 0.0
                und = np.isnan(J) | (rel_den(self, i) < 1e-6)
                for j in range(1, g.nr):
                    if und[j]: new[j] = new[j - 1]
                    new[j] = min(max(new[j], new[j - 1] - 1, lo[j]), new[j - 1] + 1, hi[j])
                moved += int(np.sum(new != s)); self.Sb[i] = new
            rr = np.concatenate([x[~np.isnan(x)] for x in rels])
            # residual weighted toward the dense part: columns within 3 a0 of the axis
            near = np.concatenate([x[:int(3.0 / self.h)][~np.isnan(x[:int(3.0 / self.h)])] for x in rels])
            hist.append((r['E'], float(np.mean(near)) if near.size else 0.0, moved))
            if verbose:
                print('    curved it %3d  E=%.6f  |J|/rho mean(rho<3) %.3f  moved %d  frozen %d/%d' % (
                    it, r['E'], hist[-1][1], moved, sum(int(f.sum()) for f in frozen.values()),
                    len(self.cfree) * g.nr), flush=True)
            if moved == 0:
                break
            key = b''.join(self.Sb[i].tobytes() for i in self.cfree)
            if key in seen:
                if verbose: print('    state cycle -> stop', flush=True)
                break
            seen[key] = it
            self.lab = self._labels()
        self.lab = self._labels(); r = self.scf()
        prof = self.boundary_profile()
        r['interp'] = dict(E=r['E'], planes=[float(prof[i][0]) if i in prof else p for i, p in enumerate(r['planes'])],
                           rcore=r.get('rcore', []))
        r['curved'] = dict(profile={str(i): [round(float(z), 4) for z in v] for i, v in prof.items()},
                           hist=hist, E_plane=res0['interp']['E'], planes_plane=res0['interp']['planes'])
        if verbose:
            for i, v in prof.items():
                js = [j for j in (0, 5, 10, 15, 20, 25, 30, 40) if j < len(v)]
                print('    boundary %d: z(rho) %s' % (i, '  '.join('%.2f:%+.3f' % ((j + 0.5) * self.h, v[j]) for j in js)))
            print('    curved E=%.6f   plane E(interp)=%.6f   gain %+.6f Ha' % (r['E'], res0['interp']['E'],
                  res0['interp']['E'] - r['E']), flush=True)
        return r

    def moments(self):
        """per-domain centroid on the axis, and the axial dipole (Debye, + = toward +z positive end)"""
        V = self.g.V; cz = [float(np.sum(self.zc * u * u * V)) for u in self.u]
        mu = sum(n['Z'] * n['z'] for n in self.nuclei) - sum(d['n'] * c for d, c in zip(self.domains, cz))
        return cz, mu * 2.541746

    def report(self, r):
        print('    planes %s  J %s%s  E=%.6f  (%d it)' % (
            ' '.join('%+.4f' % p for p in r['planes']), ' '.join('%+.5f' % j for j in r['J']),
            ''.join('  core r=%.4f Jc=%+.5f' % (rc, jc) for rc, jc in zip(r.get('rcore', []), r.get('Jc', []))),
            r['E'], r['iters']), flush=True)

    # ---------- output ----------
    def frame(self, e):
        g = self.g; lab = np.full(g.N, 255, np.uint8); rho = np.zeros(g.N)
        for k, (u, m) in enumerate(zip(self.u, self.masks())):
            lab[m] = k; rho[m] = self.domains[k]['n'] * u[m] ** 2
        q = np.clip((np.log10(np.maximum(rho, 1e-12)) + 8) / 8 * 255, 0, 255).astype(np.uint8)
        return dict(E=e['E'], planes=e['planes'], J=e['J'],
                    lab=base64.b64encode(lab.tobytes()).decode(), rho=base64.b64encode(q.tobytes()).decode())

    def meta(self):
        g = self.g
        return dict(name=self.name, nz=g.nz, nr=g.nr, h=self.h, zmin=g.zmin, L=self.L, soft=self.soft,
                    nuclei=self.nuclei, domains=self.domains, pmode=self.pmode, rho_scale='log10 rho = q/255*8-8')


def H_atom(h, L, soft=0.0, z0=None):
    """H at z0 within the cell above the face z = 0 (default: the cell centre, z0 = h/2), whole space"""
    z0 = 0.5 * h if z0 is None else z0
    m = Linear([('H', z0, 1, 0)], [('H', 1)], [], h=h, L=L, soft=soft, hfix=0.0, name='H')
    m.h = h
    return m


# ---------------- presets ----------------
# measured references (CCCBDB / NIST): these are FROM MEMORY in this file; verify before quoting.
#   BeH2 linear, R(Be-H) 1.3264 A = 2.5066 a0;  CH2 here = LINEAR HCH test case (real CH2 is bent),
#   R(C-H) 1.078 A = 2.0372 a0 (triplet CH2 bond length).
#   valence binding (sum of the IEs removed by the pseudo-nucleus): Be 2s^2 IE1+IE2 = 27.53 eV
#   = 1.0117 Ha; C (+2 pseudo: 2p^2) IE1+IE2 = 11.260+24.383 = 35.643 eV = 1.3099 Ha;
#   C (+4 pseudo: 2s^2 2p^2) IE1..IE4 = 11.260+24.383+47.888+64.494 = 148.025 eV = 5.4398 Ha.
def preset(name, o):
    R = float(o.get('R', 0))
    def xh2(X, ZX, rcX, split, Rdef, nval_half):
        RR = R or Rdef; rc = float(o.get('rc' + X, rcX))
        nuc = [('H', -RR, 1, 0), (X, 0.0, ZX, rc), ('H', RR, 1, 0)]
        sH = 0.5 * RR + 0.5 * rc
        if split:
            dom = [('H', 1), (X + 'L', nval_half), (X + 'R', nval_half), ('H', 1)]
            pl = [(-sH, 'mirror:2'), (0.0, 'fixed'), (sH, 'free')]
            frag = [(X, 1, [(X, 0.0, ZX, rc)], [(X + 'L', nval_half), (X + 'R', nval_half)], [(0.0, 'fixed')])]
        else:
            dom = [('H', 1), (X, 2 * nval_half), ('H', 1)]
            pl = [(-sH, 'mirror:1'), (sH, 'free')]
            frag = [(X, 1, [(X, 0.0, ZX, rc)], [(X, 2 * nval_half)], [])]
        return dict(nuclei=nuc, domains=dom, planes=pl, frags=frag, nH=2, hfix=RR, Robs=Rdef)
    def xh(X, ZX, outer, Rdef, target, rcfit, ref_Ha, note, rc0):
        """X-H diatomic from an Atom Simulator configuration: the inner shells frozen into a pseudo-X of charge
        ZX (r_c fitted to the sum of the IEs it removes), the OUTER shell along the axis as one domain or as
        back|bond (a plane through X, fixed), then H."""
        RR = R or Rdef; rc = float(o.get('rc' + X, rc0)); s0 = 0.5 * RR + 0.5 * rc
        if len(outer) == 1:
            dom = [(X, outer[0]), ('H', 1)]; pl = [(s0, 'free')]; fd, fp = [(X, outer[0])], []
        else:
            dom = [(X + 'back', outer[0]), (X + 'bond', outer[1]), ('H', 1)]; pl = [(0.0, 'fixed'), (s0, 'free')]
            fd, fp = [(X + 'back', outer[0]), (X + 'bond', outer[1])], [(0.0, 'fixed')]
        return dict(nuclei=[(X, 0.0, ZX, rc), ('H', RR, 1, 0)], domains=dom, planes=pl,
                    frags=[(X, 1, [(X, 0.0, ZX, rc)], fd, fp)], nH=1, hfix=RR, Robs=Rdef,
                    pseudo=[(X, target, rcfit)], ref_kcal=ref_Ha * 627.509, note=note)
    # IE sums and experimental constants below are FROM MEMORY and must be verified before quoting:
    #   B IE1 8.298 eV; N IE1..3 14.534/29.601/47.445; O IE1..3 13.618/35.121/54.936 eV.
    #   BH R_e 1.2324 A, D_e ~3.57 eV; NH R_e 1.0362 A, D_e ~3.65 eV, mu ~1.39 D (N-H+);
    #   OH R_e 0.9697 A, D_e ~4.62 eV, mu 1.66 D (O-H+).
    P = {
        'bh':         lambda: xh('B', 1, [1], 2.3289, -0.30495, (0.3, 3.0), 0.1312,
                                 'BH: B 2+2+1 -> pseudo-B +1, outer 1e | H', 1.5),
        'nh_a':       lambda: xh('N', 2, [1, 1], 1.9581, -1.6219, (0.05, 2.0), 0.1341,
                                 'NH: N 2+3+2 -> pseudo-N +2, outer 1+1 (back|bond) | H', 0.6),
        'nh_b':       lambda: xh('N', 3, [2, 1], 1.9581, -3.3655, (0.05, 2.0), 0.1341,
                                 'NH: N 2+2+3 -> pseudo-N +3, outer 2+1 (back|bond) | H', 0.6),
        'oh':         lambda: xh('O', 3, [2, 1], 1.8325, -3.8100, (0.05, 2.0), 0.1698,
                                 'OH: O 2+3+3 -> pseudo-O +3, outer 2+1 (back|bond) | H', 0.5),
        # ---- exact references (no pseudo-atom) ----
        # H2+: one electron, whole space.  Exact R_e 1.9972 a0, D_e 0.10263 Ha (to H + H+).
        'h2plus':     lambda: dict(nuclei=[('H', -0.5 * (R or 1.9972), 1, 0), ('H', 0.5 * (R or 1.9972), 1, 0)],
                                   domains=[('e', 1)], planes=[], frags=[], nH=2, Hrefs=1, hfix=0.5 * (R or 1.9972),
                                   Robs=1.9972, ref_kcal=0.10263 * 627.509, note='H2+: one electron, exact reference'),
        # HeH+: two electrons on He(2) + H(1).  Exact R_e 1.4632 a0, E -2.978707, D_e 0.07498 Ha to He + H+.
        'heh_hom':    lambda: dict(nuclei=[('He', 0.0, 2, 0), ('H', (R or 1.4632), 1, 0)], domains=[('e', 2)], planes=[],
                                   frags=[('He', 1, [('He', 0.0, 2, 0)], [('He', 2)], [])], nH=1, Hrefs=0, hfix=(R or 1.4632),
                                   Robs=1.4632, ref_kcal=0.07498 * 627.509, note='HeH+: one 2e domain (He as 2hom)'),
        'heh_split':  lambda: dict(nuclei=[('He', 0.0, 2, 0), ('H', (R or 1.4632), 1, 0)], domains=[('Heback', 1), ('Hebond', 1)],
                                   planes=[(0.25 * (R or 1.4632), 'free')],
                                   frags=[('He', 1, [('He', 0.0, 2, 0)], [('Heb', 1), ('Hef', 1)], [(0.0, 'fixed')])],
                                   nH=1, Hrefs=0, hfix=(R or 1.4632), hface=True, Robs=1.4632, ref_kcal=0.07498 * 627.509,
                                   note='HeH+: two 1e domains, the plane between them FREE (no symmetry fixes it)'),
        # ---- the Atom-Simulator recipe, every domain <= 2 electrons; nuclei on cell faces (planes through them) ----
        # N 2+3+2 -> +2, outer 1+1 (IE1+IE2 1.6219); O 2+3+3 -> +3, outer 2+1 (IE1..3 3.8100); C 2+2+2 -> +2,
        # outer 1+1 (IE1+IE2 1.3099).  Exp (from memory, verify): N2 2.0743 a0 / D_e 0.3640; CO 2.1322 a0 / 0.4127,
        # mu 0.11 D C-O+; CO2 C=O 2.1921 a0.
        'n2_r':       lambda: (lambda RR, rc: dict(
                                   nuclei=[('N', -0.5 * RR, 2, rc), ('N', 0.5 * RR, 2, rc)],
                                   domains=[('NLback', 1), ('NLbond', 1), ('NRbond', 1), ('NRback', 1)],
                                   planes=[(-0.5 * RR, 'fixed'), (0.0, 'fixed'), (0.5 * RR, 'fixed')],
                                   frags=[('N', 2, [('N', 0.0, 2, rc)], [('Nb', 1), ('Nf', 1)], [(0.0, 'fixed')])], nH=0,
                                   hfix=0.5 * RR, hface=True, Robs=2.0743, pseudo=[('N', -1.6219, (0.05, 2.0))],
                                   ref_kcal=0.3640 * 627.509, note='N2: N 2+3+2 -> pseudo-N +2, outer 1+1 per atom'))(
                                   float(o.get('R', 2.0743)), float(o.get('rcN', 0.6))),
        'co_r':       lambda: (lambda RR, rcC, rcO: dict(
                                   nuclei=[('C', 0.0, 2, rcC), ('O', RR, 3, rcO)],
                                   domains=[('Cback', 1), ('Cbond', 1), ('Obond', 1), ('Oback', 2)],
                                   planes=[(0.0, 'fixed'), (0.5 * RR, 'free'), (RR, 'fixed')],
                                   frags=[('C', 1, [('C', 0.0, 2, rcC)], [('Cb', 1), ('Cf', 1)], [(0.0, 'fixed')]),
                                          ('O', 1, [('O', 0.0, 3, rcO)], [('Of', 1), ('Ob', 2)], [(0.0, 'fixed')])], nH=0,
                                   hfix=RR, hface=True, Robs=2.1322, pseudo=[('C', -1.3099, (0.05, 2.0)), ('O', -3.8100, (0.05, 2.0))],
                                   ref_kcal=0.4127 * 627.509, note='CO: C 2+2+2 -> +2 (1+1) | O 2+3+3 -> +3 (1 bond, 2 back)'))(
                                   float(o.get('R', 2.1322)), float(o.get('rcC', 0.8)), float(o.get('rcO', 0.5))),
        'co2_r':      lambda: (lambda RR, rcC, rcO: dict(
                                   nuclei=[('O', -RR, 3, rcO), ('C', 0.0, 2, rcC), ('O', RR, 3, rcO)],
                                   domains=[('OLback', 2), ('OLbond', 1), ('CL', 1), ('CR', 1), ('ORbond', 1), ('ORback', 2)],
                                   planes=[(-RR, 'fixed'), (-0.5 * RR, 'mirror:3'), (0.0, 'fixed'), (0.5 * RR, 'free'), (RR, 'fixed')],
                                   frags=[('C', 1, [('C', 0.0, 2, rcC)], [('Cb', 1), ('Cf', 1)], [(0.0, 'fixed')]),
                                          ('O', 2, [('O', 0.0, 3, rcO)], [('Of', 1), ('Ob', 2)], [(0.0, 'fixed')])], nH=0,
                                   hfix=RR, hface=True, Robs=2.1921, pseudo=[('C', -1.3099, (0.05, 2.0)), ('O', -3.8100, (0.05, 2.0))],
                                   ref_kcal=None, note='CO2: O (2 back, 1 bond) | C 1+1 halves | O (1 bond, 2 back)'))(
                                   float(o.get('R', 2.1921)), float(o.get('rcC', 0.8)), float(o.get('rcO', 0.5))),
        'hcn_r':      lambda: (lambda CH, CN, rcC, rcN: dict(
                                   nuclei=[('H', -CH, 1, 0), ('C', 0.0, 2, rcC), ('N', CN, 2, rcN)],
                                   domains=[('H', 1), ('CL', 1), ('CR', 1), ('Nbond', 1), ('Nback', 1)],
                                   planes=[(-0.5 * CH, 'free'), (0.0, 'fixed'), (0.5 * CN, 'free'), (CN, 'fixed')],
                                   frags=[('C', 1, [('C', 0.0, 2, rcC)], [('Cb', 1), ('Cf', 1)], [(0.0, 'fixed')]),
                                          ('N', 1, [('N', 0.0, 2, rcN)], [('Nf', 1), ('Nb', 1)], [(0.0, 'fixed')])], nH=1,
                                   hfix=CN, hface=True, Robs=2.1792, geom=[('CN', 2.1792), ('CH', 2.0135)],
                                   pseudo=[('C', -1.3099, (0.05, 2.0)), ('N', -1.6219, (0.05, 2.0))],
                                   ref_kcal=None, note='HCN: H | C 1+1 halves | N (1 bond, 1 back)'))(
                                   float(o.get('CH', 2.0135)), float(o.get('CN', 2.1792)), float(o.get('rcC', 0.8)), float(o.get('rcN', 0.6))),
        'c2h2_r':     lambda: (lambda CC, CH, rcC: dict(
                                   nuclei=[('H', -0.5 * CC - CH, 1, 0), ('C', -0.5 * CC, 2, rcC), ('C', 0.5 * CC, 2, rcC), ('H', 0.5 * CC + CH, 1, 0)],
                                   domains=[('H', 1), ('C1L', 1), ('C1R', 1), ('C2L', 1), ('C2R', 1), ('H', 1)],
                                   planes=[(-0.5 * CC - 0.5 * CH, 'mirror:4'), (-0.5 * CC, 'fixed'), (0.0, 'fixed'), (0.5 * CC, 'fixed'),
                                           (0.5 * CC + 0.5 * CH, 'free')],
                                   frags=[('C', 2, [('C', 0.0, 2, rcC)], [('Cb', 1), ('Cf', 1)], [(0.0, 'fixed')])], nH=2,
                                   hfix=0.5 * CC, hface=True, Robs=2.2739, geom=[('CC', 2.2739), ('CH', 2.0088)],
                                   pseudo=[('C', -1.3099, (0.05, 2.0))], ref_kcal=None,
                                   note='C2H2: H | C 1+1 halves | C 1+1 halves | H'))(
                                   float(o.get('CC', 2.2739)), float(o.get('CH', 2.0088)), float(o.get('rcC', 0.8))),
        'beh2_split': lambda: dict(xh2('Be', 2, 1.392, True, 2.5066, 1), ref_kcal=None, X='Be', target=-1.0118,
                                   note='pseudo-Be +2, valence 1+1 (hemispheres), rc from IE1+IE2'),
        'beh2_hom':   lambda: dict(xh2('Be', 2, 1.2529, False, 2.5066, 1), ref_kcal=None, X='Be', target=-1.0118,
                                   note='pseudo-Be +2, valence one 2e domain, rc from IE1+IE2'),
        'ch2_hom':    lambda: dict(xh2('C', 2, 0.7854, False, 2.0372, 1), ref_kcal=None, X='C', target=-1.3099,
                                   note='2023 sketch model: C as +2 pseudo, 2e valence domain (linear HCH)'),
        'ch2_split':  lambda: dict(xh2('C', 2, 0.9131, True, 2.0372, 1), ref_kcal=None, X='C', target=-1.3099,
                                   note='C as +2 pseudo, valence 1+1 hemispheres (linear HCH)'),
        'ch2_4':      lambda: dict(xh2('C', 4, 0.9139, True, 2.0372, 2), ref_kcal=None, X='C', target=-5.4398,
                                   note='C as +4 pseudo (1s^2 removed), valence 2|2 hemispheres (linear HCH)'),
        # LiH: pseudo-Li +1 (1s^2 removed) with ONE valence electron, r_c from IE(Li) = 5.3917 eV = 0.19814 Ha
        # (from memory); H one electron; one free plane by density continuity.  Li on the face z = 0, H at
        # z = R on a cell centre.  Exp R_e 1.5949 A = 3.0139 a0 (R_obs below 3.0154, the value used in the
        # 3D runs), D_e 2.515 eV = 0.0924 Ha, mu 5.88 D Li+H-; all from memory, verify before quoting.
        'lih':        lambda: dict(nuclei=[('Li', 0.0, 1, float(o.get('rcLi', 1.6))), ('H', (R or 3.0154), 1, 0)],
                                   domains=[('Li', 1), ('H', 1)],
                                   planes=[(0.5 * (R or 3.0154) + 0.5 * float(o.get('rcLi', 1.6)), 'free')],
                                   frags=[('Li', 1, [('Li', 0.0, 1, float(o.get('rcLi', 1.6)))], [('Li', 1)], [])],
                                   nH=1, hfix=(R or 3.0154), Robs=3.0154, X='Li', target=-0.19814,
                                   ref_kcal=0.0924 * 627.509, note='LiH: pseudo-Li +1, r_c from IE(Li), 1e valence | H 1e'),
        # NaH: pseudo-Na +1 (1s2 2s2 2p6 removed), ONE valence electron, r_c from IE(Na) = 5.1391 eV = 0.18886 Ha;
        # exp R_e 1.8874 A = 3.5667 a0, D_e ~1.97 eV = 0.0724 Ha (from memory, verify before quoting)
        'nah':        lambda: dict(nuclei=[('Na', 0.0, 1, float(o.get('rcNa', 2.4))), ('H', (R or 3.5667), 1, 0)],
                                   domains=[('Na', 1), ('H', 1)],
                                   planes=[(0.5 * (R or 3.5667) + 0.5 * float(o.get('rcNa', 2.4)), 'free')],
                                   frags=[('Na', 1, [('Na', 0.0, 1, float(o.get('rcNa', 2.4)))], [('Na', 1)], [])],
                                   nH=1, hfix=(R or 3.5667), Robs=3.5667, X='Na', target=-0.18886, rcfit=(1.0, 3.5),
                                   ref_kcal=0.0724 * 627.509, note='NaH: pseudo-Na +1, r_c from IE(Na), 1e valence | H 1e'),
        # N2: pseudo-N +3 with its 2p^3 as ONE 3e domain per atom (the simulator's hom reading); midplane fixed.
        # r_c from IE1+IE2+IE3 = 14.534+29.601+47.445 eV = 3.3655 Ha.  Exp R_e 1.0977 A = 2.0743 a0, D_e 9.905 eV
        # = 0.3640 Ha (from memory, verify before quoting)
        'n2':         lambda: dict(nuclei=[('N', -0.5 * float(o.get('R', 2.0743)), 3, float(o.get('rcN', 0.7))),
                                           ('N', 0.5 * float(o.get('R', 2.0743)), 3, float(o.get('rcN', 0.7)))],
                                   domains=[('NL', 3), ('NR', 3)], planes=[(0.0, 'fixed')],
                                   frags=[('N', 2, [('N', 0.0, 3, float(o.get('rcN', 0.7)))], [('N', 3)], [])],
                                   nH=0, hfix=0.5 * float(o.get('R', 2.0743)), Robs=2.0743,
                                   pseudo=[('N', -3.3655, (0.2, 2.2))], ref_kcal=0.3640 * 627.509,
                                   note='N2: pseudo-N +3, 3e hom valence per atom, r_c from IE1..3'),
        # C2H2 (H-C#C-H): pseudo-C +4, 4e hom valence per C; r_c from IE1..4 = 5.4398 Ha.  Exp C-C 1.2033 A =
        # 2.2739 a0, C-H 1.0630 A = 2.0088 a0 (from memory).  Two bonds: each scanned with the other at obs.
        'c2h2':       lambda: (lambda CC, CH, rcC: dict(
                                   nuclei=[('H', -0.5 * CC - CH, 1, 0), ('C', -0.5 * CC, 4, rcC), ('C', 0.5 * CC, 4, rcC), ('H', 0.5 * CC + CH, 1, 0)],
                                   domains=[('H', 1), ('CL', 4), ('CR', 4), ('H', 1)],
                                   planes=[(-0.5 * CC - 0.5 * CH, 'mirror:2'), (0.0, 'fixed'), (0.5 * CC + 0.5 * CH, 'free')],
                                   frags=[('C', 2, [('C', 0.0, 4, rcC)], [('C', 4)], [])], nH=2,
                                   hfix=(0.5 * CC + CH) if o.get('hfixon') == 'CH' else 0.5 * CC,
                                   Robs=2.2739, geom=[('CC', 2.2739), ('CH', 2.0088)], pseudo=[('C', -5.4398, (0.3, 2.2))],
                                   ref_kcal=None, note='C2H2: pseudo-C +4 with 4e hom valence, r_c from IE1..4'))(
                                   float(o.get('CC', 2.2739)), float(o.get('CH', 2.0088)), float(o.get('rcC', 0.9))),
        # HCN (H-C#N): pseudo-C +4 (4e hom), pseudo-N +3 (3e hom), H.  Exp C-H 1.0655 A = 2.0135 a0, C-N 1.1532 A =
        # 2.1792 a0, mu 2.985 D (from memory).  Two free planes.
        'hcn':        lambda: (lambda CH, CN, rcC, rcN: dict(
                                   nuclei=[('H', -CH, 1, 0), ('C', 0.0, 4, rcC), ('N', CN, 3, rcN)],
                                   domains=[('H', 1), ('C', 4), ('N', 3)],
                                   planes=[(-0.5 * CH, 'free'), (0.5 * CN, 'free')],
                                   frags=[('C', 1, [('C', 0.0, 4, rcC)], [('C', 4)], []), ('N', 1, [('N', 0.0, 3, rcN)], [('N', 3)], [])],
                                   nH=1, hfix=CH if o.get('hfixon') == 'CH' else CN,
                                   Robs=2.1792, geom=[('CN', 2.1792), ('CH', 2.0135)],
                                   pseudo=[('C', -5.4398, (0.3, 2.2)), ('N', -3.3655, (0.2, 2.2))],
                                   ref_kcal=None, note='HCN: pseudo-C +4 (4e hom) | pseudo-N +3 (3e hom) | H'))(
                                   float(o.get('CH', 2.0135)), float(o.get('CN', 2.1792)), float(o.get('rcC', 0.9)), float(o.get('rcN', 0.7))),
        # HF: pseudo-F +2 with ONE 2e valence domain (a +1 pseudo-F cannot work: IE(F) 0.640 Ha exceeds hydrogen's
        # 0.5, which no hard core on a +1 kernel can reach).  r_c from IE1+IE2 = 17.423+34.971 eV = 1.9254 Ha.
        # Exp R_e 0.9168 A = 1.7325 a0, D_e 6.12 eV = 0.2250 Ha, mu 1.82 D H+F- (from memory, verify).
        'hf':         lambda: dict(nuclei=[('F', 0.0, 2, float(o.get('rcF', 0.5))), ('H', (R or 1.7325), 1, 0)],
                                   domains=[('F', 2), ('H', 1)],
                                   planes=[(0.5 * (R or 1.7325) + 0.5 * float(o.get('rcF', 0.5)), 'free')],
                                   frags=[('F', 1, [('F', 0.0, 2, float(o.get('rcF', 0.5)))], [('F', 2)], [])],
                                   nH=1, hfix=(R or 1.7325), Robs=1.7325, pseudo=[('F', -1.9254, (0.05, 1.5))],
                                   ref_kcal=0.2250 * 627.509, note='HF: pseudo-F +2, one 2e valence domain, r_c from IE1+IE2 | H'),
        # HF with fluorine as the Atom Simulator's 2+4+3, the 2+4 frozen into a pseudo-F +3 and the outer 3 written
        # as 2+1: a 2e domain behind F, a 1e bonding domain facing H, split by a plane through F (fixed, as BeH2's
        # 1+1).  Every domain <= 2 electrons.  r_c from IE1+IE2+IE3 = 17.4228+34.9708+62.7084 eV = 4.2299 Ha
        # (from memory, verify).  Exp as 'hf'.
        'hf21':       lambda: dict(nuclei=[('F', 0.0, 3, float(o.get('rcF', 0.6))), ('H', (R or 1.7325), 1, 0)],
                                   domains=[('Fback', 2), ('Fbond', 1), ('H', 1)],
                                   planes=[(0.0, 'fixed'), (0.5 * (R or 1.7325) + 0.5 * float(o.get('rcF', 0.6)), 'free')],
                                   frags=[('F', 1, [('F', 0.0, 3, float(o.get('rcF', 0.6)))], [('Fback', 2), ('Fbond', 1)], [(0.0, 'fixed')])],
                                   nH=1, hfix=(R or 1.7325), Robs=1.7325, pseudo=[('F', -4.2299, (0.05, 2.0))],
                                   ref_kcal=0.2250 * 627.509,
                                   note='HF: pseudo-F +3 (2+4 frozen), valence 2+1 (2e back | 1e bond) | H'),
        # H2, bare protons, split by the midplane (fixed by symmetry).  Exact R_e = 1.4011 a0, D_e 0.1745 Ha
        'h2':         lambda: dict(nuclei=[('H', -0.5 * (R or 1.4), 1, 0), ('H', 0.5 * (R or 1.4), 1, 0)],
                                   domains=[('HL', 1), ('HR', 1)], planes=[(0.0, 'fixed')],
                                   frags=[], nH=2, hfix=0.5 * (R or 1.4), ref_kcal=109.5, Robs=1.4011, Rkey='D',
                                   note='H2, bare protons, one electron per half-space'),
        # linear H3, bare protons, one electron per domain: the symmetric-stretch saddle of H + H2 is
        # at R = 1.757 a0 (exact; from memory).  The mol_fast record has no minimum outside 1.50 a0.
        'h3':         lambda: dict(nuclei=[('H', -(R or 1.757), 1, 0), ('Hc', 0.0, 1, 0), ('H', (R or 1.757), 1, 0)],
                                   domains=[('H', 1), ('Hc', 1), ('H', 1)],
                                   planes=[(-0.5 * (R or 1.757), 'mirror:1'), (0.5 * (R or 1.757), 'free')],
                                   frags=[], nH=3, hfix=(R or 1.757), ref_kcal=None, Robs=1.757,
                                   note='linear H3, bare protons, 1e per domain (central H on a cell face)'),
    }
    # ---- homonuclear X2 family: pseudo-X (charge ZX, r_c fitted to the IE sum it removes) whose outer shell is
    # [back, bond] along the axis (back = 2hom lone pair through a fixed plane at X, or none); the two bond
    # electrons as 2split (midplane, fixed by symmetry) or as ONE 2hom domain between the nuclei.  Only a sigma
    # pair is representable in axisymmetry.  Exp R_e (a0) / D_e (Ha) FROM MEMORY, verify before quoting.
    X2 = {'Li': (1, 0, -0.19814, 5.051, 0.0381, 1.99), 'Na': (1, 0, -0.18886, 5.818, 0.0276, 2.4),
          'B': (3, 2, -2.6233, 3.005, 0.110, 1.08), 'C': (2, 1, -1.3099, 2.348, 0.232, 0.9),
          'N': (3, 2, -3.3655, 2.0743, 0.3640, 0.8), 'O': (3, 2, -3.8100, 2.282, 0.1915, 0.6),
          'F': (3, 2, -4.2299, 2.668, 0.0610, 0.6),
          # Atom Simulator configurations: B 2+2+1 -> +1 [1] (IE1 8.298 eV); N 2+3+2 -> +2 [1|1] (IE1+IE2)
          'Bs': (1, 0, -0.30495, 3.005, 0.110, 1.5), 'Ns': (2, 1, -1.6219, 2.0743, 0.3640, 0.6),
          # 1D atoms with the second shell at 3 (2+4+.. over-binds 2-4%): O 2+3+1+2 -> +2 [1|1] (IE1+IE2 13.618+35.121 eV);
          # F 2+3+2+2 -> +2 [1|1] (IE1+IE2 17.4228+34.9708 eV).  An outer single electron (+1) cannot work: IE1 > 0.5 Ha.
          'Os': (2, 1, -1.7911, 2.282, 0.1915, 0.5), 'Fs': (2, 1, -1.9254, 2.668, 0.0610, 0.5)}
    def x2(key, bond):
        ZX, nb, target, Robs, De, rc0 = X2[key]; X = key[0] if key.endswith('s') else key
        RR = R or Robs; rc = float(o.get('rc' + X, rc0)); a = 0.5 * RR
        nuc = [(X, -a, ZX, rc), (X, a, ZX, rc)]
        if bond == 'split':
            dom = ([(X + 'Lb', nb)] if nb else []) + [(X + 'Lf', 1), (X + 'Rf', 1)] + ([(X + 'Rb', nb)] if nb else [])
            pl = ([(-a, 'fixed')] if nb else []) + [(0.0, 'fixed')] + ([(a, 'fixed')] if nb else [])
        else:
            dom = ([(X + 'Lb', nb)] if nb else []) + [('bond', 2)] + ([(X + 'Rb', nb)] if nb else [])
            pl = [(-a, 'fixed'), (a, 'fixed')] if nb else []
        fd = ([(X + 'b', nb)] if nb else []) + [(X + 'f', 1)]
        fp = [(0.0, 'fixed')] if nb else []
        return dict(nuclei=nuc, domains=dom, planes=pl, frags=[(X, 2, [(X, 0.0, ZX, rc)], fd, fp)], nH=0,
                    hfix=a, hface=True, Robs=Robs, pseudo=[(X, target, (0.05, 3.5))], X=X, target=target,
                    ref_kcal=De * 627.509,
                    note='%s2: pseudo-%s +%d, outer %s per atom, bond pair %s' % (X, X, ZX, ('%d|1' % nb) if nb else '1', bond))
    # C2 with pseudo-C +2 carrying its valence as ONE 2hom domain per atom (no back|bond split): two pairs meeting at the
    # midplane (fixed by symmetry); r_c fitted to IE1+IE2 in that mode.
    def c2pair():
        RR = R or 2.348; rc = float(o.get('rcC', 0.8)); a = 0.5 * RR
        return dict(nuclei=[('C', -a, 2, rc), ('C', a, 2, rc)], domains=[('CL', 2), ('CR', 2)], planes=[(0.0, 'fixed')],
                    frags=[('C', 2, [('C', 0.0, 2, rc)], [('C', 2)], [])], nH=0, hfix=a, hface=True, Robs=2.348,
                    pseudo=[('C', -1.3099, (0.05, 3.5))], X='C', target=-1.3099, ref_kcal=0.232 * 627.509,
                    note='C2: pseudo-C +2, valence one 2hom domain per atom, midplane')
    P['c2pair'] = c2pair
    # C2H2 with the C2 carbon (pseudo-C +2, 1+1 halves) and EVERY bond a 2hom pair: H electron + C outer half over a
    # bare proton, and the two facing C halves as one CC domain.  Exp CC 2.2739, CH 2.0088 a0 (from memory).
    def c2h2hom():
        CC = float(o.get('CC', 2.2739)); CH = float(o.get('CH', 2.0088)); rc = float(o.get('rcC', 0.91)); a = 0.5 * CC
        return dict(nuclei=[('H', -a - CH, 1, 0), ('C', -a, 2, rc), ('C', a, 2, rc), ('H', a + CH, 1, 0)],
                    domains=[('HC', 2), ('CC', 2), ('CH', 2)], planes=[(-a, 'fixed'), (a, 'fixed')],
                    frags=[('C', 2, [('C', 0.0, 2, rc)], [('Cb', 1), ('Cf', 1)], [(0.0, 'fixed')])], nH=2,
                    hfix=(a + CH) if o.get('hfixon') == 'CH' else a, hface=True, Robs=2.2739,
                    geom=[('CC', 2.2739), ('CH', 2.0088)], pseudo=[('C', -1.3099, (0.05, 2.0))], ref_kcal=None,
                    note='C2H2: pseudo-C +2 (1+1), all three bonds 2hom pairs')
    P['c2h2hom'] = c2h2hom
    # GENERIC LINEAR CHAIN, every bond a 2hom pair, every wall a FIXED plane through a heavy nucleus.
    # spec=H-C-N  bonds=2.0135,2.1792 (a0, in order).  Heavy atoms are pseudo-atoms: Li +1 (valence 1), Be/C/N +2 (1+1),
    # O +3 (2hom back + 1).  A middle heavy atom gives one electron to each bond (needs valence 2); a terminal one gives
    # one to its bond and keeps Z-1 behind it.  H is a bare proton whose electron is in its bond pair.
    CH_RC = {'Li': (1, 2.0008), 'Be': (2, 1.37), 'C': (2, 0.9098), 'N': (2, 0.5925), 'O': (3, 0.5925)}
    def chain():
        els = o.get('spec', 'H-C-N').split('-'); bl = [float(x) for x in o.get('bonds', '2.0135,2.1792').split(',')]
        assert len(bl) == len(els) - 1
        z = [0.0]
        for b in bl: z.append(z[-1] + b)
        c = 0.5 * (z[0] + z[-1]); z = [v - c for v in z]
        nuc, pl, dom = [], [], []
        for i, e in enumerate(els):
            if e == 'H': nuc.append(('H', z[i], 1, 0)); continue
            Z, rc0 = CH_RC[e]; rc = float(o.get('rc' + e, rc0)); nuc.append((e, z[i], Z, rc)); pl.append((z[i], 'fixed'))
        heavy = [i for i, e in enumerate(els) if e != 'H']
        if els[0] != 'H':
            nb = CH_RC[els[0]][0] - 1
            if nb: dom.append((els[0] + 'back', nb))
            else: pl = pl[1:]                                  # Li: no back domain, no plane
        for k in range(len(bl)): dom.append(('%s%s' % (els[k], els[k + 1]), 2))
        if els[-1] != 'H':
            nb = CH_RC[els[-1]][0] - 1
            if nb: dom.append((els[-1] + 'back', nb))
            else: pl = pl[:-1]
        for i in heavy[1:-1] if len(heavy) > 2 else []:
            assert CH_RC[els[i]][0] == 2, 'middle atom needs valence 2'
        frags = []
        for e in sorted(set(x for x in els if x != 'H')):
            Z, rc0 = CH_RC[e]; rc = float(o.get('rc' + e, rc0))
            fd, fp = ([(e, 1)], []) if Z == 1 else ([(e + 'b', Z - 1), (e + 'f', 1)], [(0.0, 'fixed')])
            frags.append((e, els.count(e), [(e, 0.0, Z, rc)], fd, fp))
        k = int(o.get('scan', 0)); hf = max(abs(z[k]), abs(z[k + 1]))
        return dict(nuclei=nuc, domains=dom, planes=pl, frags=frags, nH=els.count('H'), hfix=hf, hface=True,
                    Robs=bl[k], ref_kcal=None, note='chain %s, all bonds 2hom, fixed planes' % '-'.join(els))
    P['chain'] = chain
    for X in X2:
        for b in ('split', 'hom'):
            nm = ('%s2sim_%s' % (X[0].lower(), b)) if X.endswith('s') else ('%s2_%s' % (X.lower(), b))
            P[nm] = (lambda X=X, b=b: x2(X, b))
    if name not in P: raise SystemExit('presets: ' + ' '.join(P))
    return P[name]()


def run(name, o):
    p = preset(name, o)
    h = float(o.get('h', 0.1)); L = float(o.get('L', 12.0)); soft = float(o.get('soft', 0.0))
    t0 = time.time()
    iface = o.get('iface', 'hard')
    hfix = 0.0 if o.get('hfixed') == '1' else p['hfix']     # hfixed=1: h exactly as given, nuclei anywhere in a cell
    mol = Linear(p['nuclei'], p['domains'], p['planes'], h=h, L=L, soft=soft, hfix=hfix, name=name, iface=iface,
                 hface=p.get('hface', False), beta=float(o.get('beta', 0)),
                 beta_thru=(float(o['bthru']) if 'bthru' in o else None), curved=o.get('curved') == '1',
                 cfree=([int(x) for x in o['cfree'].split(',')] if 'cfree' in o else None))
    Rm = max(abs(n['z']) for n in mol.nuclei)
    for i, md in enumerate(mol.pmode):
        if md == 'free' and (name, iface, i) in PLANE_MEMO:
            mol.pk[i] = int(round(PLANE_MEMO[(name, iface, i)] * Rm / mol.h))
    mol._tie()
    print('%s: %s' % (name, p['note']))
    print('  realised: h=%.5f L=%.3f grid %dx%d  nuclei %s' % (mol.h, mol.L, mol.g.nz, mol.g.nr,
          '  '.join('%s(Z=%g rc=%g) z=%+.5f' % (n['label'], n['Z'], n['rc'], n['z']) for n in mol.nuclei)))
    print('  domains %s' % ' | '.join('%s:%de' % (d['label'], d['n']) for d in mol.domains), flush=True)
    m = mol.relax_curved() if mol.curved else mol.relax()
    frags = []
    for lab, cnt, nuc, dom, pl in p['frags']:
        a = Linear(nuc, dom, pl, h=mol.h, L=mol.L, soft=soft, hfix=0.0, name=lab, iface=iface,
                   beta=float(o.get('beta', 0)), beta_thru=(float(o['bthru']) if 'bthru' in o else None)); a.h = mol.h
        r = a.scf(); frags.append(dict(label=lab, count=cnt, E=r['E'], T=r['T'], VK=r['VK'], Vee=r['Vee']))
        print('  atom %-3s E=%.6f  T=%.4f VK=%.4f Vee=%.4f' % (lab, r['E'], r['T'], r['VK'], r['Vee']))
    # one H reference per H nucleus, at the SAME position within its cell (so grid placement cancels)
    Hn = [n for n in mol.nuclei if n['Z'] == 1 and n['rc'] == 0]
    assert len(Hn) == p['nH'], (len(Hn), p['nH'])
    if p.get('Hrefs') is not None: Hn = Hn[:p['Hrefs']]       # ions: a proton leaves bare, no H-atom reference
    hcache = {}
    for n in Hn:
        off = round((n['z'] - mol.g.zmin) % mol.h, 9)
        if off not in hcache: hcache[off] = H_atom(mol.h, mol.L, soft, z0=off).scf()
        rH = hcache[off]
        frags.append(dict(label='H', count=1, E=rH['E'], T=rH['T'], VK=rH['VK'], Vee=0.0, z0=off))
    print('  atom H   E=%s  (in-cell offsets %s)' % (' '.join('%.6f' % hcache[k]['E'] for k in hcache),
          ' '.join('%.4f' % k for k in hcache)))
    Esep = sum(f['count'] * f['E'] for f in frags)
    Ea = Esep - m['E']; Eai = Esep - m['interp']['E']
    print('  molecule E=%.6f (interp %.6f)  T=%.4f VK=%.4f Vee=%.4f Vnn=%.4f  planes %s (interp %s)' % (
        m['E'], m['interp']['E'], m['T'], m['VK'], m['Vee'], m['Vnn'],
        ' '.join('%+.4f' % s for s in m['planes']), ' '.join('%+.4f' % s for s in m['interp']['planes'])))
    print('  ATOMISATION %.5f Ha = %.1f kcal/mol = %.3f eV  (interp %.1f kcal/mol)   %.0f s' % (
        Ea, Ea * KCAL, Ea * EV, Eai * KCAL, time.time() - t0), flush=True)
    if 'E_read' in m:
        print('  WEAK interface: E_read (mol_fast readout, wd=1) = %.6f   overhang charge per domain %s' % (
            m['E_read'], ' '.join('%.4f' % x for x in m['overhang'])))
    cz, muD = mol.moments()
    print('  centroids %s   mu_z = %+.3f D' % (' '.join('%s:%+.3f' % (d['label'], c) for d, c in zip(mol.domains, cz)), muD))
    os.makedirs(OUTDIR, exist_ok=True)
    tag = name + ''.join('_%s%s' % (k, v) for k, v in sorted(o.items()))
    out = dict(meta=mol.meta(), note=p['note'], params=o, frames=mol.frames,
               result=dict(E=m['E'], E_interp=m['interp']['E'], T=m['T'], VK=m['VK'], Vee=m['Vee'], Vnn=m['Vnn'],
                           planes=m['planes'], planes_interp=m['interp']['planes'], Tdom=m['Tdom'], Vdom=m['Vdom'],
                           frags=frags, E_sep=Esep, atomisation_Ha=Ea, atomisation_kcal=Ea * KCAL,
                           atomisation_interp_kcal=Eai * KCAL, seconds=time.time() - t0,
                           centroids=cz, mu_D=muD, curved=m.get('curved'), E_read=m.get('E_read'), overhang=m.get('overhang'), iface=iface, ref_kcal=p.get('ref_kcal'),
                           cmd='python3 tools/axisym_linear.py ' + ' '.join([name] + ['%s=%s' % kv for kv in sorted(o.items())])))
    fn = os.path.join(OUTDIR, tag + '.json')
    with open(fn, 'w') as f: json.dump(out, f)
    idx = os.path.join(OUTDIR, 'index.json')
    runs = json.load(open(idx)) if os.path.exists(idx) else []
    runs = [r for r in runs if r['file'] != tag + '.json'] + [dict(file=tag + '.json', name=name, params=o,
            atomisation_kcal=round(Ea * KCAL, 2), E=m['E'], when=time.strftime('%Y-%m-%d %H:%M'))]
    json.dump(runs, open(idx, 'w'), indent=1)
    print('  wrote', fn)
    return dict(E=m['E'], E_interp=m['interp']['E'], Ea=Ea, Eai=Eai, h=mol.h, R=max(abs(n['z']) for n in mol.nuclei),
                file=tag + '.json', planes=m['planes'])


def pseudo_list(p):
    """[(X, target, (lo, hi))] -- every pseudo-atom whose r_c is fitted to its own IE sum"""
    if p.get('pseudo'): return p['pseudo']
    if p.get('target') is not None: return [(p['X'], p['target'], p.get('rcfit', (0.3, 2.2)))]
    return []


def fit_rc(name, X, target, iface, h=0.1, lo=0.3, hi=2.2, L=12.0, verbose=False):
    """bisect X's r_c so the pseudo-atom (in the preset's valence mode and THIS interface type) has E = target"""
    p = preset(name, {}); ref = Linear(p['nuclei'], p['domains'], p['planes'], h=h, L=L, hfix=p['hfix'], hface=p.get('hface', False))
    lab, cnt, nuc, dom, pl = [f for f in p['frags'] if f[0] == X][0]
    def E(rc):
        e = Linear([(nuc[0][0], 0.0, nuc[0][2], rc)], dom, pl, h=ref.h, L=ref.L, hfix=0.0, iface=iface).scf()['E']
        if verbose: print('    fit %s %s rc=%.4f E=%.6f' % (X, iface, rc, e), flush=True)
        return e
    if not (E(lo) < target < E(hi)): raise RuntimeError('r_c(%s) target %.4f not bracketed in [%g, %g]' % (X, target, lo, hi))
    for _ in range(9):
        m = 0.5 * (lo + hi)
        if E(m) < target: lo = m
        else: hi = m
    return 0.5 * (lo + hi)


def protocol(name, o):
    """GEOMETRY from the SHARP interface (an R scan per bond, the other bonds at their observed values; h ladder,
    h fitted to the scanned bond, extrapolated to h -> 0); ENERGY from the WEAK interface at h = 0.1 EXACTLY, at
    the observed geometry and at the computed one.  Each pseudo-atom's r_c is fitted to its own IE sum,
    separately for each interface type."""
    t0 = time.time(); p = preset(name, {})
    geom = p.get('geom') or [('R', p['Robs'])]
    ladder = [float(x) for x in o.get('hl', '0.1,0.07').split(',')]
    base = {k: v for k, v in o.items() if k not in ('hl',)}
    rc = {'hard': {}, 'weak': {}}
    for X, target, (lo, hi) in pseudo_list(p):
        for iface in ('hard', 'weak'):
            rc[iface][X] = fit_rc(name, X, target, iface, lo=lo, hi=hi)
            print('r_c[%s][%s] = %.4f  (atom E = %.4f, IE sum)' % (iface, X, rc[iface][X], target), flush=True)
    def opts(iface, **kw):
        d = dict(base, iface=iface, **{k: str(v) for k, v in kw.items()})
        for X, v in rc[iface].items(): d['rc' + X] = '%.4f' % v
        return d
    params = []
    for P, Pobs in geom:
        geo = []
        for h in ladder:
            pts = {}; files = {}
            def at(R):
                R = round(R, 4)
                if R not in pts:
                    r = run(name, opts('hard', h=h, hfixon=P, **{P: R})); pts[R] = r['Eai']; files[R] = r['file']
                    print('  GEOM %s h=%.3f %s=%.4f  E_at=%.6f Ha (h realised %.5f)' % (P, h, P, R, r['Eai'], r['h']), flush=True)
                return pts[R]
            step = 0.06 * Pobs
            for R in (Pobs - step, Pobs, Pobs + step): at(R)
            for _ in range(8):                         # walk until the maximum of E_at is interior
                ks = sorted(pts); best = max(ks, key=lambda k: pts[k])
                if best == ks[0]: at(best - step)
                elif best == ks[-1]: at(best + step)
                else: break
            ks = sorted(pts); best = max(ks, key=lambda k: pts[k])
            at(best - step / 2); at(best + step / 2)      # refine around the best point
            ks = sorted(pts); best = max(ks, key=lambda k: pts[k]); i = ks.index(best)
            sel = ks[max(i - 1, 0): i + 2]
            c = np.polyfit(sel, [pts[k] for k in sel], 2); Rm = -c[1] / (2 * c[0])
            geo.append(dict(h=h, R=Rm, Ea=float(np.polyval(c, Rm)), k=-2 * c[0], pts={str(k): v for k, v in pts.items()},
                            files={str(k): v for k, v in files.items()}))
            print('  GEOM %s h=%.3f  min = %.4f a0  (obs %.4f, %+.1f%%)' % (P, h, Rm, Pobs, 100 * (Rm / Pobs - 1)), flush=True)
        R0 = float(np.polyfit([g['h'] for g in geo], [g['R'] for g in geo], 1)[1]) if len(geo) >= 2 else geo[0]['R']
        params.append(dict(name=P, obs=Pobs, R0=R0, geometry=geo))
    en = {}
    for lab, kw in (('obs', {}), ('hard', {q['name']: round(q['R0'], 4) for q in params})):
        r = run(name, opts('weak', h=0.1, hfixed=1, **kw))
        en[lab] = dict(R=kw.get(params[0]['name'], params[0]['obs']), geom={q['name']: kw.get(q['name'], q['obs']) for q in params},
                       Ea=r['Ea'], Eai=r['Eai'], file=r['file'])
    q0 = params[0]
    out = dict(name=name, note=p['note'], Robs=q0['obs'], rc=rc, ladder=ladder, params=params,
               cmd='python3 tools/axisym_linear.py protocol ' + ' '.join([name] + ['%s=%s' % kv for kv in sorted(o.items())]),
               geometry=q0['geometry'], R_hard=q0['R0'], energy=en, ref_kcal=p.get('ref_kcal'), seconds=time.time() - t0)
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(out, open(os.path.join(OUTDIR, 'protocol_%s.json' % name), 'w'), indent=1)
    import glob                                      # index for static hosting (no directory listing on Pages)
    json.dump(sorted(os.path.basename(f) for f in glob.glob(os.path.join(OUTDIR, 'protocol_*.json'))),
              open(os.path.join(OUTDIR, 'protocols.json'), 'w'))
    print('\n==== PROTOCOL %s  (%s)' % (name, p['note']))
    for X in rc['hard']: print('  r_c(%s) from atom IE sum: sharp %.4f  weak %.4f' % (X, rc['hard'][X], rc['weak'][X]))
    for q in params:
        for g in q['geometry']: print('  geometry (sharp)  %s h=%.3f  min=%.4f a0' % (q['name'], g['h'], g['R']))
        print('  geometry (sharp)  %s h->0  %.4f a0 = %.4f A   obs %.4f a0  (%+.1f%%)' % (q['name'], q['R0'], q['R0'] * 0.529177,
              q['obs'], 100 * (q['R0'] / q['obs'] - 1)))
    for lab in ('obs', 'hard'):
        e = en[lab]
        print('  energy (weak, h=0.1) at %-4s geometry %s:  %.5f Ha (interp %.5f)%s' % (lab, ' '.join('%s=%.4f' % kv for kv in e['geom'].items()),
              e['Ea'], e['Eai'], '' if not p.get('ref_kcal') else '   exp %.4f Ha' % (p['ref_kcal'] / KCAL)))
    print('  %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a or a[0] == 'list': preset('?', {})
    if a[0] == 'protocol':
        protocol(a[1], dict(x.split('=', 1) for x in a[2:])); sys.exit()
    if a[0] == 'fitrc':
        # fitrc PRESET TARGET_Ha [lo hi] [h=..]: bisect the X r_c so the pseudo-atom E = TARGET
        name, target = a[1], float(a[2]); lo, hi = float(a[3]), float(a[4])
        o = dict(x.split('=', 1) for x in a[5:]); h = float(o.get('h', 0.1)); L = float(o.get('L', 12.0))
        p = preset(name, o); ref = Linear(p['nuclei'], p['domains'], p['planes'], h=h, L=L, hfix=p['hfix'])
        lab, cnt, nuc, dom, pl = p['frags'][0]
        def E(rc):
            a_ = Linear([(nuc[0][0], 0.0, nuc[0][2], rc)], dom, pl, h=ref.h, L=ref.L, hfix=0.0, iface=o.get('iface', 'hard'))
            e = a_.scf()['E']; print('  rc=%.4f  E=%.6f' % (rc, e), flush=True); return e
        elo, ehi = E(lo), E(hi)
        if not (elo < target < ehi): raise SystemExit('target not bracketed')
        for _ in range(9):
            m = 0.5 * (lo + hi)
            if E(m) < target: lo = m
            else: hi = m
        print('%s h=%.5f  rc* = %.4f  (bracket %.4f..%.4f)  target %.6f' % (name, ref.h, 0.5 * (lo + hi), lo, hi, target))
        sys.exit()
    run(a[0], dict(x.split('=', 1) for x in a[1:]))
