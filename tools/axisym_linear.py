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
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from axisym_realqm import Grid, masked_stiffness
from axisym_hom2 import _poisson_setup

KCAL, EV = 627.509, 27.2114
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(HERE, 'axisym_runs')


class Linear:
    def __init__(self, nuclei, domains, planes, h=0.1, L=12.0, soft=0.0, hfix=None, name='mol', iface='hard'):
        self.name = name; self.iface = iface
        zs = [abs(n[1]) for n in nuclei]
        ref = hfix if hfix is not None else max(zs)
        if ref > 0:
            k = int(round(ref / h - 0.5)); h = ref / (k + 0.5)
        self.h = h
        nL = int(round(L / h)); L = nL * h; self.L = L
        self.g = g = Grid(L, L, h, zmin=-L)
        self.zc = g.Z.ravel(); rr = g.R.ravel()
        self.nuclei = [dict(label=a, z=float(z), Z=float(Z), rc=float(rc)) for a, z, Z, rc in nuclei]
        self.domains = [dict(label=a, n=int(n)) for a, n in domains]
        assert len(planes) == len(domains) - 1
        K = np.zeros(g.N); ok = np.ones(g.N, bool)
        for nu in self.nuclei:
            d = np.sqrt((self.zc - nu['z']) ** 2 + rr ** 2)
            K += nu['Z'] / np.sqrt(d ** 2 + soft ** 2)
            if nu['rc'] > 0: ok &= d > nu['rc']
        self.K, self.ok, self.soft = K, ok, soft
        self.Vnn = sum(a['Z'] * b['Z'] / abs(a['z'] - b['z'])
                       for i, a in enumerate(self.nuclei) for b in self.nuclei[i + 1:])
        # planes as integer FACE indices measured from z = 0
        self.pmode = [m for _, m in planes]
        self.pk = [int(round(z / h)) for z, _ in planes]
        self._tie()
        self.solveP, self.bnd, self.rrf = _poisson_setup(g)
        self.w = 1.0 / np.sqrt(g.V); self.W = sp.diags(self.w)
        self.u = [None] * len(domains); self.frames = []

    def _tie(self):
        for i, m in enumerate(self.pmode):
            if m.startswith('mirror:'): self.pk[i] = -self.pk[int(m.split(':')[1])]

    def masks(self):
        edges = [-np.inf] + [k * self.h for k in self.pk] + [np.inf]
        return [(self.zc > edges[k]) & (self.zc < edges[k + 1]) & self.ok for k in range(len(self.domains))]

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

    def eig(self, S, Veff, keep, u0):
        A = (self.W @ (S + sp.diags(Veff * self.g.V)).tocsr() @ self.W).tocsr()[keep][:, keep]
        v0 = None if u0 is None else (u0 / self.w)[keep]
        if v0 is not None and not np.any(v0): v0 = None
        val, vec = eigsh(A, k=1, which='SA', tol=1e-10, maxiter=200000, v0=v0)
        y = np.zeros(self.g.N); y[keep] = vec[:, 0]
        u = self.w * y; u /= np.sqrt(np.sum(u * u * self.g.V))
        return -u if np.sum(u) < 0 else u

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
        n = [d['n'] for d in self.domains]
        u = [None if self.u[k] is None else np.where(np.isin(np.arange(self.g.N), keep[k]), self.u[k], 0.0)
             for k in range(len(M))]
        P = None; Eold = None
        for it in range(iters):
            for k in range(len(M)):
                Veff = -self.K.copy()
                if P is not None:
                    Veff += (n[k] - 1) * P[k] + sum(n[l] * P[l] for l in range(len(M)) if l != k)
                u[k] = self.eig(S[k], Veff * Wd[k], keep[k], u[k])     # potential gated by w (mol_fast: *wc)
            nP = [self.P(x) for x in u]
            P = nP if P is None else [(1 - mix) * a + mix * b for a, b in zip(P, nP)]
            e = self.energy(u, S)
            if Eold is not None and abs(e['E'] - Eold) < tol: break
            Eold = e['E']
        self.u = u; self.S = S
        e = self.energy(u, S); e['iters'] = it + 1; e['planes'] = [k * self.h for k in self.pk]
        e['J'] = [self.jump(i) for i in range(len(self.pk))]
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
        out = dict(E=Ttot + VK + Vee + self.Vnn, T=Ttot, VK=VK, Vee=Vee, Vnn=self.Vnn, Tdom=T, Vdom=Vk)
        if self.iface == 'weak':
            Vk1, Vee1 = parts([np.ones(self.g.N)] * len(u))
            out['E_read'] = Ttot + sum(a * b for a, b in zip(n, Vk1)) + Vee1 + self.Vnn
            out['overhang'] = [float(np.sum(x * x * (1 - w) * V)) for x, w in zip(u, Wd)]
        return out

    def jump(self, i):
        """area-weighted density jump (left domain i) - (right domain i+1) across plane i"""
        g = self.g; s = self.pk[i] * self.h
        a = np.where(np.abs(self.zc - (s - g.h / 2)) < 1e-9)[0]
        b = a + g.nr                                          # the cell across the face, same rho
        keep = self.ok[a] & self.ok[b]                        # both sides outside every core
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
        for nu in self.nuclei:
            if nu['z'] <= z0: lo = max(lo, nu['z'] + nu['rc'])
            else: hi = min(hi, nu['z'] - nu['rc'])
        lo = max(lo, zl); hi = min(hi, zr)
        return int(np.floor(lo / h)) + 1, int(np.ceil(hi / h)) - 1

    def relax(self, verbose=True, sweeps=6):
        free = [i for i, m in enumerate(self.pmode) if m == 'free']
        res = self.scf()
        if verbose: self.report(res)
        if not free: return dict(res, interp=dict(res))
        last = {}
        for sw in range(sweeps):
            changed = False
            for i in free:
                lo, hi = self.bracket(i)
                cache = {}
                def f(k):
                    if k not in cache:
                        self.pk[i] = k; self._tie(); r = self.scf(); cache[k] = r
                        if verbose: self.report(r)
                    return cache[k]['J'][i]
                old = self.pk[i]
                if f(lo) < 0:   hi = lo
                elif f(hi) > 0: lo = hi
                else:
                    while hi - lo > 1:
                        m = (lo + hi) // 2
                        if f(m) > 0: lo = m
                        else: hi = m
                a, b = cache[lo], cache[hi]
                last[i] = (a, b)
                self.pk[i] = lo if abs(a['J'][i]) < abs(b['J'][i]) else hi; self._tie()
                if self.pk[i] != old: changed = True
            if not changed: break
        res = self.scf()
        # per-plane linear interpolation to J = 0 between the bracketing faces
        dE = 0.0; s_int = list(res['planes'])
        for i, (a, b) in last.items():
            if a is b or a['J'][i] == b['J'][i]: continue
            t = a['J'][i] / (a['J'][i] - b['J'][i])
            Ei = a['E'] + t * (b['E'] - a['E'])
            dE += Ei - res['E']; s_int[i] = a['planes'][i] + t * (b['planes'][i] - a['planes'][i])
        for i, m in enumerate(self.pmode):
            if m.startswith('mirror:'): s_int[i] = -s_int[int(m.split(':')[1])]
        res['interp'] = dict(E=res['E'] + dE, planes=s_int)
        return res

    def moments(self):
        """per-domain centroid on the axis, and the axial dipole (Debye, + = toward +z positive end)"""
        V = self.g.V; cz = [float(np.sum(self.zc * u * u * V)) for u in self.u]
        mu = sum(n['Z'] * n['z'] for n in self.nuclei) - sum(d['n'] * c for d, c in zip(self.domains, cz))
        return cz, mu * 2.541746

    def report(self, r):
        print('    planes %s  J %s  E=%.6f  (%d it)' % (
            ' '.join('%+.4f' % p for p in r['planes']), ' '.join('%+.5f' % j for j in r['J']),
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
    P = {
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
    if name not in P: raise SystemExit('presets: ' + ' '.join(P))
    return P[name]()


def run(name, o):
    p = preset(name, o)
    h = float(o.get('h', 0.1)); L = float(o.get('L', 12.0)); soft = float(o.get('soft', 0.0))
    t0 = time.time()
    iface = o.get('iface', 'hard')
    hfix = 0.0 if o.get('hfixed') == '1' else p['hfix']     # hfixed=1: h exactly as given, nuclei anywhere in a cell
    mol = Linear(p['nuclei'], p['domains'], p['planes'], h=h, L=L, soft=soft, hfix=hfix, name=name, iface=iface)
    print('%s: %s' % (name, p['note']))
    print('  realised: h=%.5f L=%.3f grid %dx%d  nuclei %s' % (mol.h, mol.L, mol.g.nz, mol.g.nr,
          '  '.join('%s(Z=%g rc=%g) z=%+.5f' % (n['label'], n['Z'], n['rc'], n['z']) for n in mol.nuclei)))
    print('  domains %s' % ' | '.join('%s:%de' % (d['label'], d['n']) for d in mol.domains), flush=True)
    m = mol.relax()
    frags = []
    for lab, cnt, nuc, dom, pl in p['frags']:
        a = Linear(nuc, dom, pl, h=mol.h, L=mol.L, soft=soft, hfix=0.0, name=lab, iface=iface); a.h = mol.h
        r = a.scf(); frags.append(dict(label=lab, count=cnt, E=r['E'], T=r['T'], VK=r['VK'], Vee=r['Vee']))
        print('  atom %-3s E=%.6f  T=%.4f VK=%.4f Vee=%.4f' % (lab, r['E'], r['T'], r['VK'], r['Vee']))
    # one H reference per H nucleus, at the SAME position within its cell (so grid placement cancels)
    Hn = [n for n in mol.nuclei if n['Z'] == 1 and n['rc'] == 0]
    assert len(Hn) == p['nH'], (len(Hn), p['nH'])
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
                           centroids=cz, mu_D=muD, E_read=m.get('E_read'), overhang=m.get('overhang'), iface=iface, ref_kcal=p.get('ref_kcal'),
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


def fit_rc(name, target, iface, h=0.1, lo=0.3, hi=2.2, L=12.0, verbose=False):
    """bisect the X r_c so the pseudo-atom (in the preset's valence mode and THIS interface type) has E = target"""
    p = preset(name, {}); ref = Linear(p['nuclei'], p['domains'], p['planes'], h=h, L=L, hfix=p['hfix'])
    lab, cnt, nuc, dom, pl = p['frags'][0]
    def E(rc):
        e = Linear([(nuc[0][0], 0.0, nuc[0][2], rc)], dom, pl, h=ref.h, L=ref.L, hfix=0.0, iface=iface).scf()['E']
        if verbose: print('    fit %s rc=%.4f E=%.6f' % (iface, rc, e), flush=True)
        return e
    if not (E(lo) < target < E(hi)): raise RuntimeError('r_c target %.4f not bracketed in [%g, %g]' % (target, lo, hi))
    for _ in range(9):
        m = 0.5 * (lo + hi)
        if E(m) < target: lo = m
        else: hi = m
    return 0.5 * (lo + hi)


def protocol(name, o):
    """GEOMETRY from the SHARP interface (R scan, h ladder, h fitted to each R, extrapolated to h -> 0);
    ENERGY from the WEAK interface at h = 0.1 EXACTLY, at the observed R and at the sharp-interface R.
    r_c (if the preset has a pseudo-atom) is fitted to the atom's IE sum separately for each interface type."""
    t0 = time.time(); p = preset(name, {}); Robs = p['Robs']; rk = 'R'
    ladder = [float(x) for x in o.get('hl', '0.1,0.07').split(',')]
    base = {k: v for k, v in o.items() if k not in ('hl',)}
    rc = {}
    if p.get('target') is not None:
        for iface in ('hard', 'weak'):
            rc[iface] = fit_rc(name, p['target'], iface)
            print('r_c[%s] = %.4f  (atom E = %.4f, IE sum)' % (iface, rc[iface], p['target']), flush=True)
    def opts(iface, **kw):
        d = dict(base, iface=iface, **{k: str(v) for k, v in kw.items()})
        if iface in rc: d['rc' + p['X']] = '%.4f' % rc[iface]
        return d
    geo = []
    for h in ladder:
        pts = {}; files = {}
        def at(R):
            R = round(R, 4)
            if R not in pts:
                r = run(name, opts('hard', R=R, h=h)); pts[R] = r['Eai']; files[R] = r['file']
                print('  GEOM h=%.3f R=%.4f  E_at=%.6f Ha (h realised %.5f)' % (h, R, r['Eai'], r['h']), flush=True)
            return pts[R]
        step = 0.06 * Robs; Rs = [Robs - step, Robs, Robs + step]
        for R in Rs: at(R)
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
        geo.append(dict(h=h, R=Rm, Ea=float(np.polyval(c, Rm)), k=-2 * c[0], pts=pts, files=files))
        print('  GEOM h=%.3f  R_min = %.4f a0  (obs %.4f, %+.1f%%)' % (h, Rm, Robs, 100 * (Rm / Robs - 1)), flush=True)
    if len(geo) >= 2:
        a = np.polyfit([g['h'] for g in geo], [g['R'] for g in geo], 1); R0 = float(a[1])
    else: R0 = geo[0]['R']
    en = {}
    for lab, R in (('obs', Robs), ('hard', R0)):
        r = run(name, opts('weak', R=round(R, 4), h=0.1, hfixed=1))
        en[lab] = dict(R=R, Ea=r['Ea'], Eai=r['Eai'], file=r['file'])
    out = dict(name=name, note=p['note'], Robs=Robs, rc=rc, ladder=ladder, cmd='python3 tools/axisym_linear.py protocol ' +
               ' '.join([name] + ['%s=%s' % kv for kv in sorted(o.items())]),
               geometry=[dict(h=g['h'], R=g['R'], Ea=g['Ea'], k=g['k'], pts={str(k): v for k, v in g['pts'].items()},
                              files={str(k): v for k, v in g['files'].items()}) for g in geo],
               R_hard=R0, energy=en, ref_kcal=p.get('ref_kcal'), seconds=time.time() - t0)
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(out, open(os.path.join(OUTDIR, 'protocol_%s.json' % name), 'w'), indent=1)
    # index for static hosting (GitHub Pages has no directory listing)
    import glob
    json.dump(sorted(os.path.basename(f) for f in glob.glob(os.path.join(OUTDIR, 'protocol_*.json'))),
              open(os.path.join(OUTDIR, 'protocols.json'), 'w'))
    print('\n==== PROTOCOL %s  (%s)' % (name, p['note']))
    if rc: print('  r_c from atom IE sum: sharp %.4f  weak %.4f' % (rc['hard'], rc['weak']))
    for g in geo: print('  geometry (sharp)  h=%.3f  R_min=%.4f a0' % (g['h'], g['R']))
    print('  geometry (sharp)  h->0  R=%.4f a0 = %.4f A   obs %.4f a0  (%+.1f%%)' % (R0, R0 * 0.529177, Robs, 100 * (R0 / Robs - 1)))
    for lab in ('obs', 'hard'):
        e = en[lab]
        print('  energy (weak, h=0.1) at %-4s R=%.4f:  %.1f kcal/mol (interp %.1f)%s' % (lab, e['R'], e['Ea'] * KCAL, e['Eai'] * KCAL,
              '' if not p.get('ref_kcal') else '   exp %s' % p['ref_kcal']))
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
