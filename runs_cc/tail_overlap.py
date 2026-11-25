"""Can the long-tail over-binding of a free-wall pair be removed WITHOUT a fitted constant? (2026-11-25)
RealQM's wall couples two domains through the density AT the wall, ~e^{-kR}; exchange in stdQM goes as overlap
squared, ~e^{-2kR}. Here the overlap S is computed from RealQM's OWN converged densities: each one-electron domain's
u is continued across the wall at its own decay rate k = sqrt(-2 eps) (eps = its converged orbital eigenvalue),
u_ext(r,z) = u(r, last cell) exp(-k |z - z_last|), and S = INT u_a,ext u_b,ext dV. No fitted number.
Candidates (form chosen from the H2 GVB numbers of 2026-11-20, so H2 is NOT a test; the test is the transfer):
  D*S   (wall binding weighted by the overlap -> ~e^{-2kR} at long range)      D*S^2 for comparison.
H2: R scan vs FCI (read from h2_force_profile_h*.out). LiH, NaH, Li2, Na2: at their R_e and r_c from
beta_transfer_h*.out, vs the experimental D_e listed there (FROM MEMORY, flagged).   usage: python3 tail_overlap.py h"""
import sys, re, numpy as np
sys.path.insert(0, '/Users/cgjoh/Development/H2O/tools')
from axisym_linear import Linear, preset, H_atom
h = float(sys.argv[1]); L = 12.0; tag = {0.1: 'h01', 0.05: 'h005'}[h]
D = '/Users/cgjoh/Development/H2O/runs_cc/'

def point(name, o, R):
    p = preset(name, dict(o, R=str(R)))
    m = Linear(p['nuclei'], p['domains'], p['planes'], h=h, L=L, hfix=p['hfix'], name=name, hface=p.get('hface', False))
    lam = []; e0 = m.eig
    def eig(*a, **k):
        u, l = e0(*a, **k); lam.append(l); return u, l
    m.eig = eig
    r = m.relax(verbose=False) if any(md == 'free' for _, md in p['planes']) else m.scf()
    lam.clear(); m.scf(); m.eig = e0; eps = lam[-len(m.domains):]   # one more scf from the converged state: last eps
    Es = 0.0
    for f in p['frags']:
        lab, cnt, nuc, dom, pl = f
        a = Linear(nuc, dom, pl, h=m.h, L=m.L, hfix=0.0, name=lab); a.h = m.h; Es += cnt * a.scf()['E']
    for n in [n for n in m.nuclei if n['Z'] == 1 and n['rc'] == 0]:
        Es += H_atom(m.h, m.L, 0.0, z0=round((n['z'] - m.g.zmin) % m.h, 9)).scf()['E']
    E = m.energy(m.u, m.S)['E']
    return Es - E, m, eps

def overlap(m, eps):
    """S between the two domains bounding the (only) plane, tails continued at k = sqrt(-2 eps)."""
    g = m.g; M = m.masks(); zw = m.pk[0] * m.h
    U = [u.reshape(g.nz, g.nr).copy() for u in m.u[:2]]; Mk = [x.reshape(g.nz, g.nr) for x in M[:2]]
    z = g.z; k = [np.sqrt(-2 * e) for e in eps[:2]]
    il = int(np.max(np.where(Mk[0].any(1))[0]))           # last z-row of the left domain
    ir = int(np.min(np.where(Mk[1].any(1))[0]))           # first z-row of the right domain
    ext = []
    for s, (u, kk) in enumerate(zip(U, k)):
        e = u.copy()
        if s == 0:
            for i in range(il + 1, g.nz): e[i] = u[il] * np.exp(-kk * (z[i] - z[il]))
        else:
            for i in range(0, ir): e[i] = u[ir] * np.exp(-kk * (z[ir] - z[i]))
        ext.append(e.ravel())
    return float(np.sum(ext[0] * ext[1] * g.V)), k

print('h=%.3f   tail-continued overlap S from RealQM densities, no fitted constant' % h, flush=True)
# H2 vs FCI
ex = {}
for line in open(D + 'h2_force_profile_%s.out' % tag):
    f = line.split()
    if len(f) == 7 and f[3] == '|': ex[round(float(f[0]), 2)] = -float(f[2])
print(' sys   R     D_RQ     S      k      D*S     D*S^2   exact', flush=True)
for R in (1.4, 2.0, 3.0, 4.0, 5.0, 6.0):
    d, m, eps = point('h2', {}, R); S, k = overlap(m, eps)
    print(' h2   %.2f  %.5f  %.4f  %.3f  %.5f  %.5f  %.5f' % (R, d, S, k[0], d*S, d*S*S, ex[round(R, 2)]), flush=True)
# transfer: R_e, r_c and exact D_e from the beta_transfer run at the same h
for line in open(D + 'beta_transfer_%s.out' % tag):
    mm = re.match(r'(\w+)\s+(?:(rc\w+)=([\d.]+)\s+)?R_e ([\d.]+).*exact ([\d.]+)', line)
    if not mm or mm.group(1) == 'h2': continue
    name, rk, rv, Re, Dex = mm.group(1), mm.group(2), mm.group(3), float(mm.group(4)), float(mm.group(5))
    o = {rk: rv} if rk else {}
    d, m, eps = point(name, o, Re); S, k = overlap(m, eps)
    print(' %-9s R_e %.3f  D_RQ %.5f  S %.4f  k %.3f/%.3f  D*S %.5f (%+.0f%%)  D*S^2 %.5f (%+.0f%%)  exact %.4f' % (
          name, Re, d, S, k[0], k[1], d*S, 100*(d*S/Dex-1), d*S*S, 100*(d*S*S/Dex-1), Dex), flush=True)
print('TODONE')
