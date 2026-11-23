# Radial force on each H (bare proton or H) relative to its NEAREST heavy centre, from h2_beta's last FORCE line.
# Positive = pushed outward (bond wants to lengthen). Also the bond length actually built (check the geometry).
import sys, re, numpy as np
line = [l for l in open(sys.argv[1]) if 'FORCE ' in l and 'failed' not in l][-1]
ents = re.findall(r'(\S+)@\(([-\d.]+),([-\d.]+),([-\d.]+)\) F=\(([-\d.]+),([-\d.]+),([-\d.]+)\)', line)
P = [(e[0], np.array(list(map(float, e[1:4]))), np.array(list(map(float, e[4:7])))) for e in ents]
heavy = [p for p in P if not p[0].startswith('H')]; hs = [p for p in P if p[0].startswith('H')]
out = []
for el, x, F in hs:
    c = min(heavy, key=lambda q: np.linalg.norm(x - q[1])); d = x - c[1]; r = np.linalg.norm(d)
    out.append((r, F @ d / r))
rs = [o[0] for o in out]; fr = [o[1] for o in out]
print('H radial: R %s  Fr %s  mean Fr %+.4f' % (' '.join('%.4f' % v for v in sorted(set(np.round(rs, 4)))),
      ' '.join('%+.4f' % v for v in fr), np.mean(fr)))
# Bending component (for XH2): on each H, the force along the in-plane direction that OPENS the H-X-H angle
# (perpendicular to its own bond, away from the other H). Positive = the angle wants to open.
if len(hs) == 2:
    c = min(heavy, key=lambda q: np.linalg.norm(hs[0][1] - q[1]))
    bi = [(x - c[1]) / np.linalg.norm(x - c[1]) for _, x, _ in hs]
    ang = np.degrees(np.arccos(np.clip(bi[0] @ bi[1], -1, 1))); ft = []
    for i in range(2):
        e = bi[i]*(bi[i] @ bi[1-i]) - bi[1-i]; e /= np.linalg.norm(e); ft.append(hs[i][2] @ e)
    print('H bend: angle %.2f deg  F_open %+.4f %+.4f  mean %+.4f' % (ang, ft[0], ft[1], np.mean(ft)))
