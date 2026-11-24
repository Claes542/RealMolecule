"""RealQM's own 1s^2 core: ONE 2hom pair domain about a bare nucleus Z (He-like ion), each electron in the field of
the other (n-1 = 1), radial, solved to self-consistency.  Returns the screening term S(d) = 2/d - 2 V_1(d) so that the
valence sees K = (Z-2)/d + S(d)  (replaces the hydrogenic Clementi 2 exp(-2 zeta d)(zeta + 1/d)).  Nothing from HF."""
import numpy as np
from scipy.linalg import eigh_tridiagonal
_cache = {}
def _solve(Z, dr=0.0005, rmax=12.0):
    r = np.arange(1, int(rmax/dr)+1)*dr; VH = np.zeros_like(r); E = 0
    for it in range(200):
        d = 1/dr**2 - Z/r + VH; e = -0.5/dr**2*np.ones(len(r)-1)
        w, v = eigh_tridiagonal(d, e, select='i', select_range=(0, 0)); u = v[:, 0]/np.sqrt(dr)
        q = np.cumsum(u*u)*dr; VHn = q/r + (np.sum(u*u/r)*dr - np.cumsum(u*u/r)*dr)
        if np.max(abs(VHn - VH)) < 1e-11: break
        VH = 0.5*VH + 0.5*VHn
    T = 0.5*np.sum(np.diff(np.r_[0.0, u])**2)/dr; Eion = 2*(T - Z*np.sum(u*u/r)*dr) + np.sum(u*u*VH)*dr
    return r, VH, Eion, w[0]
def screen(Z):
    if Z not in _cache: _cache[Z] = _solve(Z)
    r, VH, _, _ = _cache[Z]
    def S(d):
        d = np.maximum(d, 1e-12); V = np.interp(d, r, VH, right=0.0)
        V = np.where(d > r[-1], 1/d, V)          # beyond the grid the pair is a point charge 1 per electron
        return 2/d - 2*V
    return S
if __name__ == '__main__':
    for Z, zc in [(3, 2.6906), (4, 3.6848), (5, 4.6795), (6, 5.6727)]:
        r, VH, Eion, eps = _solve(Z); S = screen(Z); d = np.array([0.2, 0.4, 0.8, 1.5])
        hyd = 2*np.exp(-2*zc*d)*(zc + 1/d)
        print('Z=%d  E(ion) %.5f  eps %.5f | S(d) real %s  Clementi %s' % (Z, Eion, eps, np.round(S(d), 4), np.round(hyd, 4)))
