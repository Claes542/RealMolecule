"""
Two-domain moving free-boundary flow, energy-conserving (rescaled coordinates)
=============================================================================

The genuine object: two NON-overlapping charge domains sharing a MOVING free
boundary a(t), evolved in real complex time WITHOUT grid-point transfers, so no
energy leak (the crude realqm_freeboundary.py lost ~1.5% per move).

Each domain is carried in its own rescaled coordinate (the validated moving-wall
machinery of realqm_movingwall.py):
    domain 1: [0,a],  xi=x/a,    psi_1 = a^{-1/2} chi_1(xi),   moving wall at xi=1
    domain 2: [a,L],  eta=(L-x)/b, b=L-a, psi_2 = b^{-1/2} chi_2(eta), moving wall at eta=1
with the unitary evolution
    i chi_t = -(1/2 s^2) chi'' + V chi - i (sdot/s)(u chi' + chi/2),  (s,u)=(a,xi) or (b,eta).

The free boundary a(t) is set at each instant by the VARIATIONAL (Bernoulli)
condition -- with Dirichlet interfaces the energy-density balance is slope
matching, |psi_1'(a)| = |psi_2'(a)|, i.e.
    a^{-3/2}|chi_1'(1)| = b^{-3/2}|chi_2'(1)|   =>   a = L r/(1+r),  r=(s1/s2)^{2/3},
a closed form from the current fields.  As the fields evolve the slopes change and
the boundary moves; adot is taken from the boundary's own motion.

Test: 1-D He-like atom (nucleus +Z at centre, two electron domains).  Relax to the
ground state (boundary -> centre by symmetry), then perturb and evolve, reading off
the boundary motion and -- the point -- the TOTAL ENERGY drift.

Units: atomic (hbar=1, m=1).
"""

from __future__ import annotations
import argparse
import numpy as np


def thomas(a, b, c, d):
    n = len(b); cp = np.zeros(n, complex); dp = np.zeros(n, complex)
    cp[0]=c[0]/b[0]; dp[0]=d[0]/b[0]
    for i in range(1,n):
        m=b[i]-a[i]*cp[i-1]; cp[i]=c[i]/m; dp[i]=(d[i]-a[i]*dp[i-1])/m
    x=np.zeros(n,complex); x[-1]=dp[-1]
    for i in range(n-2,-1,-1): x[i]=dp[i]-cp[i]*x[i+1]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=400)
    ap.add_argument('--L', type=float, default=24.0)
    ap.add_argument('--Z', type=float, default=2.0)
    ap.add_argument('--eps', type=float, default=0.7)
    ap.add_argument('--relax', type=int, default=8000)
    ap.add_argument('--T', type=float, default=6.0)
    ap.add_argument('--dt', type=float, default=0.0015)
    ap.add_argument('--squeeze', type=float, default=0.20)
    args = ap.parse_args()
    N, L, Z = args.N, args.L, args.Z
    xc = L/2

    u = np.linspace(0, 1, N); du = u[1]-u[0]     # rescaled coordinate (both domains)
    Npx = 1200
    xp = np.linspace(0, L, Npx); dxp = xp[1]-xp[0]
    ker = 1.0/np.sqrt(xp**2 + args.eps**2)
    kerF = np.fft.fft(np.fft.ifftshift(ker))

    def phys1(a):  return a*u                      # x from xi in domain 1
    def phys2(a):  return L - (L-a)*u              # x from eta in domain 2
    def Vnuc(x):   return -Z/np.sqrt((x-xc)**2 + args.eps**2)

    def hartree(rho_phys):
        return np.real(np.fft.ifft(np.fft.fft(rho_phys)*kerF))*dxp

    def dens_on_phys(chi1, chi2, a):
        b = L-a
        rho = np.zeros(Npx)
        x1 = phys1(a); r1 = (np.abs(chi1)**2)/a
        x2 = phys2(a); r2 = (np.abs(chi2)**2)/b
        rho += np.interp(xp, x1, r1, left=0, right=0)
        rho += np.interp(xp, x2[::-1], r2[::-1], left=0, right=0)
        return rho

    def slopes(chi1, chi2):
        # robust slope at the interface: linear fit of |chi| over the last 12 points
        nfit = 12
        uu = u[-nfit:]
        s1 = abs(np.polyfit(uu, np.abs(chi1[-nfit:]), 1)[0])
        s2 = abs(np.polyfit(uu, np.abs(chi2[-nfit:]), 1)[0])
        return s1, s2

    def boundary(chi1, chi2):
        s1, s2 = slopes(chi1, chi2)
        r = (max(s1,1e-9)/max(s2,1e-9))**(2/3)
        return L*r/(1+r)

    # ---- relax both domains (imaginary time), Dirichlet interface & outer ----
    chi1 = np.sqrt(2)*np.sin(np.pi*u).astype(complex)
    chi2 = np.sqrt(2)*np.sin(np.pi*u).astype(complex)
    a = L/2
    def normalize(c): return c/np.sqrt(np.sum(np.abs(c)**2)*du)
    chi1=normalize(chi1); chi2=normalize(chi2)
    dtau = 0.2*du**2
    for it in range(args.relax):
        b = L-a
        rho = dens_on_phys(chi1, chi2, a)
        Vh = hartree(rho)
        for (chi, ph, s) in [(chi1, phys1(a), a), (chi2, phys2(a), b)]:
            x = ph
            V = Vnuc(x) + np.interp(x, xp, Vh) - (np.abs(chi)**2)/s   # subtract own self (approx)
            lap = np.empty(N); lap[1:-1]=(chi[2:]-2*chi[1:-1]+chi[:-2])/du**2; lap[0]=lap[-1]=0
            chi[:] = chi + dtau*(0.5/s**2*lap - V*chi)
            chi[0]=0; chi[-1]=0
        chi1=normalize(chi1); chi2=normalize(chi2)
        if it % 40 == 0:
            a = 0.5*a + 0.5*boundary(chi1, chi2)   # relax boundary toward slope match
    print(f"relaxed: boundary x = {a:.3f} (centre {xc:.3f}); "
          f"slopes {slopes(chi1,chi2)[0]:.3f}/{slopes(chi1,chi2)[1]:.3f}")

    def energy(chi1, chi2, a):
        b = L-a
        rho = dens_on_phys(chi1, chi2, a); Vh = hartree(rho)
        E = 0.0
        for (chi, ph, s) in [(chi1, phys1(a), a), (chi2, phys2(a), b)]:
            d = np.gradient(chi, du)
            T = 0.5/s**2 * np.real(np.sum(np.abs(d)**2))*du
            En = np.sum(Vnuc(ph)*np.abs(chi)**2)*du
            E += T + En
        # e-e interaction (count once)
        x1=phys1(a); r1=(np.abs(chi1)**2)/a
        Eee = np.sum(np.interp(x1, xp, hartree(np.interp(xp, phys2(a)[::-1], ((np.abs(chi2)**2)/b)[::-1], left=0, right=0)))*r1)*du*a
        return E + Eee

    # ---- perturb domain 1 (push density toward the interface) ----
    chi1 = normalize(chi1*np.exp(args.squeeze*u))

    # ---- real-time evolution: moving-wall CN per domain + slope-match boundary ----
    dt = args.dt; nsteps = int(args.T/dt)
    a_prev = boundary(chi1, chi2)
    E0 = energy(chi1, chi2, a_prev)
    print(f"perturbed; E0={E0:.4f}; evolving (energy-conserving moving free boundary)...")
    print(f"  {'t':>6} {'x_bdry':>8} {'q1':>7} {'q2':>7} {'dE/E':>10}")

    def cn_step(chi, s, sdot, Vphys):
        k = 0.5/s**2/du**2
        adv = sdot/s
        sub = -k*np.ones(N,complex) - (-1j*adv)*(u/(2*du))
        sup = -k*np.ones(N,complex) + (-1j*adv)*(u/(2*du))
        diag = 2*k*np.ones(N,complex) + (-1j*adv)*0.5 + Vphys
        aL=(1j*dt/2)*np.concatenate(([0],sub[1:]))
        bD=1+(1j*dt/2)*diag
        cU=(1j*dt/2)*np.concatenate((sup[:-1],[0]))
        Hc=np.empty(N,complex)
        Hc[1:-1]=sub[1:-1]*chi[:-2]+diag[1:-1]*chi[1:-1]+sup[1:-1]*chi[2:]
        Hc[0]=0;Hc[-1]=0
        rhs=chi-(1j*dt/2)*Hc
        aL[0]=0;bD[0]=1;cU[0]=0;rhs[0]=0; aL[-1]=0;bD[-1]=1;cU[-1]=0;rhs[-1]=0
        return thomas(aL,bD,cU,rhs)

    a = a_prev
    for st in range(nsteps):
        b = L-a
        adot = (a - a_prev)/dt; bdot = -adot
        rho = dens_on_phys(chi1, chi2, a); Vh = hartree(rho)
        V1 = Vnuc(phys1(a)) + np.interp(phys1(a), xp, Vh) - (np.abs(chi1)**2)/a
        V2 = Vnuc(phys2(a)) + np.interp(phys2(a), xp, Vh) - (np.abs(chi2)**2)/b
        chi1n = cn_step(chi1, a, adot, V1)
        chi2n = cn_step(chi2, b, bdot, V2)
        chi1, chi2 = chi1n, chi2n
        a_prev = a
        target = boundary(chi1, chi2)
        vmax = 0.5                                  # rate-limit the boundary speed (bounds adot)
        a = a + np.clip(0.05*(target-a), -vmax*dt, vmax*dt)
        if st % (nsteps//8) == 0 or st == nsteps-1:
            q1=np.sum(np.abs(chi1)**2)*du; q2=np.sum(np.abs(chi2)**2)*du
            print(f"  {(st+1)*dt:6.2f} {a:8.3f} {q1:7.3f} {q2:7.3f} "
                  f"{(energy(chi1,chi2,a)-E0)/abs(E0):+10.2e}")

    print("\n  => two non-overlapping domains, MOVING free boundary set by the")
    print("     variational slope-match condition, evolved in rescaled coordinates.")
    print("     Read q1,q2 (per-domain charge) and dE/E (total energy drift): if the")
    print("     boundary moves while charge stays 1.000 and energy holds, the")
    print("     energy-conserving free-boundary flow works (vs ~1.5%/move for the")
    print("     grid-transfer prototype).")


if __name__ == '__main__':
    main()
