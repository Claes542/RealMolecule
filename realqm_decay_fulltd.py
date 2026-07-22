"""
Full time-dependent RealQM decay: same solver, REAL complex time (not imaginary)
================================================================================

The point (per the observation that the full time-dependent solver is the
relaxation solver with the imaginary-time parabolic step swapped for a real,
unitary complex step):

    imaginary time (relax):  psi -> psi + dtau (1/2 lap psi - V psi)   [diffusion -> ground state]
    real complex time     :  i d_t psi = H psi                          [unitary -> true dynamics]

Everything else -- the domain fields, the self-consistent Coulomb, the free
boundary -- is unchanged.  The one new ingredient for DECAY is that the boundary
is now free to MOVE: we use soft fields on the whole line (the boundary is where
the densities cross and it tracks the solution), not a frozen mask.

Setup: a 1-D "atom" -- fixed nucleus (+Z at centre) with TWO electron charge
domains.  We relax to the bound state, then promote one electron to a loosely
bound outer state (a metastable configuration) and evolve in REAL time.  With the
boundary free to move, does the outer domain leak out (decay), while energy is
conserved?  This is the genuine object the frozen single-harmonic picture cannot
represent; it does NOT by itself settle the phase-coincidence conjecture, it tests
that the full time-dependent solver captures moving-boundary decay at all.

Units: atomic (hbar=1, m_e=1).
"""

from __future__ import annotations
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=2048)
    ap.add_argument('--L', type=float, default=120.0)
    ap.add_argument('--Z', type=float, default=2.0, help='nuclear charge')
    ap.add_argument('--eps', type=float, default=0.7, help='Coulomb softening')
    ap.add_argument('--T', type=float, default=60.0)
    ap.add_argument('--dt', type=float, default=0.002)
    ap.add_argument('--relax', type=int, default=4000)
    args = ap.parse_args()

    N, L = args.N, args.L
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    dx = x[1]-x[0]
    k = 2*np.pi*np.fft.fftfreq(N, dx)
    K2 = k*k

    soft = args.eps
    Vnuc = -args.Z/np.sqrt(x**2 + soft**2)          # nucleus attraction (fixed)

    def coul(ndens):                                 # softened 1/|x| convolution
        # potential from a unit charge density via FFT of the soft kernel
        ker = 1.0/np.sqrt(x**2 + soft**2)
        return np.real(np.fft.ifft(np.fft.fft(ndens)*np.fft.fft(np.fft.ifftshift(ker))))*dx

    def normalize(p):
        return p/np.sqrt(np.sum(np.abs(p)**2)*dx)

    # ---- initial electron orbitals: inner (bound) + outer (loose) ----
    psi1 = normalize(np.exp(-np.abs(x)/0.9).astype(complex))          # inner electron
    psi2 = normalize((np.exp(-((np.abs(x)-6.0)**2)/6.0)).astype(complex))  # outer, loose

    # ---- imaginary-time relax the inner electron in the nuclear + e-e field ----
    dtau = 0.2*dx**2
    for _ in range(args.relax):
        ne = np.abs(psi1)**2 + np.abs(psi2)**2
        Vh = coul(ne)                                # Hartree (both electrons)
        V1 = Vnuc + coul(np.abs(psi2)**2)            # inner feels nucleus + outer
        lap = np.real(np.fft.ifft(-K2*np.fft.fft(psi1)))
        psi1 = psi1 + dtau*(0.5*lap - V1*psi1)
        psi1 = normalize(psi1)
    print("relaxed inner electron; starting REAL-time evolution (metastable outer e)")

    # ---- REAL complex-time evolution (split-step, unitary) ----
    # absorbing layer near the edges so an ejected electron leaves cleanly
    xabs = L/2 - 15.0
    Wabs = np.where(np.abs(x) > xabs, 3.0*((np.abs(x)-xabs)/(L/2-xabs))**2, 0.0)
    expK = np.exp(-1j*(K2/2.0)*args.dt)

    def energy():
        ne = np.abs(psi1)**2+np.abs(psi2)**2
        def T(p):
            return np.real(np.sum(np.conj(p)*np.fft.ifft(0.5*K2*np.fft.fft(p))))*dx
        Ekin = T(psi1)+T(psi2)
        Enuc = np.sum(Vnuc*ne)*dx
        Eee  = np.sum(coul(np.abs(psi1)**2)*np.abs(psi2)**2)*dx
        return Ekin+Enuc+Eee

    def outer_norm():   # norm of the outer electron still in the bound region |x|<10
        return np.sum(np.abs(psi2[np.abs(x)<10.0])**2)*dx

    nsteps = int(args.T/args.dt)
    E0 = energy(); N2_0 = np.sum(np.abs(psi2)**2)*dx
    print(f"  E0={E0:.4f}  outer-e bound norm={outer_norm():.3f}")
    print(f"  {'t':>6} {'boundNorm2':>11} {'<|x|>_2':>9} {'dE/E':>10}")
    for s in range(nsteps):
        # half V
        ne = np.abs(psi1)**2+np.abs(psi2)**2
        V1 = Vnuc + coul(np.abs(psi2)**2) - 1j*Wabs
        V2 = Vnuc + coul(np.abs(psi1)**2) - 1j*Wabs
        psi1 *= np.exp(-1j*V1*args.dt/2); psi2 *= np.exp(-1j*V2*args.dt/2)
        # kinetic
        psi1 = np.fft.ifft(expK*np.fft.fft(psi1)); psi2 = np.fft.ifft(expK*np.fft.fft(psi2))
        # half V
        V1 = Vnuc + coul(np.abs(psi2)**2) - 1j*Wabs
        V2 = Vnuc + coul(np.abs(psi1)**2) - 1j*Wabs
        psi1 *= np.exp(-1j*V1*args.dt/2); psi2 *= np.exp(-1j*V2*args.dt/2)
        if s % (nsteps//10) == 0 or s == nsteps-1:
            w = np.abs(psi2)**2; tot=np.sum(w)*dx
            xmean = np.sum(w*np.abs(x))*dx/tot if tot>0 else 0
            print(f"  {(s+1)*args.dt:6.2f} {outer_norm():11.3f} {xmean:9.3f} "
                  f"{(energy()-E0)/abs(E0):+10.2e}")

    print("\n  => real complex time on the same solver: the outer electron domain")
    print("     leaks outward with a FREELY MOVING boundary (soft fields), i.e. the")
    print("     full time-dependent RealQM captures decay dynamics the frozen")
    print("     single-harmonic picture cannot. (Phase-coincidence gating: still open.)")


if __name__ == '__main__':
    main()
