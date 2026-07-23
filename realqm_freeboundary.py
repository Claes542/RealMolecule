"""
Full complex RealQM with an EXPLICIT MOVING FREE BOUNDARY (minimal genuine version)
===================================================================================

Two NON-overlapping charge domains, each a distinct electron of unit charge, meet
at a free boundary whose position is set by DENSITY BALANCE (|psi_1|=|psi_2| at the
interface) and which is TRACKED and MOVED as the fields evolve.  Unlike the soft
two-field version (realqm_decay_fulltd), the domains do not overlap: each field is
confined to its side by a zero-flux (Neumann) interface, and the boundary is an
explicit dynamical interface, not just where two overlapping densities happen to
cross.  The fields evolve in REAL complex time (Visscher leapfrog) -- the same
solver as the imaginary-time relaxation, only the step changed.

Test system: a 1-D He-like atom, nucleus +2 at the centre, two electrons, one on
each side.  We relax to the ground state (the boundary finds density balance ->
centre, by symmetry), then evolve in real time and, after a symmetry-breaking
perturbation, watch the free boundary MOVE and the domains respond -- while each
domain keeps its unit charge and the total energy is tracked.

HONEST STATUS: this is a first free-boundary prototype; moving-interface schemes
are delicate, so the run also reports stability diagnostics (per-domain charge
drift, energy drift) and we read the outcome straight.

Units: atomic (hbar=1, m_e=1).
"""

from __future__ import annotations
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=600)
    ap.add_argument('--L', type=float, default=30.0)
    ap.add_argument('--Z', type=float, default=2.0)
    ap.add_argument('--eps', type=float, default=0.6)
    ap.add_argument('--relax', type=int, default=6000)
    ap.add_argument('--T', type=float, default=8.0)
    ap.add_argument('--kick', type=float, default=0.25, help='symmetry-breaking push on e1')
    ap.add_argument('--xcfrac', type=float, default=0.42, help='nucleus position / L (off-centre => asymmetric)')
    ap.add_argument('--squeeze', type=float, default=0.15, help='real density push on e1 toward the interface')
    args = ap.parse_args()

    N, L = args.N, args.L
    x = np.linspace(0, L, N); dx = x[1]-x[0]
    xc = args.xcfrac*L
    Vnuc = -args.Z/np.sqrt((x-xc)**2 + args.eps**2)

    def coul(nd):
        ker = 1.0/np.sqrt(x**2 + args.eps**2)
        return np.real(np.fft.ifft(np.fft.fft(nd)*np.fft.fft(np.fft.ifftshift(ker))))*dx

    # boundary index mb: domain1 = [0..mb], domain2 = [mb+1..N-1]
    mb = N//2

    def masks(mb):
        m1 = np.zeros(N); m1[:mb+1] = 1
        m2 = np.zeros(N); m2[mb+1:] = 1
        return m1, m2

    def lap_neumann(u, m):
        """u'' with zero-flux at the interface (ghost = own value) and Dirichlet ends."""
        up = np.empty_like(u); um = np.empty_like(u)
        up[:-1] = np.where(m[1:] > 0.5, u[1:], u[:-1]); up[-1] = 0.0
        um[1:] = np.where(m[:-1] > 0.5, u[:-1], u[1:]); um[0] = 0.0
        return (up + um - 2*u)/dx**2

    def normdom(u, m):
        s = np.sum(np.abs(u)**2 * m)*dx
        return u/np.sqrt(s) if s > 0 else u

    m1, m2 = masks(mb)
    # initial: two 1s-like bumps either side of the nucleus
    u1 = normdom(np.exp(-np.abs(x-(xc-2.0))) * m1, m1).astype(complex)
    u2 = normdom(np.exp(-np.abs(x-(xc+2.0))) * m2, m2).astype(complex)

    def update_boundary(u1, u2, mb):
        """move the interface toward density balance |u1(mb)|=|u2(mb+1)|; transfer at
        most one grid point per call, with continuity, then renormalise both."""
        d1 = np.abs(u1[mb])**2; d2 = np.abs(u2[mb+1])**2 if mb+1 < N else 0.0
        moved = 0
        if d1 > 1.15*d2 and mb < N-3:          # domain 1 denser -> it expands
            mb += 1
            u1[mb] = u1[mb-1]; u2[mb] = 0.0     # continuity for e1, drop point from e2
            moved = +1
        elif d2 > 1.15*d1 and mb > 2:          # domain 2 denser -> it expands
            u2[mb] = u2[mb+1]; u1[mb] = 0.0
            mb -= 1
            moved = -1
        m1, m2 = masks(mb)
        u1 = normdom(u1*m1, m1); u2 = normdom(u2*m2, m2)
        return u1, u2, mb, m1, m2, moved

    # ---- relax (imaginary time), boundary tracks density balance ----
    dtau = 0.2*dx**2
    for it in range(args.relax):
        ne = np.abs(u1)**2 + np.abs(u2)**2
        V1 = Vnuc + coul(np.abs(u2)**2)
        V2 = Vnuc + coul(np.abs(u1)**2)
        u1 = u1 + dtau*(0.5*lap_neumann(u1, m1) - V1*u1); u1 *= m1
        u2 = u2 + dtau*(0.5*lap_neumann(u2, m2) - V2*u2); u2 *= m2
        u1 = normdom(u1, m1); u2 = normdom(u2, m2)
        if it % 20 == 0:
            u1, u2, mb, m1, m2, _ = update_boundary(u1, u2, mb)
    print(f"relaxed: boundary at x = {x[mb]:.2f} (centre = {xc:.2f}); "
          f"q1={np.sum(np.abs(u1)**2*m1)*dx:.3f} q2={np.sum(np.abs(u2)**2*m2)*dx:.3f}")

    def energy():
        ne = np.abs(u1)**2+np.abs(u2)**2
        T = -0.5*np.real(np.sum(np.conj(u1)*lap_neumann(u1,m1))+np.sum(np.conj(u2)*lap_neumann(u2,m2)))*dx
        En = np.sum(Vnuc*ne)*dx
        Eee = np.sum(coul(np.abs(u1)**2)*np.abs(u2)**2)*dx
        return T+En+Eee

    # ---- perturb e1 (phase kick + real density push toward the interface) ----
    u1 = u1*np.exp(1j*args.kick*(x-xc))
    u1 = normdom(u1*np.exp(args.squeeze*(x-xc))*m1, m1)   # push e1 density toward larger x (interface)
    dt = 0.15*dx**2
    # Visscher: split re/im
    R1,I1 = u1.real.copy(), u1.imag.copy(); R2,I2 = u2.real.copy(), u2.imag.copy()
    def Hpsi(R, m, Vr):
        return -0.5*lap_neumann(R, m) + Vr*R
    nsteps = int(args.T/dt)
    E0 = energy(); mb0 = mb
    print(f"kick={args.kick}; E0={E0:.4f}; evolving real time, tracking free boundary...")
    print(f"  {'t':>6} {'x_bdry':>8} {'q1':>7} {'q2':>7} {'dE/E':>10} {'moves':>6}")
    V1 = Vnuc + coul(R2*R2+I2*I2); V2 = Vnuc + coul(R1*R1+I1*I1)
    I1 = I1 - 0.5*dt*Hpsi(R1,m1,V1); I2 = I2 - 0.5*dt*Hpsi(R2,m2,V2)
    totmoves = 0
    for s in range(nsteps):
        V1 = Vnuc + coul(R2*R2+I2*I2); V2 = Vnuc + coul(R1*R1+I1*I1)
        R1 = R1 + dt*Hpsi(I1,m1,V1); R1 *= m1
        R2 = R2 + dt*Hpsi(I2,m2,V2); R2 *= m2
        V1 = Vnuc + coul(R2*R2+I2*I2); V2 = Vnuc + coul(R1*R1+I1*I1)
        I1 = I1 - dt*Hpsi(R1,m1,V1); I1 *= m1
        I2 = I2 - dt*Hpsi(R2,m2,V2); I2 *= m2
        if s % 15 == 0:
            u1 = R1+1j*I1; u2 = R2+1j*I2
            u1,u2,mb,m1,m2,mv = update_boundary(u1,u2,mb)
            R1,I1 = u1.real.copy(),u1.imag.copy(); R2,I2 = u2.real.copy(),u2.imag.copy()
            totmoves += abs(mv)
        if s % (nsteps//8) == 0 or s == nsteps-1:
            u1 = R1+1j*I1; u2 = R2+1j*I2
            q1 = np.sum(np.abs(u1)**2*m1)*dx; q2 = np.sum(np.abs(u2)**2*m2)*dx
            print(f"  {(s+1)*dt:6.2f} {x[mb]:8.2f} {q1:7.3f} {q2:7.3f} "
                  f"{(energy()-E0)/abs(E0):+10.2e} {totmoves:6d}")

    print(f"\n  free boundary moved from x={x[mb0]:.2f} to x={x[mb]:.2f} "
          f"({totmoves} interface transfers)")
    print("  => two non-overlapping domains, explicit density-balanced interface,")
    print("     tracked and moved under real complex-time evolution. Read the")
    print("     charge/energy drift above for whether the scheme held.")


if __name__ == '__main__':
    main()
