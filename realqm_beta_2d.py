"""
RealQM beta-decay -- Stage 2: the 2-D test (directional emission + recoil + momentum)
=====================================================================================

Why 2-D: beta decay is directional -- the electron leaves one way, the proton
recoils the other.  A radial/1-D model is spherically symmetric and its total
momentum is identically zero, so it cannot even pose the momentum question.  The
plane is the minimal setting with a direction.

Setup: electron (m=1) and proton (m=1836) as complex fields psi_e, psi_p in a 2-D
box, coupled by a (softened) Coulomb pair interaction -- the RealQM cross term
    E_int = -\int\int n_e(r) n_p(r') / sqrt(|r-r'|^2+eps^2) dr dr'   (attraction).
Each field evolves by its time-dependent RealQM/Schroedinger equation
    i d_t psi_i = -(1/2 m_i) lap psi_i + V_i psi_i ,   V_i = q_i * phi[other],
integrated by split-step Fourier (norm-conserving, spectral kinetic).

The 1-D solve showed PHASE gates the release of a caged electron; here we inject
that release as a directional impulse on the electron and ask the vector question
the radial model could not:
    * is total momentum  p_e + p_p  conserved (and ~0 from rest)?
    * does the electron go one way and the heavy proton recoil the other,
      back-to-back (p_e = -p_p), with the proton velocity ~1/1836 of the electron?

Honest scope: this is instantaneous-Coulomb 2-body dynamics, so there is NO
radiated-field momentum here.  Two isolated bodies must come out back-to-back by
momentum conservation -- which is precisely why matching the *observed*
non-back-to-back recoil needs field momentum (the current-carrying/retarded EM
extension), beyond this solve.  The test localizes exactly where that open problem
lives.

Units: atomic (hbar=1, m_e=1).
"""

from __future__ import annotations
import argparse
import numpy as np


def gaussian(X, Y, x0, y0, s):
    g = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * s ** 2))
    return g.astype(complex)


def normalize(psi, dA):
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dA)


def momentum(psi, KX, KY, dA):
    """p = <psi| -i grad |psi> via spectral derivative; returns (px, py)."""
    ph = np.fft.fft2(psi)
    dxp = np.fft.ifft2(1j * KX * ph)
    dyp = np.fft.ifft2(1j * KY * ph)
    px = np.real(np.sum(np.conj(psi) * (-1j) * dxp)) * dA
    py = np.real(np.sum(np.conj(psi) * (-1j) * dyp)) * dA
    return px, py


def mean_pos(psi, X, Y, dA):
    n = np.abs(psi) ** 2
    tot = np.sum(n) * dA
    return np.sum(n * X) * dA / tot, np.sum(n * Y) * dA / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=128)
    ap.add_argument('--L', type=float, default=40.0)
    ap.add_argument('--mp', type=float, default=1836.0)
    ap.add_argument('--eps', type=float, default=1.0, help='Coulomb softening')
    ap.add_argument('--kick', type=float, default=1.2, help='release impulse on electron (+x)')
    ap.add_argument('--sep', type=float, default=2.0, help='initial e-p separation')
    ap.add_argument('--T', type=float, default=12.0)
    ap.add_argument('--dt', type=float, default=0.002)
    args = ap.parse_args()

    N, L, m_p = args.N, args.L, args.mp
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = x[1] - x[0]
    dA = dx * dx
    X, Y = np.meshgrid(x, x, indexing='ij')

    k = 2 * np.pi * np.fft.fftfreq(N, dx)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    K2 = KX ** 2 + KY ** 2

    # softened Coulomb kernel g(r)=1/sqrt(r^2+eps^2), FFT for convolution
    Rk = np.sqrt(X ** 2 + Y ** 2 + args.eps ** 2)
    Gk = np.fft.fft2(np.fft.ifftshift(1.0 / Rk))

    def potential_from(n):
        """electrostatic potential phi(r) = conv(n, 1/r) of a +unit density n."""
        return np.real(np.fft.ifft2(np.fft.fft2(n) * Gk)) * dA

    # initial state: electron and proton as adjacent blobs, at rest
    psi_e = normalize(gaussian(X, Y, -args.sep / 2, 0.0, 1.2), dA)
    psi_p = normalize(gaussian(X, Y, +args.sep / 2, 0.0, 0.8), dA)

    # inject the (phase-triggered) release from REST: an internal, momentum-
    # conserving impulse -- electron gets +P, proton gets -P, so p_tot = 0.
    # (Two-body momentum conservation then forces back-to-back emission; the
    # proton, 1836x heavier, recoils 1836x slower.)
    psi_e = psi_e * np.exp(+1j * args.kick * X)
    psi_p = psi_p * np.exp(-1j * args.kick * X)

    # split-step kinetic propagators
    expK_e = np.exp(-1j * (K2 / (2 * 1.0)) * args.dt)
    expK_p = np.exp(-1j * (K2 / (2 * m_p)) * args.dt)

    def energies():
        ne = np.abs(psi_e) ** 2
        npd = np.abs(psi_p) ** 2
        phi_e = potential_from(ne)      # potential from electron density (as +unit)
        phi_p = potential_from(npd)
        # kinetic via spectral
        def T(psi, m):
            ph = np.fft.fft2(psi)
            return np.real(np.sum(np.conj(psi) * np.fft.ifft2(0.5 / m * K2 * ph))) * dA
        Te, Tp = T(psi_e, 1.0), T(psi_p, m_p)
        # interaction (attraction, e=-1 p=+1): -\int n_e phi_p
        Eint = -np.sum(ne * phi_p) * dA
        return Te, Tp, Eint, Te + Tp + Eint

    # dipole record (for the radiated-field estimate, stage 3)
    c_light = 137.036            # speed of light in atomic units
    dip = []                     # dipole moment d_x(t) = q_e <x>_e + q_p <x>_p

    nsteps = int(args.T / args.dt)
    log = max(1, nsteps // 12)
    print("RealQM beta-decay 2-D test (electron emission + proton recoil + momentum)")
    print(f"  N={N} L={L} m_p={m_p} kick={args.kick}  dt={args.dt} steps={nsteps}")
    Te, Tp, Ei, E0 = energies()
    px_e0, _ = momentum(psi_e, KX, KY, dA)
    px_p0, _ = momentum(psi_p, KX, KY, dA)
    print(f"  t=0: E={E0:.4f}  p_e={px_e0:+.4f} p_p={px_p0:+.4f} "
          f"p_tot={px_e0+px_p0:+.2e}")
    print(f"  {'t':>6} {'<x>_e':>8} {'<x>_p':>9} {'p_e':>9} {'p_p':>9} "
          f"{'p_tot':>10} {'dE/E':>9}")

    for s in range(nsteps):
        # V half step
        ne = np.abs(psi_e) ** 2
        npd = np.abs(psi_p) ** 2
        Ve = -potential_from(npd)       # electron (-1) attracted to proton (+1)
        Vp = -potential_from(ne)        # proton   (+1) attracted to electron(-1)
        psi_e *= np.exp(-1j * Ve * args.dt / 2)
        psi_p *= np.exp(-1j * Vp * args.dt / 2)
        # K full step
        psi_e = np.fft.ifft2(expK_e * np.fft.fft2(psi_e))
        psi_p = np.fft.ifft2(expK_p * np.fft.fft2(psi_p))
        # V half step
        ne = np.abs(psi_e) ** 2
        npd = np.abs(psi_p) ** 2
        Ve = -potential_from(npd)
        Vp = -potential_from(ne)
        psi_e *= np.exp(-1j * Ve * args.dt / 2)
        psi_p *= np.exp(-1j * Vp * args.dt / 2)

        # dipole moment each step, for the radiated-field (Larmor) estimate
        xe_s, _ = mean_pos(psi_e, X, Y, dA)
        xp_s, _ = mean_pos(psi_p, X, Y, dA)
        dip.append(-xe_s + xp_s)          # d = q_e<x>_e + q_p<x>_p, q_e=-1 q_p=+1

        if s % log == 0 or s == nsteps - 1:
            pxe, _ = momentum(psi_e, KX, KY, dA)
            pxp, _ = momentum(psi_p, KX, KY, dA)
            _, _, _, E = energies()
            print(f"  {(s+1)*args.dt:6.2f} {xe_s:8.3f} {xp_s:9.4f} {pxe:+9.4f} "
                  f"{pxp:+9.4f} {pxe+pxp:+10.2e} {(E-E0)/abs(E0):+9.1e}")

    # final summary
    xe, _ = mean_pos(psi_e, X, Y, dA)
    xp, _ = mean_pos(psi_p, X, Y, dA)
    pxe, _ = momentum(psi_e, KX, KY, dA)
    pxp, _ = momentum(psi_p, KX, KY, dA)
    print("\n  RESULT")
    print(f"    electron drifted to <x>_e={xe:+.3f} (released +x)")
    print(f"    proton   recoiled to <x>_p={xp:+.4f} (heavy: tiny displacement)")
    print(f"    p_e={pxe:+.4f}  p_p={pxp:+.4f}  ->  p_e/p_p = {pxe/pxp:+.2f} "
          f"(=-1 => exactly back-to-back)")
    print(f"    total momentum p_e+p_p = {pxe+pxp:+.2e}  (conserved ~0)")
    print("  => 2-body RealQM conserves vector momentum; e and p emerge back-to-back.")

    # ---- STAGE 3: how much can the radiated EM field actually carry? ----
    # Larmor dipole radiation: P(t) = (2/3 c^3) |d''(t)|^2  (Gaussian-atomic, e=1).
    d = np.array(dip)
    dt = args.dt
    dd = np.gradient(np.gradient(d, dt), dt)          # d''(t)
    P_rad = (2.0 / (3.0 * c_light ** 3)) * dd ** 2
    E_rad = np.sum(P_rad) * dt                         # total radiated energy
    KE_e = 0.5 * pxe ** 2 / 1.0                        # electron kinetic energy
    p_field_max = E_rad / c_light                      # |p_field| <= E_rad/c (photons)
    print("\n  STAGE 3 -- radiated EM field (Larmor estimate from the dipole):")
    print(f"    electron kinetic energy  KE_e     = {KE_e:.4e}")
    print(f"    total radiated energy    E_rad    = {E_rad:.4e}")
    print(f"    radiated energy fraction E_rad/KE = {E_rad/KE_e:.2e}")
    print(f"    max field momentum  E_rad/c       = {p_field_max:.2e}  "
          f"(vs |p_e|={abs(pxe):.3f})")
    print(f"    field-momentum fraction (E_rad/c)/|p_e| = {p_field_max/abs(pxe):.2e}")
    print("  => if this fraction is ~1e-2 or smaller, the radiated field is far too")
    print("     weak to carry the ~60% of Q and the momentum the neutrino accounts")
    print("     for -- i.e. the field cannot replace the neutrino.")


if __name__ == '__main__':
    main()
