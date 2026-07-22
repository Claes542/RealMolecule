"""
RealQM beta-decay prototype -- Stage 1: radial "neutron" (electron caged by one proton)
=======================================================================================

Geometry (spherically symmetric, s-wave):
    electron : core   r in [0, a]      (charge -1)
    proton   : shell  r in [a, R]      (charge +1)
non-overlapping radial domains, hard partition at the free boundary r = a with a
zero-flux (Neumann) interface, exactly as in the static RealQM/RealNucleus solver.

This prototype has ONE job: test whether the *complex time-dependent* RealQM
evolution

    i d_t psi_i = -(1/2 m_i) lap psi_i + q_i * phi[rho] * psi_i          (Eq. 2)

with phi the self-consistent electrostatic potential of the total charge density
rho = -|psi_e|^2 + |psi_p|^2, conserves the total energy

    E = sum_i (1/2 m_i) integral |grad psi_i|^2  +  (1/2) integral rho phi         (Eq. E)

numerically -- the load-bearing claim of the beta-decay article.  It also relaxes
(imaginary time) to the caged ground state so we can read the per-domain energies.

What it does NOT do (needs 2-D, stage 2): directional emission, proton recoil,
the electron actually separating past the shell, momentum balance.

Units: atomic (hbar = 1, m_e = 1, e = 1).  Proton mass m_p = 1836.

Numerics: reduced radial amplitude u(r) = r * psi(r), so the 3-D s-wave Laplacian
    lap psi = psi'' + (2/r) psi'   becomes   lap psi = u''/r,
and normalization / kinetic / Coulomb integrals are 1-D in u.  Real-time stepping
uses the Visscher staggered (real/imag) leapfrog, which is norm- and
energy-stable for dt < ~2/|H_max|.
"""

from __future__ import annotations
import argparse
import numpy as np

FOURPI = 4.0 * np.pi


# ---------------------------------------------------------------------------
# grid and operators (reduced radial: work with u = r*psi on cell centres)
# ---------------------------------------------------------------------------

def make_grid(N, R):
    h = R / N
    r = (np.arange(N) + 0.5) * h        # staggered: r>0 everywhere, no r=0 node
    return r, h


def radial_density(u, r):
    """number density n(r) = |psi|^2 = |u|^2 / r^2  from reduced amplitude u."""
    return (np.abs(u) ** 2) / (r ** 2)


def normalize(u, r, h):
    """scale u so that  integral n 4pi r^2 dr = 1, i.e. 4pi integral |u|^2 dr = 1."""
    norm = FOURPI * np.sum(np.abs(u) ** 2) * h
    return u / np.sqrt(norm)


def poisson_phi(rho, r, h):
    """Electrostatic potential phi(r) of a spherically symmetric charge density rho,
    via Gauss:  Q_enc(r) = integral_0^r rho 4pi r'^2 dr',  E(r) = Q_enc/r^2,
    phi(r) = integral_r^R E dr' + Q_tot/R.
    """
    shell_charge = rho * FOURPI * r ** 2 * h          # dQ in each shell
    Q_enc = np.cumsum(shell_charge)                   # enclosed charge at outer edge of cell
    # centre the enclosed charge at the cell (subtract half the local shell)
    Q_enc_c = Q_enc - 0.5 * shell_charge
    E_field = Q_enc_c / r ** 2                         # radial field
    Q_tot = Q_enc[-1]
    # phi(r) = integral_r^R E dr' + phi(R),  phi(R) = Q_tot / R
    # reverse cumulative integral of E_field
    phi = np.zeros_like(rho)
    tail = np.cumsum(E_field[::-1] * h)[::-1]         # integral_r^R E dr'
    phi = tail + Q_tot / (r[-1] + 0.5 * h)
    return phi, Q_tot


def lap_u(u, mask, h):
    """u'' with a zero-flux (Neumann) interface: neighbours outside the domain are
    replaced by the cell's own value (so no flux crosses the free boundary), and
    Dirichlet u=0 at the outer box edge and at the origin side (u(0)=0)."""
    up = np.empty_like(u)
    um = np.empty_like(u)
    up[:-1] = np.where(mask[1:] > 0.5, u[1:], u[:-1])
    up[-1] = 0.0                                       # Dirichlet at outer edge
    um[1:] = np.where(mask[:-1] > 0.5, u[:-1], u[1:])
    um[0] = 0.0                                        # u(0)=0 regularity (reduced radial)
    return (up + um - 2.0 * u) / h ** 2


# ---------------------------------------------------------------------------
# energy
# ---------------------------------------------------------------------------

def energies(ue, up_, r, h, m_p, mask_e, mask_p):
    """Return dict of energy components (kinetic e, kinetic p, Coulomb, total).

    Kinetic uses the SAME discrete Laplacian as the evolution, so it is the
    quantity the Visscher scheme actually conserves:
        T_i = (1/2 m_i) * 4pi * < u_i, -lap u_i >.
    """
    ne = radial_density(ue, r)
    np_ = radial_density(up_, r)
    rho = -ne + np_                                    # charge density (e:-1, p:+1)
    phi, Qtot = poisson_phi(rho, r, h)

    def kinetic(u, m, mask):
        lu = lap_u(u, mask, h)
        return (1.0 / (2.0 * m)) * FOURPI * np.real(np.sum(np.conj(u) * (-lu))) * h

    Te = kinetic(ue, 1.0, mask_e)
    Tp = kinetic(up_, m_p, mask_p)
    # Coulomb (full RealQM functional):  (1/2) integral rho phi 4pi r^2 dr
    E_coul = 0.5 * np.sum(rho * phi * FOURPI * r ** 2) * h
    E_tot = Te + Tp + E_coul
    return dict(Te=Te, Tp=Tp, Ecoul=E_coul, Etot=E_tot, Qtot=Qtot)


# ---------------------------------------------------------------------------
# static relaxation (imaginary time) -> caged ground state
# ---------------------------------------------------------------------------

def relax(N, R, a, m_p, steps, verbose=False, confine_e=True):
    r, h = make_grid(N, R)
    if confine_e:
        mask_e = (r <= a).astype(float)
    else:
        mask_e = np.ones(N); mask_e[-1] = 0.0          # electron free on whole grid
    mask_p = (r > a).astype(float)

    # initial guesses: electron ~ exp(-r) ball, proton ~ shell bump
    ue = mask_e * (r * np.exp(-r))
    up_ = mask_p * (r * np.exp(-((r - (a + R) / 2) ** 2)))
    ue = normalize(ue, r, h)
    up_ = normalize(up_, r, h)

    dtau = 0.2 * h ** 2
    Eprev = 1e9
    for s in range(steps):
        ne = radial_density(ue, r)
        np_ = radial_density(up_, r)
        rho = -ne + np_
        phi, _ = poisson_phi(rho, r, h)
        Ve = -phi          # electron charge -1
        Vp = +phi          # proton   charge +1

        ue = ue + dtau * (0.5 * lap_u(ue, mask_e, h) - Ve * ue)
        up_ = up_ + dtau * (0.5 / m_p * lap_u(up_, mask_p, h) - Vp * up_)
        ue *= mask_e
        up_ *= mask_p
        ue = normalize(ue, r, h)
        up_ = normalize(up_, r, h)

        if s % 200 == 0 or s == steps - 1:
            E = energies(ue, up_, r, h, m_p, mask_e, mask_p)
            if verbose:
                print(f"    relax {s:5d}: E={E['Etot']:.5f}  Te={E['Te']:.4f} "
                      f"Tp={E['Tp']:.5f} Ecoul={E['Ecoul']:.4f}")
            if abs(E['Etot'] - Eprev) < 1e-7:
                break
            Eprev = E['Etot']
    return r, h, mask_e, mask_p, ue, up_


# ---------------------------------------------------------------------------
# real-time complex evolution (Visscher staggered leapfrog) -> conservation test
# ---------------------------------------------------------------------------

def evolve(r, h, mask_e, mask_p, ue, up_, m_p, T, dt, kick=0.0,
           mask_e_run=None, mask_p_run=None):
    """Real-time evolution of Eq.(2).  Returns time series of energy for the
    conservation check, plus the electron mean radius <r>_e and the escaped
    fraction (norm beyond r=8), to see whether the electron floats out.

    mask_e_run / mask_p_run : optional evolution masks that differ from the
    relaxation masks -- e.g. release the electron to the full grid ('the decay').
    Optional 'kick': multiply the electron by exp(i k r) to give outward momentum.
    """
    if mask_e_run is None:
        mask_e_run = mask_e
    if mask_p_run is None:
        mask_p_run = mask_p
    if kick != 0.0:
        ue = ue * np.exp(1j * kick * r)

    # split into real/imag, staggered by dt/2 for the imaginary parts
    Re_e, Im_e = ue.real.copy(), ue.imag.copy()
    Re_p, Im_p = up_.real.copy(), up_.imag.copy()

    def H(Re, Im, mask, m, phi, q):
        V = q * phi
        HR = -(0.5 / m) * lap_u(Re, mask, h) + V * Re
        HI = -(0.5 / m) * lap_u(Im, mask, h) + V * Im
        return HR, HI

    def potential():
        ne = (Re_e ** 2 + Im_e ** 2) / r ** 2
        np_ = (Re_p ** 2 + Im_p ** 2) / r ** 2
        rho = -ne + np_
        phi, _ = poisson_phi(rho, r, h)
        return phi

    nsteps = int(T / dt)
    ts, Es, Ns, Rmean, Esc = [], [], [], [], []

    def observe():
        we = Re_e ** 2 + Im_e ** 2                     # |u_e|^2  (radial weight)
        tot = np.sum(we)
        rmean = np.sum(we * r) / tot if tot > 0 else 0.0
        escaped = np.sum(we[r > 8.0]) / tot if tot > 0 else 0.0
        return rmean, escaped

    # prime the imaginary parts by half a step:  I(dt/2) = I(0) - (dt/2) H R(0)
    phi = potential()
    Im_e = Im_e - 0.5 * dt * (-(0.5) * lap_u(Re_e, mask_e_run, h) + (-phi) * Re_e)
    Im_p = Im_p - 0.5 * dt * (-(0.5 / m_p) * lap_u(Re_p, mask_p_run, h) + (phi) * Re_p)

    for s in range(nsteps):
        phi = potential()
        # R(t+dt) = R(t) + dt H I(t+dt/2)
        Re_e = Re_e + dt * (-(0.5) * lap_u(Im_e, mask_e_run, h) + (-phi) * Im_e)
        Re_p = Re_p + dt * (-(0.5 / m_p) * lap_u(Im_p, mask_p_run, h) + (phi) * Im_p)
        Re_e *= mask_e_run; Re_p *= mask_p_run
        phi = potential()
        # I(t+3dt/2) = I(t+dt/2) - dt H R(t+dt)
        Im_e = Im_e - dt * (-(0.5) * lap_u(Re_e, mask_e_run, h) + (-phi) * Re_e)
        Im_p = Im_p - dt * (-(0.5 / m_p) * lap_u(Re_p, mask_p_run, h) + (phi) * Re_p)
        Im_e *= mask_e_run; Im_p *= mask_p_run

        if s % 50 == 0 or s == nsteps - 1:
            ue_c = Re_e + 1j * Im_e
            up_c = Re_p + 1j * Im_p
            E = energies(ue_c, up_c, r, h, m_p, mask_e_run, mask_p_run)
            Nnorm = FOURPI * (np.sum(np.abs(ue_c) ** 2) + np.sum(np.abs(up_c) ** 2)) * h
            rmean, escaped = observe()
            ts.append(s * dt); Es.append(E['Etot']); Ns.append(Nnorm)
            Rmean.append(rmean); Esc.append(escaped)

    return (np.array(ts), np.array(Es), np.array(Ns),
            np.array(Rmean), np.array(Esc))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=400)
    ap.add_argument('--R', type=float, default=20.0)
    ap.add_argument('--a', type=float, default=2.5, help='core/shell boundary radius')
    ap.add_argument('--mp', type=float, default=1836.0)
    ap.add_argument('--relax-steps', type=int, default=8000)
    ap.add_argument('--T', type=float, default=2.0, help='real-time duration')
    ap.add_argument('--kick', type=float, default=0.0)
    ap.add_argument('--dtfrac', type=float, default=0.2, help='dt = dtfrac * h^2')
    ap.add_argument('--decay', action='store_true',
                    help='release the electron cage and watch it float out (the decay)')
    ap.add_argument('--free-relax', action='store_true',
                    help='relax the electron UNconfined (clean bound/unbound test)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    print("RealQM beta-decay prototype -- stage 1 (radial, electron caged by one proton)")
    print(f"  N={args.N} R={args.R} a={args.a} m_p={args.mp}")
    print("  [1] relaxing to caged ground state ...")
    r, h, me, mp_, ue, up_ = relax(args.N, args.R, args.a, args.mp,
                                   args.relax_steps, args.verbose,
                                   confine_e=not args.free_relax)
    if args.free_relax:
        we = np.abs(ue) ** 2
        rmean = np.sum(we * r) / np.sum(we)
        esc = np.sum(we[r > 8.0]) / np.sum(we)
        print(f"      [free relax] electron <r>_e={rmean:.3f}, "
              f"frac beyond r=8: {esc:.3f}  (large/box-filling => unbound, floats out)")
    E0 = energies(ue, up_, r, h, args.mp, me, mp_)
    print(f"      ground state: E={E0['Etot']:.5f}  Te={E0['Te']:.4f} "
          f"Tp={E0['Tp']:.6f}  Ecoul={E0['Ecoul']:.4f}  Qtot={E0['Qtot']:.2e}")

    dt = args.dtfrac * h ** 2
    me_run = mp_run = None
    if args.decay:
        # release the electron: it may now occupy the whole grid (Dirichlet only
        # at the box edge); the heavy proton stays on its shell.  'The decay.'
        me_run = np.ones_like(me)
        me_run[-1] = 0.0
        mp_run = mp_
        print(f"  [2] THE DECAY: cage released, T={args.T} -- does the electron float out?")
    else:
        print(f"  [2] real-time evolution (Eq.2), T={args.T}, checking dE/dt ~ 0 ...")

    ts, Es, Ns, Rm, Esc = evolve(r, h, me, mp_, ue, up_, args.mp, args.T, dt,
                                 kick=args.kick, mask_e_run=me_run, mask_p_run=mp_run)
    dE = Es - Es[0]
    print(f"      steps logged: {len(ts)}   dt={dt:.2e}")
    print(f"      E(0)   = {Es[0]:.6f}   E(end) = {Es[-1]:.6f}")
    print(f"      max|dE|/|E(0)|     = {np.max(np.abs(dE))/abs(Es[0]):.2e}")
    print(f"      norm drift         = {np.max(np.abs(Ns - Ns[0])):.2e}")
    if args.decay:
        print(f"      <r>_e :  {Rm[0]:.3f} (start)  ->  {Rm[-1]:.3f} (end)"
              f"   [grows => electron floats out]")
        print(f"      escaped fraction (norm beyond r=8):  "
              f"{Esc[0]:.3f} -> {Esc[-1]:.3f}")
        # a few intermediate points
        idx = np.linspace(0, len(ts) - 1, 6).astype(int)
        print("      t, <r>_e, escaped:")
        for i in idx:
            print(f"        t={ts[i]:6.2f}  <r>_e={Rm[i]:6.3f}  esc={Esc[i]:.3f}")
    else:
        print("  => energy conservation is the pass/fail for the article's key claim.")


if __name__ == '__main__':
    main()
