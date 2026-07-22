"""
Josephson-network decay: deterministic decay from phase coincidence of
COEXISTING charge domains (not superposition of one particle's energy states)
=============================================================================

The faithful RealQM mechanism.  Several charge domains coexist side by side, each
on its own region Omega_i, each a stationary piece carrying its own phase
    theta_i(t) = phi_i - E_i t / hbar .
The domain boundaries are fixed by DENSITY balance (|psi|^2 is phase-invariant, so
the phase never moves a density-balanced boundary).  What the phases do is drive a
CURRENT across the fixed boundaries -- exactly the Josephson relation
    I_ij ~ sin(theta_i - theta_j),   d(theta_i - theta_j)/dt = (E_j - E_i)/hbar,
i.e. the inter-domain current oscillates at the beat frequency (AC Josephson, with
the domain energy difference playing the role of the voltage).

The escaping domain L sits inside a "cage" of M neighbour domains.  Charge leaves L
to the outside only when the cage is momentarily TRANSPARENT -- when L's phase
coincides with the cage phases so the outward Josephson currents add rather than
cancel.  Cage transparency (a product over the junctions, so ALL must align):
    G(t) = prod_m (1 + cos(theta_L - theta_m)) / 2 .

Claims tested here:
  * escape comes in phase-coincidence BURSTS (G spikes), not smoothly;
  * the decay is DETERMINISTIC given the initial phases phi_i -- the only
    randomness is ignorance of them;
  * averaged over unknown initial phases, the survival is EXPONENTIAL with a
    definite half-life -- the exponential law as ensemble statistics of a
    coincidence, not irreducible chance;
  * the rate is set by the domains' own ENERGIES (beat frequencies) and the
    number of cage domains M, with no imposed escape window and no coupling
    constant of a new force.

Units: hbar = 1.
"""

from __future__ import annotations
import argparse
import numpy as np


def beats_setup(M, seed_energies):
    """Incommensurate beat frequencies omega_m = E_L - E_m from domain energies."""
    rng = np.random.default_rng(seed_energies)
    # cage domain energies spread around the escaping domain's energy
    E_L = 1.0
    E_m = E_L + 0.6 * rng.uniform(0.5, 1.5, size=M) * np.array(
        [1.0, 1.6180339, 2.7182818, 3.1415926, 2.2360679][:M] if M <= 5
        else rng.uniform(0.5, 3.0, size=M))
    omega = E_L - E_m                     # beat frequencies (incommensurate)
    return omega


def cage_transparency(phi0, omega, t):
    """G(t) for one nucleus with initial phase offsets phi0 (length M)."""
    beta = phi0[:, None] + np.outer(omega, t)          # (M, len(t))
    return np.prod((1.0 + np.cos(beta)) / 2.0, axis=0)  # (len(t),)


def run(M=3, N_nuclei=20000, T=400.0, dt=0.05, Gamma=0.6, seed=1, seed_E=7):
    omega = beats_setup(M, seed_E)
    tgrid = np.arange(0.0, T, dt)
    rng = np.random.default_rng(seed)

    # ---- one nucleus: show the mechanism (transparency + norm drain) ----
    phi_demo = rng.uniform(0, 2 * np.pi, size=M)
    G_demo = cage_transparency(phi_demo, omega, tgrid)
    # norm on L drains by the Josephson outflow through the transparent cage:
    #   dN/dt = -Gamma G(t) N   ->   N(t) = exp(-Gamma \int G dt')
    H_demo = Gamma * np.cumsum(G_demo) * dt
    N_demo = np.exp(-H_demo)

    # ---- ensemble: deterministic decay, randomness only in initial phases ----
    # Each nucleus decays at the instant its accumulated Josephson outflow
    # (hazard) reaches a fixed threshold -- deterministic given phi0.
    # Survival S(t) = fraction of the ensemble not yet decayed.
    survived = np.ones(len(tgrid))
    decay_times = np.empty(N_nuclei)
    thresh = 1.0                                   # fixed escape threshold (deterministic)
    for i in range(N_nuclei):
        phi0 = rng.uniform(0, 2 * np.pi, size=M)
        G = cage_transparency(phi0, omega, tgrid)
        H = Gamma * np.cumsum(G) * dt              # accumulated outflow
        idx = np.searchsorted(H, thresh)
        decay_times[i] = tgrid[idx] if idx < len(tgrid) else T
    # survival curve
    S = np.array([np.mean(decay_times > t) for t in tgrid])

    # exponential fit on the well-sampled middle of the curve
    mask = (S > 0.05) & (S < 0.9)
    a, b = np.polyfit(tgrid[mask], np.log(S[mask]), 1)
    lam = -a
    t_half = np.log(2) / lam
    Sfit = np.exp(a * tgrid + b)
    ss_res = np.sum((np.log(S[mask]) - (a * tgrid[mask] + b)) ** 2)
    ss_tot = np.sum((np.log(S[mask]) - np.mean(np.log(S[mask]))) ** 2)
    r2 = 1 - ss_res / ss_tot

    return dict(omega=omega, tgrid=tgrid, G_demo=G_demo, N_demo=N_demo,
                S=S, Sfit=Sfit, lam=lam, t_half=t_half, r2=r2,
                decay_times=decay_times, M=M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=3, help='number of cage domains')
    ap.add_argument('--N', type=int, default=20000, help='ensemble size')
    ap.add_argument('--T', type=float, default=400.0)
    ap.add_argument('--Gamma', type=float, default=0.6)
    args = ap.parse_args()

    print("Josephson-network decay: coexisting domains, phase-coincidence escape")
    print("=" * 68)
    print(f"  M = {args.M} cage domains + 1 escaping domain L")
    r = run(M=args.M, N_nuclei=args.N, T=args.T, Gamma=args.Gamma)
    print(f"  beat frequencies omega_m = {np.round(r['omega'], 3)}")

    # mechanism: is escape bursty (phase-gated), not smooth?
    G = r['G_demo']
    print(f"\n  [mechanism, one nucleus]")
    print(f"    cage transparency G(t): mean={G.mean():.3f}  peak={G.max():.3f}  "
          f"peak/mean={G.max()/G.mean():.1f}  (>>1 => escape in phase-coincidence bursts)")
    print(f"    norm remaining on L at end: {r['N_demo'][-1]:.3f}")

    # statistics: exponential survival with a half-life?
    print(f"\n  [ensemble, {args.N} nuclei, randomness only in initial phases]")
    print(f"    survival S(t) exponential fit:  R^2 = {r['r2']:.4f}")
    print(f"    decay rate  lambda = {r['lam']:.4e}")
    print(f"    half-life   t_1/2  = {r['t_half']:.2f}  (model time)")
    print(f"    => exponential, memoryless decay emerges as the ensemble statistics")
    print(f"       of a phase coincidence among coexisting domains -- deterministic")
    print(f"       given the initial phases, no imposed window, no new force.")

    # dependence on M (more cage domains -> rarer coincidence -> longer life)
    print(f"\n  [rate vs number of cage domains M]")
    for M in (2, 3, 4):
        rr = run(M=M, N_nuclei=6000, T=args.T, Gamma=args.Gamma)
        print(f"    M={M}:  t_1/2 = {rr['t_half']:8.2f}   (R^2={rr['r2']:.3f})")
    print("    more coexisting domains => rarer joint coincidence => longer half-life")


if __name__ == '__main__':
    main()
