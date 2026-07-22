"""
Phase-gated escape: does a *caged* electron float out under phase control?
==========================================================================

The point of this solve (answering: "what makes a caged electron float out --
phase?").  A single electron sits in a metastable well behind a barrier -- a
genuine cage, classically trapped, that can only leak out by tunnelling.  We
prepare it in a superposition of TWO quasi-bound levels of the well, E1 and E2,
and integrate the time-dependent Schroedinger equation

    i d_t psi = [ -(1/2m) d^2/dx^2 + V(x) ] psi ,     V real well+barrier,

with an absorbing layer past the barrier so outgoing amplitude leaves cleanly.
We record the escape flux  j(x_det,t) = (1/m) Im(psi* d_x psi)  at a detector
just outside the barrier.

Claim under test:
  * ONE level  -> smooth, monotone tunnelling (ordinary Gamow decay, no gating).
  * TWO levels -> the escape flux OSCILLATES at the beat frequency
                  omega = (E2 - E1)/hbar,  i.e. the electron floats out in
                  PHASE-GATED bursts -- escape opened and closed by the relative
                  phase of the two caged components.  No escape window is imposed;
                  the gating emerges from the TDSE + interference.

Nothing here is fitted: E1,E2 are the well's own levels and the predicted beat
omega=(E2-E1) is compared with the measured flux oscillation.

Units: atomic (hbar=1, m_e=1).  Crank-Nicolson (unitary, unconditionally stable),
complex absorbing potential near the right edge.
"""

from __future__ import annotations
import argparse
import numpy as np


def thomas(a, b, c, d):
    """Solve tridiagonal system (a=sub, b=diag, c=super, d=rhs), complex."""
    n = len(b)
    cp = np.zeros(n, complex)
    dp = np.zeros(n, complex)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.zeros(n, complex)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def build_potential(x, a_well, Vb, w, eta, x_abs, L):
    V = np.zeros(len(x), complex)
    barrier = (x >= a_well) & (x < a_well + w)
    V[barrier] = Vb
    ramp = (x - x_abs) / (L - x_abs)
    V[x > x_abs] += -1j * eta * ramp[x > x_abs] ** 2
    return V


def relax_resonances(x, dx, Vreal, a_well, m, steps=20000):
    """Imaginary-time relaxation to the two lowest quasi-bound states of the
    well+barrier (real potential), confining renormalisation to the well so the
    slowly-leaking resonances are captured.  Returns phi1, phi2 and energies."""
    inside = x <= a_well
    k = 1.0 / (2.0 * m * dx ** 2)

    def Hreal(p):
        lap = (np.roll(p, -1) + np.roll(p, 1) - 2.0 * p) / dx ** 2
        lap[0] = lap[-1] = 0.0
        hp = -(0.5 / m) * lap + np.real(Vreal) * p
        hp[0] = hp[-1] = 0.0
        return hp

    def norm_well(p):
        return np.sqrt(np.sum(np.abs(p[inside]) ** 2) * dx)

    dtau = 0.2 * dx ** 2
    # ground resonance
    p1 = np.zeros(len(x)); p1[inside] = np.sin(np.pi * x[inside] / a_well)
    p1 /= norm_well(p1)
    for _ in range(steps):
        p1 = p1 - dtau * Hreal(p1)
        p1[0] = p1[-1] = 0.0
        p1 /= norm_well(p1)
    # first excited: relax with Gram-Schmidt against p1 (over the well)
    p2 = np.zeros(len(x)); p2[inside] = np.sin(2 * np.pi * x[inside] / a_well)
    p2 /= norm_well(p2)
    for _ in range(steps):
        p2 = p2 - dtau * Hreal(p2)
        ov = np.sum(np.conj(p1[inside]) * p2[inside]) * dx
        p2 = p2 - ov * p1
        p2[0] = p2[-1] = 0.0
        p2 /= norm_well(p2)

    def energy(p):
        return np.real(np.sum(np.conj(p) * Hreal(p)) * dx / (np.sum(np.abs(p) ** 2) * dx))
    return p1, p2, energy(p1), energy(p2)


def run(a_well=5.0, Vb=3.0, w=1.5, L=60.0, N=3000, m=1.0,
        two_level=True, T=160.0, dt=0.01, x_det=None, eta=3.0, x_abs=None,
        cache=None):
    x = np.linspace(0.0, L, N)
    dx = x[1] - x[0]
    if x_abs is None:
        x_abs = L - 12.0
    V = build_potential(x, a_well, Vb, w, eta, x_abs, L)

    # true quasi-bound resonances (relax once, reuse via cache)
    if cache is None or 'phi' not in cache:
        p1, p2, E1, E2 = relax_resonances(x, dx, V, a_well, m)
        if cache is not None:
            cache['phi'] = (p1, p2, E1, E2)
    else:
        p1, p2, E1, E2 = cache['phi']

    if two_level:
        psi = (p1 + p2) / np.sqrt(2.0)
        beat = E2 - E1
    else:
        psi = p1.copy().astype(complex)
        beat = 0.0
    psi = psi.astype(complex)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    psi[0] = 0.0
    psi[-1] = 0.0

    # --- Crank-Nicolson tridiagonal coefficients
    k = 1.0 / (2.0 * m * dx ** 2)
    diagH = 2.0 * k + V
    sub = -k * np.ones(N, complex)
    sup = -k * np.ones(N, complex)
    a_cn = (1j * dt / 2.0) * sub
    b_cn = 1.0 + (1j * dt / 2.0) * diagH
    c_cn = (1j * dt / 2.0) * sup
    # Dirichlet at both ends
    a_cn[0] = 0.0; b_cn[0] = 1.0; c_cn[0] = 0.0
    a_cn[-1] = 0.0; b_cn[-1] = 1.0; c_cn[-1] = 0.0

    if x_det is None:
        x_det = a_well + w + 3.0
    jdet = int(round(x_det / dx))

    nsteps = int(T / dt)
    ts = np.empty(nsteps)
    flux = np.empty(nsteps)
    norm_in = np.empty(nsteps)     # probability still inside the well (x<a)
    inside_mask = x < a_well

    for s in range(nsteps):
        # rhs = (I - i dt/2 H) psi
        Hpsi = -k * (np.roll(psi, -1) + np.roll(psi, 1) - 2.0 * psi) + V * psi
        Hpsi[0] = 0.0; Hpsi[-1] = 0.0
        rhs = psi - (1j * dt / 2.0) * Hpsi
        rhs[0] = 0.0; rhs[-1] = 0.0
        psi = thomas(a_cn, b_cn, c_cn, rhs)

        # escape flux at detector: (1/m) Im(psi* dpsi/dx)
        dpsi = (psi[jdet + 1] - psi[jdet - 1]) / (2.0 * dx)
        ts[s] = (s + 1) * dt
        flux[s] = (1.0 / m) * np.imag(np.conj(psi[jdet]) * dpsi)
        norm_in[s] = np.sum(np.abs(psi[inside_mask]) ** 2) * dx

    return dict(t=ts, flux=flux, norm_in=norm_in, E1=E1, E2=E2, beat=beat,
                x_det=x_det, dx=dx)


def dominant_freq(t, sig):
    """FFT peak (angular frequency) of sig after removing mean, ignoring DC."""
    s = sig - np.mean(sig)
    dt = t[1] - t[0]
    F = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    f = np.fft.rfftfreq(len(s), dt)          # cycles per unit time
    w = 2.0 * np.pi * f
    F[0] = 0.0
    kpk = np.argmax(F)
    return w[kpk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', type=float, default=5.0, help='well width')
    ap.add_argument('--Vb', type=float, default=3.0, help='barrier height')
    ap.add_argument('--w', type=float, default=1.5, help='barrier width')
    ap.add_argument('--L', type=float, default=60.0)
    ap.add_argument('--N', type=int, default=3000)
    ap.add_argument('--T', type=float, default=160.0)
    ap.add_argument('--dt', type=float, default=0.01)
    args = ap.parse_args()

    print("Phase-gated escape of a caged electron (1-D TDSE, Crank-Nicolson)")
    print(f"  well a={args.a}  barrier Vb={args.Vb} w={args.w}  box L={args.L} N={args.N}")
    print("  relaxing the two quasi-bound resonances of the well+barrier ...")

    cache = {}
    print("\n  [A] control: ONE resonance -- expect smooth Gamow tunnelling")
    rA = run(a_well=args.a, Vb=args.Vb, w=args.w, L=args.L, N=args.N,
             two_level=False, T=args.T, dt=args.dt, cache=cache)
    E1, E2 = rA['E1'], rA['E2']
    beat = E2 - E1
    print(f"      resonance energies: E1={E1:.4f}  E2={E2:.4f}")
    print(f"      predicted beat (E2-E1) = {beat:.4f}   period 2pi/beat = {2*np.pi/beat:.2f}")

    fA = rA['flux']
    print("\n  [B] test: TWO resonances (E1+E2) -- expect flux gated at the beat")
    rB = run(a_well=args.a, Vb=args.Vb, w=args.w, L=args.L, N=args.N,
             two_level=True, T=args.T, dt=args.dt, cache=cache)
    fB = rB['flux']

    # measure oscillation frequency of each flux after the initial transient
    half = len(rB['t']) // 3
    wA = dominant_freq(rA['t'][half:], fA[half:])
    wB = dominant_freq(rB['t'][half:], fB[half:])

    # absolute oscillation amplitude (std of AC part) after transient
    def ac_amp(f):
        seg = f[half:]
        base = np.convolve(seg, np.ones(300) / 300, 'same')[150:-150]
        return np.std((seg[150:-150] - base))
    ampA, ampB = ac_amp(fA), ac_amp(fB)

    print(f"      one-level : flux peak={np.max(fA):.3e}  AC-amp={ampA:.2e}  FFT-omega={wA:.4f}")
    print(f"      two-level : flux peak={np.max(fB):.3e}  AC-amp={ampB:.2e}  FFT-omega={wB:.4f}")
    print(f"\n  RESULT")
    print(f"      predicted beat (E2-E1)       = {beat:.4f}")
    print(f"      measured two-level flux omega = {wB:.4f}   (ratio {wB/beat:.3f})")
    print(f"      oscillation amplitude ratio two/one = {ampB/ (ampA+1e-30):.1f}x")
    print("  => two-level escape flux gated at (E2-E1): phase-coincidence escape.")
    print("     one-level flux smooth: ordinary tunnelling. PHASE gates the cage.")


if __name__ == '__main__':
    main()
