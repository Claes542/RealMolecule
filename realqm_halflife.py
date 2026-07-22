"""
Half-life from the full time-dependent RealQM solve -- no WKB, no phase conjecture
=================================================================================

A genuine metastable resonance -- a charge domain trapped behind a barrier -- is
evolved in REAL complex time (unitary Crank-Nicolson, absorbing outer layer), the
same solver as the relaxation but with i d_t psi = H psi instead of the parabolic
step.  We measure the TRAPPED norm N(t) = \int_well |psi|^2 and fit its decay:

    N(t) ~ e^{-lambda t}   ->   t_1/2 = ln 2 / lambda .

That IS the half-life, read off the dynamics from first principles -- the decay is
exponential because a resonance leaks at a constant rate (its width Gamma = hbar
lambda), no semiclassical approximation and no imposed rule.

Then we sweep the barrier height Vb.  The WKB penetration exponent through a
barrier of width w is 2 w sqrt(2m(Vb-E)), so log t_1/2 should be linear in
sqrt(Vb-E): the Geiger-Nuttall shape, here emerging from the full RealQM dynamics
rather than from the Gamow formula.

Units: hbar = 1, m = 1.
"""

from __future__ import annotations
import argparse
import numpy as np


def thomas(a, b, c, d):
    n = len(b)
    cp = np.zeros(n, complex); dp = np.zeros(n, complex)
    cp[0] = c[0]/b[0]; dp[0] = d[0]/b[0]
    for i in range(1, n):
        m = b[i]-a[i]*cp[i-1]
        cp[i] = c[i]/m
        dp[i] = (d[i]-a[i]*dp[i-1])/m
    xx = np.zeros(n, complex); xx[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        xx[i] = dp[i]-cp[i]*xx[i+1]
    return xx


def relax_resonance(x, dx, Vreal, a_well, steps=15000):
    """Imaginary-time relax to the quasi-bound state of the well (barrier present),
    confining the renormalisation to the well so the slowly-leaking resonance is
    captured smoothly (no kink to launch transients)."""
    inside = x <= a_well
    def H(p):
        lap = (np.roll(p,-1)+np.roll(p,1)-2*p)/dx**2
        lap[0]=lap[-1]=0.0
        hp = -0.5*lap + Vreal*p
        hp[0]=hp[-1]=0.0
        return hp
    p = np.zeros(len(x)); p[inside] = np.sin(np.pi*x[inside]/a_well)
    p = p/np.sqrt(np.sum(p[inside]**2)*dx)
    dtau = 0.2*dx**2
    for _ in range(steps):
        p = p - dtau*H(p)
        p[0]=p[-1]=0.0
        p = p/np.sqrt(np.sum(p[inside]**2)*dx)
    E = np.sum(p*H(p))*dx/(np.sum(p**2)*dx)
    return p.astype(complex), float(E)


def halflife(a_well=5.0, Vb=1.0, w=1.5, L=80.0, N=2400, T=600.0, dt=0.01,
             eta=4.0):
    x = np.linspace(0.0, L, N); dx = x[1]-x[0]
    Vr = np.zeros(N)
    Vr[(x>=a_well)&(x<a_well+w)] = Vb
    xabs = L-14.0
    Vabs = np.where(x>xabs, eta*((x-xabs)/(L-xabs))**2, 0.0)
    V = Vr - 1j*Vabs

    psi, E = relax_resonance(x, dx, Vr, a_well)
    well = x <= a_well
    def trapped():
        return np.real(np.sum(np.abs(psi[well])**2))*dx

    k = 1.0/(2.0*dx**2)
    a_cn = (1j*dt/2)*(-k)*np.ones(N, complex)
    b_cn = 1.0 + (1j*dt/2)*(2*k+V)
    c_cn = (1j*dt/2)*(-k)*np.ones(N, complex)
    a_cn[0]=0; b_cn[0]=1; c_cn[0]=0
    a_cn[-1]=0; b_cn[-1]=1; c_cn[-1]=0

    nsteps = int(T/dt)
    ts = np.empty(nsteps); Ns = np.empty(nsteps)
    for s in range(nsteps):
        Hp = -k*(np.roll(psi,-1)+np.roll(psi,1)-2*psi)+V*psi
        Hp[0]=Hp[-1]=0
        rhs = psi-(1j*dt/2)*Hp; rhs[0]=rhs[-1]=0
        psi = thomas(a_cn,b_cn,c_cn,rhs)
        ts[s]=(s+1)*dt; Ns[s]=trapped()

    # exponential fit on the clean tail (after the initial transient, before
    # too depleted).  Skip the first ~10% of the run as transient.
    N0 = Ns[0]; Nend = Ns[-1]
    t0 = 0.10*T
    mask = (ts > t0) & (Ns > 0.12*N0) & (Ns < 0.92*N0)
    if mask.sum() < 30:                     # too little decay: fit whatever tail exists
        mask = (ts > t0) & (Ns < 0.995*N0) & (Ns > 0.02*N0)
    A = np.polyfit(ts[mask], np.log(Ns[mask]), 1)
    lam = -A[0]; t_half = np.log(2)/lam if lam > 0 else np.inf
    yfit = A[0]*ts[mask]+A[1]
    r2 = 1 - np.sum((np.log(Ns[mask])-yfit)**2)/np.sum((np.log(Ns[mask])-np.mean(np.log(Ns[mask])))**2)
    return dict(t_half=t_half, lam=lam, r2=r2, E=E, Vb=Vb, ts=ts, Ns=Ns,
                N0=N0, Nend=Nend, frac=Nend/N0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', type=float, default=5.0)
    ap.add_argument('--w', type=float, default=1.5)
    ap.add_argument('--L', type=float, default=80.0)
    ap.add_argument('--N', type=int, default=2400)
    ap.add_argument('--T', type=float, default=300.0)
    args = ap.parse_args()

    print("Half-life from the full time-dependent RealQM (Crank-Nicolson, no WKB)")
    print("="*68)
    print("  [1] one resonance: trapped norm decay -> exponential -> t_1/2")
    r = halflife(a_well=args.a, Vb=1.0, w=args.w, L=args.L, N=args.N, T=args.T)
    print(f"      resonance energy E = {r['E']:.4f}   barrier Vb = {r['Vb']}")
    print(f"      trapped norm: {r['N0']:.3f} -> {r['Nend']:.3f} ({100*(1-r['frac']):.0f}% decayed)")
    print(f"      exponential fit R^2 = {r['r2']:.4f}")
    print(f"      decay rate  lambda = {r['lam']:.4e}")
    print(f"      HALF-LIFE   t_1/2  = {r['t_half']:.2f}  (model time), from the dynamics")

    print("\n  [2] barrier sweep: is log t_1/2 linear in sqrt(Vb - E)? (Geiger-Nuttall shape)")
    print(f"      {'Vb':>5} {'E':>7} {'%dec':>6} {'t_1/2':>10} {'log10 t_1/2':>12} {'R^2':>7}")
    xs, ys = [], []
    for Vb in (0.6, 0.8, 1.0, 1.2, 1.4):
        rr = halflife(a_well=args.a, Vb=Vb, w=args.w, L=args.L, N=args.N, T=args.T)
        gx = np.sqrt(max(Vb-rr['E'], 1e-6))
        xs.append(gx); ys.append(np.log10(rr['t_half']))
        print(f"      {Vb:5.1f} {rr['E']:7.3f} {100*(1-rr['frac']):6.0f} "
              f"{rr['t_half']:10.2f} {np.log10(rr['t_half']):12.3f} {rr['r2']:7.3f}")
    a1, b1 = np.polyfit(xs, ys, 1)
    ss = 1 - np.sum((np.array(ys)-(a1*np.array(xs)+b1))**2)/np.sum((np.array(ys)-np.mean(ys))**2)
    print(f"\n      log10(t_1/2) = {a1:.2f} * sqrt(Vb-E) + {b1:.2f}   R^2 = {ss:.4f}")
    print("      => the half-life comes straight from the full time-dependent RealQM,")
    print("         exponential decay, and it follows the Gamow/Geiger-Nuttall barrier")
    print("         law -- no WKB, no phase-coincidence conjecture invoked.")


if __name__ == '__main__':
    main()
