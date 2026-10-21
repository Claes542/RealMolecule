"""
PHASE 1 of the Hilbert-6th programme: verify the MADELUNG IDENTITY, the right way.

WHY NOT INTEGRATE THE MADELUNG PAIR DIRECTLY.  For a SINGLE domain the Madelung pair is equivalent
to the Schrodinger equation, so a bespoke Madelung integrator buys no new physics -- only a much
harder numerical problem.  Measured, in tools/madelung_1d.py: Q = -1/2(phi'' + (phi')^2) puts a
THIRD derivative into the momentum equation through Q', making the pair dispersive like KdV, so
explicit stepping needs dt <~ C dx^3.  That is why the "missing half" is not a mere implementation
task, and it is a fact about the route, not about the code.

WHAT IS TESTED HERE INSTEAD.  Propagate the SCHRODINGER equation spectrally (split-step Fourier,
which is accurate and unconditionally stable), then ask whether the hydrodynamic quantities read off
that solution satisfy the Madelung pair:

    R1 = rho_t + (rho v)_x                                  (continuity)
    R2 = v_t + v v_x + (V + Q)_x,    Q = -1/2 (sqrt(rho))_xx / sqrt(rho)    (momentum)

Both should vanish identically.  R1 is probability conservation and is the easy half.  R2 is the
claim that matters: that the quantum potential IS the stress of a continuum mechanics.

THE POINT OF METHOD: every time derivative is taken ANALYTICALLY from the Schrodinger right-hand
side at a single instant,
    psi_t = i( (1/2) psi_xx - V psi ),
so  rho_t = 2 Re(conj(psi) psi_t),  and  J = Im(conj(psi) psi_x),  v = J/rho,
    J_t = Im(conj(psi_t) psi_x + conj(psi) psi_xt).
No finite differencing in time anywhere, so the residuals measure the IDENTITY and not an
integrator.  Residuals are reported relative to the size of the individual terms they cancel
between, which is the only honest normalisation: a small residual means nothing if the terms are
small too.

v = J/rho is meaningless where rho underflows, so norms are restricted to the bulk,
rho > tol*max(rho).  TRAP FOUND HERE, worth recording: the first version zeroed v and Q outside
that mask and then differentiated SPECTRALLY.  A spectral derivative is GLOBAL, so the artificial
jump at the mask edge polluted the whole domain, and the momentum residual came out at 80% of the
terms and GREW proportionally to N -- the 1/dx signature of differencing across a discontinuity.
The physics was never involved.  Residual derivatives are therefore taken with LOCAL 4th-order
finite differences, and the residual is evaluated on the mask ERODED by the stencil half-width, so
no stencil ever straddles the edge.  The Schrodinger propagation itself is still spectral, which is
legitimate: it acts on psi, which is smooth and decays.

REPRODUCIBILITY, settled 2026-10-21.  The numbers quoted in the paper are obtained on a FIXED
PHYSICAL WINDOW |x - x_c| < 2, not on the density-threshold mask that used to be the default.  The
distinction matters: a threshold mask MOVES WITH THE MESH -- refining pulls more low-density points
in, 216/436/875 bulk points at tol=1e-8 for N=512/1024/2048 -- so the residual is measured on a
larger and worse-conditioned set at every refinement and cannot converge.  On the fixed window it
does, at the order of the operators:
    rel R2 = 1.10e-5, 8.08e-7, 8.05e-8 at N = 256, 512, 1024   (ratios 13.6 and 10.0, i.e. 4th order)
and then RISES to 2.45e-7 at N=2048, which is the round-off floor in grad^2 sqrt(rho)/sqrt(rho)
being reached.  Wider windows reach that floor earlier and higher -- 1.9e-5 at |x-x_c|<4 -- exactly
as the round-off explanation requires.  The default below is now the window the paper quotes, so
running this file reproduces the published numbers.
"""
import numpy as np


def psi_coherent(x, t, w=1.0, x0=2.0):
    """Exact coherent state of V = w^2 x^2/2: rigid oscillation, width fixed at 1/sqrt(2w)."""
    xt = x0*np.cos(w*t)
    pt = -x0*w*np.sin(w*t)
    ph = -0.5*w*t + pt*(x - 0.5*xt)        # global + dynamical phase
    return (w/np.pi)**0.25*np.exp(-0.5*w*(x-xt)**2)*np.exp(1j*ph)


def run(N=1024, L=20.0, w=1.0, x0=2.0, T=0.7, dt=1e-4, tol=1e-8, verbose=False, xwin=None):
    x = -L/2 + (L/N)*np.arange(N)
    k = 2*np.pi*np.fft.fftfreq(N, d=L/N)
    V = 0.5*w*w*x*x

    def ddx(f):  return np.fft.ifft(1j*k*np.fft.fft(f))
    def ddxx(f): return np.fft.ifft(-(k*k)*np.fft.fft(f))

    # --- propagate Schrodinger by split-step Fourier -------------------------
    psi = psi_coherent(x, 0.0, w, x0).astype(complex)
    nst = int(round(T/dt))
    eV  = np.exp(-1j*V*dt/2.0)
    eK  = np.exp(-1j*(k*k/2.0)*dt)
    for _ in range(nst):
        psi = eV*psi
        psi = np.fft.ifft(eK*np.fft.fft(psi))
        psi = eV*psi
    t = nst*dt

    # --- how good is the propagation itself? (exact answer known) -----------
    ex  = psi_coherent(x, t, w, x0)
    rho = np.abs(psi)**2
    m   = rho.sum()*(L/N)
    xc  = (x*rho).sum()*(L/N)/m
    sg  = np.sqrt(((x-xc)**2*rho).sum()*(L/N)/m)
    err_psi = np.abs(np.abs(psi) - np.abs(ex)).max()

    # --- hydrodynamic quantities, and the two residuals ---------------------
    px, pxx = ddx(psi), ddxx(psi)
    psit    = 1j*(0.5*pxx - V*psi)           # Schrodinger RHS, exact
    psixt   = ddx(psit)

    h = L/N
    def fd1(f):      # 4th-order central first derivative, LOCAL stencil
        return (-np.roll(f,-2) + 8*np.roll(f,-1) - 8*np.roll(f,1) + np.roll(f,2))/(12*h)
    def fd2(f):
        return (-np.roll(f,-2) + 16*np.roll(f,-1) - 30*f + 16*np.roll(f,1) - np.roll(f,2))/(12*h*h)

    J     = np.imag(np.conj(psi)*px)
    rho_t = 2*np.real(np.conj(psi)*psit)
    J_t   = np.imag(np.conj(psit)*px + np.conj(psi)*psixt)

    # xwin: a FIXED PHYSICAL window about the packet centre.  Needed because a mask defined by a
    # density threshold MOVES WITH THE MESH -- refining pulls more low-density points in (bulk
    # points 216/436/875 at tol=1e-8 for N=512/1024/2048), so the residual is measured on a larger,
    # worse-conditioned set each time and cannot converge.  With xwin the set is fixed and the
    # refinement measures the identity rather than the mask.
    if xwin is not None:
        msk = np.abs(x - x0*np.cos(w*t)) < xwin
    else:
        msk  = rho > tol*rho.max()
    flo  = np.where(msk, rho, 1.0)              # floor only to avoid 0/0; no zeroing of v or Q
    v    = J/flo
    v_t  = (J_t*rho - J*rho_t)/(flo*flo)
    u    = np.sqrt(rho)
    Q    = -0.5*fd2(u)/np.where(msk, u, 1.0)

    R1 = rho_t + fd1(rho*v)
    R2 = v_t + v*fd1(v) + fd1(V + Q)

    # erode the mask by the stencil half-width so no stencil straddles the mask edge
    ev = msk.copy()
    for sh in (-2,-1,1,2): ev &= np.roll(msk, sh)

    def nrm(f): return np.abs(f[ev]).max()
    scale1 = max(nrm(rho_t), nrm(fd1(rho*v)))
    scale2 = max(nrm(v_t), nrm(v*fd1(v)), nrm(fd1(V+Q)))
    msk = ev
    return dict(t=t, N=N, mass=m, centroid=xc, sigma=sg,
                exact_xc=x0*np.cos(w*t), exact_sg=1.0/np.sqrt(2*w), err_psi=err_psi,
                R1=nrm(R1), R2=nrm(R2), s1=scale1, s2=scale2,
                rel1=nrm(R1)/scale1, rel2=nrm(R2)/scale2, bulk=int(msk.sum()))


if __name__ == '__main__':
    import sys
    print(__doc__)
    print("=== A: does the Schrodinger propagation itself reproduce the exact coherent state? ===")
    print("%6s %10s %12s %12s %12s %12s" % ("N","t","centroid","exact","sigma","exact"))
    r = run(N=1024, T=0.7, dt=1e-4)
    print("%6d %10.4f %12.8f %12.8f %12.8f %12.8f"
          % (r['N'], r['t'], r['centroid'], r['exact_xc'], r['sigma'], r['exact_sg']))
    print("   |psi| error %.2e | mass %.12f | centroid err %.2e | width err %.2e"
          % (r['err_psi'], r['mass'], abs(r['centroid']-r['exact_xc']),
             abs(r['sigma']-r['exact_sg'])))

    print("\n=== B: do the hydrodynamic quantities satisfy the MADELUNG PAIR? ===")
    print("Residuals relative to the magnitude of the terms they cancel between.")
    print("%6s %12s %12s %10s | %12s %12s %10s | %7s" %
          ("N","|R1|","terms","rel","|R2|","terms","rel","bulk pts"))
    for N in [256, 512, 1024, 2048]:
        r = run(N=N, T=0.7, dt=1e-4, xwin=2.0)
        print("%6d %12.3e %12.3e %10.2e | %12.3e %12.3e %10.2e | %7d"
              % (N, r['R1'], r['s1'], r['rel1'], r['R2'], r['s2'], r['rel2'], r['bulk']))
        sys.stdout.flush()
    print("\nR1 = continuity (probability conservation). R2 = momentum balance, i.e. the claim that")
    print("Q is the stress. If rel2 falls with N, the hydrodynamic form of RealQM is verified in 1D.")
