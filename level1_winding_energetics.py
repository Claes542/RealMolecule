import numpy as np
# Level 1: relax m=0 and m=1 states (2D radial, imaginary time) in a harmonic well,
# get winding cost DeltaE = E(m=1)-E(m=0); then scale to free (a0) vs caged (fm).
# atomic units hbar=m_e=1; 1 Hartree=27.2114 eV; a0=5.29177e4 fm.  2D h.o.: E=(2n_r+|m|+1)*omega.
N=4000; L=12.0; ds=L/N; s=(np.arange(N)+0.5)*ds
def relax(mq, V, steps=60000, dt=None):
    if dt is None: dt=0.3*ds**2
    R=np.exp(-0.5*s**2)*(s**mq)          # sensible init (node on axis if m=1)
    for _ in range(steps):
        Rp1=np.roll(R,-1); Rm1=np.roll(R,1); Rm1[0]=R[0]; Rp1[-1]=0.0
        lap=(Rp1-2*R+Rm1)/ds**2 + (Rp1-Rm1)/(2*ds*s)
        H=-0.5*lap + (0.5*mq**2/s**2 + V)*R
        R=R-dt*H
        R[-1]=0.0
        nrm=np.sqrt(np.sum(R*R*s)*ds*2*np.pi); R/=nrm
    Rp1=np.roll(R,-1); Rm1=np.roll(R,1); Rm1[0]=R[0]; Rp1[-1]=0.0
    Rprime=(Rp1-Rm1)/(2*ds)
    E=np.sum((0.5*Rprime**2 + 0.5*mq**2/s**2*R**2 + V*R**2)*s)*ds*2*np.pi
    return E
V=0.5*s**2                      # omega=1 harmonic well (dimensionless)
E0=relax(0,V); E1=relax(1,V)
print(f"dimensionless harmonic well (omega=1):  E(m=0)={E0:.4f}  E(m=1)={E1:.4f}  (exact 1, 2)")
print(f"winding cost  DeltaE = E(m=1)-E(m=0) = {E1-E0:.4f} * hbar*omega   (exact 1)")
print(f"m=1 state magnetic moment mu_z = -mu_B  (L_z=+1, electron q=-1), by construction\n")

Ha=27.2114; a0_fm=5.29177e4
def cost_eV(sigma_fm):
    # harmonic sigma = 1/sqrt(omega) (a.u.) ; DeltaE = hbar*omega = 1/sigma^2 (Hartree), sigma in a0
    sig_a0=sigma_fm/a0_fm
    return (1.0/sig_a0**2)*Ha
print("scaling the (dimensionless) winding cost to physical confinement scales:")
for name,sig in [("free electron  sigma~1 a0", a0_fm),
                 ("caged e (deuteron) sigma=2 fm", 2.0)]:
    E=cost_eV(sig)
    s_=f"{E:.1f} eV" if E<1e3 else (f"{E/1e6:.2f} MeV" if E<1e9 else f"{E/1e9:.2f} GeV")
    print(f"  {name:32s} DeltaE(winding) = {s_}")
Ec=cost_eV(2.0)
print(f"\n  caged winding {Ec/1e9:.1f} GeV vs deuteron binding 2.2 MeV  ->  ~{Ec/2.2e6:.0e}x : forbidden -> caged electron locked in m=0 -> NO moment")
print(f"  free winding {cost_eV(a0_fm):.0f} eV ~ level spacing: accessible (but the free ground state is still m=0; its mu_B is intrinsic/Compton, not this orbital winding)")
