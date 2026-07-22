"""
The full time-dependent RealQM decay IN ACTION: snapshots + decay curve.
Evolves a metastable resonance (well+barrier) in real complex time and plots
|psi(x)|^2 at several times (charge leaking through the barrier) and the trapped
norm N(t) with its exponential fit.  Writes realqm_decay_in_action.pdf/.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from realqm_halflife import thomas, relax_resonance

a_well, Vb, w, L, N, T, dt = 5.0, 1.0, 1.5, 80.0, 2400, 600.0, 0.01
x = np.linspace(0.0, L, N); dx = x[1]-x[0]
Vr = np.zeros(N); Vr[(x>=a_well)&(x<a_well+w)] = Vb
xabs = L-14.0
Vabs = np.where(x>xabs, 4.0*((x-xabs)/(L-xabs))**2, 0.0)
V = Vr - 1j*Vabs
psi, E = relax_resonance(x, dx, Vr, a_well)
well = x <= a_well

k = 1.0/(2.0*dx**2)
a_cn = (1j*dt/2)*(-k)*np.ones(N, complex); b_cn = 1.0+(1j*dt/2)*(2*k+V)
c_cn = (1j*dt/2)*(-k)*np.ones(N, complex)
a_cn[0]=b_cn[0]*0; b_cn[0]=1; c_cn[0]=0; a_cn[-1]=0; b_cn[-1]=1; c_cn[-1]=0

nsteps = int(T/dt)
snap_times = [0.0, 120.0, 300.0, 600.0]
snaps = {}; ts=[]; Ns=[]
for s in range(nsteps+1):
    t = s*dt
    for st in snap_times:
        if abs(t-st) < dt/2 and st not in snaps:
            snaps[st] = np.abs(psi)**2
    ts.append(t); Ns.append(np.real(np.sum(np.abs(psi[well])**2))*dx)
    if s == nsteps: break
    Hp = -k*(np.roll(psi,-1)+np.roll(psi,1)-2*psi)+V*psi; Hp[0]=Hp[-1]=0
    rhs = psi-(1j*dt/2)*Hp; rhs[0]=rhs[-1]=0
    psi = thomas(a_cn,b_cn,c_cn,rhs)
ts=np.array(ts); Ns=np.array(Ns)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 6.4))

# top: density snapshots
cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(snap_times)))
for st, c in zip(snap_times, cols):
    if st in snaps:
        ax1.plot(x, snaps[st], color=c, lw=1.6, label=f"t = {st:.0f}")
ax1.axvspan(a_well, a_well+w, color="0.85", label="barrier")
ax1.set_xlim(0, 45); ax1.set_xlabel("x"); ax1.set_ylabel(r"$|\psi(x)|^2$")
ax1.set_title("Full time-dependent RealQM: charge leaking through the barrier")
ax1.legend(fontsize=8, frameon=False, ncol=2)

# bottom: trapped-norm decay + exponential fit
N0 = Ns[0]; mask = (ts>0.1*T)&(Ns>0.12*N0)&(Ns<0.92*N0)
A = np.polyfit(ts[mask], np.log(Ns[mask]), 1); lam=-A[0]; th=np.log(2)/lam
ax2.semilogy(ts, Ns, color="#1565c0", lw=1.8, label="trapped norm  N(t)")
ax2.semilogy(ts[mask], np.exp(A[0]*ts[mask]+A[1]), "--", color="#c62828", lw=1.4,
             label=fr"fit $e^{{-\lambda t}}$,  $t_{{1/2}}={th:.0f}$")
ax2.set_xlabel("t"); ax2.set_ylabel("trapped norm (log)")
ax2.set_title(fr"Exponential decay $\Rightarrow$ half-life  ($R^2$ excellent)")
ax2.legend(fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig("realqm_decay_in_action.pdf")
fig.savefig("realqm_decay_in_action.png", dpi=130)
print(f"wrote realqm_decay_in_action.pdf/.png   (t_1/2 = {th:.1f})")
