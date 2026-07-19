import numpy as np
# Does joining the spinor field across the shared boundary FORCE antiparallel (diamagnetic)?
# Spin orientation angle theta(x); spinor chi=(cos(theta/2),sin(theta/2)). up=0, down=pi.
# Spin-texture kinetic energy T_spin = (1/8) integral (theta')^2 dx.
L=10.0; N=4001; x=np.linspace(-L,L,N); dx=x[1]-x[0]
Tspin=lambda th: 0.125*np.sum(np.gradient(th,dx)**2)*dx
th_par=np.zeros_like(x)                          # parallel: both up
print("PARALLEL vs ANTIPARALLEL (domain wall) spin-texture kinetic energy:")
for w in (0.5,1.0,2.0):
    th_anti=0.5*np.pi*(1+np.tanh(x/w))           # up->down wall = antiparallel
    print(f"  w={w}: parallel={Tspin(th_par):.4f}  antiparallel={Tspin(th_anti):.4f}  (pi^2/24w={np.pi**2/(24*w):.4f})")
print("=> antiparallel (domain wall) costs energy; parallel costs 0 -> geometry FAVORS PARALLEL.")
print("   Plus: separate non-overlapping domains -> spinors uncoupled -> degenerate. Either way,")
print("   geometry gives NO antiparallel pairing; closed-shell diamagnetism needs added exchange.")
