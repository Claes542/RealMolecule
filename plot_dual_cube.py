"""Plot the dual solution for drag on a cube: slices and profiles."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHI = np.load("dual_cube_3d_phi.npy")        # (3, nx, ny, nz)
p, q, r = PHI
nx, ny, nz = p.shape
AR, CUBE, XC = 2, 0.25, 0.30
hx, h = AR/nx, 1.0/ny
x = (np.arange(nx)+0.5)*hx; y = (np.arange(ny)+0.5)*h
jz = nz//2                                    # mid-plane z = 0.5
mag = np.sqrt(p**2+q**2+r**2)

fig, ax = plt.subplots(2, 2, figsize=(12, 7.2))

def body(a):
    x0, s = XC*x[-1], CUBE/2
    a.add_patch(plt.Rectangle((x0-s, 0.5-s), CUBE, CUBE, fill=False, ec='w', lw=1.6))

im = ax[0,0].imshow(p[:,:,jz].T, origin='lower', aspect='auto', cmap='RdBu_r',
                    extent=[x[0],x[-1],y[0],y[-1]], vmin=-1.2, vmax=1.2)
ax[0,0].set_title(r'$\phi_x$  (drag direction),  $z=0.5$'); body(ax[0,0])
plt.colorbar(im, ax=ax[0,0])

im = ax[0,1].imshow(np.log10(mag[:,:,jz].T+1e-6), origin='lower', aspect='auto',
                    cmap='magma', extent=[x[0],x[-1],y[0],y[-1]])
ax[0,1].set_title(r'$\log_{10}|\phi|$  — where the output is sensitive'); body(ax[0,1])
plt.colorbar(im, ax=ax[0,1])

g = np.gradient(p, hx, h, h)
gm = np.sqrt(sum(c**2 for c in g))
im = ax[1,0].imshow(np.log10(gm[:,:,jz].T+1e-6), origin='lower', aspect='auto',
                    cmap='viridis', extent=[x[0],x[-1],y[0],y[-1]])
ax[1,0].set_title(r'$\log_{10}|\nabla\phi_x|$  — the weight on the residual'); body(ax[1,0])
plt.colorbar(im, ax=ax[1,0])

ax[1,1].semilogy(x, np.abs(p[:, ny//2, jz])+1e-12, label=r'$|\phi_x|$ on the axis')
ax[1,1].semilogy(x, np.sqrt((mag**2).mean(axis=(1,2)))+1e-12, label=r'rms $|\phi|$ over $y,z$')
ax[1,1].axvline(XC*x[-1], color='k', ls='--', lw=1, label='cube')
ax[1,1].set_xlabel('x  (flow left to right; dual propagates right to left)')
ax[1,1].legend(); ax[1,1].grid(alpha=.3)
ax[1,1].set_title('upstream decay of the dual')

for a in ax.ravel()[:3]: a.set_xlabel('x'); a.set_ylabel('y')
fig.suptitle('Dual solution for drag on a cube — 3D, frozen base flow, $Re_d\\approx125$')
fig.tight_layout()
fig.savefig("dual_cube_3d.png", dpi=120)
print("written dual_cube_3d.png")
print(f"  max|phi_x| = {np.abs(p).max():.3f}   mean|phi| far upstream (x<0.2) = "
      f"{mag[x<0.2].mean():.3e}   downstream (x>1.5) = {mag[x>1.5].mean():.3e}")
