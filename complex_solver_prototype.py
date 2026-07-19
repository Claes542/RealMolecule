import numpy as np
# Prototype ComplexRealQM real-time solver, Cartesian (u,v) = Re,Im, Visscher leapfrog.
# hbar=1, m=1 -> mu_B = 1/2. Single electron, 2D, harmonic V=1/2 r^2.
# Init: m=1 winding psi=(x+iy) exp(-r^2/2)  (an L_z=+1 eigenstate) -> mu_z should hold at -mu_B.
N=161; L=8.0; dx=2*L/(N-1); m=1.0; muB=0.5
x=np.linspace(-L,L,N); X,Y=np.meshgrid(x,x,indexing='ij'); r2=X**2+Y**2
V=0.5*r2
def lap(f):
    f=f.copy()
    return (np.roll(f,1,0)+np.roll(f,-1,0)+np.roll(f,1,1)+np.roll(f,-1,1)-4*f)/dx**2
def H(f): return -0.5/m*lap(f) + V*f

# initial complex state, normalized
psi = (X+1j*Y)*np.exp(-r2/2.0)
psi/= np.sqrt(np.sum(np.abs(psi)**2)*dx*dx)
u=psi.real.copy(); v=psi.imag.copy()

def moment(u,v):
    # J = q/m Im(psi* grad psi) = -(1/m)(u grad v - v grad u), electron q=-1
    gy_v=(np.roll(v,-1,1)-np.roll(v,1,1))/(2*dx); gx_v=(np.roll(v,-1,0)-np.roll(v,1,0))/(2*dx)
    gy_u=(np.roll(u,-1,1)-np.roll(u,1,1))/(2*dx); gx_u=(np.roll(u,-1,0)-np.roll(u,1,0))/(2*dx)
    Jx=-(u*gx_v - v*gx_u)/m; Jy=-(u*gy_v - v*gy_u)/m
    mu_z=0.5*np.sum(X*Jy - Y*Jx)*dx*dx
    return mu_z

dt=0.2*dx**2   # CFL: dt < ~1/H_max ~ dx^2
# Visscher: v staggered half step behind
v_half = v - 0.5*dt*H(u)
print(f"grid {N}x{N} dx={dx:.4f} dt={dt:.2e}   mu_B={muB}")
print(f"{'step':>6} {'t':>7} {'norm':>10} {'mu_z':>10} {'mu_z/muB':>10}")
norm0=None
for n in range(0,4001):
    if n%800==0:
        v_int=v_half+0.5*dt*H(u)                 # v at integer step (approx)
        norm=np.sum(u*u+v_half*v_int)*dx*dx      # Visscher conserved norm
        if norm0 is None: norm0=norm
        mu=moment(u,v_int)
        print(f"{n:6d} {n*dt:7.3f} {norm:10.6f} {mu:10.6f} {mu/muB:10.5f}")
    u = u + dt*H(v_half)
    v_half = v_half - dt*H(u)
