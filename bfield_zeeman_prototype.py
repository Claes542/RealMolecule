import numpy as np
# Circulating RealQM charge density in a magnetic field, via minimal coupling p -> p - qA.
# hbar=m=1, electron q=-1, muB=1/2. Uniform B along z: A = (B/2)(-y, x).
# Show: m=+1/-1 (circulating) shift linearly (Zeeman, +-muB*B); m=0 (no current) does not.
N=241; L=8.0; dx=2*L/(N-1); muB=0.5; q=-1.0
x=np.linspace(-L,L,N); X,Y=np.meshgrid(x,x,indexing='ij'); r2=X**2+Y**2
V=0.5*r2
def state(m):
    psi = (np.exp(-r2/2)*(1+0j)) if m==0 else ((X+1j*np.sign(m)*Y)**abs(m))*np.exp(-r2/2)
    return psi/np.sqrt(np.sum(np.abs(psi)**2)*dx*dx)
def grad(f,ax): return (np.roll(f,-1,ax)-np.roll(f,1,ax))/(2*dx)
def energy(psi,B):
    Ax=-B/2*Y; Ay=B/2*X
    Dx=-1j*grad(psi,0)-q*Ax*psi          # (-i d - qA) psi
    Dy=-1j*grad(psi,1)-q*Ay*psi
    T=0.5*np.sum(np.abs(Dx)**2+np.abs(Dy)**2)*dx*dx
    return (T+np.sum(V*np.abs(psi)**2)*dx*dx).real
Bs=[0.0,0.05,0.10,0.20]
print(f"{'B':>6}"+"".join(f"  E(m={m:+d})" for m in (-1,0,1))+"   E(+1)-E(-1)   2*muB*B")
for B in Bs:
    e={m:energy(state(m),B) for m in (-1,0,1)}
    print(f"{B:6.2f}  {e[-1]:8.4f}  {e[0]:8.4f}  {e[1]:8.4f}   {e[1]-e[-1]:+9.4f}   {2*muB*B:7.4f}")
print("\ncirculating m=+-1: E shifts LINEARLY in B (Zeeman -mu.B, moment reacting to the field)")
print("non-circulating m=0: E ~ flat to linear order (no current, no moment) -- only weak diamagnetic B^2")
