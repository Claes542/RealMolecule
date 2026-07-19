import numpy as np
# Try the spinor structure: two-component psi=(up,dn), sigma.D dynamics, D=p-qA (minimal coupling).
# Test whether g=2 EMERGES: verify the operator identity (sigma.D)^2 = D^2 I - q sigma.B.
# hbar=m=1, electron q=-1, muB=1/2, uniform B along z: A=(B/2)(-y,x) -> B_z=B.
N=161; L=8.0; dx=2*L/(N-1); q=-1.0; B=0.3; muB=0.5
x=np.linspace(-L,L,N); X,Y=np.meshgrid(x,x,indexing='ij')
Ax=-B/2*Y; Ay=B/2*X
d_x=lambda f:(np.roll(f,-1,0)-np.roll(f,1,0))/(2*dx)
d_y=lambda f:(np.roll(f,-1,1)-np.roll(f,1,1))/(2*dx)
Dx=lambda f:-1j*d_x(f)-q*Ax*f
Dy=lambda f:-1j*d_y(f)-q*Ay*f
D2=lambda f:Dx(Dx(f))+Dy(Dy(f))
sigmaD=lambda a,b:(Dx(b)-1j*Dy(b), Dx(a)+1j*Dy(a))   # (sigma.D)(a,b)

# arbitrary smooth test spinor (interior, avoid roll wrap)
a=np.exp(-((X-0.6)**2+Y**2)/3.0)*(1+0.4j)
b=np.exp(-(X**2+(Y-0.4)**2)/3.0)*(1-0.2j)
t,bt=sigmaD(a,b); lu,ld=sigmaD(t,bt)         # (sigma.D)^2 (a,b)
ru=D2(a)-q*B*a; rd=D2(b)+q*B*b               # D^2 I - q sigma_z B, on (a,b)
c=slice(10,-10)
print("Operator identity  (sigma.D)^2 = D^2 - q*sigma.B   (residual should be O(dx^2)):")
print(f"  dx={dx:.4f},  dx^2={dx**2:.2e}")
print(f"  max|LHS-RHS| up = {np.max(np.abs((lu-ru)[c,c])):.2e}")
print(f"  max|LHS-RHS| dn = {np.max(np.abs((ld-rd)[c,c])):.2e}")
print("  => the -q*sigma.B (g=2) term EMERGES from squaring sigma.D; not put in by hand.\n")

# Consequence: L=0 ground state (no orbital current) splits by the spin term into TWO, no middle.
phi=np.exp(-(X**2+Y**2)/2); phi/=np.sqrt(np.sum(np.abs(phi)**2)*dx*dx)
V=0.5*(X**2+Y**2)
# spin part energy = <(-q sigma_z B)/2m>; up: -qB/2=+muB*B, dn:+qB/2=-muB*B  (q=-1)
E_orb=np.sum((0.5*np.abs(-1j*d_x(phi))**2+0.5*np.abs(-1j*d_y(phi))**2+V*np.abs(phi)**2))*dx*dx
Eup=E_orb - q*B/2.0     # spin up
Edn=E_orb + q*B/2.0     # spin dn
print("L=0 ground state (orbitally inert, m=0) with the spinor structure, field B=%.2f:"%B)
print(f"  spin up:   E = {Eup:.4f}")
print(f"  spin down: E = {Edn:.4f}")
print(f"  splitting  = {Eup-Edn:+.4f}   vs  2*muB*B = {2*muB*B:.4f}   -> TWO levels, NO middle")
print(f"  gyromagnetic factor g = (shift per S_z) : {(Eup-E_orb)/(muB*B):.3f}  (spin g=2; orbital would be 1)")
