import numpy as np
from scipy import integrate
z=1.6875  # variational He 1s exponent
Phi=lambda r: 2*np.exp(-2*z*r)*(z+1/r)          # potential of neutral frozen He (nucleus + 2e)
n=lambda r: 2*z**3/np.pi*np.exp(-2*z*r)          # 2-electron density
def E(R):
    # E_es = Z_B Phi_A(R) - int n_B(r) Phi_A(|r - R|) d^3r ; integrate in coords centred on B
    f=lambda u,r: n(r)*Phi(np.sqrt(r*r+R*R-2*r*R*u)+1e-14)*2*np.pi*r*r
    I,_=integrate.dblquad(f,0,30,-1,1,epsabs=1e-11)
    return 2*Phi(R)-I
for R in [2.0,3.0,4.0,5.6]: print(R, "%+.6f"%E(R))
