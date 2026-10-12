import sys, numpy as np; sys.path.insert(0,'.')
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, factorized
from axisym_realqm import *
from scipy.optimize import brentq
np.seterr(all='ignore')
GRID=np.sqrt(0.081)      # grid softening coefficient: eps_grid = 0.285*h
_c={}
def eps_tot(Rk,h): return np.sqrt(Rk*Rk + (GRID*h)**2)
def one(h,L,kern,Rk,f):
    """f = 2*kappa ; f=1 is the physical kinetic coefficient 1/2."""
    g=Grid(L,L,h); K=kernel_potential(g,list(kern),eps_tot(Rk,h))
    S=stiffness(g)*f
    D=sp.diags((-K)*g.V); H=(S+D).tocsr()
    w=1.0/np.sqrt(g.V); W=sp.diags(w)
    val,vec=eigsh((W@H@W).tocsr(),k=1,which='SA',tol=1e-11,maxiter=300000)
    return val[0]
def two(h,L,kern,nn,Rk,f):
    nzh=int(round(L/h)); L=nzh*h
    gh=Grid(L,L,h,zmin=0.0); gf=Grid(L,L,h,zmin=-L)
    key=(h,L)
    if key not in _c:
        Lp=laplacian(gf).tolil(); bnd=np.zeros(gf.N,bool)
        bnd[gf.idx(0,np.arange(gf.nr))]=True; bnd[gf.idx(gf.nz-1,np.arange(gf.nr))]=True
        bnd[gf.idx(np.arange(gf.nz),gf.nr-1)]=True
        for k in np.where(bnd)[0]: Lp.rows[k]=[k]; Lp.data[k]=[1.0]
        _c[key]=(stiffness(gh),factorized(Lp.tocsc()),bnd,np.sqrt(gf.Z**2+gf.R**2).ravel(),gh,gf)
    S0,solveP,bnd,rrf,gh,gf=_c[key]
    S=S0*f; K=kernel_potential(gh,list(kern),eps_tot(Rk,h))
    w=1.0/np.sqrt(gh.V); W=sp.diags(w); mir=np.arange(gh.nz)[::-1]
    P1=np.zeros(gh.N); u=None; Eo=None
    for it in range(90):
        A=(W@(S+sp.diags((-K+P1)*gh.V)).tocsr()@W).tocsr()
        val,vec=eigsh(A,k=1,which='SA',tol=1e-11,maxiter=300000,v0=None if u is None else u/w)
        u=w*vec[:,0]; u=u/np.sqrt(np.sum(u*u*gh.V))
        densf=np.zeros((gf.nz,gf.nr)); densf[gh.nz:,:]=(u*u).reshape(gh.nz,gh.nr)
        b=4*np.pi*densf.ravel()*gf.V; b[bnd]=1.0/rrf[bnd]
        P1=solveP(b).reshape(gf.nz,gf.nr)[mir,:].ravel()
        T=float(u@(S@u)); VK=-float(np.sum(K*u*u*gh.V)); U12=float(np.sum(P1*u*u*gh.V))
        E=2*(T+VK)+U12+nn
        if Eo is not None and abs(E-Eo)<1e-9: break
        Eo=E
    return E
h=0.12; L=12.
print("ONE-PARAMETER FAMILY, EVERY MEMBER CALIBRATED TO E(H) = -0.5 EXACTLY")
print("kernel radius R_k, with the kinetic factor f=2*kappa solved to restore -0.5.")
print("H2 is then a PREDICTION.  Standard model = (R_k=0, f=1) -> H2 = -1.2769 at h=0.12\n")
print("  R_k     f        E(H)       H2 @D=1.401   H2 @D=1.30   H2 @D=1.60   exact -1.17448")
for Rk in (0.0,0.1,0.2,0.3,0.4,0.5):
    try:
        f=brentq(lambda t: one(h,L,[(0.,1.)],Rk,t)+0.5, 0.25, 3.0, xtol=1e-7)
    except Exception as e:
        print("  %.2f   no f found (%s)" % (Rk,e.__class__.__name__)); continue
    eH=one(h,L,[(0.,1.)],Rk,f)
    row=[two(h,L,[(-D/2,1.),(D/2,1.)],1.0/D,Rk,f) for D in (1.401,1.30,1.60)]
    print("  %.2f  %7.4f  %9.5f   %10.5f  %10.5f  %10.5f"
          % (Rk,f,eH,row[0],row[1],row[2]), flush=True)
