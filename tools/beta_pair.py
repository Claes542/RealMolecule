import sys, numpy as np; sys.path.insert(0,'.')
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, factorized
from axisym_realqm import *
from scipy.optimize import brentq
np.seterr(all='ignore')
SOFT=np.sqrt(0.081); _c={}
def solve(h,L,kernels,nn,beta=0.0):
    """U12 -> (1+beta)*U12 ; the variational derivative scales P1 by the same factor."""
    nzh=int(round(L/h)); L=nzh*h
    gh=Grid(L,L,h,zmin=0.0); gf=Grid(L,L,h,zmin=-L)
    key=(h,L)
    if key not in _c:
        Lp=laplacian(gf).tolil(); bnd=np.zeros(gf.N,bool)
        bnd[gf.idx(0,np.arange(gf.nr))]=True; bnd[gf.idx(gf.nz-1,np.arange(gf.nr))]=True
        bnd[gf.idx(np.arange(gf.nz),gf.nr-1)]=True
        for k in np.where(bnd)[0]: Lp.rows[k]=[k]; Lp.data[k]=[1.0]
        _c[key]=(stiffness(gh),factorized(Lp.tocsc()),bnd,np.sqrt(gf.Z**2+gf.R**2).ravel(),gh,gf)
    S,solveP,bnd,rrf,gh,gf=_c[key]
    K=kernel_potential(gh,list(kernels),SOFT*h)
    w=1.0/np.sqrt(gh.V); W=sp.diags(w); mir=np.arange(gh.nz)[::-1]
    P1=np.zeros(gh.N); u=None; Eo=None
    for it in range(90):
        A=(W@(S+sp.diags((-K+(1.0+beta)*P1)*gh.V)).tocsr()@W).tocsr()
        val,vec=eigsh(A,k=1,which='SA',tol=1e-11,maxiter=300000,v0=None if u is None else u/w)
        u=w*vec[:,0]; u=u/np.sqrt(np.sum(u*u*gh.V))
        densf=np.zeros((gf.nz,gf.nr)); densf[gh.nz:,:]=(u*u).reshape(gh.nz,gh.nr)
        b=4*np.pi*densf.ravel()*gf.V; b[bnd]=1.0/rrf[bnd]
        P1=solveP(b).reshape(gf.nz,gf.nr)[mir,:].ravel()
        T=float(u@(S@u)); VK=-float(np.sum(K*u*u*gh.V)); U12=float(np.sum(P1*u*u*gh.V))
        E=2*(T+VK)+(1.0+beta)*U12+nn
        if Eo is not None and abs(E-Eo)<1e-9: break
        Eo=E
    return E,U12
h=0.08; L=12.
g=Grid(L,L,h); eH=solve_one_electron(g,kernel_potential(g,[(0.,1.)],SOFT*h),np.zeros(g.N))[0]
print("BULK PAIR CORRECTION  U12 -> (1+beta)*U12 .  beta is DIMENSIONLESS, so it is")
print("scale-covariant by construction (U12 scales as Z like every other term).\n")
D0=1.401
beta=brentq(lambda b: solve(h,L,[(-D0/2,1.),(D0/2,1.)],1.0/D0,b)[0]+1.174476, 0.0, 2.0, xtol=1e-6)
print("beta CALIBRATED ON H2 at D=%.3f  ->  beta = %.5f\n" % (D0,beta))
print("  system            beta=0      with beta    exact       err(beta)   rel")
EX={'He':-2.903724,'Li+':-7.279913,'Be2+':-13.655566}
for nm,Z in (('He',2),('Li+',3),('Be2+',4)):
    LL=12.0 if Z==2 else 10.0
    e0=solve(h,LL,[(0.,float(Z))],0.0,0.0)[0]
    e1=solve(h,LL,[(0.,float(Z))],0.0,beta)[0]
    print("  %-6s %13.5f %12.5f %11.5f  %+9.5f  %6.2f%%"
          % (nm,e0,e1,EX[nm],e1-EX[nm],100*abs(e1-EX[nm])/abs(EX[nm])), flush=True)
print("\n  H2 across D (exact E_bind: 0.1745 at 1.401, ~0.057 at 3.0, ~0.0004 at 6.0)")
print("    D      beta=0 E_bind   with beta E_bind")
for D in (1.401,2.0,3.0,6.0):
    a=solve(h,L,[(-D/2,1.),(D/2,1.)],1.0/D,0.0)[0]
    b=solve(h,L,[(-D/2,1.),(D/2,1.)],1.0/D,beta)[0]
    print("   %5.2f    %9.5f        %9.5f" % (D,2*eH-a,2*eH-b), flush=True)
