import sys, numpy as np; sys.path.insert(0,'.')
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, factorized
from axisym_realqm import *
np.seterr(all='ignore')
SOFT=np.sqrt(0.081); _c={}
def solve(h,L,kernels,nn,lam=0.0):
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
    dS=np.zeros(gh.N); dS[gh.idx(0,np.arange(gh.nr))]=2*np.pi*gh.r*h
    w=1.0/np.sqrt(gh.V); W=sp.diags(w); mir=np.arange(gh.nz)[::-1]
    P1=np.zeros(gh.N); u=None; Eo=None
    for it in range(80):
        A=(W@(S+sp.diags((-K+P1)*gh.V+lam*dS)).tocsr()@W).tocsr()
        val,vec=eigsh(A,k=1,which='SA',tol=1e-11,maxiter=300000,v0=None if u is None else u/w)
        u=w*vec[:,0]; u=u/np.sqrt(np.sum(u*u*gh.V))
        densf=np.zeros((gf.nz,gf.nr)); densf[gh.nz:,:]=(u*u).reshape(gh.nz,gh.nr)
        b=4*np.pi*densf.ravel()*gf.V; b[bnd]=1.0/rrf[bnd]
        P1=solveP(b).reshape(gf.nz,gf.nr)[mir,:].ravel()
        T=float(u@(S@u)); VK=-float(np.sum(K*u*u*gh.V)); U12=float(np.sum(P1*u*u*gh.V))
        E=2*(T+VK+float(np.sum(lam*dS*u*u)))+U12+nn
        if Eo is not None and abs(E-Eo)<1e-9: break
        Eo=E
    I  = float(np.sum(dS*u*u))                       # int_plane u^2 dS
    I1 = float(np.sum(dS*np.abs(u)))                 # int_plane |u| dS, a weaker weighting
    return E,I,I1
h=0.08; L=12.
# exact BO energies. He from Pekeris; H2 at 1.401 is the standard minimum.
# *** D=2.0 is from reference knowledge (Kolos-Wolniewicz) and NEEDS A SOURCE CHECK ***
CASES=[("He  (D=0)", [(0.,2.)], 0.0, -2.903724),
       ("H2  D=1.401",[(-0.7005,1.),(0.7005,1.)], 1.0/1.401, -1.174476),
       ("H2  D=2.0",  [(-1.0,1.),(1.0,1.)],       1.0/2.0,   -1.138133)]
print("DOES THE lam*int u^2 dS FORM HAVE THE RIGHT DENSITY WEIGHTING?")
print("If yes, dE/(2I) must be the SAME for every row.\n")
print("  system        E(lam=0)    exact      dE       I=int u^2 dS   dE/(2I)   dE/(2*int|u|dS)")
rows=[]
for nm,kern,nn,ex in CASES:
    E,I,I1=solve(h,L,kern,nn,0.0)
    dE=ex-E
    rows.append((nm,E,ex,dE,I,I1))
    print("  %-12s %9.5f %10.5f %+8.5f   %9.5f    %8.5f   %8.5f"
          % (nm,E,ex,dE,I,dE/(2*I),dE/(2*I1)), flush=True)
r=[x[3]/(2*x[4]) for x in rows]
print("\n  lam implied by int u^2 dS : %s   spread %.0f%%"
      % (", ".join("%.4f"%v for v in r), 100*(max(r)-min(r))/np.mean(r)))
r1=[x[3]/(2*x[5]) for x in rows]
print("  lam implied by int |u| dS : %s   spread %.0f%%"
      % (", ".join("%.4f"%v for v in r1), 100*(max(r1)-min(r1))/np.mean(r1)))
