import sys, numpy as np; sys.path.insert(0,'.')
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, factorized
from axisym_realqm import *
np.seterr(all='ignore')
SOFT=np.sqrt(0.081)
def two_elec(h,L,kernels,nn):
    nzh=int(round(L/h)); L=nzh*h
    gh=Grid(L,L,h,zmin=0.0); gf=Grid(L,L,h,zmin=-L)
    S=stiffness(gh); K=kernel_potential(gh,list(kernels),SOFT*h)
    Lp=laplacian(gf).tolil(); rrf=np.sqrt(gf.Z**2+gf.R**2).ravel()
    bnd=np.zeros(gf.N,bool)
    bnd[gf.idx(0,np.arange(gf.nr))]=True; bnd[gf.idx(gf.nz-1,np.arange(gf.nr))]=True
    bnd[gf.idx(np.arange(gf.nz),gf.nr-1)]=True
    for k in np.where(bnd)[0]: Lp.rows[k]=[k]; Lp.data[k]=[1.0]
    solveP=factorized(Lp.tocsc())
    w=1.0/np.sqrt(gh.V); W=sp.diags(w); mir=np.arange(gh.nz)[::-1]
    P1=np.zeros(gh.N); u=None; Eo=None
    for it in range(80):
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
    return E,2*T,2*VK+U12+nn
print("VIRIAL CHECK.  The half-space through the kernel is SCALE-INVARIANT, so the true minimiser")
print("of this constrained problem must satisfy 2T + V = 0 exactly (for H2, at the equilibrium D).")
print("  system        h      E          2T        V        2T+V     |2T+V|/2T")
for nm,h,kern,nn in (("He",0.12,[(0.,2.)],0.0),("He",0.08,[(0.,2.)],0.0),
                     ("He",0.06,[(0.,2.)],0.0)):
    E,T2,V=two_elec(h,12.,kern,nn)
    print("  %-6s  %6.2f %9.5f %9.5f %9.5f  %+9.5f   %7.4f"
          % (nm,h,E,T2,V,T2+V,abs(T2+V)/T2), flush=True)
print()
for D in (1.35,1.40,1.45):
    E,T2,V=two_elec(0.08,12.,[(-D/2,1.),(D/2,1.)],1.0/D)
    print("  H2 D=%.2f  h=0.08 %9.5f %9.5f %9.5f  %+9.5f   %7.4f"
          % (D,E,T2,V,T2+V,abs(T2+V)/T2), flush=True)
