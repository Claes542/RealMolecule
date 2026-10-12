import sys, numpy as np, time; sys.path.insert(0,'.')
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, factorized
from axisym_realqm import *
np.seterr(all='ignore')
def two_elec(h,L,kernels,nn,soft=0.0):
    nzh=int(round(L/h)); L=nzh*h
    gh=Grid(L,L,h,zmin=0.0); gf=Grid(L,L,h,zmin=-L)
    S=stiffness(gh); K=kernel_potential(gh,list(kernels),soft)
    Lp=laplacian(gf).tolil(); rrf=np.sqrt(gf.Z**2+gf.R**2).ravel()
    bnd=np.zeros(gf.N,bool)
    bnd[gf.idx(0,np.arange(gf.nr))]=True; bnd[gf.idx(gf.nz-1,np.arange(gf.nr))]=True
    bnd[gf.idx(np.arange(gf.nz),gf.nr-1)]=True
    for k in np.where(bnd)[0]: Lp.rows[k]=[k]; Lp.data[k]=[1.0]
    solveP=factorized(Lp.tocsc())
    w=1.0/np.sqrt(gh.V); W=sp.diags(w); mir=np.arange(gh.nz)[::-1]
    P1=np.zeros(gh.N); u=None; Eo=None
    for it in range(60):
        A=(W@(S+sp.diags((-K+P1)*gh.V)).tocsr()@W).tocsr()
        val,vec=eigsh(A,k=1,which='SA',tol=1e-10,maxiter=300000,v0=None if u is None else u/w)
        u=w*vec[:,0]; u=u/np.sqrt(np.sum(u*u*gh.V))
        densf=np.zeros((gf.nz,gf.nr)); densf[gh.nz:,:]=(u*u).reshape(gh.nz,gh.nr)
        b=4*np.pi*densf.ravel()*gf.V; b[bnd]=1.0/rrf[bnd]
        P1=solveP(b).reshape(gf.nz,gf.nr)[mir,:].ravel()
        T=float(u@(S@u)); VK=-float(np.sum(K*u*u*gh.V)); U12=float(np.sum(P1*u*u*gh.V))
        E=2*(T+VK)+U12+nn
        if Eo is not None and abs(E-Eo)<1e-8: break
        Eo=E
    return E
def one(h,L,kern,soft=0.0):
    g=Grid(L,L,h); return solve_one_electron(g,kernel_potential(g,list(kern),soft),np.zeros(g.N))[0]
print("MESH LADDER, NO SOFTENING.  Does the two-electron over-binding SURVIVE h->0?")
print("  h      E(H)      E(He)     err_He    E(H2,D=1.4)   E_bind(H2)   err_bind")
for h in (0.16,0.12,0.08,0.06):
    t0=time.time()
    eH=one(h,12.,[(0.,1.)])
    eHe=two_elec(h,12.,[(0.,2.)],0.0)
    eH2=two_elec(h,12.,[(-0.7,1.),(0.7,1.)],1.0/1.4)
    b=2*eH-eH2
    print("  %.2f  %9.5f  %9.5f  %+8.5f   %9.5f    %8.5f   %+8.5f   [%.0fs]"
          % (h,eH,eHe,eHe+2.90372,eH2,b,b-0.17447,time.time()-t0), flush=True)
