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
    for it in range(80):
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
    return E,U12
# exact He-like ground states (two electrons, one kernel of charge Z)
EX={2:-2.903724,3:-7.279913,4:-13.655566,5:-22.030972}
print("IS THE ERROR A CONSTANT PER ELECTRON PAIR?  two electrons, one kernel, increasing Z")
print("  Z   exact      h=0.12     h=0.08    limit(2pt)   offset   U12(h=.08)")
for Z in (2,3,4,5):
    L = 12.0 if Z==2 else 8.0
    vals=[]
    for h in (0.12,0.08):
        E,U=two_elec(h,L,[(0.,float(Z))],0.0); vals.append((E,U))
    e12,e08=vals[0][0],vals[1][0]
    lim = e08 + (e08-e12)/(0.08-0.12)*(0.0-0.08)
    print("  %d  %9.4f  %9.4f  %9.4f   %9.4f   %+7.4f   %6.4f"
          % (Z,EX[Z],e12,e08,lim,lim-EX[Z],vals[1][1]), flush=True)
