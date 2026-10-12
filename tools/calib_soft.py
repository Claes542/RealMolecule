import sys, numpy as np, time; sys.path.insert(0,'.')
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, factorized
from axisym_realqm import *
from scipy.optimize import brentq
np.seterr(all='ignore')

def build(h, L, zmin_half=0.0):
    nzh=int(round(L/h)); L=nzh*h
    gh=Grid(L,L,h,zmin=zmin_half); gf=Grid(L,L,h,zmin=-L)
    return gh, gf, L

def two_elec(h, L, kernels, nn, soft, S=None, cache={}):
    gh,gf,L = build(h,L)
    key=('S',h,L)
    if key not in cache: cache[key]=(stiffness(gh), laplacian(gf))
    S,Lap = cache[key]
    K=kernel_potential(gh,list(kernels),soft)
    kb=('P',h,L)
    if kb not in cache:
        Lp=Lap.tolil(); bnd=np.zeros(gf.N,bool)
        bnd[gf.idx(0,np.arange(gf.nr))]=True; bnd[gf.idx(gf.nz-1,np.arange(gf.nr))]=True
        bnd[gf.idx(np.arange(gf.nz),gf.nr-1)]=True
        for k in np.where(bnd)[0]: Lp.rows[k]=[k]; Lp.data[k]=[1.0]
        cache[kb]=(factorized(Lp.tocsc()), bnd, np.sqrt(gf.Z**2+gf.R**2).ravel())
    solveP,bnd,rrf = cache[kb]
    w=1.0/np.sqrt(gh.V); W=sp.diags(w); mir=np.arange(gh.nz)[::-1]
    P1=np.zeros(gh.N); u=None; Eo=None
    for it in range(60):
        A=(W@(S+sp.diags((-K+P1)*gh.V)).tocsr()@W).tocsr()
        val,vec=eigsh(A,k=1,which='SA',tol=1e-10,maxiter=200000,v0=None if u is None else u/w)
        u=w*vec[:,0]; u=u/np.sqrt(np.sum(u*u*gh.V))
        densf=np.zeros((gf.nz,gf.nr)); densf[gh.nz:,:]=(u*u).reshape(gh.nz,gh.nr)
        b=4*np.pi*densf.ravel()*gf.V; b[bnd]=1.0/rrf[bnd]
        P1=solveP(b).reshape(gf.nz,gf.nr)[mir,:].ravel()
        T=float(u@(S@u)); VK=-float(np.sum(K*u*u*gh.V)); U12=float(np.sum(P1*u*u*gh.V))
        E=2*(T+VK)+U12+nn
        if Eo is not None and abs(E-Eo)<1e-8: break
        Eo=E
    return E

def one_elec(h, L, kernels, soft):
    g=Grid(L if abs(L/h-round(L/h))<1e-9 else round(L/h)*h, round(L/h)*h, h, zmin=-round(L/h)*h)
    K=kernel_potential(g,list(kernels),soft)
    e,u=solve_one_electron(g,K,np.zeros(g.N)); return e

HE_REF=-2.90372
for h in (0.16,0.12):
    t0=time.time()
    f=lambda s: two_elec(h,12.0,[(0.,2.)],0.0,s)-HE_REF
    lo,hi=0.0,1.2
    eps=brentq(f,lo,hi,xtol=1e-4)
    EH =one_elec(h,12.0,[(0.,1.)],eps)
    print("h=%.2f   eps_calib = %.4f au   ->  He = %.5f (target %.5f)   H = %.5f (exact -0.5)  [%.0fs]"
          % (h,eps,two_elec(h,12.0,[(0.,2.)],0.0,eps),HE_REF,EH,time.time()-t0), flush=True)
    print("         H2 with the SAME eps:", flush=True)
    for D in (1.2,1.4,1.6,2.0):
        E=two_elec(h,12.0,[(-D/2,1.),(D/2,1.)],1.0/D,eps)
        print("            D=%.2f   E = %9.5f   E_bind = %7.4f   (exact E=-1.1745 at 1.40, bind 0.1745)"
              % (D,E,2*EH-E), flush=True)
