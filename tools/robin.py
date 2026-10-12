import sys, numpy as np; sys.path.insert(0,'.')
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, factorized
from axisym_realqm import *
from scipy.optimize import brentq
np.seterr(all='ignore')
SOFT=np.sqrt(0.081); _c={}
def two_elec(h,L,kernels,nn,lam=0.0):
    """Robin at the shared interface, imposed per electron: d u/dn + 2*lam*u = 0,
       equivalently a surface energy lam*int_plane u^2 dS added for EACH electron.
       Continuity is automatic here by mirror symmetry, so the free boundary cannot move."""
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
    dS=np.zeros(gh.N); dS[gh.idx(0,np.arange(gh.nr))]=2*np.pi*gh.r*h   # midplane area element
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
        Ei=float(np.sum(lam*dS*u*u))
        E=2*(T+VK+Ei)+U12+nn
        if Eo is not None and abs(E-Eo)<1e-9: break
        Eo=E
    return E
print("ROBIN AT THE INTERFACE, per electron:  du/dn + 2*lam*u = 0")
print("calibrate lam on He (-2.903724), then PREDICT H2 with no further adjustment")
print("H is a one-electron system with no interface, so it is untouched.\n")
for h in (0.12,0.08):
    lam=brentq(lambda t: two_elec(h,12.,[(0.,2.)],0.0,t)+2.903724, 0.0, 5.0, xtol=1e-6)
    g=Grid(12.,12.,h); eH=solve_one_electron(g,kernel_potential(g,[(0.,1.)],SOFT*h),np.zeros(g.N))[0]
    print("h=%.2f   lam = %.5f   He = %.6f   H = %.5f" %
          (h,lam,two_elec(h,12.,[(0.,2.)],0.0,lam),eH), flush=True)
    for D in (1.25,1.35,1.40,1.45,1.55,3.0):
        E=two_elec(h,12.,[(-D/2,1.),(D/2,1.)],1.0/D,lam)
        print("    D=%.2f  E = %9.5f  E_bind = %8.5f   (exact: -1.17447 / 0.17447 at D=1.401)"
              % (D,E,2*eH-E), flush=True)
    print(flush=True)
