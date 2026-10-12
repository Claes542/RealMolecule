import sys, numpy as np, time; sys.path.insert(0,'.')
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, factorized
from axisym_realqm import *
np.seterr(all='ignore')
SOFT=np.sqrt(0.081); _c={}
def solve(h,L,D):
    nzh=int(round(L/h)); L=nzh*h
    gh=Grid(L,L,h,zmin=0.0); gf=Grid(L,L,h,zmin=-L)
    # COMMENSURATION CHECK: the kernel at +D/2 must sit exactly on a cell centre (i+0.5)h
    m=(D/2)/h-0.5
    assert abs(m-round(m))<1e-9, "D=%.4f is NOT commensurate with h=%.4f (m=%.6f)"%(D,h,m)
    key=(h,L)
    if key not in _c:
        Lp=laplacian(gf).tolil(); bnd=np.zeros(gf.N,bool)
        bnd[gf.idx(0,np.arange(gf.nr))]=True; bnd[gf.idx(gf.nz-1,np.arange(gf.nr))]=True
        bnd[gf.idx(np.arange(gf.nz),gf.nr-1)]=True
        for k in np.where(bnd)[0]: Lp.rows[k]=[k]; Lp.data[k]=[1.0]
        _c[key]=(stiffness(gh),factorized(Lp.tocsc()),bnd,np.sqrt(gf.Z**2+gf.R**2).ravel(),gh,gf)
    S,solveP,bnd,rrf,gh,gf=_c[key]
    eps=SOFT*h
    K=kernel_potential(gh,[(-D/2,1.),(D/2,1.)],eps)
    w=1.0/np.sqrt(gh.V); W=sp.diags(w); mir=np.arange(gh.nz)[::-1]
    P1=np.zeros(gh.N); u=None; Eo=None
    for it in range(120):
        A=(W@(S+sp.diags((-K+P1)*gh.V)).tocsr()@W).tocsr()
        val,vec=eigsh(A,k=1,which='SA',tol=1e-12,maxiter=400000,v0=None if u is None else u/w)
        u=w*vec[:,0]; u=u/np.sqrt(np.sum(u*u*gh.V))
        densf=np.zeros((gf.nz,gf.nr)); densf[gh.nz:,:]=(u*u).reshape(gh.nz,gh.nr)
        b=4*np.pi*densf.ravel()*gf.V; b[bnd]=1.0/rrf[bnd]
        P1=solveP(b).reshape(gf.nz,gf.nr)[mir,:].ravel()
        T=float(u@(S@u)); VK=-float(np.sum(K*u*u*gh.V)); U12=float(np.sum(P1*u*u*gh.V))
        E=2*(T+VK)+U12+1.0/D
        if Eo is not None and abs(E-Eo)<1e-11: break
        Eo=E
    d2=(u*u).reshape(gh.nz,gh.nr)
    rho=np.vstack([d2[::-1,:],d2]); zf=np.concatenate([-gh.z[::-1],gh.z])
    Zf,Rf=np.meshgrid(zf,gh.r,indexing='ij'); dV=2*np.pi*Rf*h*h
    dz=(D/2-Zf); r3=(dz*dz+Rf*Rf+eps*eps)**1.5
    return E, 1.0/(D*D)-float(np.sum(rho*dz/r3*dV))
def report(h,L,Ds):
    print("\n=== h=%.3f, L=%.1f ;  all D commensurate (kernels exactly on cell centres) ===" % (h,L))
    print("     D         E            F_elec")
    E=[];F=[]
    for D in Ds:
        t0=time.time(); e,f=solve(h,L,D); E.append(e); F.append(f)
        print("   %6.3f  %12.7f  %+11.6f   [%.0fs]" % (D,e,f,time.time()-t0), flush=True)
    E=np.array(E); F=np.array(F); Ds=np.array(Ds)
    i=int(np.argmin(E))
    if 0<i<len(Ds)-1:
        c=np.polyfit(Ds[i-1:i+2],E[i-1:i+2],2)
        Dm=-c[1]/(2*c[0]); k=2*c[0]
        print("   -> D_min = %.4f au  (exact 1.401)   k = d2E/dD2 = %.4f Ha/bohr^2 (exact ~0.370)"
              % (Dm,k))
    # force zero by linear interpolation through the sign change
    for j in range(len(F)-1):
        if F[j]>0>=F[j+1]:
            Dz=Ds[j]+(Ds[j+1]-Ds[j])*F[j]/(F[j]-F[j+1])
            print("   -> force zero at D = %.4f au  (exact 1.401)" % Dz)
    print("   monotone force? %s" % ("YES" if np.all(np.diff(F)<0) else "NO -- still noisy"))
report(0.05, 8.0, [1.15,1.25,1.35,1.45,1.55])
report(0.04, 8.0, [1.16,1.24,1.32,1.40,1.48,1.56])
