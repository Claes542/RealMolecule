import sys, numpy as np; sys.path.insert(0,'.')
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, factorized
from axisym_realqm import *
np.seterr(all='ignore')
SOFT=np.sqrt(0.081); _c={}
def solve(h,L,D):
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
    eps=SOFT*h
    K=kernel_potential(gh,[(-D/2,1.),(D/2,1.)],eps)
    w=1.0/np.sqrt(gh.V); W=sp.diags(w); mir=np.arange(gh.nz)[::-1]
    P1=np.zeros(gh.N); u=None; Eo=None
    for it in range(90):
        A=(W@(S+sp.diags((-K+P1)*gh.V)).tocsr()@W).tocsr()
        val,vec=eigsh(A,k=1,which='SA',tol=1e-12,maxiter=400000,v0=None if u is None else u/w)
        u=w*vec[:,0]; u=u/np.sqrt(np.sum(u*u*gh.V))
        densf=np.zeros((gf.nz,gf.nr)); densf[gh.nz:,:]=(u*u).reshape(gh.nz,gh.nr)
        b=4*np.pi*densf.ravel()*gf.V; b[bnd]=1.0/rrf[bnd]
        P1=solveP(b).reshape(gf.nz,gf.nr)[mir,:].ravel()
        T=float(u@(S@u)); VK=-float(np.sum(K*u*u*gh.V)); U12=float(np.sum(P1*u*u*gh.V))
        E=2*(T+VK)+U12+1.0/D
        if Eo is not None and abs(E-Eo)<1e-10: break
        Eo=E
    # ELECTROSTATIC (Hellmann-Feynman) force on the kernel at z=+D/2, z-component.
    # rho_total(z,rho) = u^2(|z|,rho) by mirror symmetry; integrate over the FULL domain.
    d2=(u*u).reshape(gh.nz,gh.nr)
    rho_full=np.vstack([d2[::-1,:], d2])                 # z<0 mirror, then z>0
    zf=np.concatenate([-gh.z[::-1], gh.z])
    Zf,Rf=np.meshgrid(zf,gh.r,indexing='ij')
    dV=2*np.pi*Rf*h*h
    dz=(D/2 - Zf)
    r3=(dz*dz + Rf*Rf + eps*eps)**1.5
    F_elec = -float(np.sum(rho_full*dz/r3*dV))           # Z_A = 1
    F_nuc  = 1.0/(D*D)
    return E, F_nuc+F_elec, F_nuc, F_elec
h=0.08; L=12.
# *** KW reference energies, from model knowledge -- NEED A SOURCE CHECK ***
KW={1.0:-1.124540,1.2:-1.164935,1.4:-1.174475,1.6:-1.168583,1.8:-1.155060}
print("FORCES FROM THE POTENTIALS (Hellmann-Feynman, density only) vs FORCES FROM THE EXACT CURVE")
print("F on one kernel = Z^2/D^2 - Z*int rho*(z_A-z)/|r_A-r|^3 dV ;  F_exact = -dE_exact/dD")
print("(KW reference energies are from model knowledge and NEED CHECKING)\n")
print("   D      E(model)    F_nuc     F_elec    F_total    F_exact    ratio")
for D in (1.2,1.4,1.6):
    E,F,Fn,Fe=solve(h,L,D)
    Dm,Dp=round(D-0.2,1),round(D+0.2,1)
    Fex=-(KW[Dp]-KW[Dm])/0.4
    print("  %4.2f  %9.5f  %8.5f  %9.5f  %+8.5f  %+8.5f   %s"
          % (D,E,Fn,Fe,F,Fex,("%6.2f"%(F/Fex)) if abs(Fex)>1e-3 else "  n/a"), flush=True)
