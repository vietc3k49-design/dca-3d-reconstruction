"""Spec: lower-bound diagnostic for transverse nonadjacent triangle crossings.

Test strict segment/triangle interior hits in both directions. Exclude pairs
sharing a vertex. Coplanar overlaps, tangencies and adjacent-face overlap are
NOT certified by this check; zero hits is not a proof of an embedded solid.
"""
import numpy as np


def transverse_intersections(X,F,chunk=16,stop_after=None):
    X=np.asarray(X,float);F=np.asarray(F,int)
    scale=max(float(np.ptp(X,axis=0).max()),1e-15);X=(X-X.mean(0))/scale
    T=X[F];e1=T[:,1]-T[:,0];e2=T[:,2]-T[:,0];hits=set();N=len(F);tol=1e-9
    for start in range(0,N,chunk):
        ids=np.arange(start,min(start+chunk,N));U=T[ids]
        for j in range(3):
            O=U[:,j];D=U[:,(j+1)%3]-O
            h=np.cross(D[:,None],e2[None]);det=np.sum(e1[None]*h,axis=-1)
            valid=np.abs(det)>1e-12;inv=np.zeros_like(det);inv[valid]=1/det[valid]
            rel=O[:,None]-T[None,:,0];u=np.sum(rel*h,axis=-1)*inv
            q=np.cross(rel,e1[None]);v=np.sum(D[:,None]*q,axis=-1)*inv
            t=np.sum(e2[None]*q,axis=-1)*inv
            shared=(F[ids,None,:,None]==F[None,:,None,:]).any(axis=(-1,-2))
            mask=valid&~shared&(u>tol)&(v>tol)&(u+v<1-tol)&(t>tol)&(t<1-tol)
            ii,jj=np.nonzero(mask)
            for a,b in zip(ids[ii],jj):hits.add(tuple(sorted((int(a),int(b)))))
            if stop_after and len(hits)>=stop_after:return [list(pair) for pair in sorted(hits)]
    return [list(pair) for pair in sorted(hits)]
