"""Intrinsic solid moments and sampled surface C_n symmetry for any closed mesh.

No category model or vertex permutation is supplied. Solid moments are exact
polyhedral integrals, hence unchanged by planar triangle subdivision.
"""
import numpy as np
from .rotations import exp_rotation


def moments(X,F):
    T=X[F];s=T.sum(1)
    vol=np.einsum('ij,ij->i',T[:,0],np.cross(T[:,1],T[:,2]))/6
    V=vol.sum()
    if V<=0:raise ValueError('Positive signed solid volume required.')
    first=(vol[:,None]*s/4).sum(0);mu=first/V
    second=(s[:,:,None]*s[:,None,:]+np.einsum('fki,fkj->fij',T,T))/20
    M2=np.einsum('f,fij->ij',vol,second)/V
    cov=M2-np.outer(mu,mu)
    return mu,cov,dict(T=T,s=s,vol=vol,V=V,mu=mu,M2=M2,second=second)


def pullback(X,F,cache,dcov=None,dmu=None):
    T=cache['T'];s=cache['s'];V=cache['V'];mu=cache['mu']
    G=np.zeros((3,3)) if dcov is None else np.asarray(dcov)
    z=np.zeros(3) if dmu is None else np.asarray(dmu)
    dm=(z-2*G@mu)/V
    dv=(-np.sum(G*cache['M2'])+2*mu@G@mu-z@mu)/V
    coeff=np.einsum('ij,fij->f',G/V,cache['second'])+s@dm/4+dv
    grad=np.zeros_like(X)
    for j in range(3):
        dvol=np.cross(T[:,(j+1)%3],T[:,(j+2)%3])/6
        direct=(s+T[:,j])@G/(10*V)+dm/4
        np.add.at(grad,F[:,j],coeff[:,None]*dvol+cache['vol'][:,None]*direct)
    return grad


def transverse_balance(X,F,axis):
    mu,M,cache=moments(X,F);axis=np.asarray(axis);P=np.eye(3)-np.outer(axis,axis)
    B=P@M@P;t=float(np.trace(B))
    if t<=1e-12:raise ValueError('Collapsed transverse solid covariance.')
    D=B-.5*t*P;d2=np.sum(D*D)
    loss=float(2*d2/t**2)
    G=4*D/t**2-4*d2*P/t**3
    eigen=np.linalg.eigvalsh(B);ratio=np.sqrt(max(eigen[-1],0)/max(eigen[-2],1e-15))
    return loss,pullback(X,F,cache,dcov=G),dict(transverse_ratio=float(ratio),
         axial_variance=float(axis@M@axis),transverse_variance=t/2)


def closest_surface(points,X,F,chunk=64):
    """Exact nearest triangle among all faces; plane-interior + edge candidates."""
    T=X[F];A=T[:,0];e=T[:,1]-A;f=T[:,2]-A
    n=np.cross(e,f);n2=np.maximum(np.sum(n*n,axis=1),1e-30)
    ee=np.sum(e*e,axis=1);ef=np.sum(e*f,axis=1);ff=np.sum(f*f,axis=1)
    den=np.maximum(ee*ff-ef*ef,1e-30)
    face_ids=[];weights=[]
    for start in range(0,len(points),chunk):
        q=points[start:start+chunk];rel=q[:,None]-A[None]
        distance=np.einsum('pfi,fi->pf',rel,n)/n2
        onplane=rel-distance[...,None]*n[None]
        de=np.einsum('pfi,fi->pf',onplane,e);df=np.einsum('pfi,fi->pf',onplane,f)
        u=(ff*de-ef*df)/den;v=(ee*df-ef*de)/den
        b=np.stack([1-u-v,u,v],axis=-1)
        nearest=np.einsum('pfj,fji->pfi',b,T)
        best=np.sum((nearest-q[:,None])**2,axis=-1)
        best[(b.min(-1)<0)|(np.sum(n*n,axis=1)[None]<1e-24)]=np.inf
        best_b=b.copy()
        for j in range(3):
            end=(j+1)%3;edge=T[:,end]-T[:,j];length=np.maximum(np.sum(edge*edge,axis=1),1e-30)
            t=np.clip(np.einsum('pfi,fi->pf',q[:,None]-T[None,:,j],edge)/length,0,1)
            point=T[None,:,j]+t[...,None]*edge[None];d=np.sum((point-q[:,None])**2,axis=-1)
            better=d<best;bb=np.zeros_like(b);bb[...,j]=1-t;bb[...,end]=t
            best_b=np.where(better[...,None],bb,best_b);best=np.minimum(best,d)
        fi=best.argmin(1);rows=np.arange(len(q));face_ids.extend(fi.tolist());weights.extend(best_b[rows,fi].tolist())
    return np.array(face_ids,int),np.array(weights,float)


def update_surface_correspondences(X,F,axis,order=8):
    """C8 is a DEFAULT prior, not an order inferred from a bilateral image."""
    mu,_,_=moments(X,F);points=X[F].mean(1)
    rotations=np.array([exp_rotation(np.asarray(axis)*2*np.pi*j/order) for j in range(1,order)])
    transformed=np.einsum('aij,pj->api',rotations,points-mu)+mu
    fi,bary=closest_surface(transformed.reshape(-1,3),X,F)
    return dict(rotations=rotations,source_faces=F,target_vertices=F[fi],target_bary=bary)


def surface_group_loss(X,F,axis,association):
    mu,M,cache=moments(X,F);P=np.eye(3)-np.outer(axis,axis);scale=float(np.sum(P*M))
    if scale<=1e-12:raise ValueError('Collapsed transverse scale.')
    source_faces=association['source_faces'];p=X[source_faces].mean(1);R=association['rotations']
    rotated=np.einsum('aij,pj->api',R,p-mu)+mu
    target=np.einsum('pj,pji->pi',association['target_bary'],X[association['target_vertices']])
    residual=rotated-target.reshape(rotated.shape);raw=float(np.mean(np.sum(residual**2,axis=-1)))
    dr=2*residual/(residual.shape[0]*residual.shape[1]*scale)
    source_grad=np.einsum('aij,api->pj',R,dr);grad=np.zeros_like(X)
    for j in range(3):np.add.at(grad,source_faces[:,j],source_grad/3)
    flat=-dr.reshape(-1,3)
    for j in range(3):
        np.add.at(grad,association['target_vertices'][:,j],flat*association['target_bary'][:,j,None])
    dmu=np.sum(dr,axis=(0,1))-source_grad.sum(0)
    grad+=pullback(X,F,cache,dcov=-raw/scale**2*P,dmu=dmu)
    return raw/scale,grad
