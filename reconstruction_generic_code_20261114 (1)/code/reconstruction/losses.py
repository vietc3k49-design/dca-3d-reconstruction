"""Shared, object-independent losses extracted from the original star experiment.

These functions accept arbitrary vertex/triangle counts; no object-specific indices.
"""
import numpy as np

def sample_field(fields,uv,width,coefficients=None):
    """Bilinear scalar-field values AND exact local spatial derivatives."""
    if coefficients is not None:
        from .fields import sample_cubic
        return sample_cubic(coefficients,uv,width)
    K,H,W=fields.shape
    xy=uv*width+np.array([W/2-.5,H/2-.5])
    # Outside the image is NOT zero loss. Extend the clamped distance field
    # by distance to the image rectangle, so off-screen geometry cannot escape.
    upper=np.array([W-1.000001,H-1.000001]);clamped=np.clip(xy,0,upper)
    x=clamped[...,0];y=clamped[...,1]
    ix=x.astype(int);iy=y.astype(int);fx=x-ix;fy=y-iy
    view=np.arange(K)[:,None]
    v00=fields[view,iy,ix];v10=fields[view,iy,ix+1]
    v01=fields[view,iy+1,ix];v11=fields[view,iy+1,ix+1]
    value=(1-fx)*(1-fy)*v00+fx*(1-fy)*v10+(1-fx)*fy*v01+fx*fy*v11
    derivative=np.stack([(1-fy)*(v10-v00)+fy*(v11-v01),
                         (1-fx)*(v01-v00)+fx*(v11-v10)],axis=-1)*width
    derivative*=((xy>=0)&(xy<=upper))
    offset=xy-clamped;extra=np.linalg.norm(offset,axis=-1)
    value+=extra/width
    derivative+=offset/np.maximum(extra[...,None],1e-15)
    return value,derivative


def silhouette_terms(pred,a):
    """Analytic two-sided distance gradients wrt all projected mesh vertices."""
    K,P,_=pred.shape;faces=a['faces'];tri=pred[:,faces]
    bary=a['bary'];T=len(faces);S=len(bary)
    samples=np.einsum('sj,ktjd->ktsd',bary,tri).reshape(K,T*S,2)
    distance,field_gradient=sample_field(a['fields'],samples,a['width'],a.get('field_coefficients'))
    outside=np.maximum(distance-a['pixel_tolerance'],0)
    loss_out=float(np.mean(outside**2))
    gs=(2*outside[...,None]*field_gradient/(K*T*S)).reshape(K,T,S,2)
    gt=np.einsum('ktsd,sj->ktjd',gs,bary)
    grad=np.zeros_like(pred)
    for corner in range(3):
        for v in range(K): np.add.at(grad[v],faces[:,corner],gt[v,:,corner])
    # Reverse direction: each foreground pixel must be covered by the projected
    # triangle union. If not, its closest triangle edge supplies the derivative.
    p=a['targets'];M=p.shape[1]
    A=tri[:,:,0];B=tri[:,:,1];C=tri[:,:,2]
    v0=B-A;v1=C-A;v2=p[:,:,None,:]-A[:,None]
    den=v0[...,0]*v1[...,1]-v0[...,1]*v1[...,0]
    safe=np.where(np.abs(den)>1e-14,den,1.)
    u=(v2[...,0]*v1[:,None,:,1]-v2[...,1]*v1[:,None,:,0])/safe[:,None]
    v=(v0[:,None,:,0]*v2[...,1]-v0[:,None,:,1]*v2[...,0])/safe[:,None]
    inside=((u>=0)&(v>=0)&(u+v<=1)&(np.abs(den[:,None])>1e-14)).any(axis=-1)
    starts=tri;ends=np.roll(tri,-1,axis=2);edge=ends-starts
    relative=p[:,:,None,None,:]-starts[:,None]
    t=np.clip(np.sum(relative*edge[:,None],axis=-1)/np.maximum(np.sum(edge**2,axis=-1)[:,None],1e-15),0,1)
    closest=starts[:,None]+t[...,None]*edge[:,None]
    delta=closest-p[:,:,None,None,:]
    d2=np.sum(delta**2,axis=-1)
    which=d2.reshape(K,M,-1).argmin(axis=-1);fi=which//3;ei=which%3
    kk=np.arange(K)[:,None];mm=np.arange(M)[None]
    d=np.sqrt(d2[kk,mm,fi,ei]);active=np.maximum(d-a['pixel_tolerance'],0)*(~inside)
    loss_in=float(np.mean(active**2))
    selected_delta=delta[kk,mm,fi,ei];selected_t=t[kk,mm,fi,ei]
    gd=2*active[...,None]*selected_delta/np.maximum(d[...,None],1e-15)/(K*M)
    va=faces[fi,ei];vb=faces[fi,(ei+1)%3]
    for view in range(K):
        np.add.at(grad[view],va[view],gd[view]*(1-selected_t[view,:,None]))
        np.add.at(grad[view],vb[view],gd[view]*selected_t[view,:,None])
    return loss_out,loss_in,grad


def visibility_term(q,a):
    """An observed vertex must be the FIRST surface point on its own ray.

    Segment q_point*t intersects a nonincident triangle at t<1 -> occlusion.
    Differentiate the selected intersection implicitly, using its barycentrics.
    Intrinsics do not affect physical visibility; this term has no f gradient.
    """
    ci,pi=a['ci'],a['pi'];F=a['faces']
    if len(ci)==0:
        result=(0.,np.zeros_like(q),0)
        return (*result,np.empty(0)) if a.get('return_constraints') else result
    tri=q[ci][:,F]
    D=q[ci,pi];e1=tri[:,:,1]-tri[:,:,0];e2=tri[:,:,2]-tri[:,:,0]
    h=np.cross(D[:,None],e2);det=np.sum(e1*h,axis=-1)
    valid=np.abs(det)>1e-12;inv=np.zeros_like(det);inv[valid]=1/det[valid]
    S=-tri[:,:,0];u=inv*np.sum(S*h,axis=-1);cross=np.cross(S,e1)
    v=inv*np.sum(D[:,None]*cross,axis=-1);t=inv*np.sum(e2*cross,axis=-1)
    incident=(F[None]==pi[:,None,None]).any(axis=-1)
    rho=a.get('penalty',0.)
    ok=valid&~incident&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(t>1e-7)
    if not rho:ok&=(t<1)
    candidates=np.where(ok,t,np.inf);face=candidates.argmin(axis=1)
    rows=np.arange(len(ci));near=candidates[rows,face]
    relative=a.get('relative_tolerance')
    tolerance=(np.full(len(D),relative) if relative is not None else
               .002*a['height']/np.maximum(D[:,2],.01))
    constraint=1-near-tolerance;residual=np.maximum(constraint,0)
    base=a.get('base_weight',1.)
    loss=float(base*np.mean(residual**2));derivative=2*base*residual
    if rho:
        dual=np.asarray(a.get('dual',np.zeros(len(ci))))
        active_dual=np.maximum(dual+rho*constraint,0)
        loss+=float(np.mean((active_dual**2-dual**2)/(2*rho)))
        derivative+=active_dual
    grad=np.zeros_like(q);active=np.flatnonzero(derivative>0)
    if len(active):
        f=face[active];views=ci[active];points=pi[active]
        normal=np.cross(e1[active,f],e2[active,f])
        denom=np.sum(normal*D[active],axis=-1)
        normal=normal/denom[:,None]
        factor=derivative[active]/len(ci)
        np.add.at(grad,(views,points),factor[:,None]*near[active,None]*normal)
        alpha=np.column_stack([1-u[active,f]-v[active,f],u[active,f],v[active,f]])
        for corner in range(3):
            np.add.at(grad,(views,F[f,corner]),-factor[:,None]*alpha[:,corner,None]*normal)
        # Tolerance depends on D_z. Include its derivative for exact gradients.
        if relative is None:
            np.add.at(grad,(views,points,2),factor*.002*a['height']/np.maximum(D[active,2],.01)**2)
    result=(loss,grad,int(np.sum(residual>a.get('feasibility_tolerance',0.))))
    return (*result,constraint) if a.get('return_constraints') else result

