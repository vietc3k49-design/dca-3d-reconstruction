"""Scale-invariant priors for arbitrary vertex/triangle counts; no class labels."""
import numpy as np


def compactness(X,faces):
    """Log isoperimetric quotient: scale/rotation invariant, no target category.

    A^3/(36*pi*V^2) is 1 for a sphere, grows for elongation AND flattening.
    This is a preference for compact completion, not information from images.
    Requires an outward, closed mesh with positive signed volume.
    """
    tri=X[faces];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
    cross=np.cross(e1,e2);length=np.maximum(np.linalg.norm(cross,axis=1),1e-15)
    A=.5*length.sum();V=np.sum(tri[:,0]*np.cross(tri[:,1],tri[:,2]))/6
    if V<=0 or A<=0:raise ValueError('Compactness needs positive area and signed volume.')
    normal=cross/length[:,None]
    db=.5*np.cross(e2,normal);dc=.5*np.cross(normal,e1)
    ga=np.zeros_like(X);gv=np.zeros_like(X)
    for j,da in enumerate((-db-dc,db,dc)):np.add.at(ga,faces[:,j],da)
    for j in range(3):
        np.add.at(gv,faces[:,j],np.cross(tri[:,(j+1)%3],tri[:,(j+2)%3])/6)
    raw=float(3*np.log(A)-2*np.log(V)-np.log(36*np.pi))
    return raw,3*ga/A-2*gv/V,dict(surface_area=A,signed_volume=V,isoperimetric_quotient=float(np.exp(raw)))


def symmetry(X,H,Q0):
    centre=X.mean(0);Y0=X-centre;Q=H@Q0@H.T;Y=Y0@Q.T+centre
    delta=X[:,None]-Y[None];d2=np.sum(delta**2,axis=-1);N=len(X)
    ia=np.arange(N);ja=d2.argmin(axis=1);jb=np.arange(N);ib=d2.argmin(axis=0)
    gx=np.zeros_like(X);gy=np.zeros_like(X)
    for ii,jj in [(ia,ja),(ib,jb)]:
        dr=delta[ii,jj]/N
        np.add.at(gx,ii,dr);np.add.at(gy,jj,-dr)
    raw=float((d2.min(1).mean()+d2.min(0).mean())/2)
    grad=gx+gy@Q+np.sum(gy@(np.eye(3)-Q),axis=0)/N
    # Left tangent perturbation of the candidate symmetry frame H.
    gh=np.sum(np.cross(Y0@Q.T,gy)-np.cross(Y0,gy@Q),axis=0)
    scale=float(np.mean(np.sum(Y0**2,axis=1)))+1e-12
    grad=grad/scale-raw/(scale**2)*2*Y0/N
    return raw/scale,grad,gh/scale


def regularization(s,a):
    X=s['X'];N=len(X);Y=X-X.mean(0);scale=float(np.mean(np.sum(Y**2,axis=1)))+1e-12
    L=a['L'];r=L@X;raw=float(np.mean(np.sum(r*r,axis=1)))
    smooth=raw/scale;gx=a['smooth_weight']*(2*L.T@r/(N*scale)-2*raw*Y/(N*scale**2))
    total=a['smooth_weight']*smooth;gh=np.zeros(3);stats=dict(smoothness=smooth)
    central,cg,_=symmetry(X,np.eye(3),-np.eye(3))
    total+=a['prior_weight']*central;gx+=a['prior_weight']*cg;stats['central_symmetry']=central
    # One best-supported mirror / rotational hypothesis; never force every
    # possible group or a predefined object category. Their frame is optimized.
    for family in ('mirror','rotation'):
        if family=='rotation' and a.get('axis_constraint') is not None:continue
        choices=[]
        for label,Q in a['symmetry_candidates'][family]:
            value,grad,frame_grad=symmetry(X,s['h'],Q);choices.append((value,grad,frame_grad,label))
        value,grad,frame_grad,label=min(choices,key=lambda item:item[0])
        total+=a['prior_weight']*value;gx+=a['prior_weight']*grad;gh+=a['prior_weight']*frame_grad
        stats[family+'_symmetry']=value;stats[family+'_candidate']=label
    records=a['folds'];faces=a['faces'];tri=X[faces]
    e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];cross=np.cross(e1,e2)
    length=np.maximum(np.linalg.norm(cross,axis=1),1e-12);normal=cross/length[:,None]
    fi,opposite,neighbour=records.T;triangle=faces[fi]
    w=X[opposite]-tri[fi,0];signed=np.sum(normal[fi]*w,axis=1)
    # Soft local convexity preference on all adjacent patches. Merely facing
    # a camera is not evidence that a fold is real. Strong image evidence can
    # still outweigh this weak prior for genuinely concave objects.
    active=np.ones(len(fi),bool);positive=np.maximum(signed,0)*active
    raw=float(np.mean(positive**2));convexity=raw/scale
    fac=2*positive/(len(records)*scale)
    dn=(w-normal[fi]*signed[:,None])/length[fi,None]
    db=np.cross(e2[fi],dn);dc=np.cross(dn,e1[fi]);da=-normal[fi]-db-dc
    cg=np.zeros_like(X)
    np.add.at(cg,opposite,fac[:,None]*normal[fi])
    for corner,derivative in enumerate((da,db,dc)):
        np.add.at(cg,triangle[:,corner],fac[:,None]*derivative)
    cg-=2*raw*Y/(N*scale**2)
    total+=a['prior_weight']*convexity;gx+=a['prior_weight']*cg
    stats['local_concavity']=convexity
    constraint=a.get('axis_constraint')
    if constraint is not None:
        from .axis_geometry import transverse_balance,surface_group_loss
        balance,bg,details=transverse_balance(X,faces,np.asarray(constraint['axis']))
        group,sg=surface_group_loss(X,faces,np.asarray(constraint['axis']),constraint['association'])
        total+=a['prior_weight']*(balance+group);gx+=a['prior_weight']*(bg+sg)
        stats.update(axis_transverse_balance=balance,axis_surface_group_error=group,
                     axis_rotation_order=constraint['order'],**details)
    compact_weight=a.get('compactness_weight',0.)
    if compact_weight:
        value,gradient,diagnostics=compactness(X,faces)
        total+=compact_weight*value;gx+=compact_weight*gradient
        stats.update(compactness=value,**diagnostics)
    return total,gx,gh,stats


def candidates():
    from .rotations import exp_rotation
    mirror=[];rotation=[]
    for i,axis in enumerate(np.eye(3)):
        mirror.append((f'frame_axis_{i}',np.eye(3)-2*np.outer(axis,axis)))
        for n in (2,3,4,5,6):
            rotation.append((f'frame_axis_{i}_order_{n}',exp_rotation(axis*(2*np.pi/n))))
    return dict(mirror=mirror,rotation=rotation)
