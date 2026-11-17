"""Category-independent mesh/camera objective using the original shared kernels."""
import copy
import numpy as np
from scipy.ndimage import binary_erosion,distance_transform_edt
from .losses import silhouette_terms,visibility_term
from .mesh import laplacian,adjacent_folds,ray_anchors
from .priors import regularization,candidates
from .rotations import exp_rotation
from .point_terms import reprojection_term


def arrays(images,masks,pairs,s,faces):
    K,size=images.shape[:2];rng=np.random.default_rng(592);fields=[];targets=[]
    for mask in masks:
        fields.append(distance_transform_edt(~mask).astype(np.float32)/size)
        picks=[]
        for region,n in ((mask&~binary_erosion(mask),96),(mask,24)):
            yy,xx=np.nonzero(region);take=rng.choice(len(xx),n,replace=len(xx)<n)
            picks.append(np.column_stack([xx[take]+.5,yy[take]+.5]))
        targets.append((np.vstack(picks)-size/2)/size)
    tracks={};observations=[];observed=set()
    for pair in pairs:
        for a,b in zip(pair['pixels_a'],pair['pixels_b']):
            key=tuple(np.round(a,2))
            if key not in tracks:
                tracks[key]=len(tracks);observations.append((0,tracks[key],a))
            identity=(pair['view_b'],tracks[key])
            if identity not in observed:
                observations.append((pair['view_b'],tracks[key],b));observed.add(identity)
    first=np.array(list(tracks),float).reshape(-1,2)/size-.5
    anchor_face,bary=ray_anchors(s['X'],faces,s['C'][0],s['g'][0],np.exp(s['lf'][0]),s['pp'][0],first)
    return dict(k=K,width=size,height=1.,faces=faces,fields=np.array(fields),targets=np.array(targets),
                masks=masks,pixel_tolerance=1/size,bary=np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[0,.5,.5],[.5,0,.5],[1/3]*3]),
                ci=np.array([o[0] for o in observations],int),pi=np.array([o[1] for o in observations],int),
                uv=np.array([o[2] for o in observations],float).reshape(-1,2)/size-.5,
                anchor_vertices=faces[anchor_face],anchor_bary=bary,
                L=laplacian(len(s['X']),faces),folds=adjacent_folds(faces),
                smooth_weight=1e-4,prior_weight=1e-3,compactness_weight=1/size**2,
                symmetry_candidates=candidates(),
                gauge_camera_distance=float(np.linalg.norm(s['C'][0])),
                initial_pp=s['pp'].copy(),initial_lf=s['lf'].copy(),
                calibration_weight=1/size**2,principal_sigma=.25,log_focal_sigma=np.log(2.))


def queries(s,a):
    if len(a['anchor_vertices'])==0:return np.empty((0,3))
    return np.einsum('tj,tjd->td',a['anchor_bary'],s['X'][a['anchor_vertices']])


def objective(s,a,gauge=None):
    X=s['X'];K=len(s['C']);P=len(X);q=np.einsum('kij,kpj->kpi',s['g'],X[None]-s['C'][:,None])
    if np.any(q[...,2]<=.02):raise ValueError('Mesh crosses a camera plane.')
    f=np.exp(s['lf']);uv=f[:,None,None]*q[...,:2]/q[...,2,None]
    projected=uv+s['pp'][:,None]
    out,inside,gp=silhouette_terms(projected,a)
    gq=np.zeros_like(q);gf=np.sum(gp*uv,axis=(1,2));gpp=gp.sum(axis=1)
    gq[...,:2]=gp*f[:,None,None]/q[...,2,None]
    gq[...,2]=-np.sum(gp*uv,axis=-1)/q[...,2]
    ci,pi=a['ci'],a['pi'];T=len(a['anchor_vertices']);point=0.;point_rmse=0.;violations=0;vis=0.;viscost=0.;max_violation=0.
    point_stats=dict(observations=0,outside_tolerance=0,max_px=0.)
    if len(ci):
        xt=queries(s,a);qt=np.einsum('kij,ktj->kti',s['g'],xt[None]-s['C'][:,None])
        pred=f[ci,None]*qt[ci,pi,:2]/qt[ci,pi,2,None]+s['pp'][ci]
        point,dr,point_stats=reprojection_term(pred-a['uv'],a['width'],a.get('point_tolerance_pixels',0.))
        point_rmse=point_stats['rms_px'];dr*=1.5
        dqt=np.zeros_like(qt);depth=qt[ci,pi,2]
        du=dr*f[ci,None]/depth[:,None]
        dz=-np.sum(dr*(pred-s['pp'][ci]),axis=1)/depth
        np.add.at(dqt,(ci,pi,slice(0,2)),du)
        np.add.at(dqt,(ci,pi,2),dz)
        np.add.at(gf,ci,np.sum(dr*(pred-s['pp'][ci]),axis=1));np.add.at(gpp,ci,dr)
        # Reuse the exact original first-hit kernel with query points appended
        # as arbitrary indices, then chain their gradients to mesh barycentrics.
        # Unknown metric scale: tolerance must be invariant to the similarity
        # gauge, instead of the old fixed absolute-depth tolerance.
        va=dict(ci=ci,pi=P+pi,faces=a['faces'],relative_tolerance=.001,base_weight=2.,return_constraints=True,
                penalty=a.get('visibility_penalty',0.),dual=a.get('visibility_dual',np.zeros(len(ci))),
                feasibility_tolerance=a.get('visibility_feasibility_tolerance',0.))
        viscost,dall,violations,constraints=visibility_term(np.concatenate([q,qt],axis=1),va)
        positive=np.maximum(constraints,0);vis=float(np.mean(positive**2));max_violation=float(np.max(positive))
        gq+=dall[:,:P];dqt+=dall[:,P:]
        for v in range(K):
            for j in range(3):
                np.add.at(gq[v],a['anchor_vertices'][:,j],dqt[v]*a['anchor_bary'][:,j,None])
    gx=np.einsum('kij,kpi->kpj',s['g'],gq).sum(0)
    gc=-np.einsum('kij,ki->kj',s['g'],gq.sum(1));gg=np.cross(q,gq).sum(1);gg[0]=0
    prior,pr,gh,parts=regularization(s,a);gx+=pr
    # Weak image-independent calibration priors select among underdetermined
    # crop/focal configurations; these values are NOT known camera parameters.
    # Explicit weak camera-selection prior, in the SAME image-noise units as
    # compactness: 1 equivalent pixel by default. The target is the generic
    # initialization assumption, not a moving estimate from the last stage.
    # It does not assert these cropped photographs really have centered K.
    reference_lf=np.full(K,np.log(2.))
    cw=a.get('calibration_weight',1/a['width']**2)
    pw=cw/a.get('principal_sigma',.25)**2
    fw=cw/a.get('log_focal_sigma',np.log(2.))**2
    pp_pen=pw*np.mean(s['pp']**2);lf_pen=fw*np.mean((s['lf']-reference_lf)**2)
    gpp+=2*pw*s['pp']/s['pp'].size;gf+=2*fw*(s['lf']-reference_lf)/len(f)
    contact_cost=0.;contact_stats={}
    if a.get('contact_model') is not None:
        raw,contact_gradient,contact_stats=a['contact_model'].term(X)
        weight=a.get('contact_weight',1/a['width']**2);contact_cost=weight*raw;gx+=weight*contact_gradient
    total=out+inside+1.5*point+viscost+prior+pp_pen+lf_pen+contact_cost
    stats=dict(total=total,point_rmse_px=point_rmse,model_to_image_px=np.sqrt(out)*a['width'],
               image_to_model_px=np.sqrt(inside)*a['width'],visibility_violations=violations,
               visibility_loss=vis,visibility_max_residual=max_violation,**parts)
    stats['data_cost']=out+inside+1.5*point+2*vis
    stats['image_cost']=out+inside+1.5*point
    stats['prior_cost']=prior
    stats['intrinsic_cost']=pp_pen+lf_pen
    stats.update(contact_cost=contact_cost,**contact_stats)
    stats.update(point_max_error_px=point_stats['max_px'],point_observations=point_stats['observations'],
                 point_outside_tolerance=point_stats['outside_tolerance'],point_cost=1.5*point)
    return total,dict(X=gx,C=gc,g=gg,lf=gf,pp=gpp,h=gh),stats


def first_hit_constraints(s,a):
    q=np.einsum('kij,kpj->kpi',s['g'],s['X'][None]-s['C'][:,None]);xt=queries(s,a)
    qt=np.einsum('kij,ktj->kti',s['g'],xt[None]-s['C'][:,None])
    va=dict(ci=a['ci'],pi=len(s['X'])+a['pi'],faces=a['faces'],relative_tolerance=.001,
            penalty=1.,dual=np.zeros(len(a['ci'])),return_constraints=True)
    return visibility_term(np.concatenate([q,qt],axis=1),va)[-1]


def step(s,delta,a,gauge=None):
    t={key:value.copy() for key,value in s.items()}
    for key,d in delta.items():
        if key in ('g','h'):t[key]=exp_rotation(d)@s[key]
        else:t[key]+=d
    t['g'][0]=s['g'][0]
    # Pure similarity gauge: translates/scales BOTH geometry and cameras. It
    # cannot change projections and supplies no metric/object-size information.
    centre=t['X'].mean(0);t['X']-=centre;t['C']-=centre
    scale=a['gauge_camera_distance']/max(np.linalg.norm(t['C'][0]),1e-8)
    t['X']*=scale;t['C']*=scale
    t['lf']=np.clip(t['lf'],np.log(.4),np.log(12.));t['pp']=np.clip(t['pp'],-.8,.8)
    return t


def valid(s,a):
    q=np.einsum('kij,kpj->kpi',s['g'],s['X'][None]-s['C'][:,None])
    if not (np.isfinite(q).all() and np.all(q[...,2]>.02)):return False
    if a.get('compactness_weight',0.) or a.get('require_positive_volume',False):
        X=s['X'];tri=X[a['faces']];V=np.sum(tri[:,0]*np.cross(tri[:,1],tri[:,2]))/6
        radius=np.sqrt(np.mean(np.sum((X-X.mean(0))**2,axis=1)))
        if V<=1e-8*radius**3:return False
    if a.get('axis_constraint') is not None:
        from .axis_geometry import moments
        try:_,cov,_=moments(s['X'],a['faces'])
        except ValueError:return False
        if np.linalg.eigvalsh(cov)[0]<=1e-10*np.trace(cov):return False
    if a.get('reject_transverse_intersections',False):
        from .intersections import transverse_intersections
        tri=s['X'][a['faces']];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
        scale=np.mean(np.sum((s['X']-s['X'].mean(0))**2,axis=1))
        if np.min(area)<=1e-10*scale:return False
        if a.get('contact_model') is None and transverse_intersections(s['X'],a['faces'],stop_after=1):return False
    if a.get('contact_model') is not None and a['contact_model'].audit(s['X'],first_only=True)['found']:return False
    return True
