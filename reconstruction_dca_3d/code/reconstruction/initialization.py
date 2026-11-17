"""Image-only, category-independent initialization of poses and a neutral mesh."""
import numpy as np
from scipy.optimize import least_squares
from .geometry import fundamental,essential_pose
from .mesh import sphere
from .rotations import exp_rotation


def _mask_bounds(mask,width):
    yy,xx=np.nonzero(mask)
    lo=np.array([xx.min(),yy.min()])/width-.5
    hi=np.array([xx.max(),yy.max()])/width-.5
    return lo,hi


def _ring_sign_from_calibrated(C,calibrated,centre,u_up,K):
    r0=C[0]-centre;r0h=r0-(r0@u_up)*u_up
    norm0=np.linalg.norm(r0h)
    if norm0<1e-8:return 1.,False
    eb=r0h/norm0;er=np.cross(u_up,eb)
    score=0.
    for v in calibrated:
        if v==0:continue
        rv=C[v]-centre;rvh=rv-(rv@u_up)*u_up
        if np.linalg.norm(rvh)<1e-8:continue
        theta=np.arctan2(rvh@er,rvh@eb)
        phi=2.*np.pi*v/K
        score+=np.sin(theta)*np.sin(phi)
    return (1. if score>=0. else -1.),abs(score)>.05


def _place_ring_cameras(C,g,f,pp,masks,width,centre,views,K,ring_sign):
    lo0,hi0=_mask_bounds(masks[0],width)
    h0=max(float(hi0[1]-lo0[1]),1e-4)
    c0=(lo0+hi0)/2.
    u_up=-g[0,1]/max(np.linalg.norm(g[0,1]),1e-8)
    offset0=C[0]-centre
    for v in views:
        theta=ring_sign*2.*np.pi*v/K
        R=exp_rotation(theta*u_up)
        lov,hiv=_mask_bounds(masks[v],width)
        hv=max(float(hiv[1]-lov[1]),1e-4)
        dist_ratio=float(np.clip(h0/hv,.5,2.))
        C[v]=centre+dist_ratio*(R@offset0)
        g[v]=g[0]@R.T
        f[v]=f[0]
        pp[v]=np.clip(pp[0]+((lov+hiv)/2.-c0),-.6,.6)


def build_uniform_ring_state(images,masks,subdivisions=1,direction=1.,f_init=2.):
    """Neutral circular turntable camera initialization without prior SIFT poses."""
    K=len(images);width=images.shape[2]
    C=np.zeros((K,3));g=np.tile(np.eye(3),(K,1,1));f=np.full(K,float(f_init));pp=np.zeros((K,2))
    lo0,hi0=_mask_bounds(masks[0],width)
    median_depth=3.
    factor=np.sqrt(np.prod(hi0-lo0))/(2*f[0]);depth=median_depth/max(1-.6*factor,.4)
    centre=np.r_[((lo0+hi0)/2-pp[0])*depth/f[0],depth]
    radii=np.r_[(hi0-lo0)*depth/(2*f[0]),factor*depth]
    unit,faces=sphere(subdivisions);X=unit*radii+centre
    _place_ring_cameras(C,g,f,pp,masks,width,centre,range(1,K),K,float(direction))
    mean=X.mean(0);distance=np.linalg.norm(C[0]-mean);scale=3/max(distance,1e-8)
    X=(X-mean)*scale;C=(C-mean)*scale
    return dict(X=X,C=C,g=g,lf=np.log(f),pp=pp),faces


def camera_hypotheses(state,images=None,masks=None,init_mode='star',ring_direction='auto',
                      ring_direction_detected=False,subdivisions=1):
    """Object-independent focal/crop/translation ambiguity starts.

    Centering a principal point can be offset by a lateral camera translation
    at the mesh-centre plane; focal changes can be offset by camera distance.
    These preserve each projected centre exactly, not the whole 3D projection.
    All variables are free again during joint refinement. No camera is known.
    """
    starts=[('feature_calibration',{key:value.copy() for key,value in state.items()})]
    centre=state['X'].mean(0)
    for factor in (.5,1.,2.):
        t={key:value.copy() for key,value in state.items()}
        old_f=np.exp(state['lf']);t['lf']=np.clip(state['lf']+np.log(factor),np.log(.4),np.log(12.))
        ratios=np.exp(t['lf'])/old_f
        for v in range(len(t['C'])):
            q=state['g'][v]@(centre-state['C'][v]);new=q.copy()
            new[:2]+=state['pp'][v]*q[2]/old_f[v];new[2]*=ratios[v]
            t['C'][v]=centre-state['g'][v].T@new
        t['pp'][:]=0
        translation=t['X'].mean(0);t['X']-=translation;t['C']-=translation
        scale=3/max(np.linalg.norm(t['C'][0]),1e-8);t['X']*=scale;t['C']*=scale
        starts.append((f'centred_crop_focal_x{factor:g}',t))
    if init_mode=='ring' and images is not None and masks is not None and len(images)>=3:
        K=len(images)
        if ring_direction=='ccw':dirs=[(1.,'uniform_ring_ccw')]
        elif ring_direction=='cw':dirs=[(-1.,'uniform_ring_cw')]
        elif ring_direction_detected:
            u_up=-state['g'][0,1]/max(np.linalg.norm(state['g'][0,1]),1e-8)
            sign,_=_ring_sign_from_calibrated(state['C'],range(K),centre,u_up,K)
            dirs=[(sign,'uniform_ring_ccw' if sign>0 else 'uniform_ring_cw')]
        else:
            dirs=[(1.,'uniform_ring_ccw'),(-1.,'uniform_ring_cw')]
        for sign,label in dirs:
            ring_s,_=build_uniform_ring_state(images,masks,subdivisions=subdivisions,direction=sign)
            if 'h' in state:ring_s['h']=state['h'].copy()
            starts.append((label,ring_s))
    return starts


def calibrate_pair(pair,width,fixed_reference=None):
    a=(pair['pixels_a']-width/2)/width;b=(pair['pixels_b']-width/2)/width
    F,_=fundamental(a,b)
    def K(params):
        f=np.exp(params[0]);return np.array([[f,0,params[1]],[0,f,params[2]],[0,0,1.]])
    def residual(z):
        ka,kb=K(z[:3] if fixed_reference is None else fixed_reference),K(z[3:]);E=kb.T@F@ka;E/=np.linalg.norm(E)
        metric=(E@E.T@E-.5*np.trace(E@E.T)*E).ravel()
        # Cropped web photos have unknown principal points. These are weak,
        # explicit calibration priors, not product-specific camera guesses.
        prior=np.r_[.003*(z[[0,3]]-np.log(2.)),.006*z[[1,2,4,5]]]
        return np.r_[metric,prior]
    lower=np.tile([np.log(.5),-.6,-.6],2);upper=np.tile([np.log(10.),.6,.6],2)
    solutions=[]
    for f0 in [1.,2.,4.]:
        z0=np.array([np.log(f0),0,0,np.log(f0),0,0])
        r=least_squares(residual,z0,bounds=(lower,upper),max_nfev=250)
        ka,kb=K(r.x[:3] if fixed_reference is None else fixed_reference),K(r.x[3:]);E=kb.T@F@ka
        va=(a-ka[:2,2])/ka[0,0];vb=(b-kb[:2,2])/kb[0,0]
        count,C,g,X=essential_pose(va,vb,1.,E)
        score=np.linalg.norm(residual(r.x))+2*(1-count/len(a))
        z=r.x.copy()
        if fixed_reference is not None:z[:3]=fixed_reference
        solutions.append((score,z,C,g,X,count))
    score,z,C,g,X,count=min(solutions,key=lambda item:item[0])
    return dict(C=C,g=g,X=X,f=np.exp(z[[0,3]]),pp=np.array([z[1:3],z[4:6]]),
                positive_depth=count,tracks=len(a),calibration_residual=float(score),
                prior='centred principal points and focal around 2*image width, softly; crop offsets optimized')


def _oriented_pixels(pair,u,v):
    va=int(pair.get('view_a',0));vb=int(pair['view_b'])
    if va==u and vb==v:return pair['pixels_a'],pair['pixels_b']
    if va==v and vb==u:return pair['pixels_b'],pair['pixels_a']
    raise ValueError(f'Pair ({va}, {vb}) does not connect ({u}, {v}).')


def _estimate_baseline_scale(shared,C_u,g_v,dC_world,seed,width,robust=False):
    if len(shared)<2:return -1.
    px_v=np.array([z[0] for z in shared],float)
    world=np.array([z[1] for z in shared],float)
    bearing=np.column_stack([((px_v/width-.5)-seed['pp'][1])/seed['f'][1],np.ones(len(shared))])
    A=np.cross(bearing,g_v@dC_world)
    B=np.cross(bearing,(world-C_u)@g_v.T)
    num=np.sum(A*B,axis=1);den=np.sum(A*A,axis=1)
    if not robust or len(shared)<4:
        return float(np.sum(num)/max(np.sum(den),1e-12))
    per=num/np.maximum(den,1e-12);pos=per>0
    if not np.any(pos):return -1.
    s_med=float(np.median(per[pos]))
    keep=pos&(per>=.25*s_med)&(per<=4.*s_med)
    if not np.any(keep):return -1.
    return float(np.sum(num[keep])/max(np.sum(den[keep]),1e-12))


def initialize(images,masks,pairs,subdivisions=1,init_mode='star',ring_direction='auto'):
    K=len(images);width=images.shape[2]
    C=np.zeros((K,3));g=np.tile(np.eye(3),(K,1,1));f=np.full(K,2.);pp=np.zeros((K,2))
    pointcloud=[];pair_info=[];known={};reference=None
    calibrated={0}
    valid_pairs=[p for p in pairs if p['inliers']>=8]
    if init_mode=='star':
        ordered=[p for p in sorted(valid_pairs,key=lambda p:-p['inliers']) if int(p.get('view_a',0))==0]
        for pair in ordered:
            v=int(pair['view_b'])
            if v in calibrated:continue
            seed=calibrate_pair(pair,width,reference)
            if reference is None:
                f[0]=seed['f'][0];pp[0]=seed['pp'][0]
                reference=np.r_[np.log(f[0]),pp[0]]
            shared=[(pair['pixels_b'][i],known[(0,tuple(np.round(p,2)))])
                    for i,p in enumerate(pair['pixels_a']) if (0,tuple(np.round(p,2))) in known]
            estimated=_estimate_baseline_scale(shared,C[0],seed['g'],seed['C'],seed,width,robust=False)
            uncertain_scale=bool(pointcloud and (len(shared)<2 or estimated<=0))
            baseline_scale=estimated if (len(shared)>=2 and estimated>0) else 1.
            seed['C']*=baseline_scale;seed['X']*=baseline_scale
            C[v]=seed['C'];g[v]=seed['g'];f[v]=seed['f'][1];pp[v]=seed['pp'][1]
            front=(seed['X'][:,2]>0)&(((seed['X']-C[v])@g[v].T)[:,2]>0)
            pointcloud.append(seed['X'][front])
            for p,Xp,keep in zip(pair['pixels_a'],seed['X'],front):
                if keep:known.setdefault((0,tuple(np.round(p,2))),Xp)
            calibrated.add(v)
            pair_info.append(dict(view_a=0,view=v,positive_depth=seed['positive_depth'],tracks=seed['tracks'],
                                  calibration_residual=seed['calibration_residual'],prior=seed['prior'],
                                  baseline_scale_uncertain=uncertain_scale))
    elif init_mode=='ring':
        remaining=list(valid_pairs)
        while True:
            best_idx=None;best_key=None;best_uv=None
            for idx,pair in enumerate(remaining):
                va=int(pair.get('view_a',0));vb=int(pair['view_b'])
                if (va in calibrated)==(vb in calibrated):continue
                u,v=(va,vb) if va in calibrated else (vb,va)
                cdist=min((v-u)%K,(u-v)%K)
                key=(pair['inliers'],-cdist,-v)
                if best_key is None or key>best_key:
                    best_key=key;best_idx=idx;best_uv=(u,v)
            if best_idx is None:break
            pair=remaining.pop(best_idx);u,v=best_uv
            px_u,px_v=_oriented_pixels(pair,u,v)
            ref_u=None if (u==0 and reference is None) else np.r_[np.log(f[u]),pp[u]]
            seed=calibrate_pair(dict(pixels_a=px_u,pixels_b=px_v),width,ref_u)
            if u==0 and reference is None:
                f[0]=seed['f'][0];pp[0]=seed['pp'][0]
                reference=np.r_[np.log(f[0]),pp[0]]
            g_v=seed['g']@g[u]
            dC_world=g[u].T@seed['C']
            shared=[]
            for p_pair in valid_pairs:
                pa,pb=int(p_pair.get('view_a',0)),int(p_pair['view_b'])
                if pa in calibrated and pb==v:w=pa;qw,qv=p_pair['pixels_a'],p_pair['pixels_b']
                elif pb in calibrated and pa==v:w=pb;qw,qv=p_pair['pixels_b'],p_pair['pixels_a']
                else:continue
                for pw,pv in zip(qw,qv):
                    k=(w,tuple(np.round(pw,2)))
                    if k in known:shared.append((pv,known[k]))
            estimated=_estimate_baseline_scale(shared,C[u],g_v,dC_world,seed,width,robust=True)
            uncertain_scale=bool(pointcloud and (len(shared)<2 or estimated<=0))
            if len(shared)>=2 and estimated>0:
                baseline_scale=estimated
            elif pointcloud:
                all_pts=np.vstack(pointcloud)
                du=((all_pts-C[u])@g[u].T)[:,2];du_pos=du[du>0]
                loc_z=seed['X'][:,2];loc_pos=loc_z[loc_z>0]
                if len(du_pos) and len(loc_pos):
                    baseline_scale=float(np.clip(np.median(du_pos)/max(np.median(loc_pos),1e-6),.1,10.))
                else:baseline_scale=1.
            else:
                baseline_scale=1.
            C[v]=C[u]+baseline_scale*dC_world;g[v]=g_v;f[v]=seed['f'][1];pp[v]=seed['pp'][1]
            X_world=C[u]+baseline_scale*(seed['X']@g[u])
            front=(((X_world-C[u])@g[u].T)[:,2]>0)&(((X_world-C[v])@g[v].T)[:,2]>0)
            pointcloud.append(X_world[front])
            for pu,pv,Xp,keep in zip(px_u,px_v,X_world,front):
                if keep:
                    known.setdefault((u,tuple(np.round(pu,2))),Xp)
                    known.setdefault((v,tuple(np.round(pv,2))),Xp)
            calibrated.add(v)
            pair_info.append(dict(view_a=u,view=v,positive_depth=seed['positive_depth'],tracks=seed['tracks'],
                                  shared_scale_points=len(shared),calibration_residual=seed['calibration_residual'],
                                  prior=seed['prior'],baseline_scale_uncertain=uncertain_scale))
    else:
        raise ValueError(f'Unknown init_mode: {init_mode}')
    lo,hi=_mask_bounds(masks[0],width)
    if pointcloud:
        d0=((pointcloud[0]-C[0])@g[0].T)[:,2];d0=d0[d0>0]
        median_depth=float(np.median(d0)) if len(d0) else 3.
        if init_mode=='ring':median_depth=float(np.clip(median_depth,.8,60.))
    else:
        median_depth=3.
    factor=np.sqrt(np.prod(hi-lo))/(2*f[0]);depth=median_depth/max(1-.6*factor,.4)
    centre=np.r_[((lo+hi)/2-pp[0])*depth/f[0],depth]
    radii=np.r_[(hi-lo)*depth/(2*f[0]),factor*depth]
    unit,faces=sphere(subdivisions);X=unit*radii+centre
    missing=[v for v in range(1,K) if v not in calibrated]
    ring_detected=False;ring_sign_used=None
    if init_mode=='ring':
        u_up=-g[0,1]/max(np.linalg.norm(g[0,1]),1e-8)
        auto_sign,ring_detected=_ring_sign_from_calibrated(C,calibrated,centre,u_up,K)
        if ring_direction=='ccw':ring_sign_used=1.
        elif ring_direction=='cw':ring_sign_used=-1.
        else:ring_sign_used=auto_sign
        if missing:
            _place_ring_cameras(C,g,f,pp,masks,width,centre,missing,K,ring_sign_used)
    else:
        # No per-class camera orientations: missing-view seeds use the SAME initial
        # view and are explicitly marked uncalibrated, not asserted to be recovered.
        for v in missing:C[v]=0;g[v]=np.eye(3);f[v]=f[0]
    mean=X.mean(0);distance=np.linalg.norm(C[0]-mean);scale=3/max(distance,1e-8)
    X=(X-mean)*scale;C=(C-mean)*scale
    return dict(X=X,C=C,g=g,lf=np.log(f),pp=pp),faces,dict(
        initialization='neutral subdivided icosphere; affine extents from first silhouette',
        camera_initialization=('360-degree ring pose-graph + circular fill-in' if init_mode=='ring'
                               else 'feature-based two-view geometry with crop/focal priors'),
        init_mode=init_mode,
        calibrated_views=sorted(calibrated),
        uncalibrated_views=missing,
        ring_direction_used=('ccw' if ring_sign_used and ring_sign_used>0 else 'cw') if ring_sign_used is not None else None,
        ring_direction_detected=bool(ring_detected) if init_mode=='ring' else None,
        pairs=pair_info,metric_scale_known=False,
        note='two-view calibration may be ill-conditioned; this is an initial estimate, not unique camera recovery')
