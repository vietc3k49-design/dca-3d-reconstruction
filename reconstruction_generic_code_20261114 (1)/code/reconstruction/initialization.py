"""Image-only, category-independent initialization of poses and a neutral mesh."""
import numpy as np
from scipy.optimize import least_squares
from .geometry import fundamental,essential_pose
from .mesh import sphere


def camera_hypotheses(state):
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


def initialize(images,masks,pairs,subdivisions=1):
    K=len(images);width=images.shape[2]
    C=np.zeros((K,3));g=np.tile(np.eye(3),(K,1,1));f=np.full(K,2.);pp=np.zeros((K,2))
    pointcloud=[];pair_info=[];known={};reference=None
    for pair in sorted(pairs,key=lambda p:-p['inliers']):
        if pair['inliers']<8:continue
        seed=calibrate_pair(pair,width,reference);v=pair['view_b']
        if reference is None:
            f[0]=seed['f'][0];pp[0]=seed['pp'][0]
            reference=np.r_[np.log(f[0]),pp[0]]
        # Pairwise essential geometry has an arbitrary baseline scale. Shared
        # image-0 tracks provide a 1D least-squares baseline alignment, avoiding
        # silently overwriting camera-0 intrinsics for every additional view.
        shared=[(i,known[tuple(np.round(p,2))]) for i,p in enumerate(pair['pixels_a']) if tuple(np.round(p,2)) in known]
        baseline_scale=1.;uncertain_scale=bool(pointcloud and len(shared)<2)
        if len(shared)>=2:
            ids=np.array([z[0] for z in shared]);world=np.array([z[1] for z in shared])
            bearing=np.column_stack([((pair['pixels_b'][ids]/width-.5)-seed['pp'][1])/seed['f'][1],np.ones(len(ids))])
            A=np.cross(bearing,seed['g']@seed['C']);B=np.cross(bearing,world@seed['g'].T)
            estimated=np.sum(A*B)/max(np.sum(A*A),1e-12)
            if estimated>0:baseline_scale=float(estimated)
            else:uncertain_scale=True
        seed['C']*=baseline_scale;seed['X']*=baseline_scale
        C[v]=seed['C'];g[v]=seed['g'];f[v]=seed['f'][1];pp[v]=seed['pp'][1]
        front=(seed['X'][:,2]>0)&(((seed['X']-C[v])@g[v].T)[:,2]>0)
        pointcloud.append(seed['X'][front])
        for p,Xp,keep in zip(pair['pixels_a'],seed['X'],front):
            if keep:known.setdefault(tuple(np.round(p,2)),Xp)
        pair_info.append(dict(view=v,positive_depth=seed['positive_depth'],tracks=seed['tracks'],
                              calibration_residual=seed['calibration_residual'],prior=seed['prior'],baseline_scale_uncertain=uncertain_scale))
    yy,xx=np.nonzero(masks[0]);lo=np.array([xx.min(),yy.min()])/width-.5;hi=np.array([xx.max(),yy.max()])/width-.5
    median_depth=np.median(pointcloud[0][:,2]) if pointcloud else 3.
    factor=np.sqrt(np.prod(hi-lo))/(2*f[0]);depth=median_depth/max(1-.6*factor,.4)
    centre=np.r_[((lo+hi)/2-pp[0])*depth/f[0],depth]
    radii=np.r_[(hi-lo)*depth/(2*f[0]),factor*depth]
    unit,faces=sphere(subdivisions);X=unit*radii+centre
    # No per-class camera orientations: missing-view seeds use the SAME initial
    # view and are explicitly marked uncalibrated, not asserted to be recovered.
    missing=[v for v in range(1,K) if not any(p['view_b']==v for p in pairs)]
    for v in missing:C[v]=0;g[v]=np.eye(3);f[v]=f[0]
    mean=X.mean(0);distance=np.linalg.norm(C[0]-mean);scale=3/max(distance,1e-8)
    X=(X-mean)*scale;C=(C-mean)*scale
    return dict(X=X,C=C,g=g,lf=np.log(f),pp=pp),faces,dict(
        initialization='neutral subdivided icosphere; affine extents from first silhouette',
        camera_initialization='feature-based two-view geometry with crop/focal priors',
        uncalibrated_views=missing,pairs=pair_info,metric_scale_known=False,
        note='two-view calibration may be ill-conditioned; this is an initial estimate, not unique camera recovery')
