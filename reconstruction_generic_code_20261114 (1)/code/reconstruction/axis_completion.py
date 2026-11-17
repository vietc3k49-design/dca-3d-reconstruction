"""Image-supported axes with an explicitly ASSUMED default C8 completion.

Bilateral silhouette support identifies axis candidates, NOT rotational order.
The conditional default (45 degree invariance) is tested against a 2D budget.
Axes remain fixed in the reference camera gauge and cannot escape by rotating
to a depth-elongation direction merely to minimize the shape penalty.
"""
import copy
import cv2
import numpy as np
from .engine import objective,step,valid,first_hit_constraints
from .optimizer import optimize
from .axis_geometry import update_surface_correspondences,transverse_balance
from .weak_priors import pixel_normalized_shape_priors


def reflection_support(mask,direction,centre):
    d=np.asarray(direction,float);d/=np.linalg.norm(d)
    R=2*np.outer(d,d)-np.eye(2)
    transform=np.column_stack([R,centre-R@centre])
    reflected=cv2.warpAffine(np.uint8(mask),transform,(mask.shape[1],mask.shape[0]),flags=cv2.INTER_NEAREST)>0
    return float(np.sum(mask&reflected)/max(np.sum(mask|reflected),1))


def image_axes(mask):
    yy,xx=np.nonzero(mask);centre=np.array([xx.mean(),yy.mean()]);P=np.column_stack([xx,yy])-centre
    eigen,V=np.linalg.eigh(np.cov(P.T));candidates=[]
    # Search the full image-angle range. PCA supplies additional exact candidate
    # directions and a transparent longitudinal tie-breaker, never a class ID.
    directions=[np.array([np.cos(t),np.sin(t)]) for t in np.deg2rad(np.arange(0,180,2))]
    directions.extend([V[:,0],V[:,1]])
    records=[]
    for d in directions:
        score=reflection_support(mask,d,centre);extent=float(np.mean((P@d)**2))
        records.append(dict(direction=d,score=score,extent=extent))
    best=max(r['score'] for r in records)
    # Within one percentage point of support, prefer the longer observed axis.
    # This is an explicit default when axis identity is ambiguous in images.
    records.sort(key=lambda r:(r['score']>=best-.01,r['extent'] if r['score']>=best-.01 else r['score']),reverse=True)
    for r in records:
        if any(abs(r['direction']@q['direction'])>np.cos(np.deg2rad(25)) for q in candidates):continue
        candidates.append(r)
        if len(candidates)==2:break
    return centre,candidates


def axis_hypotheses(masks,state):
    profiles=[image_axes(m) for m in masks];hypotheses=[]
    for rank,r in enumerate(profiles[0][1]):
        d=r['direction'];axis=state['g'][0].T@np.r_[d,0.];axis/=np.linalg.norm(axis)
        evidence=[]
        for view,(centre,_) in enumerate(profiles):
            midpoint=state['X'].mean(0);q=state['g'][view]@(midpoint-state['C'][view]);ac=state['g'][view]@axis
            projected=ac[:2]-q[:2]*ac[2]/q[2]
            length=np.linalg.norm(projected)
            score=0. if length<1e-8 else reflection_support(masks[view],projected/length,centre)
            evidence.append(dict(view=view+1,mask_reflection_IoU=score))
        hypotheses.append(dict(label=f'image_axis_{rank+1}',axis=axis.tolist(),
            reference_direction=d.tolist(),reference_support=r['score'],
            observed_longitudinal_variance=r['extent'],view_evidence=evidence,
            evidence_score=float(np.mean([v['mask_reflection_IoU'] for v in evidence])),
            assumption='axis parallel to reference image plane; tilt not identified as physical truth',
            rotation_order=8,rotation_order_status='conditional_default_prior_NOT_detected_from_reflection'))
    return hypotheses


def final_axis_support(masks,state,axis):
    result=[];midpoint=state['X'].mean(0)
    for view,mask in enumerate(masks):
        yy,xx=np.nonzero(mask);centre=np.array([xx.mean(),yy.mean()])
        q=state['g'][view]@(midpoint-state['C'][view]);ac=state['g'][view]@axis
        d=ac[:2]-q[:2]*ac[2]/q[2]
        result.append(0. if np.linalg.norm(d)<1e-8 else reflection_support(mask,d,centre))
    return result


def fit_axis(state,a,hypothesis,steps=900,block_steps=100,verbose=True,optimizer_fn=None):
    solve=optimizer_fn or optimize
    work=copy.deepcopy(state);blocks=[];history=[];outer=[]
    axis=np.array(hypothesis['axis']);b=dict(a,compactness_weight=0.,require_positive_volume=True,
                                          visibility_penalty=0.,visibility_dual=np.zeros(len(a['ci'])))
    for start in range(0,steps,block_steps):
        association=update_surface_correspondences(work['X'],a['faces'],axis,8)
        b['axis_constraint']=dict(axis=axis,order=8,association=association)
        beginning=objective(work,b)[0];outer.append(float(beginning))
        result=solve(work,b,steps=min(block_steps,steps-start),verbose=verbose,
            objective_fn=objective,step_fn=step,valid_fn=valid,
            learning_rates=dict(X=.003,C=.002,g=.0006,lf=.0008,pp=.0005,h=.001))
        work=result['state'];blocks.append(result['optimization'])
        history.extend([dict(block=start//block_steps,**h) for h in result['history']])
        outer.append(float(result['stats']['total']))
        if verbose:print(f' axis {hypothesis["label"]}, {start+min(block_steps,steps-start)}: '
                         f'transverse ratio {result["stats"]["transverse_ratio"]:.3f}',flush=True)
        if result['optimization']['accepted_steps']==0:break
    feasibility=[];dual=np.zeros(len(a['ci']));b['visibility_feasibility_tolerance']=1e-6
    # Proper inequality augmented Lagrangian for c = 1-t_first-0.001 <= 0.
    # Keep the original geometric tolerance; only a separate 1e-6 numerical
    # feasibility tolerance is allowed. Never remove inconvenient observations.
    for rho in (10.,100.,1000.):
        c=first_hit_constraints(work,b)
        if not len(c) or np.max(c)<=1e-6:break
        dual=np.maximum(0,dual+rho*c)
        b.update(visibility_penalty=rho,visibility_dual=dual)
        b['axis_constraint']['association']=update_surface_correspondences(work['X'],a['faces'],axis,8)
        polished=solve(work,b,steps=150,verbose=verbose,objective_fn=objective,step_fn=step,valid_fn=valid,
            learning_rates=dict(X=.002,C=.001,g=.0004,lf=.0005,pp=.0003,h=.0007))
        work=polished['state'];feasibility.append(dict(penalty=rho,status=polished['status'],
                             optimization=polished['optimization'],statistics=polished['stats']))
        history.extend([dict(feasibility_penalty=rho,**h) for h in polished['history']])
    b['axis_constraint']['association']=update_surface_correspondences(work['X'],a['faces'],axis,8)
    _,_,stats=objective(work,b);outer.append(float(stats['total']))
    result.update(state=work,stats=stats,history=history,
                  optimization=dict(blocks=blocks,true_objective_at_updates=outer,
                                    monotone_outer=bool(np.all(np.diff(outer[:-1])<=1e-12)),
                                    feasibility_blocks=feasibility))
    return result,b


def select_axis_completion(state,a,masks,steps=900,noise_pixels=1.,verbose=True,control_steps=None):
    a=pixel_normalized_shape_priors(a)
    hypotheses=axis_hypotheses(masks,state);runs=[];records=[]
    # The null model remains available when images contradict the default.
    base=dict(a,compactness_weight=0.,require_positive_volume=True,axis_constraint=None,
              visibility_feasibility_tolerance=1e-6,visibility_penalty=0.,visibility_dual=np.zeros(len(a['ci'])))
    if verbose:print('Axis selection: no-axis control',flush=True)
    if control_steps==0:
        _,_,stats=objective(state,base)
        baseline=dict(state=copy.deepcopy(state),stats=stats,status='cached_control',steps=0,
                      history=[dict(step=0,**stats)],optimization=dict(method='cached image fit',accepted_losses=[stats['total']]))
    else:
        baseline=optimize(copy.deepcopy(state),base,steps=steps if control_steps is None else control_steps,verbose=verbose,
            objective_fn=objective,step_fn=step,valid_fn=valid,
            learning_rates=dict(X=.003,C=.002,g=.0006,lf=.0008,pp=.0005,h=.001))
    runs.append((baseline,base));records.append(dict(label='no_axis',evidence_score=None,
        image_cost_px_squared=baseline['stats']['image_cost']*a['width']**2,
        state={k:v.tolist() for k,v in baseline['state'].items()},
        visibility_violations=baseline['stats']['visibility_violations']))
    for hypothesis in hypotheses:
        if hypothesis['evidence_score']<.85:
            records.append(dict(**hypothesis,status='weak_image_axis_support',tested=False));continue
        if hypothesis['evidence_score']<max(h['evidence_score'] for h in hypotheses)-.01:
            records.append(dict(**hypothesis,status='less_image_axis_support',tested=False));continue
        if verbose:print('Axis selection:',hypothesis['label'],'support',hypothesis['evidence_score'],flush=True)
        # A tiny prior must refine the already fitted data solution, rather
        # than spend its budget re-fitting images from the earlier initialization.
        result,b=fit_axis(baseline['state'],base,hypothesis,steps=steps,verbose=verbose)
        final_support=final_axis_support(masks,result['state'],np.array(hypothesis['axis']))
        # Compare both states under the SAME weak axis prior. The unconstrained
        # control must pay that prior too; comparing different objectives would
        # make epsilon meaningless. Drop AL constants for physical model scoring.
        reference=dict(b,visibility_penalty=0.,visibility_dual=np.zeros(len(a['ci'])))
        reference['axis_constraint']=dict(axis=np.array(hypothesis['axis']),order=8,
            association=update_surface_correspondences(baseline['state']['X'],a['faces'],np.array(hypothesis['axis']),8))
        baseline_common=objective(baseline['state'],reference)[0]
        physical=dict(b,visibility_penalty=0.,visibility_dual=np.zeros(len(a['ci'])))
        candidate_common=objective(result['state'],physical)[0]
        runs.append((result,b));records.append(dict(**hypothesis,tested=True,run_index=len(runs)-1,
            image_cost_px_squared=result['stats']['image_cost']*a['width']**2,
            visibility_violations=result['stats']['visibility_violations'],
            transverse_ratio=result['stats']['transverse_ratio'],statistics=result['stats'],
            final_mask_reflection_IoU=final_support,
            baseline_common_objective_px_squared=baseline_common*a['width']**2,
            candidate_common_objective_px_squared=candidate_common*a['width']**2,
            state={k:v.tolist() for k,v in result['state'].items()},
            axis_constraint=b['axis_constraint'],visibility_dual=b.get('visibility_dual'),
            visibility_penalty=b.get('visibility_penalty',0.),
            symmetry_satisfied=(result['stats']['axis_transverse_balance']<=.1**2 and
                                result['stats']['axis_surface_group_error']<=.1**2),
            optimization=result['optimization']))
    minimum=min(r['image_cost_px_squared'] for r in records if 'image_cost_px_squared' in r)
    budget=minimum+noise_pixels**2;baseline_vis=baseline['stats']['visibility_violations']
    eligible=[]
    for r in records:
        if r['label']=='no_axis':r['admissible']=r['image_cost_px_squared']<=budget
        else:r['admissible']=bool(r.get('tested',False) and r['image_cost_px_squared']<=budget and
                              r['visibility_violations']<=baseline_vis and
                              np.mean(r['final_mask_reflection_IoU'])>=.85 and
                              r['candidate_common_objective_px_squared']<=r['baseline_common_objective_px_squared']+1e-10)
        if r.get('tested') and r['admissible']:eligible.append(r)
    # Choose supported axes by image evidence, NEVER by the smallest 3D symmetry
    # penalty or by switching to a cheap two-fold rotation.
    selected=max(eligible,key=lambda r:r['evidence_score']) if eligible else records[0]
    index=selected.get('run_index',0);result,b=runs[index]
    for r in records:r['selected']=r is selected
    result['axis_symmetry_selection']=dict(selected=selected['label'],accepted_axis=index>0,
        allowed_image_cost_px_squared=budget,minimum_image_cost_px_squared=minimum,
        noise_pixels=noise_pixels,rotation_order=8,rotation_angle_degrees=45.,
        rotational_order_is_assumed=True,axis_is_fixed_after_image_proposal=True,
        compactness_weight=0.,hypotheses=records,physical_truth_certified=False,
        first_hit_geometric_tolerance=.001,first_hit_numerical_tolerance=1e-6,
        relative_symmetry_rms_tolerance=None,symmetry_is_soft_only=True,
        symmetry_epsilon=a['symmetry_epsilon'],shape_prior_units=a['shape_prior_units'],
        note='Axis from images, C8 is a weak default. No hard symmetry gate; common pixel-unit objective decides whether a shape update is justified.')
    if verbose:print('Axis selection chose',selected['label'],'image budget',budget,flush=True)
    return result,b
