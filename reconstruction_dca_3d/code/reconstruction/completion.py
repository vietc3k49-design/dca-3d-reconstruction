"""Generic compact completion selected under an explicit 2D discrepancy budget.

This is a finite, nonconvex regularization-path heuristic, not a uniqueness or
global optimality theorem. Images may reject compactness for long/thin objects.
"""
import copy
import numpy as np
from .engine import objective,step,valid
from .optimizer import optimize
from .priors import compactness


def select_completion(state,a,steps=750,noise_pixels=1.,strengths=(0.,.5,1.,2.,4.),verbose=True,optimizer_fn=None):
    solve=optimizer_fn or optimize
    width=a['width'];work=copy.deepcopy(state);candidates=[];records=[]
    for strength in sorted(set(strengths)):
        b=dict(a,compactness_weight=(strength/width)**2,require_positive_volume=True)
        if verbose:print(f'Completion path: compactness {strength:g}px, budget {noise_pixels:g}px',flush=True)
        result=solve(work,b,steps=steps,verbose=verbose,objective_fn=objective,step_fn=step,valid_fn=valid,
            learning_rates=dict(X=.003,C=.002,g=.0006,lf=.0008,pp=.0005,h=.001))
        work=result['state'];stats=result['stats']
        value,_,diagnostics=compactness(work['X'],b['faces'])
        eig=np.linalg.eigvalsh(np.cov(work['X'].T))
        image_cost=stats['image_cost']*width**2
        record=dict(strength_pixels=float(strength),image_cost_px_squared=float(image_cost),
            feature_rmse_px=stats['point_rmse_px'],visibility_violations=stats['visibility_violations'],
            compactness=value,pca_long_to_short=float(np.sqrt(eig[-1]/max(eig[0],1e-15))),
            isoperimetric_quotient=diagnostics['isoperimetric_quotient'],status=result['status'],
            state={key:value.tolist() for key,value in work.items()},
            optimization=result['optimization'])
        records.append(record);candidates.append((copy.deepcopy(work),result,b))
    minimum_data=min(r['image_cost_px_squared'] for r in records)
    budget=minimum_data+noise_pixels**2
    best_visibility=min(r['visibility_violations'] for r in records)
    feasible=[i for i,r in enumerate(records) if r['image_cost_px_squared']<=budget and
              r['visibility_violations']==best_visibility]
    if not feasible:
        # A visibility/data conflict is reported rather than silently ignored.
        feasible=[int(np.argmin([r['image_cost_px_squared'] for r in records]))]
        conflict=True
    else:conflict=False
    selected=min(feasible,key=lambda i:records[i]['compactness'])
    for i,r in enumerate(records):r['admissible']=i in feasible;r['selected']=i==selected
    seed,result,b=candidates[selected];result['state']=seed
    summary=dict(policy='minimum compactness among tested candidates within an additive 2D discrepancy budget',
        noise_pixels=noise_pixels,minimum_image_cost_px_squared=minimum_data,
        allowed_image_cost_px_squared=budget,selected_strength_pixels=records[selected]['strength_pixels'],
        selected_index=selected,visibility_data_conflict=conflict,path=records,
        selected_at_strength_upper_bound=records[selected]['strength_pixels']==max(strengths),
        visibility_not_resolved=best_visibility>0,
        admissible_axis_ratio_range=[min(records[i]['pca_long_to_short'] for i in feasible),
                                     max(records[i]['pca_long_to_short'] for i in feasible)],
        uncertainty_note='tested regularization sensitivity, NOT a confidence interval or full ambiguity bound',
        global_optimum_certified=False,physical_truth_certified=False,
        path_steps_per_strength=steps)
    if verbose:print(f'Selected completion {summary["selected_strength_pixels"]:g}px: '
        f'image cost {records[selected]["image_cost_px_squared"]:.3f} <= {budget:.3f} px^2, '
        f'axis ratio {records[selected]["pca_long_to_short"]:.3f}',flush=True)
    result['completion_selection']=summary
    return result,b
