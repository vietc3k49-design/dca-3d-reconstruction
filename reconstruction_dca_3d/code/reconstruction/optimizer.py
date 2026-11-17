"""Adam-preconditioned Armijo descent on the post-retraction objective.

Momentum may point uphill; in that case use the preconditioned current
negative gradient. Feasibility alone never suffices to accept a step.
"""
import copy
import time
import numpy as np


def optimize(s,a,gauge=None,steps=3500,verbose=True,objective_fn=None,step_fn=None,valid_fn=None,
             learning_rates=None):
    if objective_fn is None or step_fn is None:raise ValueError('Objective and step hooks required.')
    evaluate=objective_fn
    # A weak prior must not disappear below numerical stopping/Adam eps merely
    # because image coordinates were divided by image width. This is equivalent
    # to optimizing the globally rescaled pixel-unit objective, with unchanged minima.
    objective_scale=a.get('optimization_scale',1.)
    current=evaluate(s,a,gauge);g=current[1]
    m={key:np.zeros_like(v) for key,v in g.items()};v=copy.deepcopy(m)
    rates=dict(X=.0015,C=.001,g=.00025,lf=.0004)
    if learning_rates:rates.update(learning_rates)
    history=[];losses=[float(current[0])];start=time.perf_counter();status='max_steps'
    backtracks=0;fallbacks=0;accepted=0;next_alpha=1.;evaluations=1
    evaluation_budget=max(200,5*steps+1)
    ccd_limited=0;minimum_ccd_fraction=1.
    for iteration in range(1,steps+1):
        loss,gradient,stats=current
        if iteration==1 or iteration%100==0:history.append(dict(step=iteration,**stats))
        if verbose and (iteration==1 or iteration%250==0):
            print(f' {iteration}: points {stats["point_rmse_px"]:.3f}px, out {stats["model_to_image_px"]:.3f}px, '
                  f'in {stats["image_to_model_px"]:.3f}px, blocked {stats["visibility_violations"]}',flush=True)
        if iteration>200 and max(np.max(np.abs(z)) for z in gradient.values())<3e-8/objective_scale:
            status='small_gradient';break
        for key in gradient:
            m[key]=.9*m[key]+.1*gradient[key];v[key]=.999*v[key]+.001*gradient[key]**2
        preconditioner={key:rates.get(key,.001)/(np.sqrt(v[key]/(1-.999**iteration))+1e-9/objective_scale)
                        /(1+iteration/1800) for key in gradient}
        momentum={key:-preconditioner[key]*m[key]/(1-.9**iteration) for key in gradient}
        steepest={key:-preconditioner[key]*gradient[key] for key in gradient}
        success=False
        for kind,direction in enumerate((momentum,steepest)):
            slope=float(sum(np.sum(gradient[key]*d) for key,d in direction.items()))
            if slope>=0:continue
            alpha=min(1.,next_alpha*1.5) if kind==0 else 1.
            for _ in range(22):
                if evaluations>=evaluation_budget:break
                # Do not accept a vanishing momentum step indefinitely when a
                # fresh negative-gradient direction can make useful progress.
                if kind==0 and alpha<1e-4:break
                delta={key:alpha*d for key,d in direction.items()};fraction=1.
                if a.get('contact_model') is not None:
                    fraction=a['contact_model'].safe_fraction(s['X'],s['X']+delta['X'])
                    if fraction<1.:ccd_limited+=1;minimum_ccd_fraction=min(minimum_ccd_fraction,fraction)
                    delta={key:fraction*d for key,d in delta.items()}
                trial=step_fn(s,delta,a,gauge)
                if valid_fn is None or valid_fn(trial,a):
                    candidate=evaluate(trial,a,gauge)
                    evaluations+=1
                    if fraction>0 and np.isfinite(candidate[0]) and candidate[0]<=loss+1e-4*alpha*fraction*slope:
                        s=trial;current=candidate;losses.append(float(candidate[0]));accepted+=1
                        if kind:fallbacks+=1
                        next_alpha=alpha if kind==0 else .5
                        success=True;break
                alpha*=.5;backtracks+=1
            if success:break
        if not success:
            status='line_search_budget' if evaluations>=evaluation_budget else 'no_descent_step'
            break
        if len(losses)>80 and losses[-80]-losses[-1]<1e-6*max(abs(losses[-80]),1e-12):
            status='stalled_cost';break
    history.append(dict(step=iteration,**current[2]))
    return dict(state=s,stats=current[2],history=history,steps=iteration,status=status,
                seconds=time.perf_counter()-start,a=a,gauge=gauge,
                optimization=dict(method='Adam-preconditioned Armijo descent on the retracted objective',
                    accepted_steps=accepted,backtracks=backtracks,gradient_fallbacks=fallbacks,
                    ccd_limited_trials=ccd_limited,minimum_ccd_fraction=minimum_ccd_fraction,
                    objective_evaluations=evaluations,evaluation_budget=evaluation_budget,
                    accepted_losses=losses,monotone=bool(np.all(np.diff(losses)<=0))))
