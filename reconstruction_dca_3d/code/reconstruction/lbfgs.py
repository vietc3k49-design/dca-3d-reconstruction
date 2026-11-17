"""Spec: limited-memory gradient optimization of the UNCHANGED objective.

Analytic SO(3) chart derivatives, existing focal/crop bounds and validity
checks. Track the best valid evaluation and never return a higher true cost.
Termination messages are recorded, not interpreted as global convergence.
"""
import copy
import time
import numpy as np
from scipy.optimize import minimize
from .chart import Chart


def optimize_lbfgs(s,a,gauge=None,steps=900,verbose=True,objective_fn=None,step_fn=None,valid_fn=None,
                   learning_rates=None):
    if a.get('contact_model') is not None:raise ValueError('Use the CCD-aware Adam solver for contact trajectories.')
    chart=Chart(s);scale=a.get('optimization_scale',1.);start=time.perf_counter()
    initial=objective_fn(s,a,gauge);best=dict(value=initial[0],state=copy.deepcopy(s),stats=initial[2])
    losses=[float(initial[0])];history=[dict(step=0,**initial[2])];evaluations=0;invalid=0
    def evaluate(z):
        nonlocal evaluations,invalid
        state=chart.decode(z);evaluations+=1
        if valid_fn and not valid_fn(state,a):
            invalid+=1;return 1e20,np.zeros_like(z)
        try:value,gradient,stats=objective_fn(state,a,gauge)
        except (ValueError,FloatingPointError):
            invalid+=1;return 1e20,np.zeros_like(z)
        if not np.isfinite(value):
            invalid+=1;return 1e20,np.zeros_like(z)
        if value<best['value']:
            best.update(value=value,state=state,stats=stats);losses.append(float(value))
        return value*scale,chart.pullback(z,gradient)*scale
    count=0
    def callback(z):
        nonlocal count
        count+=1
        if count==1 or count%100==0:
            history.append(dict(step=count,**best['stats']))
            if verbose:print(f' LBFGS {count}: cost {best["value"]*scale:.8g} px^2, '
                             f'points {best["stats"]["point_rmse_px"]:.3f} px',flush=True)
    result=minimize(evaluate,chart.initial,jac=True,method='L-BFGS-B',bounds=chart.bounds,callback=callback,
                    options=dict(maxiter=steps,maxfun=max(200,5*steps+1),maxls=35,maxcor=12,ftol=1e-12,gtol=1e-6))
    state=step_fn(best['state'],{},a,gauge)
    final=objective_fn(state,a,gauge)
    # Pure gauge normalization must preserve cost; fall back if numerical loss
    # of gauge precision or a geometry validity threshold prevents this.
    if abs(final[0]-best['value'])>1e-10/scale or (valid_fn and not valid_fn(state,a)):
        state=best['state'];final=objective_fn(state,a,gauge)
    history.append(dict(step=int(result.nit),**final[2]))
    return dict(state=state,stats=final[2],history=history,steps=int(result.nit),
        status='lbfgs_success' if result.success else 'lbfgs_stopped',seconds=time.perf_counter()-start,
        a=a,gauge=gauge,optimization=dict(method='L-BFGS-B with analytic SO(3) charts',
            accepted_steps=int(result.nit),best_evaluation_updates=len(losses)-1,accepted_losses=losses,
            objective_evaluations=evaluations,invalid_evaluations=invalid,monotone=bool(np.all(np.diff(losses)<=0)),
            loss_history_kind='best_valid_evaluation',scipy_success=bool(result.success),
            scipy_status=int(result.status),scipy_message=str(result.message)))
