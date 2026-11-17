"""Spec: same-objective block descent on model-hidden, unobserved vertices only.

Visibility is a current-model hypothesis. Protected barycentric feature
supports are never moved in this block except by the projection-invariant
similarity gauge. All mask/feature/first-hit/contact terms are still evaluated.
"""
import numpy as np
from .losses import visibility_term
from .optimizer import optimize
from .axis_completion import fit_axis


def eligible_vertices(state,arrays):
    P=len(state['X']);K=len(state['C'])
    q=np.einsum('kij,kpj->kpi',state['g'],state['X'][None]-state['C'][:,None])
    a=dict(ci=np.repeat(np.arange(K),P),pi=np.tile(np.arange(P),K),faces=arrays['faces'],
           relative_tolerance=.001,return_constraints=True)
    constraints=visibility_term(q,a)[-1].reshape(K,P)
    free=(constraints>0).all(0)
    free[np.unique(arrays['anchor_vertices'])]=False
    return free


def restricted_objective(objective,free):
    def evaluate(s,a,gauge=None):
        value,gradient,stats=objective(s,a,gauge)
        masked={key:np.zeros_like(g) for key,g in gradient.items()}
        masked['X'][free]=gradient['X'][free]
        return value,masked,stats
    return evaluate


def complete_hidden(state,arrays,steps=900,verbose=True):
    def solve(s,a,**kwargs):
        free=eligible_vertices(s,a)
        kwargs['objective_fn']=restricted_objective(kwargs['objective_fn'],free)
        result=optimize(s,a,**kwargs)
        result['optimization']['hidden_vertex_count']=int(free.sum())
        if verbose:print(' hidden unanchored vertices',int(free.sum()),flush=True)
        return result
    hypothesis=dict(label='hidden_completion',axis=np.array(arrays['axis_constraint']['axis']).tolist())
    return fit_axis(state,arrays,hypothesis,steps=steps,verbose=verbose,optimizer_fn=solve)


def append_hidden_stage(result,arrays,history,optimization_stages,steps):
    if steps<=0 or arrays.get('axis_constraint') is None:return result,arrays
    refined,b=complete_hidden(result['state'],arrays,steps=steps)
    for key in ('axis_symmetry_selection','completion_selection'):
        if key in result:refined[key]=result[key]
    refined['hidden_completion']=dict(same_objective=True,steps_budget=steps,
        hidden_counts=[block['hidden_vertex_count'] for block in refined['optimization']['blocks']],
        note='Model-hidden unanchored vertices only; protected feature projections unchanged up to similarity gauge.')
    history.extend(dict(stage='hidden_completion',**h) for h in refined['history'])
    optimization_stages.append(dict(stage='hidden_completion',**refined['optimization']))
    return refined,b
