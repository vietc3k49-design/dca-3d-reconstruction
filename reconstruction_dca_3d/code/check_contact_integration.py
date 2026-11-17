"""Test exact contact objective composition and actual CCD-scaled descent steps."""
import copy
from pathlib import Path
import numpy as np
from reconstruction.records import read_case
from reconstruction.contact import ContactSystem
from reconstruction.engine import objective
from reconstruction.optimizer import optimize
from check_generic_reconstruction import check_gradient


def check_step():
    class QuarterStep:
        def safe_fraction(self,X,Y):return .25
    def cost(s,a,gauge=None):
        value=float(np.sum(s['X']**2))
        return value,dict(X=2*s['X']),dict(total=value,point_rmse_px=0.,model_to_image_px=0.,image_to_model_px=0.,visibility_violations=0)
    def advance(s,delta,a,gauge=None):return dict(X=s['X']+delta['X'])
    source=dict(X=np.ones((1,3)))
    kwargs=dict(steps=1,verbose=False,objective_fn=cost,step_fn=advance,learning_rates=dict(X=.2))
    full=optimize(copy.deepcopy(source),{},**kwargs)
    limited=optimize(copy.deepcopy(source),dict(contact_model=QuarterStep()),**kwargs)
    assert np.allclose(limited['state']['X']-source['X'],.25*(full['state']['X']-source['X']))
    assert limited['optimization']['ccd_limited_trials']>0


def main():
    root=Path(__file__).resolve().parents[1]/'generic_can_axis_symmetry_soft_xeon'
    _,_,_,_,_,s,a=read_case(root);model=ContactSystem(s['X'],a['faces'])
    s=copy.deepcopy(s);s['X'][:,2]*=.03
    s['X']+=np.random.default_rng(51).normal(0,1e-5,s['X'].shape)
    value,gradient,stats=objective(s,a);contact,cg,_=model.term(s['X']);assert contact>0
    b=dict(a,contact_model=model,contact_weight=1/a['width']**2)
    v,g,st=objective(s,b)
    assert abs(v-value-contact*b['contact_weight'])<1e-12
    assert np.max(abs(g['X']-gradient['X']-cg*b['contact_weight']))<1e-10
    for key in ('C','g','lf','pp','h'):assert np.array_equal(g[key],gradient[key])
    check_gradient(s,b,full=False);check_step()
    print('PASS separate contact term, full analytic chain and actual CCD-scaled Armijo direction')


if __name__=='__main__':main()
