"""The hidden block must preserve every observed query and use a restricted gradient."""
from pathlib import Path
import numpy as np
from reconstruction.records import read_case
from reconstruction.engine import objective,queries,step
from reconstruction.hidden_completion import eligible_vertices,restricted_objective


def project(s,a):
    q=queries(s,a);camera=np.einsum('kij,ktj->kti',s['g'],q[None]-s['C'][:,None])
    return np.exp(s['lf'])[:,None,None]*camera[...,:2]/camera[...,2,None]+s['pp'][:,None]


def main():
    root=Path(__file__).resolve().parents[1]/'generic_can_axis_symmetry_soft_xeon'
    _,_,_,_,s,_,a=read_case(root);free=eligible_vertices(s,a)
    protected=np.unique(a['anchor_vertices']);assert not free[protected].any() and free.any()
    value,g,stats=objective(s,a);restricted=restricted_objective(objective,free)
    v,gg,st=restricted(s,a)
    assert v==value and st==stats
    assert np.array_equal(gg['X'][free],g['X'][free]) and not gg['X'][~free].any()
    for key in ('C','g','lf','pp','h'):assert not gg[key].any()
    delta={key:np.zeros_like(x) for key,x in g.items()}
    delta['X'][free]=np.random.default_rng(19).normal(0,.001,(int(free.sum()),3))
    after=step(s,delta,a)
    assert np.max(np.abs(project(after,a)-project(s,a)))<1e-12
    print('PASS hidden unanchored block, unchanged full cost/measurements, exact protected feature projections')


if __name__=='__main__':main()
