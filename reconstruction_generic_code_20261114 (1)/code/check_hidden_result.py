"""Replay same-cost hidden completion and verify no feature/camera drift."""
import argparse
import json
from pathlib import Path
import numpy as np
from reconstruction.records import read_case
from reconstruction.engine import objective,first_hit_constraints,valid


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);args=p.parse_args()
    before,_,_,_,seed,_,aa=read_case(args.source)
    after,_,_,_,state,_,a=read_case(args.root/'results')
    for key,value in before['observations'].items():assert np.array_equal(value,after['observations'][key]),key
    for key in ('g','lf','pp'):assert np.max(abs(seed[key]-state[key]))<1e-12,key
    for key in ('prior_weight','smooth_weight','compactness_weight','calibration_weight','point_tolerance_pixels','contact_weight'):
        assert aa[key]==a[key],key
    assert valid(state,a) and a['contact_model'].audit(state['X'])['intersecting_pairs']==0
    value,_,stats=objective(state,a);old=objective(seed,aa)[0]
    assert value<=old+1e-10/a['width']**2 and abs(value-after['objective']['total'])<1e-12
    assert abs(stats['point_rmse_px']-before['objective']['point_rmse_px'])<1e-9
    assert stats['visibility_violations']==0 and np.max(first_hit_constraints(state,a))<=1e-6
    record=dict(same_objective=True,observations=len(a['ci']),protected_projections_unchanged=True,
        first_hit_violations=0,libigl_intersections=0,before_px_squared=old*a['width']**2,
        after_px_squared=value*a['width']**2,raw_point_rms_px=stats['point_rmse_px'],physical_accuracy_known=False)
    (args.root/'results/verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('PASS HIDDEN RESULT',args.root.name,json.dumps(record))


if __name__=='__main__':main()
