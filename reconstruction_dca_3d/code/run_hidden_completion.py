"""Same-cost completion of hidden geometry, with frozen observed features/cameras."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
from reconstruction.records import read_case
from reconstruction.engine import objective
from reconstruction.hidden_completion import complete_hidden
from reconstruction.ablation import summarize
from reconstruct_images import save_report,jsonable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=900);args=p.parse_args()
    if (args.output/'result.json').exists():raise FileExistsError('Preserve prior result')
    payload,images,masks,pairs,state,_,a=read_case(args.source)
    start=time.perf_counter();before=objective(state,a)[0]*a['width']**2
    result,b=complete_hidden(state,a,steps=args.steps)
    output=save_report(args.output,images,masks,payload['input_files'],pairs,payload['feature_frontend'],
                       payload['initializer'],state,result,b,result['history'])
    weight=a['calibration_weight']*a['width']**2
    row=summarize(output,dict(label='hidden_completion',camera_coefficient=weight,point_tolerance_pixels=1.),time.perf_counter()-start)
    row.update(host=platform.node(),contact_audit=a['contact_model'].audit(result['state']['X']),
        contact_cost_px_squared=result['stats']['contact_cost']*a['width']**2,
        before_objective_px_squared=before,after_objective_px_squared=result['stats']['total']*a['width']**2,
        prior_point_rms_px=payload['features']['rms_reprojection_px'],
        hidden_counts=[block['hidden_vertex_count'] for block in result['optimization']['blocks']],
        source=str(args.source.resolve()),shape_epsilon=.001,same_objective=True)
    assert abs(row['feature_rms_px']-row['prior_point_rms_px'])<1e-9
    assert row['contact_audit']['intersecting_pairs']==0
    assert row['after_objective_px_squared']<=before+1e-10
    (args.output/'hidden_audit.json').write_text(json.dumps(jsonable(row),indent=2),encoding='utf-8')
    print(json.dumps(jsonable(row)),flush=True)


if __name__=='__main__':main()
