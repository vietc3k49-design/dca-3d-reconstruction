"""Run one predeclared case on a cached image problem, then export ALL geometry."""
import argparse
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
from reconstruction.records import read_case
from reconstruction.engine import objective
from reconstruction.axis_completion import fit_axis
from reconstruction.ablation import case_grid,configure,summarize
from reconstruct_images import save_report,jsonable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',required=True,choices=[c['label'] for c in case_grid()])
    p.add_argument('--steps',type=int,default=900)
    p.add_argument('--optimizer',choices=('adam','lbfgs'),default='adam')
    args=p.parse_args()
    if args.steps<1:p.error('Positive step budget required.')
    if (args.output/'result.json').exists():raise FileExistsError('Preserve existing result; choose a new output.')
    payload,images,masks,pairs,state,initial,a=read_case(args.source)
    case=next(c for c in case_grid() if c['label']==args.case);a=configure(a,case)
    axis=np.array(a['axis_constraint']['axis']);hypothesis=dict(label=args.case,axis=axis.tolist())
    start=time.perf_counter();physical=dict(a,visibility_penalty=0.,visibility_dual=np.zeros(len(a['ci'])))
    _,_,before=objective(state,physical)
    from reconstruction.lbfgs import optimize_lbfgs
    result,b=fit_axis(state,a,hypothesis,steps=args.steps,optimizer_fn=optimize_lbfgs if args.optimizer=='lbfgs' else None)
    output=save_report(args.output,images,masks,payload['input_files'],pairs,payload['feature_frontend'],
                       payload['initializer'],state,result,b,result['history'])
    summary=summarize(output,case,time.perf_counter()-start)
    physical=dict(b,visibility_penalty=0.,visibility_dual=np.zeros(len(b['ci'])))
    final_physical=objective(result['state'],physical)[0]*a['width']**2
    summary.update(host=platform.node(),pid=os.getpid(),steps_budget=args.steps,
        source=str(args.source.resolve()),shape_epsilon=.001,axis=axis.tolist(),
        point_tolerance_status='explicit uncertainty assumption, not measured confidence',
        starting_image_cost_px_squared=before['image_cost']*a['width']**2,optimizer=args.optimizer,
        starting_objective_px_squared=before['total']*a['width']**2,
        final_objective_px_squared=final_physical,
        physical_objective_improved=final_physical<=before['total']*a['width']**2+1e-10,
        solver_messages=[b.get('scipy_message','') for b in result['optimization']['blocks']])
    (args.output/'ablation.json').write_text(json.dumps(jsonable(summary),indent=2),encoding='utf-8')
    print(json.dumps(jsonable(summary)),flush=True)


if __name__=='__main__':main()
