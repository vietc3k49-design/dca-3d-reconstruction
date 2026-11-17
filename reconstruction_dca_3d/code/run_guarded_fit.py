"""Same generic free-mesh model, now reject transverse crossings and test field smoothing."""
import argparse
import json
import platform
import time
from pathlib import Path
from reconstruction.records import read_case
from reconstruction.guarded import make_guarded_problem
from reconstruction.axis_completion import fit_axis
from reconstruction.intersections import transverse_intersections
from reconstruction.ablation import summarize
from reconstruct_images import save_report,jsonable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--field',choices=('bilinear','cubic'),required=True);p.add_argument('--steps',type=int,default=1800)
    args=p.parse_args()
    if (args.output/'result.json').exists():raise FileExistsError('Preserve prior result')
    payload,images,masks,pairs,_,neutral,a=read_case(args.source)
    state,a,hypothesis=make_guarded_problem(neutral,a,args.field);start=time.perf_counter()
    result,b=fit_axis(state,a,hypothesis,steps=args.steps)
    init=dict(payload['initializer'],guarded_restart='cached neutral image estimates; all UVs/track IDs retained; surface anchors reinitialized')
    output=save_report(args.output,images,masks,payload['input_files'],pairs,payload['feature_frontend'],
                       init,state,result,b,result['history'])
    row=summarize(output,dict(label='guard_'+args.field,camera_coefficient=1.,point_tolerance_pixels=1.),time.perf_counter()-start)
    crossing=transverse_intersections(result['state']['X'],b['faces'])
    row.update(host=platform.node(),field_interpolation=args.field,source=str(args.source.resolve()),
        transverse_crossing_pairs=len(crossing),crossing_face_ids=crossing,guard_enabled=True,
        full_embeddedness_certified=False,steps_budget=args.steps,
        final_status=result['status'],monotone_outer=result['optimization']['monotone_outer'])
    (args.output/'guard_audit.json').write_text(json.dumps(jsonable(row),indent=2),encoding='utf-8')
    if crossing:raise AssertionError('Intersection guard failed')
    print(json.dumps(jsonable(row)),flush=True)


if __name__=='__main__':main()
