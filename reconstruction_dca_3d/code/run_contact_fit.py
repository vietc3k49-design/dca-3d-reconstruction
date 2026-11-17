"""Generic contact-feasible free-mesh fit, using established IPC and libigl tools."""
import argparse
import json
import platform
import time
from pathlib import Path
from reconstruction.records import read_case
from reconstruction.guarded import make_guarded_problem
from reconstruction.contact import ContactSystem
from reconstruction.axis_completion import fit_axis
from reconstruction.ablation import summarize
from reconstruct_images import save_report,jsonable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage-steps',type=int,default=600);p.add_argument('--camera-coefficient',type=float,default=.001)
    args=p.parse_args()
    if args.camera_coefficient<0:p.error('Camera coefficient must be nonnegative')
    if (args.output/'result.json').exists():raise FileExistsError('Preserve existing result')
    payload,images,masks,pairs,_,neutral,a=read_case(args.source)
    state,a,hypothesis=make_guarded_problem(neutral,a,'cubic');initial={k:v.copy() for k,v in state.items()}
    a['calibration_weight']=args.camera_coefficient/a['width']**2
    model=ContactSystem(state['X'],a['faces']);a['contact_model']=model
    start=time.perf_counter();stages=[];history=[];blocks=[];feasibility=[]
    for mu in (1.,.1,.01):
        a['contact_weight']=mu/a['width']**2
        print('CONTACT STAGE',mu,flush=True)
        result,a=fit_axis(state,a,hypothesis,steps=args.stage_steps);state=result['state']
        audit=model.audit(state['X']);assert audit['intersecting_pairs']==0,audit
        stages.append(dict(mu=mu,status=result['status'],statistics=result['stats'],audit=audit,
                           optimization=result['optimization']))
        history.extend(dict(contact_mu=mu,**h) for h in result['history'])
        blocks.extend(result['optimization']['blocks']);feasibility.extend(result['optimization']['feasibility_blocks'])
    result['optimization']=dict(blocks=blocks,feasibility_blocks=feasibility,contact_stages=stages,
        monotone_outer=all(s['optimization']['monotone_outer'] for s in stages),
        note='Compare losses within a fixed mu stage; barrier coefficients differ between stages.')
    init=dict(payload['initializer'],restart='neutral image-only initialization; same UVs and tracks; latent anchors reset')
    output=save_report(args.output,images,masks,payload['input_files'],pairs,payload['feature_frontend'],
                       init,initial,result,a,history)
    row=summarize(output,dict(label='contact_cubic',camera_coefficient=args.camera_coefficient,point_tolerance_pixels=1.),time.perf_counter()-start)
    row.update(host=platform.node(),contact_audit=model.audit(state['X']),contact_mu_schedule=[1.,.1,.01],
               contact_cost_px_squared=result['stats']['contact_cost']*a['width']**2,
               stages=stages,source=str(args.source.resolve()),shape_epsilon=.001,
               actual_method='CCD-aware Adam-preconditioned Armijo descent, IPC barrier, SO(3); not DCA')
    (args.output/'contact_audit.json').write_text(json.dumps(jsonable(row),indent=2),encoding='utf-8')
    print(json.dumps(jsonable({k:v for k,v in row.items() if k!='stages'})),flush=True)


if __name__=='__main__':main()
