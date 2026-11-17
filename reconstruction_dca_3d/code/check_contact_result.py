"""Replay and independently audit the real contact-aware result; never assume truth."""
import argparse
import json
from pathlib import Path
import numpy as np
from reconstruction.records import read_case
from reconstruction.engine import objective,first_hit_constraints,valid


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path);args=p.parse_args()
    root=args.root;folder=args.source or root/'generic_can_axis_symmetry_soft_xeon'
    source=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    payload,_,_,_,state,neutral,a=read_case(root/'results')
    for key in ('camera_ids','track_ids','pixels_normalized'):
        assert np.array_equal(payload['observations'][key],source['observations'][key]),key
    assert np.array_equal(payload['initial_state']['X'],source['initial_state']['X'])
    assert abs(a['prior_weight']*a['width']**2-.001)<1e-15 and a['point_tolerance_pixels']==1
    assert a['field_interpolation']=='cubic' and abs(a['contact_weight']*a['width']**2-.01)<1e-15
    row=json.loads((root/'results/contact_audit.json').read_text(encoding='utf-8'))
    assert abs(a['calibration_weight']*a['width']**2-row['camera_coefficient'])<1e-15
    assert a['compactness_weight']==0 and valid(state,a)
    _,_,stats=objective(state,a)
    for key in ('total','image_cost','prior_cost','intrinsic_cost','contact_cost','point_rmse_px'):
        assert abs(stats[key]-payload['objective'][key])<1e-10,key
    assert a['contact_model'].audit(state['X'])['intersecting_pairs']==0
    assert stats['visibility_violations']==0 and np.max(first_hit_constraints(state,a))<=1e-6
    F=a['faces'];E=np.sort(np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1)
    edges,counts=np.unique(E,axis=0,return_counts=True)
    assert np.all(counts==2) and len(state['X'])-len(edges)+len(F)==2
    assert payload['geometry_error_3d'] is None and not payload['metric_scale_known']
    for stage in payload['optimization']['contact_stages']:
        assert stage['audit']['intersecting_pairs']==0
        for block in stage['optimization']['blocks']:assert np.all(np.diff(block['accepted_losses'])<=0)
    report=dict(observations=len(a['ci']),tracks=len(a['anchor_vertices']),field='cubic',shape_epsilon=.001,
                raw_point_rms_px=stats['point_rmse_px'],mask_IoU=[r['mask_IoU'] for r in payload['masks_after']],
                exact_predicate_crossings=0,first_hit_violations=0,closed_mesh=True,metric_scale_known=False,
                physical_accuracy_known=False,objective_replay_error=abs(stats['total']-payload['objective']['total']))
    (root/'results/verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS REAL CONTACT FIT',json.dumps(report))


if __name__=='__main__':main()
