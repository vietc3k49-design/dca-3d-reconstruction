"""Spec: same fixed axis/seed/observations, a predeclared 2x3 objective study.

Only camera-prior coefficient and point uncertainty vary. Shape epsilon stays
0.001 in pixel-squared units. No compactness, object type, size or template.
All cases are reported; differing objectives cannot rank physical accuracy.
"""
import copy
import numpy as np
from .weak_priors import pixel_normalized_shape_priors


def case_grid():
    return [dict(label=f'camera_{name}_tolerance_{radius}',camera_coefficient=weight,
                 point_tolerance_pixels=float(radius))
            for radius in (0,1) for name,weight in [('1',1.),('0p001',.001),('0',0.)]]


def configure(arrays,case):
    b=pixel_normalized_shape_priors(copy.deepcopy(arrays))
    if b.get('axis_constraint') is None:raise ValueError('A saved image-proposed axis is required.')
    if abs(b['symmetry_epsilon']-.001)>1e-15:raise ValueError('Study requires shape epsilon=0.001.')
    b.update(calibration_weight=case['camera_coefficient']/b['width']**2,
             camera_prior_coefficient_px_squared=case['camera_coefficient'],
             point_tolerance_pixels=case['point_tolerance_pixels'],compactness_weight=0.)
    return b


def summarize(payload,case,seconds):
    s=payload['objective'];width=payload['configuration']['optimization_scale']**.5
    X=np.array(payload['display_vertices']);extent=np.ptp(X,axis=0)
    state=payload['state'];pp=np.array(state['pp']);f=np.exp(state['lf'])*width
    blocks=payload['optimization']['blocks'];feasible=payload['optimization']['feasibility_blocks']
    return dict(**case,seconds=seconds,feature_rms_px=s['point_rmse_px'],feature_max_px=s['point_max_error_px'],
        mask_IoU=[m['mask_IoU'] for m in payload['masks_after']],
        observations=s['point_observations'],tracks=payload['features']['track_count'],
        point_outside_tolerance=s['point_outside_tolerance'],first_hit_violations=s['visibility_violations'],
        first_hit_max_residual=s['visibility_max_residual'],depth_to_width=float(extent[1]/extent[0]),
        height_to_width=float(extent[2]/extent[0]),transverse_ratio=s['transverse_ratio'],
        c8_rms=float(np.sqrt(s['axis_surface_group_error'])),
        local_concavity=s['local_concavity'],smoothness=s['smoothness'],
        image_cost_px_squared=s['image_cost']*width**2,
        point_cost_px_squared=s['point_cost']*width**2,
        shape_cost_px_squared=s['prior_cost']*width**2,
        camera_cost_px_squared=s['intrinsic_cost']*width**2,
        focal_px=f.tolist(),principal_point_px=((pp+.5)*width).tolist(),
        accepted_steps=sum(b['accepted_steps'] for b in blocks)+sum(b['optimization']['accepted_steps'] for b in feasible),
        block_statuses=[b.get('monotone') for b in blocks],feasibility_blocks=len(feasible),
        physical_accuracy_known=False,metric_scale_known=False)
