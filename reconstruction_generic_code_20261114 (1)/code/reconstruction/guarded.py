"""Spec: restart from neutral cached image estimates, preserving ALL measurements.

Reference rays reinitialize latent surface anchors; observed UVs and track
IDs are unchanged. No physical camera, dimensions or class template is used.
"""
import copy
import numpy as np
from .mesh import ray_anchors
from .weak_priors import pixel_normalized_shape_priors
from .axis_geometry import update_surface_correspondences
from .fields import field_coefficients
from .engine import valid


def make_guarded_problem(neutral,arrays,interpolation='cubic'):
    if interpolation not in ('bilinear','cubic'):raise ValueError('Unknown field interpolation')
    s=copy.deepcopy(neutral);b=pixel_normalized_shape_priors(copy.deepcopy(arrays))
    reference={i:uv for i,uv in zip(b['pi'][b['ci']==0],b['uv'][b['ci']==0])}
    pixels=np.array([reference[i] for i in range(len(b['anchor_vertices']))])
    ids,bary=ray_anchors(s['X'],b['faces'],s['C'][0],s['g'][0],np.exp(s['lf'][0]),s['pp'][0],pixels)
    b.update(anchor_vertices=b['faces'][ids],anchor_bary=bary,
        gauge_camera_distance=float(np.linalg.norm(s['C'][0])),compactness_weight=0.,calibration_weight=1/b['width']**2,
        point_tolerance_pixels=1.,reject_transverse_intersections=True,field_interpolation=interpolation,
        visibility_penalty=0.,visibility_dual=np.zeros(len(b['ci'])),visibility_feasibility_tolerance=1e-6)
    b.pop('field_coefficients',None)
    if interpolation=='cubic':b['field_coefficients']=field_coefficients(b['fields'])
    axis=np.array(b['axis_constraint']['axis'])
    b['axis_constraint']=dict(axis=axis,order=8,association=update_surface_correspondences(s['X'],b['faces'],axis,8))
    if not valid(s,b):raise ValueError('Neutral initialization is invalid; no guarded run launched')
    return s,b,dict(label='guard_'+interpolation,axis=axis.tolist())
