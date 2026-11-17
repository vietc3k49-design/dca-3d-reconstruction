"""Read a saved raw-image fitting problem without simulator or class data."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .engine import arrays


def read_case(source):
    source=Path(source);p=json.loads((source/'result.json').read_text(encoding='utf-8'))
    images=np.array([np.array(Image.open(source/f'input_{i+1:02d}.png')) for i in range(len(p['input_files']))])
    masks=np.array([np.array(Image.open(source/f'mask_{i+1:02d}.png'))>127 for i in range(len(images))])
    state={k:np.array(v,float) for k,v in p['state'].items()}
    initial={k:np.array(v,float) for k,v in p['initial_state'].items()}
    obs=p['observations'];ci=np.array(obs['camera_ids'],int);ids=np.array(obs['track_ids'],int)
    uv=np.array(obs['pixels_normalized'],float).reshape(-1,2);pairs=[];width=images.shape[1]
    for view in range(1,len(images)):
        ref={i:q for i,q in zip(ids[ci==0],uv[ci==0])}
        values=[(ref[i],q) for i,q in zip(ids[ci==view],uv[ci==view]) if i in ref]
        if values:
            aa,bb=np.array(values).transpose(1,0,2)
            pairs.append(dict(view_a=0,view_b=view,inliers=len(aa),pixels_a=(aa+.5)*width,pixels_b=(bb+.5)*width))
    a=arrays(images,masks,pairs,state,np.array(p['faces'],int))
    a.update(ci=ci,pi=ids,uv=uv,anchor_vertices=np.array(obs['anchor_vertices'],int).reshape(-1,3),
             anchor_bary=np.array(obs['anchor_bary'],float).reshape(-1,3))
    for key in ('prior_weight','smooth_weight','compactness_weight','calibration_weight','principal_sigma','log_focal_sigma'):
        if p['configuration'].get(key) is not None:a[key]=p['configuration'][key]
    for key in ('shape_prior_units','symmetry_epsilon','smoothness_epsilon','optimization_scale',
                'point_tolerance_pixels','camera_prior_coefficient_px_squared','reject_transverse_intersections'):
        if p['configuration'].get(key) is not None:a[key]=p['configuration'][key]
    if p.get('axis_constraint'):
        c=p['axis_constraint'];association=c['association']
        a['axis_constraint']=dict(axis=np.array(c['axis']),order=c['order'],association={
            key:np.array(value,int if key in ('source_faces','target_vertices') else float)
            for key,value in association.items()})
    solver=p.get('visibility_constraint_solver') or {}
    a['visibility_penalty']=solver.get('penalty',0.)
    if solver.get('dual') is not None:a['visibility_dual']=np.array(solver['dual'])
    a['visibility_feasibility_tolerance']=solver.get('numerical_tolerance',0.)
    a['field_interpolation']=p['configuration'].get('field_interpolation','bilinear')
    if a['field_interpolation']=='cubic':
        from .fields import field_coefficients
        a['field_coefficients']=field_coefficients(a['fields'])
    if p.get('contact_configuration'):
        from .contact import ContactSystem
        c=p['contact_configuration'];a['contact_model']=ContactSystem(np.array(c['reference_vertices']),a['faces'],c['dhat'])
        a['contact_weight']=p['configuration']['contact_weight']
    return p,images,masks,pairs,state,initial,a
