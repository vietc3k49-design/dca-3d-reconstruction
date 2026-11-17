"""Spec: epsilon is small relative to ONE PIXEL squared, not raw image width.

Engine image residuals are divided by W. Dividing dimensionless shape-prior
weights by W^2 gives the equivalent objective D_pixels + epsilon*R_shape.
This is an idempotent unit conversion, not object-specific tuning.
"""


def pixel_normalized_shape_priors(arrays):
    result=dict(arrays)
    if result.get('shape_prior_units')=='one_pixel_squared':return result
    width=result['width']
    result['symmetry_epsilon']=result['prior_weight']
    result['smoothness_epsilon']=result['smooth_weight']
    result['prior_weight']/=width**2
    result['smooth_weight']/=width**2
    result['shape_prior_units']='one_pixel_squared'
    result['optimization_scale']=width**2
    return result
