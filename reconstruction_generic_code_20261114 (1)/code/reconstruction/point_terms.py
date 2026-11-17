"""Spec: optional observation disk; preserve squared loss at zero tolerance.

Inputs are residuals normalized by image width and a nonnegative pixel radius.
Returns mean squared distance outside the disk, its analytic gradient, and RAW
pixel errors. This changes observation uncertainty, never drops observations.
"""
import numpy as np


def reprojection_term(residual,width,tolerance_pixels=0.):
    if width<=0 or tolerance_pixels<0:raise ValueError('Invalid width or tolerance.')
    residual=np.asarray(residual,float)
    if residual.ndim!=2 or residual.shape[1]!=2:raise ValueError('Expected N x 2 residuals.')
    n=len(residual)
    if not n:return 0.,np.zeros_like(residual),dict(observations=0,outside_tolerance=0,rms_px=0.,max_px=0.)
    squared=np.sum(residual**2,axis=1);distance=np.sqrt(squared)
    if tolerance_pixels==0:
        value=float(np.mean(squared));gradient=2*residual/n
    else:
        outside=np.maximum(distance-tolerance_pixels/width,0.)
        value=float(np.mean(outside**2))
        gradient=2*residual*(outside/np.maximum(distance,1e-15))[:,None]/n
    stats=dict(observations=n,outside_tolerance=int(np.sum(distance>tolerance_pixels/width)),
               rms_px=float(np.sqrt(np.mean(squared))*width),max_px=float(np.max(distance)*width))
    return value,gradient,stats
