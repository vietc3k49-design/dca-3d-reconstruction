"""Spec: C2 interior cubic interpolation of the same sampled distance field.

Keep exact integer-pixel samples after spline prefiltering. Return analytic
normalized-coordinate derivatives. Extend outside the image by distance to
its rectangle, so geometry cannot escape the field-of-view penalty.
"""
import numpy as np
from scipy.ndimage import spline_filter


def field_coefficients(fields):
    return np.array([spline_filter(f,order=3,mode='mirror') for f in fields])


def weights(t):
    value=np.stack([(1-t)**3,3*t**3-6*t**2+4,-3*t**3+3*t**2+3*t+1,t**3],axis=-1)/6
    derivative=np.stack([-.5*(1-t)**2,1.5*t*t-2*t,-1.5*t*t+t+.5,.5*t*t],axis=-1)
    return value,derivative


def mirror(indices,size):
    period=2*(size-1);q=indices%period
    return np.where(q<size,q,period-q)


def sample_cubic(coefficients,uv,width):
    K,H,W=coefficients.shape;xy=uv*width+np.array([W/2-.5,H/2-.5])
    upper=np.array([W-1.000001,H-1.000001]);clamped=np.clip(xy,0,upper)
    base=np.floor(clamped).astype(int);fraction=clamped-base
    wx,dx=weights(fraction[...,0]);wy,dy=weights(fraction[...,1]);shift=np.arange(-1,3)
    ix=mirror(base[...,0,None]+shift,W);iy=mirror(base[...,1,None]+shift,H)
    kk=np.arange(K)[:,None,None,None]
    values=coefficients[kk,iy[...,None],ix[...,None,:]]
    value=np.einsum('kmij,kmi,kmj->km',values,wy,wx)
    gx=np.einsum('kmij,kmi,kmj->km',values,wy,dx)*width
    gy=np.einsum('kmij,kmi,kmj->km',values,dy,wx)*width
    gradient=np.stack([gx,gy],axis=-1)*((xy>=0)&(xy<=upper))
    offset=xy-clamped;extra=np.linalg.norm(offset,axis=-1)
    value+=extra/width;gradient+=offset/np.maximum(extra[...,None],1e-15)
    return value,gradient
