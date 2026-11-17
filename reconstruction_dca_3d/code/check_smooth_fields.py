"""Cubic field interpolation: exact samples, analytic gradient, no pixel-grid kink."""
import numpy as np
from reconstruction.fields import field_coefficients,sample_cubic


def main():
    W=32;y,x=np.mgrid[:W,:W];field=(np.sqrt((x-12.)**2+(y-13.)**2)/W)[None]
    coefficients=field_coefficients(field)
    uv=(np.array([[[10.,14.],[11.,14.],[12.,14.]]])+0.5-W/2)/W
    value,gradient=sample_cubic(coefficients,uv,W)
    assert np.max(np.abs(value-field[0,14,[10,11,12]]))<1e-12
    worst=0.
    for index in np.ndindex(uv.shape):
        plus=uv.copy();minus=uv.copy();plus[index]+=1e-7;minus[index]-=1e-7
        numerical=(sample_cubic(coefficients,plus,W)[0]-sample_cubic(coefficients,minus,W)[0])/2e-7
        worst=max(worst,abs(numerical[index[:2]]-gradient[index]))
    assert worst<1e-7,worst
    at=uv[:,1:2].copy();left=at.copy();right=at.copy();left[...,0]-=1e-8;right[...,0]+=1e-8
    assert np.max(abs(sample_cubic(coefficients,left,W)[1]-sample_cubic(coefficients,right,W)[1]))<1e-5
    off=np.array([[[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]]])
    value,g=sample_cubic(coefficients,off,W)
    assert np.all(value>0) and np.all(np.linalg.norm(g,axis=-1)>0)
    print(f'PASS cubic sample reconstruction, exact field gradients ({worst:.3g}), continuous pixel-grid gradient, offscreen penalty')


if __name__=='__main__':main()
