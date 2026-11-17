"""Observation uncertainty changes residual cost, never the observation list."""
import numpy as np
from reconstruction.point_terms import reprojection_term


def main():
    width=384
    r=np.array([[.2,.3],[2.,.4],[0.,0.],[-3.,1.]])/width
    squared,grad,stats=reprojection_term(r,width,0.)
    assert squared==np.mean(np.sum(r*r,axis=1))
    assert np.array_equal(grad,2*r/len(r))
    value,grad,stats=reprojection_term(r,width,1.)
    assert np.all(grad[[0,2]]==0) and np.linalg.norm(grad[1])>0
    assert stats['observations']==4 and stats['outside_tolerance']==2
    worst=0.
    for i in np.ndindex(r.shape):
        plus=r.copy();minus=r.copy();plus[i]+=1e-7;minus[i]-=1e-7
        derivative=(reprojection_term(plus,width,1.)[0]-reprojection_term(minus,width,1.)[0])/2e-7
        worst=max(worst,abs(derivative-grad[i]))
    assert worst<1e-9,worst
    assert reprojection_term(np.empty((0,2)),width,1.)[0]==0
    print(f'PASS exact squared-loss compatibility, uncertainty gradients, all observations retained; error {worst:.3g}')


if __name__=='__main__':main()
