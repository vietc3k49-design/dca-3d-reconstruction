"""IPC integration tests, including an endpoint-valid but colliding trajectory."""
import numpy as np
from reconstruction.contact import ContactSystem


def main():
    X=np.array([[0.,0.,0.],[1.,0.,0.],[.4,1.,0.],[.15,.1,.02],[.8,.15,.022],[.45,.7,.021]])
    F=np.array([[0,1,2],[3,4,5]],int);model=ContactSystem(X,F,dhat=.05)
    value,gradient,stats=model.term(X);assert value>0
    worst=0.
    for index in np.ndindex(X.shape):
        plus=X.copy();minus=X.copy();plus[index]+=1e-7;minus[index]-=1e-7
        numeric=(model.term(plus)[0]-model.term(minus)[0])/2e-7
        worst=max(worst,abs(numeric-gradient[index]))
    assert worst<1e-4,worst
    assert abs(model.term(3.2*(X+[2.,-1.,.7]))[0]-value)<1e-10
    assert model.audit(X)['intersecting_pairs']==0
    Y=X.copy();Y[3:,2]-=.08
    assert model.audit(Y)['intersecting_pairs']==0
    fraction=model.safe_fraction(X,Y)
    assert 0<fraction<.3,fraction
    assert model.audit(X+fraction*(Y-X))['intersecting_pairs']==0
    coplanar=X.copy();coplanar[3:,2]=0
    assert model.audit(coplanar)['coplanar_pairs']>0
    print(f'PASS IPC normalized barrier gradient ({worst:.3g}), similarity, CCD trajectory and coplanar audit; fraction {fraction:.5f}')


if __name__=='__main__':main()
