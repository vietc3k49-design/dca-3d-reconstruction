"""Spec: reuse IPC barrier/CCD and independent libigl intersection predicates.

Barrier geometry uses current RMS-radius normalization and its full analytic
pullback. Fixed rest geometry defines IPC edge mollifiers; all parameters are
recorded. CCD operates on the actual unnormalized vertex trajectory.
"""
import numpy as np
import ipctk
from igl.predicates import find_self_intersections


def normalize(X):
    centred=X-X.mean(0);radius=float(np.sqrt(np.mean(np.sum(centred**2,axis=1))))
    if radius<=1e-12:raise ValueError('Collapsed contact geometry')
    return centred/radius,radius


class ContactSystem:
    def __init__(self,reference,faces,dhat=.02):
        ipctk.set_num_threads(1)
        self.reference=np.array(reference,float);self.faces=np.array(faces,np.int64);self.dhat=float(dhat)
        edges=np.unique(np.sort(np.concatenate([self.faces[:,[0,1]],self.faces[:,[1,2]],self.faces[:,[2,0]]]),axis=1),axis=0)
        rest,_=normalize(self.reference)
        self.mesh=ipctk.CollisionMesh(np.asfortranarray(rest),np.asfortranarray(edges,dtype=np.int32),
                                     np.asfortranarray(self.faces,dtype=np.int32))
        self.potential=ipctk.BarrierPotential(self.dhat,1/self.dhat**4)
        if self.audit(self.reference)['intersecting_pairs']:raise ValueError('Contact initialization intersects')

    def term(self,X):
        Y,radius=normalize(X);Y=np.asfortranarray(Y)
        collisions=ipctk.NormalCollisions();collisions.use_area_weighting=False
        collisions.build(self.mesh,Y,self.dhat)
        value=float(self.potential(collisions,self.mesh,Y))
        gradient=np.asarray(self.potential.gradient(collisions,self.mesh,Y)).reshape(X.shape)
        grad=(gradient-gradient.mean(0)-Y*np.sum(gradient*Y)/len(X))/radius
        count=sum(len(getattr(collisions,key)) for key in ('vv_collisions','ev_collisions','ee_collisions','fv_collisions'))
        return value,grad,dict(contact_pairs=count,contact_barrier_raw=value)

    def hessian_data(self,X):
        """Return PSD-projected sparse IPC barrier Hessian wrt normalized Y and normalization radius."""
        Y,radius=normalize(X);Y=np.asfortranarray(Y)
        collisions=ipctk.NormalCollisions();collisions.use_area_weighting=False
        collisions.build(self.mesh,Y,self.dhat)
        H=self.potential.hessian(collisions,self.mesh,Y,ipctk.PSDProjectionMethod.CLAMP)
        return H,Y,radius

    def safe_fraction(self,X,Y):
        fraction=float(ipctk.compute_collision_free_stepsize(self.mesh,np.asfortranarray(X),np.asfortranarray(Y)))
        return fraction*.99 if fraction<1. else 1.

    def audit(self,X,first_only=False):
        found,pairs,coplanar=find_self_intersections(np.ascontiguousarray(X,dtype=float),
                                                    np.ascontiguousarray(self.faces),first_only)
        return dict(intersecting_pairs=len(pairs),coplanar_pairs=int(np.sum(coplanar)),
                    face_pairs=np.asarray(pairs).tolist(),found=bool(found),backend='libigl exact predicates')

    def configuration(self):
        return dict(backend='ipctk 1.6.1; libigl 2.6.3',dhat=self.dhat,
                    reference_vertices=self.reference.tolist(),normalization='current RMS radius with analytic pullback',
                    stiffness=1/self.dhat**4,area_weighting=False)
