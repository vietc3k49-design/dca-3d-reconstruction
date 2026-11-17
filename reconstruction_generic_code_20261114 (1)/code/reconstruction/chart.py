"""Spec: ambient mesh/camera coordinates and exact local SO(3) chart gradients.

Camera zero's rotation stays fixed. Translation/scale gauge is normalized
only after solving; the objective is invariant to this joint similarity.
No object category, symmetry parameterization or vertex restrictions are added.
"""
import copy
import numpy as np
from .rotations import exp_rotation


def left_jacobian(w):
    w=np.asarray(w,float);t2=np.sum(w*w,axis=-1);t=np.sqrt(t2)
    K=np.zeros(w.shape[:-1]+(3,3))
    K[...,0,1],K[...,0,2]=-w[...,2],w[...,1]
    K[...,1,0],K[...,1,2]=w[...,2],-w[...,0]
    K[...,2,0],K[...,2,1]=-w[...,1],w[...,0]
    small=t<1e-4;safe=np.maximum(t,1e-15)
    alpha=np.where(small,.5-t2/24+t2*t2/720,(1-np.cos(t))/safe**2)
    beta=np.where(small,1/6-t2/120+t2*t2/5040,(t-np.sin(t))/safe**3)
    return np.eye(3)+alpha[...,None,None]*K+beta[...,None,None]*(K@K)


class Chart:
    def __init__(self,state):
        self.reference=copy.deepcopy(state);self.slices={};self.shapes={};parts=[];bounds=[];start=0
        for key in ('X','C','lf','pp','g','h'):
            value=(np.zeros((len(state['g'])-1,3)) if key=='g' else
                   np.zeros(3) if key=='h' else state[key])
            size=value.size;self.slices[key]=slice(start,start+size);self.shapes[key]=value.shape;start+=size
            parts.append(value.ravel())
            bound=((np.log(.4),np.log(12.)) if key=='lf' else (-.8,.8) if key=='pp' else (None,None))
            bounds.extend([bound]*size)
        self.initial=np.concatenate(parts);self.bounds=bounds

    def decode(self,vector):
        result=copy.deepcopy(self.reference)
        for key,sl in self.slices.items():
            value=vector[sl].reshape(self.shapes[key])
            if key=='g':result[key][1:]=exp_rotation(value)@self.reference[key][1:]
            elif key=='h':result[key]=exp_rotation(value)@self.reference[key]
            else:result[key]=value.copy()
        return result

    def pullback(self,vector,gradient):
        result=[]
        for key,sl in self.slices.items():
            g=gradient[key][1:] if key=='g' else gradient[key]
            if key in ('g','h'):
                w=vector[sl].reshape(self.shapes[key]);J=left_jacobian(w)
                g=np.einsum('...ji,...j->...i',J,g)
            result.append(g.ravel())
        return np.concatenate(result)
