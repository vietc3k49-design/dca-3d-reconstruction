"""SO(3) updates shared by all object categories."""
import numpy as np


def exp_rotation(w):
    w=np.asarray(w,float);theta=np.linalg.norm(w,axis=-1)
    k=np.zeros(w.shape[:-1]+(3,3))
    k[...,0,1],k[...,0,2]=-w[...,2],w[...,1]
    k[...,1,0],k[...,1,2]=w[...,2],-w[...,0]
    k[...,2,0],k[...,2,1]=-w[...,1],w[...,0]
    return np.eye(3)+np.sinc(theta/np.pi)[...,None,None]*k+(.5*np.sinc(theta/(2*np.pi))**2)[...,None,None]*(k@k)


def look_at(C,target):
    forward=np.asarray(target,float)-C;forward/=np.linalg.norm(forward)
    up=np.array([0.,1.,0.])
    if abs(up@forward)>.999:up=np.array([0.,0.,1.])
    right=np.cross(forward,up);right/=np.linalg.norm(right)
    return np.stack([right,np.cross(forward,right),forward])
