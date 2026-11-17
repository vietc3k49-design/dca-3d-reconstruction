"""Neutral mesh construction and generic mesh operations (no object templates)."""
import numpy as np
from scipy.spatial import ConvexHull


def sphere(subdivisions=1):
    t=(1+np.sqrt(5))/2
    vertices=np.array([[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],
                       [0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],
                       [t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]],float)
    vertices/=np.linalg.norm(vertices,axis=1,keepdims=True)
    faces=ConvexHull(vertices).simplices
    for i,f in enumerate(faces):
        if np.cross(vertices[f[1]]-vertices[f[0]],vertices[f[2]]-vertices[f[0]])@vertices[f].mean(0)<0:
            faces[i]=f[::-1]
    for _ in range(subdivisions):
        vertices,faces=subdivide(vertices,faces)
        vertices/=np.linalg.norm(vertices,axis=1,keepdims=True)
    return vertices,faces


def subdivide(vertices,faces):
    result=vertices.tolist();cache={};new_faces=[]
    def middle(i,j):
        key=tuple(sorted((int(i),int(j))))
        if key not in cache:
            cache[key]=len(result);result.append(((vertices[i]+vertices[j])/2).tolist())
        return cache[key]
    for a,b,c in faces:
        ab,bc,ca=middle(a,b),middle(b,c),middle(c,a)
        new_faces.extend([[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]])
    return np.array(result),np.array(new_faces,int)


def laplacian(vertex_count,faces):
    adjacency=np.zeros((vertex_count,vertex_count))
    for i,j in [(0,1),(1,2),(2,0)]:
        adjacency[faces[:,i],faces[:,j]]=1;adjacency[faces[:,j],faces[:,i]]=1
    return np.eye(vertex_count)-adjacency/np.maximum(adjacency.sum(1,keepdims=True),1)


def adjacent_folds(faces):
    edges={};records=[]
    for fi,face in enumerate(faces):
        for j in range(3):
            key=tuple(sorted((int(face[j]),int(face[(j+1)%3]))))
            if key in edges:
                other,opposite=edges[key]
                records.extend([(fi,opposite,other),(other,int(face[(j+2)%3]),fi)])
            else:edges[key]=(fi,int(face[(j+2)%3]))
    return np.array(records,int)


def ray_anchors(vertices,faces,C,g,f,pp,pixels):
    """Attach image tracks to first-hit mesh surface using barycentric weights.

    A missed ray is attached to the closest projected FRONT-facing edge. This
    is initialization only; matched-point and first-hit costs then refine it.
    """
    q=(vertices-C)@g.T;projected=pp+f*q[:,:2]/q[:,2,None]
    triangles=q[faces];e1=triangles[:,1]-triangles[:,0];e2=triangles[:,2]-triangles[:,0]
    face_ids=[];weights=[]
    for pixel in pixels:
        D=np.r_[(pixel-pp)/f,1.]
        h=np.cross(D,e2);det=np.sum(e1*h,axis=1);inv=np.zeros_like(det)
        valid=np.abs(det)>1e-12;inv[valid]=1/det[valid]
        s=-triangles[:,0];u=inv*np.sum(s*h,axis=1);cross=np.cross(s,e1)
        v=inv*np.sum(D*cross,axis=1);t=inv*np.sum(e2*cross,axis=1)
        ok=valid&(u>=0)&(v>=0)&(u+v<=1)&(t>0)
        if ok.any():
            face=int(np.argmin(np.where(ok,t,np.inf)));bary=np.array([1-u[face]-v[face],u[face],v[face]])
        else:
            tri=projected[faces];start=tri;end=np.roll(tri,-1,axis=1);edge=end-start
            ts=np.clip(np.sum((pixel-start)*edge,axis=2)/np.maximum(np.sum(edge**2,axis=2),1e-14),0,1)
            closest=start+ts[...,None]*edge;d2=np.sum((closest-pixel)**2,axis=2)
            front=np.sum(np.cross(e1,e2)*triangles.mean(1),axis=1)<0
            d2[~front]=np.inf
            face,corner=np.unravel_index(d2.argmin(),d2.shape)
            screen=np.zeros(3);screen[corner]=1-ts[face,corner];screen[(corner+1)%3]=ts[face,corner]
            bary=screen/q[faces[face],2];bary/=bary.sum()
        face_ids.append(face);weights.append(bary)
    return np.array(face_ids,int),np.array(weights).reshape(-1,3)
