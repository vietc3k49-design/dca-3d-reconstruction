"""General projection, rendering, mesh I/O and two-view geometry.

Extracted from the original implementation, with no object template.
"""
import json
import struct
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar

def project(X,C,g,f,size):
    q=(X-C)@g.T
    return np.asarray(size)/2+f*q[:,:2]/q[:,2,None],q[:,2]


def render(X,faces,C,g,f,size=(1024,1024)):
    """Small perspective-correct z-buffer renderer, opaque triangles."""
    width,height=size
    image=np.full((height,width,3),[235.,239.,241.])
    zbuffer=np.full((height,width),np.inf)
    uv,z=project(X,C,g,f,size)
    light=np.array([-.35,-.45,1.]); light/=np.linalg.norm(light)
    for face in faces:
        if np.any(z[face]<=0): continue
        p=uv[face]; v=X[face]
        normal=np.cross(v[1]-v[0],v[2]-v[0]); norm=np.linalg.norm(normal)
        if norm<1e-12: continue
        normal/=norm
        color=np.array([173.,190.,202.])*(.40+.60*max(normal@light,0))
        xmin=max(0,int(np.floor(p[:,0].min()))); xmax=min(width-1,int(np.ceil(p[:,0].max())))
        ymin=max(0,int(np.floor(p[:,1].min()))); ymax=min(height-1,int(np.ceil(p[:,1].max())))
        if xmin>xmax or ymin>ymax: continue
        yy,xx=np.mgrid[ymin:ymax+1,xmin:xmax+1]
        xx=xx+.5; yy=yy+.5
        denominator=(p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
        if abs(denominator)<1e-10: continue
        a=((p[1,1]-p[2,1])*(xx-p[2,0])+(p[2,0]-p[1,0])*(yy-p[2,1]))/denominator
        b=((p[2,1]-p[0,1])*(xx-p[2,0])+(p[0,0]-p[2,0])*(yy-p[2,1]))/denominator
        c=1-a-b
        inverse=a/z[face[0]]+b/z[face[1]]+c/z[face[2]]
        depth=1/np.maximum(inverse,1e-12)
        old=zbuffer[ymin:ymax+1,xmin:xmax+1]
        use=(a>=-1e-9)&(b>=-1e-9)&(c>=-1e-9)&(depth<old)
        old[use]=depth[use]
        image[ymin:ymax+1,xmin:xmax+1][use]=color
    return Image.fromarray(np.uint8(np.clip(image,0,255)))


def export_mesh(path,X,faces):
    path=Path(path)
    obj=['# Units: relative; no metric scale was supplied.',
         'mtllib surface.mtl','usemtl surface']
    obj+=['v '+' '.join(f'{x:.10f}' for x in point) for point in X]
    obj+=['f '+' '.join(str(int(v)+1) for v in face) for face in faces]
    path.with_suffix('.obj').write_text('\n'.join(obj)+'\n',encoding='ascii')
    (path.parent/'surface.mtl').write_text('newmtl surface\nKa 0.2 0.12 0.03\nKd 0.65 0.70 0.78\nKs 0.3 0.3 0.3\nNs 40\n',encoding='ascii')
    header=['ply','format ascii 1.0',f'element vertex {len(X)}','property float x','property float y','property float z',
            f'element face {len(faces)}','property list uchar int vertex_indices','end_header']
    rows=[' '.join(f'{v:.10f}' for v in p) for p in X]
    rows+=['3 '+' '.join(map(str,face)) for face in faces]
    path.with_suffix('.ply').write_text('\n'.join(header+rows)+'\n',encoding='ascii')
    # Self-contained glTF binary, flat-shaded gold. Root rotates Z-up to Y-up.
    positions=np.asarray(X[faces].reshape(-1,3),dtype='<f4')
    face_normals=np.cross(X[faces[:,1]]-X[faces[:,0]],X[faces[:,2]]-X[faces[:,0]])
    face_normals/=np.maximum(np.linalg.norm(face_normals,axis=1,keepdims=True),1e-15)
    normals=np.asarray(np.repeat(face_normals,3,axis=0),dtype='<f4')
    binary=positions.tobytes()+normals.tobytes()
    gltf=dict(asset=dict(version='2.0',generator='Generic image reconstruction'),
              scene=0,scenes=[dict(nodes=[0])],nodes=[dict(mesh=0,rotation=[-2**-.5,0,0,2**-.5])],
              meshes=[dict(primitives=[dict(attributes=dict(POSITION=0,NORMAL=1),material=0,mode=4)])],
              materials=[dict(name='Surface',doubleSided=True,pbrMetallicRoughness=dict(
                  baseColorFactor=[.65,.70,.78,1],metallicFactor=0.,roughnessFactor=.5))],
              buffers=[dict(byteLength=len(binary))],
              bufferViews=[dict(buffer=0,byteOffset=0,byteLength=positions.nbytes,target=34962),
                           dict(buffer=0,byteOffset=positions.nbytes,byteLength=normals.nbytes,target=34962)],
              accessors=[dict(bufferView=0,componentType=5126,count=len(positions),type='VEC3',
                              min=positions.min(axis=0).tolist(),max=positions.max(axis=0).tolist()),
                         dict(bufferView=1,componentType=5126,count=len(normals),type='VEC3')])
    json_bytes=json.dumps(gltf,separators=(',',':')).encode('utf-8')
    json_bytes+=b' '*((-len(json_bytes))%4)
    binary+=b'\0'*((-len(binary))%4)
    total=12+8+len(json_bytes)+8+len(binary)
    content=(struct.pack('<4sII',b'glTF',2,total)+struct.pack('<I4s',len(json_bytes),b'JSON')+json_bytes
             +struct.pack('<I4s',len(binary),b'BIN\0')+binary)
    path.with_suffix('.glb').write_bytes(content)


def normalize2d(points):
    c=points.mean(axis=0); s=np.sqrt(2)/np.sqrt(np.mean(np.sum((points-c)**2,axis=1)))
    H=np.array([[s,0,-s*c[0]],[0,s,-s*c[1]],[0,0,1.]])
    return (points-c)*s,H


def fundamental(a,b):
    a,Ha=normalize2d(a); b,Hb=normalize2d(b)
    x,y=a.T; u,v=b.T
    design=np.column_stack([u*x,u*y,u,v*x,v*y,v,x,y,np.ones(len(x))])
    _,singular,vt=np.linalg.svd(design)
    F=vt[-1].reshape(3,3); U,S,V=np.linalg.svd(F); S[-1]=0
    F=Hb.T@(U@np.diag(S)@V)@Ha
    return F/np.linalg.norm(F),float(singular[-2]/max(singular[0],1e-15))


def focal_from_F(F):
    def cost(logf):
        f=np.exp(logf); K=np.diag([f,f,1.]); S=np.linalg.svd(K@F@K,compute_uv=False)
        return ((S[0]-S[1])/(S[0]+S[1]))**2
    grid=np.linspace(np.log(.25),np.log(3),100); costs=np.array([cost(t) for t in grid])
    i=int(np.argmin(costs)); lo=grid[max(i-1,0)]; hi=grid[min(i+1,len(grid)-1)]
    solved=minimize_scalar(cost,bounds=(lo,hi),method='bounded')
    return float(np.exp(solved.x)),float(solved.fun)


def triangulate(uv,C,g,f):
    rows=[]
    for pixel,center,rotation in zip(uv,C,g):
        P=np.column_stack([rotation,-rotation@center])
        rows.extend([pixel[0]/f*P[2]-P[0],pixel[1]/f*P[2]-P[1]])
    _,_,vt=np.linalg.svd(rows); p=vt[-1]
    return p[:3]/p[3]


def essential_pose(a,b,f,F):
    K=np.diag([f,f,1.]); U,S,V=np.linalg.svd(K@F@K)
    if np.linalg.det(U)<0: U[:,-1]*=-1
    if np.linalg.det(V)<0: V[-1]*=-1
    W=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    best=None
    for R in (U@W@V,U@W.T@V):
        for t in (U[:,2],-U[:,2]):
            C=-R.T@t
            points=np.array([triangulate([p,q],[np.zeros(3),C],[np.eye(3),R],f) for p,q in zip(a,b)])
            front=(points[:,2]>0)&(((points-C)@R.T)[:,2]>0)
            if best is None or front.sum()>best[0]: best=(int(front.sum()),C,R,points)
    return best

