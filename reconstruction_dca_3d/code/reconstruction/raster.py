"""Same opaque z-buffer renderer, with arbitrary intrinsics offsets."""
import numpy as np
from PIL import Image
from .geometry import project

def render_mesh(X,faces,C,g,f,principal_point,size=(1024,1024)):
    """Small perspective-correct z-buffer renderer, opaque triangles."""
    width,height=size
    image=np.full((height,width,3),[235.,239.,241.])
    zbuffer=np.full((height,width),np.inf)
    uv,z=project(X,C,g,f,size)
    uv+=np.asarray(principal_point)-np.asarray(size)/2
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
