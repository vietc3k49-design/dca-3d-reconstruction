"""Generic plain-background segmentation and image-feature correspondences.

No object names, CAD models, geometric landmarks, dimensions or truth inputs.
Automatic masks assume a nearly uniform background; explicit masks may replace
them for general photographs.
"""
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from scipy.ndimage import binary_fill_holes,binary_closing,label
import cv2


def largest_component(mask):
    labels,n=label(mask)
    if n==0:raise ValueError('No foreground found; supply an explicit object mask.')
    counts=np.bincount(labels.ravel());counts[0]=0
    return labels==counts.argmax()


def load_images(paths,size=320,mask_paths=None):
    images=[];masks=[];transforms=[]
    for i,path in enumerate(paths):
        image=ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        rgb=np.array(image)
        corners=np.vstack([rgb[:8,:8].reshape(-1,3),rgb[:8,-8:].reshape(-1,3),
                           rgb[-8:,:8].reshape(-1,3),rgb[-8:,-8:].reshape(-1,3)])
        background=np.median(corners,axis=0)
        if mask_paths:
            mask=np.array(Image.open(mask_paths[i]).convert('L'))>127
            if mask.shape!=rgb.shape[:2]:raise ValueError('Mask and photo sizes differ.')
        else:
            if np.percentile(np.linalg.norm(corners-background,axis=1),90)>25:
                raise ValueError('Background is not uniform; supply foreground masks.')
            mask=np.max(np.abs(rgb.astype(float)-background),axis=2)>18
            mask=largest_component(binary_closing(mask,iterations=1))
            mask=binary_fill_holes(mask)
        scale=(size-20)/max(image.size)
        wh=np.maximum(np.round(np.array(image.size)*scale).astype(int),1)
        small=image.resize(tuple(wh),Image.Resampling.LANCZOS)
        sm=Image.fromarray(mask).resize(tuple(wh),Image.Resampling.NEAREST)
        offset=(np.array([size,size])-wh)//2
        canvas=Image.new('RGB',(size,size),tuple(np.uint8(background)))
        canvas.paste(small,tuple(offset))
        mm=np.zeros((size,size),bool);mm[offset[1]:offset[1]+wh[1],offset[0]:offset[0]+wh[0]]=np.array(sm)>0
        images.append(np.array(canvas));masks.append(mm)
        transforms.append(dict(path=str(Path(path).resolve()),original_size=image.size,
                               resize_scale=float(scale),offset_px=offset.tolist(),background_rgb=background.tolist()))
    return np.array(images),np.array(masks),transforms


def feature_pairs(images,masks):
    cv2.setRNGSeed(592)
    sift=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.02)
    keypoints=[];descriptors=[]
    for image,mask in zip(images,masks):
        kp,desc=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),np.uint8(mask)*255)
        if desc is not None:
            desc=np.sqrt(desc/(np.sum(desc,axis=1,keepdims=True)+1e-12)).astype(np.float32)
        keypoints.append(kp);descriptors.append(desc)
    pairs=[];matcher=cv2.BFMatcher(cv2.NORM_L2)
    for v in range(1,len(images)):
        da,db=descriptors[0],descriptors[v]
        if da is None or db is None:continue
        forward=matcher.knnMatch(da,db,k=2);backward=matcher.knnMatch(db,da,k=2)
        good_b={x.queryIdx:x.trainIdx for x,y in backward if x.distance<.76*y.distance}
        good=[x for x,y in forward if x.distance<.76*y.distance and good_b.get(x.trainIdx)==x.queryIdx]
        if len(good)<8:continue
        a=np.array([keypoints[0][m.queryIdx].pt for m in good]);b=np.array([keypoints[v][m.trainIdx].pt for m in good])
        F,inliers=cv2.findFundamentalMat(a,b,cv2.USAC_MAGSAC,1.,.999,10000)
        if F is None or F.shape!=(3,3) or inliers is None:continue
        keep=inliers.ravel().astype(bool)
        # SIFT can report multiple orientations at the SAME image position.
        # Keep one observation per physical pixel pair, independent of category.
        unique=[];seen=set()
        for index in np.flatnonzero(keep):
            key=tuple(np.round(np.r_[a[index],b[index]],2))
            if key not in seen:seen.add(key);unique.append(index)
        selected=np.array(unique,int)
        if len(selected)<8:continue
        H,hi=cv2.findHomography(a,b,cv2.RANSAC,1.5)
        pairs.append(dict(view_a=0,view_b=v,pixels_a=a[selected],pixels_b=b[selected],F_pixel=F,
                          raw_matches=len(good),inliers=len(selected),ransac_inlier_detections=int(keep.sum()),
                          homography_inliers=int(hi.sum()) if hi is not None else 0))
    return pairs,dict(keypoints_per_image=[len(k) for k in keypoints])
