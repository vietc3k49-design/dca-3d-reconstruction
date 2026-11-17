"""One object-independent entry point for images -> mesh + unknown cameras.

Same neutral icosphere initialization, feature frontend, shared two-way mask
losses, first-hit kernel, symmetry hypotheses and Adam loop for every input.
Default contact handling uses IPC barrier/CCD with libigl intersection audits.
Default C8 completion uses image-supported fixed axes and a 2D discrepancy
budget. Compactness completion remains available as an alternative.
No product/category switches, object templates, known dimensions or true cameras.
Current limits: closed sphere-topology mesh; plain background or supplied masks;
known correspondences are NOT required, but poor feature matches limit calibration.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from reconstruction.inputs import load_images,feature_pairs
from reconstruction.initialization import initialize,camera_hypotheses
from reconstruction.mesh import subdivide
from reconstruction.engine import arrays,objective,step,valid,queries
from reconstruction.optimizer import optimize
from reconstruction.dca import optimize_dca
from reconstruction.geometry import render,export_mesh
from reconstruction.rotations import look_at
from reconstruction.completion import select_completion
from reconstruction.axis_completion import select_axis_completion
from reconstruction.weak_priors import pixel_normalized_shape_priors
from reconstruction.report_html import build_report_html
from reconstruction.hidden_completion import append_hidden_stage


def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    return x


def metrics(state,a):
    result=[];rendered=[]
    for v in range(a['k']):
        # The retained renderer uses centered principal point in pixel units.
        # Explicit K below includes the optimized crop/principal-point offset.
        image=render_calibrated(state,a,v)
        mask=np.any(np.array(image)!=[235,239,241],axis=2);target=a['masks'][v]
        union=(mask|target).sum()
        result.append(dict(view=v+1,mask_IoU=float((mask&target).sum()/max(union,1)),
                           extra_pixels=int((mask&~target).sum()),missing_pixels=int((target&~mask).sum())))
        rendered.append(image)
    q=queries(state,a)
    if len(q):
        ci,pi=a['ci'],a['pi'];camera=np.einsum('nij,nj->ni',state['g'][ci],q[pi]-state['C'][ci])
        uv=np.exp(state['lf'][ci,None])*camera[:,:2]/camera[:,2,None]+state['pp'][ci]
        residual=np.linalg.norm(uv-a['uv'],axis=1)*a['width']
        feature=dict(median_reprojection_px=float(np.median(residual)),mean_reprojection_px=float(residual.mean()),
                     rms_reprojection_px=float(np.sqrt(np.mean(residual**2))),fraction_within_3px=float(np.mean(residual<3)),
                     observation_count=len(ci),track_count=len(q))
    else:feature=dict(track_count=0,observation_count=0)
    return result,feature,rendered


def render_calibrated(state,a,view,size=None):
    from reconstruction.raster import render_mesh
    width=a['width'];factor=(size or width)/width
    return render_mesh(state['X'],a['faces'],state['C'][view],state['g'][view],
                       np.exp(state['lf'][view])*width*factor,
                       (width/2+state['pp'][view]*width)*factor,
                       (int(width*factor),int(width*factor)))


def display_frame(state):
    # Display up follows image 0, identically for every input. PCA longest axis
    # can be an uncertain depth direction, so it must not be called vertical.
    X=state['X'];centre=X.mean(0);up=-state['g'][0,1].copy()
    toward=state['C'][0]-centre;toward-=up*(toward@up);toward/=np.linalg.norm(toward)
    right=np.cross(up,toward);right/=np.linalg.norm(right)
    Q=np.stack([right,-toward,up]);Y=(X-centre)@Q.T
    factor=1/max(np.ptp(Y[:,2]),1e-8)
    return Y*factor,dict(world_to_display_rotation=Q.tolist(),centre=centre.tolist(),
                        display_scale=factor,height_normalized_to=1.,units='relative, not metres',
                        display_up='camera-0 image vertical, not measured world gravity')


def save_report(root,images,masks,transforms,pairs,feature_info,init_info,initial,result,a,history):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    state=result['state'];before,bf,bimages=metrics(initial,a);after,af,renders=metrics(state,a)
    model,frame=display_frame(state);export_mesh(root/'reconstruction',model,a['faces'])
    F=a['faces'];C=np.array([1.,-1.,np.sqrt(2.)]);g=look_at(C,np.zeros(3),up=(0.,0.,1.))
    render(model,F,C,g,1000,(1024,1024)).save(root/'model_45deg_2units.png')
    from reconstruction.raster import render_mesh
    turns=[]
    for j,angle in enumerate([0,90,180,270]):
        theta=np.radians(angle);C=np.array([2*np.sin(theta),-2*np.cos(theta),.65])
        im=render_mesh(model,F,C,look_at(C,np.zeros(3),up=(0.,0.,1.)),900,[512,512],(1024,1024))
        im.save(root/f'view_{angle:03d}.png');turns.append(im)
    for v,(image,mask,reproj) in enumerate(zip(images,masks,renders)):
        Image.fromarray(image).save(root/f'input_{v+1:02d}.png')
        Image.fromarray(np.uint8(mask)*255).save(root/f'mask_{v+1:02d}.png')
        reproj.save(root/f'reprojection_{v+1:02d}.png')
    fig,axs=plt.subplots(len(images),4,figsize=(15,5*len(images)),squeeze=False)
    for v in range(len(images)):
        pred=np.any(np.array(renders[v])!=[235,239,241],axis=2);target=masks[v]
        diff=np.full((*target.shape,3),[240,243,245],np.uint8)
        diff[pred&target]=[72,153,120];diff[pred&~target]=[220,62,70];diff[target&~pred]=[44,111,203]
        overlay=images[v].copy();overlay[pred^target]=np.uint8(.35*overlay[pred^target]+.65*diff[pred^target])
        panels=[(images[v],f'Ảnh {v+1}: đầu vào'),(np.array(renders[v]),'Mô hình chiếu lại qua camera ước lượng'),
                (overlay,f'Đối chiếu: IoU {after[v]["mask_IoU"]*100:.2f}%'),(diff,'Xanh: trùng; đỏ: thừa; lam: thiếu')]
        for ax,(im,title) in zip(axs[v],panels):ax.imshow(im);ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Pipeline dùng chung: lưới tự do, không dùng mẫu hình học theo loại vật',fontsize=13)
    fig.text(.5,.02,'Đang khớp hình học, mặt nạ và điểm tương ứng; màu, chữ và giọt nước không được dựng lại bằng cost RGB.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(root/'input_reprojection_comparison.png',dpi=145);plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(14,4))
    for ax,im,angle in zip(axs,turns,[0,90,180,270]):ax.imshow(im);ax.set_title(f'Góc xem {angle}°');ax.axis('off')
    fig.tight_layout();fig.savefig(root/'model_views.png',dpi=135);plt.close(fig)
    # Show data-derived feature correspondences, not simulated/hand-labeled IDs.
    if len(images)==2:
        fig,ax=plt.subplots(figsize=(12,6))
        canvas=np.concatenate([images[0],images[1]],axis=1);ax.imshow(canvas)
        rng=np.random.default_rng(11)
        for pair in pairs:
            for p,q in zip(pair['pixels_a'],pair['pixels_b']):
                color=rng.uniform(.1,.9,3);ax.plot([p[0],q[0]+a['width']],[p[1],q[1]],color=color,lw=.6,alpha=.65)
        ax.set_title(f'Điểm tương ứng tự động: {sum(p["inliers"] for p in pairs)} cặp sau lọc hình học');ax.axis('off')
        fig.tight_layout();fig.savefig(root/'feature_matches.png',dpi=150);plt.close(fig)
    elif pairs:
        shown=sorted(pairs,key=lambda p:-p['inliers'])[:4]
        fig,axs=plt.subplots(len(shown),1,figsize=(10,4.5*len(shown)),squeeze=False)
        rng=np.random.default_rng(11)
        for row,pair in enumerate(shown):
            va,vb=int(pair.get('view_a',0)),int(pair['view_b'])
            canvas=np.concatenate([images[va],images[vb]],axis=1);ax=axs[row,0];ax.imshow(canvas)
            for p,q in zip(pair['pixels_a'],pair['pixels_b']):
                color=rng.uniform(.1,.9,3);ax.plot([p[0],q[0]+a['width']],[p[1],q[1]],color=color,lw=.7,alpha=.75)
            ax.set_title(f'Cặp ảnh {va+1} ↔ {vb+1}: {pair["inliers"]} điểm tương ứng sau lọc hình học');ax.axis('off')
        fig.tight_layout();fig.savefig(root/'feature_matches.png',dpi=140);plt.close(fig)
    else:
        fig,ax=plt.subplots(figsize=(8,2.5))
        ax.text(.5,.5,'Không có cặp ảnh nào đạt >= 8 SIFT inliers; tái dựng bằng ràng buộc đường bao đa góc nhìn.',
                ha='center',va='center',fontsize=11)
        ax.axis('off');fig.tight_layout();fig.savefig(root/'feature_matches.png',dpi=120);plt.close(fig)
    impl_label=('free mesh, Joint DCA + Cauchy IRLS, image-proposed fixed-axis C8 or alternative compact completion'
                if a.get('cauchy_scale_pixels') is not None else
                'free mesh, Armijo descent, image-proposed fixed-axis C8 or alternative compact completion')
    payload=dict(implementation=impl_label,
                 input_files=transforms,mesh_vertex_count=len(model),mesh_face_count=len(F),
                 initializer=init_info,feature_frontend=feature_info,features=af,masks_before=before,masks_after=after,
                 state={k:v.tolist() for k,v in state.items()},display_frame=frame,display_vertices=model.tolist(),faces=F.tolist(),
                 observations=dict(camera_ids=a['ci'].tolist(),track_ids=a['pi'].tolist(),pixels_normalized=a['uv'].tolist(),
                                   anchor_vertices=a['anchor_vertices'].tolist(),anchor_bary=a['anchor_bary'].tolist()),
                 initial_state={k:v.tolist() for k,v in initial.items()},
                 history=history,objective=result['stats'],status=result['status'],steps=result['steps'],
                 optimization=result.get('optimization'),
                 completion_selection=result.get('completion_selection'),
                 axis_symmetry_selection=result.get('axis_symmetry_selection'),
                 hidden_completion=result.get('hidden_completion'),
                 axis_constraint=a.get('axis_constraint'),
                 contact_configuration=(a['contact_model'].configuration() if a.get('contact_model') is not None else None),
                 visibility_constraint_solver=dict(penalty=a.get('visibility_penalty',0.),
                    dual=a.get('visibility_dual'),numerical_tolerance=a.get('visibility_feasibility_tolerance',0.)),
                 metric_scale_known=False,ground_truth_available=False,geometry_error_3d=None,
                 scope='free closed sphere-topology mesh, not a category-specific or rotational-surface model',
                 limitations=['no metric dimensions from these images alone','no true geometry/cameras available for accuracy evaluation',
                              'unseen surfaces remain prior-dependent','mask/feature fitting is not full RGB matching',
                              'fixed sphere topology cannot infer new holes or handles','camera calibration may be ambiguous',
                              'symmetry operators are hypotheses, not guaranteed object properties',
                              'assumes photos share one rigid geometry; not verified for these web images',
                              'sampled silhouette costs and sparse first-hit checks do not prove a visible-surface bijection'])
    from scipy.spatial import ConvexHull
    X=state['X'];tri=X[F]
    signed_volume=float(np.sum(tri[:,0]*np.cross(tri[:,1],tri[:,2]))/6)
    hull_volume=float(ConvexHull(X).volume)
    axes=np.linalg.eigvalsh(np.cov(X.T))
    payload['mesh_diagnostics']=dict(signed_volume_relative_units=signed_volume,convex_hull_volume_relative_units=hull_volume,
         signed_to_hull_volume_ratio=signed_volume/max(hull_volume,1e-15),
         pca_min_to_max_axis_ratio=float(np.sqrt(max(axes[0],0)/max(axes[-1],1e-15))),
         self_intersections_checked=False,
         note='topology/volume/2D consistency are diagnostics, not proof of correct physical 3D shape')
    if a.get('contact_model') is not None:payload['mesh_diagnostics'].update(self_intersections_checked=True,**a['contact_model'].audit(X))
    payload['configuration']=dict(initial_mesh='icosphere',coarse_vertices=42,final_vertices=len(X),
        point_loss=('cauchy IRLS robust loss outside observation disk' if a.get('cauchy_scale_pixels') and a.get('point_tolerance_pixels',0.) else
                    'cauchy IRLS robust loss' if a.get('cauchy_scale_pixels') else
                    'squared distance outside observation disk' if a.get('point_tolerance_pixels',0.) else
                    'squared after geometric feature filtering'),mask_loss='shared sampled bidirectional distance',
        priors=('central, best mirror, fixed image-proposed C8 surface and transverse moments, concavity, Laplacian'
                if a.get('axis_constraint') else 'scale-invariant central, best mirror and rotation hypotheses, local concavity, Laplacian'),
        prior_weight=a['prior_weight'],smooth_weight=a['smooth_weight'],pixel_tolerance=a['pixel_tolerance']*a['width'],
        compactness_weight=a.get('compactness_weight',0.),
        compactness_prior='log(A^3/(36*pi*V^2)); compact completion preference, not measured depth',
        calibration_weight=a.get('calibration_weight',0.),principal_sigma=a.get('principal_sigma'),
        nominal_normalized_focal=2.,log_focal_sigma=a.get('log_focal_sigma'),
        first_hit_relative_tolerance=.001)
    payload['configuration'].update(shape_prior_units=a.get('shape_prior_units','legacy_raw_image_units'),
         symmetry_epsilon=a.get('symmetry_epsilon',a['prior_weight']),smoothness_epsilon=a.get('smoothness_epsilon',a['smooth_weight']),
         optimization_scale=a.get('optimization_scale',1.),point_tolerance_pixels=a.get('point_tolerance_pixels',0.),
         cauchy_scale_pixels=a.get('cauchy_scale_pixels'),
         camera_prior_coefficient_px_squared=a.get('calibration_weight',0.)*a['width']**2,
         reject_transverse_intersections=a.get('reject_transverse_intersections',False),
         field_interpolation=a.get('field_interpolation','bilinear'),contact_weight=a.get('contact_weight',0.))
    (root/'result.json').write_text(json.dumps(jsonable(payload),indent=2),encoding='utf-8')
    html=build_report_html(before,after,result)
    (root/'index.html').write_text(html,encoding='utf-8')
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--images',nargs='+',type=Path,required=True);p.add_argument('--masks',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--size',type=int,default=384)
    p.add_argument('--optimizer',choices=('adam','dca'),default='adam')
    p.add_argument('--cauchy-scales',nargs='+',type=float,default=[30.,10.,3.])
    p.add_argument('--coarse-steps',type=int,default=None);p.add_argument('--steps',type=int,default=None)
    p.add_argument('--prior-weight',type=float,default=.001)
    p.add_argument('--point-tolerance-pixels',type=float,default=0.)
    p.add_argument('--distance-field-interpolation',choices=('bilinear','cubic'),default='cubic')
    p.add_argument('--contact-mode',choices=('ipc','diagnostic'),default='ipc',help='IPC requires ipctk/libigl; diagnostic reproduces the incomplete transverse guard.')
    p.add_argument('--allow-transverse-intersections',action='store_true',help='Legacy diagnostic mode; new fits reject transverse crossings.')
    p.add_argument('--compactness-pixels',type=float,default=1.,
                   help='Compact-completion prior strength in equivalent working-image pixels; 0 disables.')
    p.add_argument('--camera-starts',choices=('multiple','single'),default='multiple',
                   help='Compare generic focal/crop/translation hypotheses using the same objective.')
    p.add_argument('--calibration-prior-pixels',type=float,default=np.sqrt(.001),
                   help='Amplitude of focal/crop prior; its square is the pixel-unit coefficient (default 0.001); 0 disables.')
    p.add_argument('--completion-selection',choices=('axis','discrepancy','fixed'),default='axis')
    p.add_argument('--axis-steps',type=int,default=None)
    p.add_argument('--hidden-completion-steps',type=int,default=0,help='Optional same-cost completion of model-hidden unanchored geometry after accepted axis fitting.')
    p.add_argument('--completion-steps',type=int,default=None)
    p.add_argument('--data-budget-pixels',type=float,default=1.)
    p.add_argument('--init-ring',action='store_true',help='Enable 360-degree ring feature matching and circular camera initialization.')
    p.add_argument('--match-mode',choices=('auto','star','ring','all'),default='auto',
                   help='Feature pair graph topology (auto selects ring when --init-ring is set, else star).')
    p.add_argument('--ring-window',type=int,default=2,help='Circular neighbor step window for ring feature matching.')
    p.add_argument('--ring-direction',choices=('auto','ccw','cw'),default='auto',
                   help='Camera orbit direction around -g[0,1] for circular initialization.')
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    if args.coarse_steps is None:args.coarse_steps=90 if args.optimizer=='dca' else 650
    if args.steps is None:args.steps=180 if args.optimizer=='dca' else 1600
    if args.axis_steps is None:args.axis_steps=120 if args.optimizer=='dca' else 900
    if args.completion_steps is None:args.completion_steps=120 if args.optimizer=='dca' else 750
    if args.size<128 or args.ring_window<1 or min(args.coarse_steps,args.steps,args.completion_steps,args.axis_steps)<1 or min(args.prior_weight,args.compactness_pixels,args.calibration_prior_pixels,args.data_budget_pixels,args.point_tolerance_pixels)<0:
        p.error('size>=128, ring_window>=1, positive iteration counts and nonnegative prior weight required.')
    if args.optimizer=='dca' and (not args.cauchy_scales or min(args.cauchy_scales)<=0):
        p.error('Positive --cauchy-scales required for DCA.')
    if args.masks and len(args.masks)!=len(args.images):p.error('One mask per image required.')
    init_mode='ring' if args.init_ring else 'star'
    pair_mode=('ring' if args.init_ring else 'star') if args.match_mode=='auto' else args.match_mode
    images,masks,transforms=load_images(args.images,args.size,args.masks)
    pairs,feature_info=feature_pairs(images,masks,pair_mode=pair_mode,ring_window=args.ring_window)
    print('Image features:',feature_info,'inlier pairs:',[(p.get('view_a',0),p['view_b'],p['inliers']) for p in pairs],flush=True)
    state,faces,init_info=initialize(images,masks,pairs,init_mode=init_mode,ring_direction=args.ring_direction);state['h']=np.eye(3)
    start=time.perf_counter();coarse_faces=faces
    starts=(camera_hypotheses(state,images=images,masks=masks,init_mode=init_mode,
                              ring_direction=args.ring_direction,
                              ring_direction_detected=bool(init_info.get('ring_direction_detected')))
            if args.camera_starts=='multiple' else [('feature_calibration',state)])
    fitted=[];search=[]
    use_dca=(args.optimizer=='dca')
    solve_fn=optimize_dca if use_dca else optimize
    def fit(seed,faces,steps,cauchy_schedule=None):
        a=arrays(images,masks,pairs,seed,faces);a['prior_weight']=args.prior_weight
        a['compactness_weight']=(0. if args.completion_selection=='axis' else
                                 (args.compactness_pixels/args.size)**2)
        a['calibration_weight']=(args.calibration_prior_pixels/args.size)**2
        a.update(point_tolerance_pixels=args.point_tolerance_pixels,reject_transverse_intersections=not args.allow_transverse_intersections)
        a['field_interpolation']=args.distance_field_interpolation
        if args.distance_field_interpolation=='cubic':
            from reconstruction.fields import field_coefficients
            a['field_coefficients']=field_coefficients(a['fields'])
        if args.contact_mode=='ipc':
            from reconstruction.contact import ContactSystem
            a.update(contact_model=ContactSystem(seed['X'],faces),contact_weight=1/args.size**2)
        if args.completion_selection=='axis':a=pixel_normalized_shape_priors(a)
        if use_dca:
            sched=tuple(cauchy_schedule) if cauchy_schedule is not None else tuple(args.cauchy_scales)
            a['cauchy_scale_pixels']=float(sched[-1])
            result=optimize_dca(seed,a,steps=steps,objective_fn=objective,step_fn=step,valid_fn=valid,
                                cauchy_schedule=sched)
            a=result['a']
        else:
            result=optimize(seed,a,steps=steps,objective_fn=objective,step_fn=step,valid_fn=valid,
                            learning_rates=dict(X=.003,C=.002,g=.0006,lf=.0008,pp=.0005,h=.001))
        return result,a
    for label,seed in starts:
        print(f'Coarse hypothesis {label}: {len(seed["X"])} vertices',flush=True)
        probe=arrays(images,masks,pairs,seed,coarse_faces)
        if not valid(seed,probe):
            search.append(dict(hypothesis=label,status='invalid_initial_geometry'));continue
        result,a=fit(seed,coarse_faces,args.coarse_steps,cauchy_schedule=args.cauchy_scales if use_dca else None)
        # Compare hypotheses on a common REFINED topology, not merely on coarse
        # meshes whose approximation errors can differ by camera hypothesis.
        exploratory_steps=min(args.steps,max(1,args.steps//2)) if len(starts)>1 else 0
        exploratory=None
        if exploratory_steps:
            refined=copy.deepcopy(result['state']);refined['X'],fine_faces=subdivide(refined['X'],coarse_faces)
            print(f'Exploratory refinement {label}: {exploratory_steps} steps',flush=True)
            exp_sched=args.cauchy_scales[-2:] if (use_dca and len(args.cauchy_scales)>=2) else (args.cauchy_scales if use_dca else None)
            exploratory,a=fit(refined,fine_faces,exploratory_steps,cauchy_schedule=exp_sched)
        candidate=exploratory or result
        fitted.append((candidate['stats']['total'],label,copy.deepcopy(seed),result,exploratory,a,exploratory_steps))
        search.append(dict(hypothesis=label,status=candidate['status'],cost=candidate['stats']['total'],
                           statistics=candidate['stats'],optimization=candidate['optimization'],
                           coarse_optimization=result['optimization']))
    if not fitted:raise RuntimeError('No valid camera initialization.')
    selected_cost,label,initial,coarse,exploratory,a,used_steps=min(fitted,key=lambda item:item[0])
    init_info['camera_hypothesis_search']=search;init_info['selected_hypothesis']=label
    print(f'Selected {label} by common objective, cost {selected_cost:.8g}',flush=True)
    if exploratory:state=exploratory['state'];_,faces=subdivide(initial['X'],coarse_faces)
    else:state=coarse['state'];state['X'],faces=subdivide(state['X'],coarse_faces)
    initial['X'],_=subdivide(initial['X'],coarse_faces)
    print(f'Refinement: {len(state["X"])} vertices / {len(faces)} triangles',flush=True)
    if args.steps>used_steps:result,a=fit(state,faces,args.steps-used_steps,cauchy_schedule=[args.cauchy_scales[-1]] if use_dca else None)
    else:result=exploratory
    history=([dict(stage=1,**h) for h in coarse['history']]+
             [dict(stage=2,**h) for h in result['history']])
    optimization_stages=[dict(stage=1,**coarse['optimization']),dict(stage=2,**result['optimization'])]
    if exploratory:
        history=([dict(stage=1,**h) for h in coarse['history']]+
                 [dict(stage=2,**h) for h in exploratory['history']]+
                 [dict(stage=3,**h) for h in result['history']])
        optimization_stages=[dict(stage=1,**coarse['optimization']),dict(stage=2,**exploratory['optimization']),
                             dict(stage=3,**result['optimization'])]
    if args.completion_selection=='discrepancy':
        result,a=select_completion(result['state'],a,steps=args.completion_steps,noise_pixels=args.data_budget_pixels,optimizer_fn=solve_fn if use_dca else None)
        history.extend([dict(stage='selected_completion',**h) for h in result['history']])
        optimization_stages.append(dict(stage='selected_completion',**result['optimization']))
    if args.completion_selection=='axis':
        result,a=select_axis_completion(result['state'],a,masks,steps=args.axis_steps,noise_pixels=args.data_budget_pixels,optimizer_fn=solve_fn if use_dca else None)
        history.extend([dict(stage='axis_selection',**h) for h in result['history']])
        optimization_stages.append(dict(stage='axis_selection',**result['optimization']))
    result,a=append_hidden_stage(result,a,history,optimization_stages,args.hidden_completion_steps,optimizer_fn=solve_fn if use_dca else None)
    result['optimization']=dict(stages=optimization_stages)
    payload=save_report(args.output,images,masks,transforms,pairs,feature_info,init_info,initial,result,a,history)
    print('Mask metrics:',payload['masks_after'],flush=True);print('Feature reprojection:',payload['features'],flush=True)
    print('Runtime seconds:',time.perf_counter()-start,'Saved:',args.output,flush=True)


if __name__=='__main__':main()
