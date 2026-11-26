import sys,json,subprocess,time,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'dcacolmap_benchmark'))
from benchmark import native,evaluate,json_default
from stochastic_dca import solve_stochastic,coverage_core

def read_state(path):
    with open(path,'rb') as f:
        assert f.read(8)==b'SDCAOUT1'
        m,n=np.fromfile(f,'<u4',2); a=np.fromfile(f,'<f8')
    return a[:m*9].reshape(m,3,3),a[m*9:m*12].reshape(m,3),a[m*12:].reshape(n,3)

def main():
    d=dict(np.load(ROOT.parent/'helmet_benchmark/problem.npz')); rows=[]; checks=[]
    order=np.lexsort((d['ci'],d['pi'])); ci=d['ci'][order]; pi=d['pi'][order]
    ptr=np.r_[0,np.cumsum(np.bincount(pi,minlength=len(d['X'])))].astype(np.int64)
    core,_=coverage_core(ci,ptr,len(d['X']),len(d['R']),d['anchor_idx'],6)
    solve_stochastic(d['R'],d['C'],d['X'],d['ci'],d['pi'],d['uv'],d['K'],{},d['anchor_idx'],max_outer=1)
    for seed in [11,22,33,44,55]:
        for marked in [False,True]:
            anchors={int(i):x.copy() for i,x in zip(d['anchor_idx'],d['anchor_xyz'])} if marked else {}
            methods=['ceres','numba','cpp','cpp_full']; offset=seed%4
            for method in methods[offset:]+methods[:offset]:
                print('START',seed,marked,method,flush=True)
                if method=='ceres':
                    R,C,X,info=native(d['R'],d['C'],d['X'],d,anchors,max_iterations=1000)
                    info['linear_solver_seconds']=info['native_summary']['linear_solver_time_in_seconds']
                    info['assembly_and_control_seconds']=info['seconds']-info['linear_solver_seconds']
                elif method=='numba':
                    R,C,X,info=solve_stochastic(d['R'],d['C'],d['X'],d['ci'],d['pi'],d['uv'],d['K'],anchors,d['anchor_idx'],seed=seed)
                else:
                    stem=ROOT/f'{method}_a{int(marked)}_s{seed}'
                    args=[str(ROOT/'stochastic_dca.exe'),'--input',str(ROOT/'problem.bin'),'--output',str(stem.with_suffix('.json')),'--state',str(stem.with_suffix('.bin')),'--plan',str(stem.with_suffix('.plan')),'--anchors',str(int(marked)),'--seed',str(seed)]
                    if method=='cpp_full':args+=['--full-reuse']
                    subprocess.run(args,check=True,capture_output=True)
                    info=json.loads(stem.with_suffix('.json').read_text()); R,C,X=read_state(stem.with_suffix('.bin'))
                    if method=='cpp':
                        plan=np.fromfile(stem.with_suffix('.plan'),'<u4'); nc=int(plan[0]); cc=plan[1:1+nc]; perm=plan[nc+2:]
                        assert np.array_equal(cc,np.flatnonzero(core))
                        assert np.array_equal(perm,np.random.RandomState(seed).permutation(np.flatnonzero(~core)))
                        checks.append({'seed':seed,'anchors':marked,'sampling_exact_match':True})
                    if marked:assert np.max(np.abs(X[d['anchor_idx']]-d['anchor_xyz']))==0
                    assert np.max(np.abs(R@R.transpose(0,2,1)-np.eye(3)))<1e-10
                metrics=evaluate(R,C,X,d)
                row={'seed':seed,'method':method,'landmarks':marked,'solver':info,'metrics':metrics}; rows.append(row)
                (ROOT/'comparison.json').write_text(json.dumps({'runs':rows,'checks':checks},indent=2,default=json_default))
                print('RESULT',seed,marked,method,info['seconds'],info['final_cost'],flush=True)
    summary=[]
    for marked in [False,True]:
        for method in ['ceres','numba','cpp','cpp_full']:
            rr=[r for r in rows if r['method']==method and r['landmarks']==marked]
            s={'anchors':8 if marked else 0,'method':method,'runs':len(rr)}
            for key in ['seconds','linear_solver_seconds','assembly_and_control_seconds','final_cost']:
                s[key]=float(np.median([r['solver'][key] for r in rr]))
            for key in ['camera_center_rmse_mm_sim3','rotation_mean_deg','heldout_reprojection_rmse_px']:
                s[key]=float(np.median([r['metrics'][key] for r in rr]))
            summary.append(s)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    with (ROOT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
