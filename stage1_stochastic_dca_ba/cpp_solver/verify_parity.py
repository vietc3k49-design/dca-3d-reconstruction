import json,subprocess
import numpy as np
from compare import ROOT,read_state,solve_stochastic,evaluate

d=dict(np.load(ROOT.parent/'helmet_benchmark/problem.npz')); results=[]
for marked in [False,True]:
    anchors={int(i):x.copy() for i,x in zip(d['anchor_idx'],d['anchor_xyz'])} if marked else {}
    R,C,X,info=solve_stochastic(d['R'],d['C'],d['X'],d['ci'],d['pi'],d['uv'],d['K'],anchors,d['anchor_idx'],seed=11)
    stem=ROOT/f'cpp_a{int(marked)}_s11'; cr,cc,cx=read_state(stem.with_suffix('.bin')); cpp=json.loads(stem.with_suffix('.json').read_text())
    errs={k:float(np.max(np.abs(a-b))) for k,a,b in [('R',R,cr),('C',C,cc),('X',X,cx)]}
    assert max(errs.values())<1e-6
    assert abs(cpp['final_cost']-info['final_cost'])<1e-6
    assert cpp['accepted']==info['accepted'] and cpp['Q_rebuilds']==info['Q_rebuilds']
    hist=max(abs(a['batch_cost']-b['batch_cost']) for a,b in zip(cpp['history'],info['history']))
    assert hist<1e-5
    metrics=evaluate(cr,cc,cx,d)
    metricerr={k:abs(v-metrics[k]) for k,v in cpp['metrics'].items()}
    assert max(metricerr.values())<1e-6
    tests=subprocess.run([str(ROOT/'stochastic_dca.exe'),'--input',str(ROOT/'problem.bin'),'--self-test','--anchors',str(int(marked))],capture_output=True,text=True,check=True).stdout.strip()
    results.append({'anchors':marked,'state_max_abs_error':errs,'history_max_cost_error':hist,'metric_errors':metricerr,'self_test':tests})
(ROOT/'verification.json').write_text(json.dumps({'passed':True,'cases':results},indent=2))
print(json.dumps(results,indent=2))
