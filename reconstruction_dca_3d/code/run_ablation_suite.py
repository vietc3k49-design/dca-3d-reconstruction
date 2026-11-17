"""Spec: run all six declared cases with bounded concurrency and per-case logs.

Never rerun a completed case or discard a failed case. The parent worker
records job lifetime; this runner additionally records each process/exit code.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from reconstruction.ablation import case_grid


def run_case(case,source,root,steps,optimizer='adam',warm_start_cases=False):
    folder=root/case['label'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'ablation.json').exists():
        return dict(label=case['label'],status='cached_complete',exit_code=0)
    if (folder/'result.json').exists():raise FileExistsError(f'Incomplete existing case: {folder}')
    source=source/case['label'] if warm_start_cases else source
    command=[sys.executable,'-u',str(Path(__file__).with_name('run_axis_ablation.py')),
             '--source',str(source),'--output',str(folder),'--case',case['label'],'--steps',str(steps),'--optimizer',optimizer]
    with (folder/'stdout.log').open('w',encoding='utf-8') as out,(folder/'stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen(command,stdout=out,stderr=err)
        (folder/'process.json').write_text(json.dumps(dict(pid=child.pid,host=platform.node(),started=time.time())),encoding='utf-8')
        code=child.wait()
    return dict(label=case['label'],status='complete' if code==0 else 'failed',exit_code=code)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=900);p.add_argument('--parallel',type=int,default=3)
    p.add_argument('--optimizer',choices=('adam','lbfgs'),default='adam');p.add_argument('--warm-start-cases',action='store_true')
    args=p.parse_args()
    if args.steps<1 or args.parallel<1:p.error('Positive steps and concurrency required.')
    root=args.output.resolve();source=args.source.resolve();root.mkdir(parents=True,exist_ok=True)
    status=dict(host=platform.node(),pid=os.getpid(),started=time.time(),cases=[],planned=case_grid())
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures=[pool.submit(run_case,case,source,root,args.steps,args.optimizer,args.warm_start_cases) for case in case_grid()]
        for future in as_completed(futures):
            record=future.result();status['cases'].append(record)
            temporary=root/'suite_state.tmp';temporary.write_text(json.dumps(status,indent=2),encoding='utf-8')
            temporary.replace(root/'suite_state.json');print(json.dumps(record),flush=True)
    status['finished']=time.time()
    (root/'suite_state.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    if any(c['exit_code'] for c in status['cases']):raise RuntimeError('Some declared cases failed; preserve logs.')


if __name__=='__main__':main()
