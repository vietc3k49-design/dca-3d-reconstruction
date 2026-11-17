"""Matched optimizer continuation: same six saved states and same objectives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def run(method,source,root,steps):
    arguments=[sys.executable,'-u',str(Path(__file__).with_name('run_ablation_suite.py')),
               '--source',str(source),'--output',str(root/f'results_{method}'),
               '--steps',str(steps),'--parallel','3','--optimizer',method,'--warm-start-cases']
    with (root/f'{method}_stdout.log').open('w',encoding='utf-8') as out,(root/f'{method}_stderr.log').open('w',encoding='utf-8') as err:
        code=subprocess.call(arguments,stdout=out,stderr=err)
    return dict(optimizer=method,exit_code=code)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=900)
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,m,args.source.resolve(),root,args.steps) for m in ('adam','lbfgs')]
        results=[f.result() for f in futures]
    (root/'study_state.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    if any(r['exit_code'] for r in results):raise RuntimeError('Preserve failed optimizer-check logs.')
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
