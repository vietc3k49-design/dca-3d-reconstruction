"""Run the two declared guarded restarts with at most two fitting processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def run(field,source,root,steps):
    folder=root/field;folder.mkdir(parents=True,exist_ok=True)
    arguments=[sys.executable,'-u',str(Path(__file__).with_name('run_guarded_fit.py')),
               '--source',str(source),'--output',str(folder),'--field',field,'--steps',str(steps)]
    with (folder/'stdout.log').open('w',encoding='utf-8') as out,(folder/'stderr.log').open('w',encoding='utf-8') as err:
        code=subprocess.call(arguments,stdout=out,stderr=err)
    return dict(field=field,exit_code=code)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=1800)
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,f,args.source.resolve(),root,args.steps) for f in ('bilinear','cubic')]
        records=[f.result() for f in futures]
    (root/'suite_state.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    if any(r['exit_code'] for r in records):raise RuntimeError('Guarded comparison failed; preserve logs')


if __name__=='__main__':main()
