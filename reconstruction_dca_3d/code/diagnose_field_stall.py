"""Compare one-sided directional derivatives at a saved stalled state.

Each interpolant defines its own surrogate. Compare directional derivatives
to THAT surrogate's analytic slope, never compare their objective values as
if they were identical losses. No fitting or point removal is performed.
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from reconstruction.records import read_case
from reconstruction.engine import objective
from reconstruction.chart import Chart
from reconstruction.fields import field_coefficients


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    _,_,_,_,state,_,a=read_case(args.source);chart=Chart(state);z=chart.initial;records=[]
    fig,ax=plt.subplots(figsize=(9,5))
    for field in ('bilinear','cubic'):
        b=dict(a);b.pop('field_coefficients',None)
        if field=='cubic':b['field_coefficients']=field_coefficients(a['fields'])
        cost,g,stats=objective(state,b);gg=chart.pullback(z,g)*a['width']**2;norm=np.linalg.norm(gg);direction=-gg/norm
        data=[]
        for h in np.logspace(-12,-3,19):
            value=objective(chart.decode(z+h*direction),b)[0]
            data.append(dict(step=float(h),forward_derivative=float((value-cost)*a['width']**2/h)))
        record=dict(interpolation=field,analytic_slope=float(-norm),samples=data);records.append(record)
        line,=ax.semilogx([r['step'] for r in data],[r['forward_derivative'] for r in data],'o-',label=field)
        ax.axhline(-norm,color=line.get_color(),ls='--',alpha=.6)
    ax.axhline(0,color='black',lw=.8);ax.set_ylim(-1.5,2)
    ax.set_xlabel('Step length along normalized negative gradient');ax.set_ylabel('One-sided cost derivative (pixel squared units)')
    ax.set_title('Saved stalled geometry: negative analytic slope vs finite feasible steps\nDashed: own analytic slope; negative derivative means actual descent')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(args.output/'field_stall_diagnosis.png',dpi=150);plt.close(fig)
    (args.output/'field_stall_diagnosis.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    for row in records:print(row['interpolation'],'analytic slope',row['analytic_slope'],'forward at 1e-7',row['samples'][10])


if __name__=='__main__':main()
