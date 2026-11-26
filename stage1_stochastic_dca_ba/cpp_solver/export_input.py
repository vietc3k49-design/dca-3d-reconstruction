"""One-time data-format conversion only; no optimization is performed here."""
import struct
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def main():
    d=dict(np.load(ROOT.parent/'helmet_benchmark'/'problem.npz',allow_pickle=False))
    with (ROOT/'problem.bin').open('wb') as f:
        f.write(b'SDCABIN1');f.write(struct.pack('<5I',len(d['R']),len(d['X']),len(d['ci']),len(d['ci_test']),len(d['anchor_idx'])))
        for k in ['R','C','X','K']: f.write(np.asarray(d[k],dtype='<f8').tobytes())
        for k in ['ci','pi']:f.write(np.asarray(d[k],dtype='<u4').tobytes())
        f.write(np.asarray(d['uv'],dtype='<f8').tobytes())
        for k in ['Rgt','Cgt']:f.write(np.asarray(d[k],dtype='<f8').tobytes())
        for k in ['ci_test','pi_test']:f.write(np.asarray(d[k],dtype='<u4').tobytes())
        f.write(np.asarray(d['uv_test'],dtype='<f8').tobytes())
        f.write(np.asarray(d['anchor_idx'],dtype='<u4').tobytes())
        f.write(np.asarray(d['anchor_xyz'],dtype='<f8').tobytes());f.write(struct.pack('<d',.1))
    print(ROOT/'problem.bin')
if __name__=='__main__':main()
