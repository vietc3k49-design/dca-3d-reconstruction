"""Finite-difference and data-integrity checks for the category-independent path.

Run after the supplied can/control experiments. No known 3D object is passed to
the reconstruction; saved observations are used only for numerical verification.
"""
import ast
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
from reconstruction.engine import arrays, objective, step, queries, valid
from reconstruction.mesh import sphere
from reconstruction.rotations import exp_rotation

ROOT = Path(__file__).resolve().parents[1]


def load_case(folder, initial=False):
    folder = ROOT / folder
    if not folder.exists():
        folder = ROOT / 'results' / ('can' if folder.name == 'generic_can_refined' else 'star_control')
    payload = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    images = np.array([np.array(Image.open(p)) for p in sorted(folder.glob('input_*.png'))
                       if p.stem.split('_')[-1].isdigit()])
    masks = np.array([np.array(Image.open(p)) > 127 for p in sorted(folder.glob('mask_*.png'))])
    state = {k: np.array(v, float) for k, v in payload['state'].items()}
    obs = payload['observations']; ci = np.array(obs['camera_ids'], int)
    ids = np.array(obs['track_ids'], int); uv = np.array(obs['pixels_normalized']).reshape(-1, 2)
    pairs = []
    for view in range(1, len(images)):
        reference = {i: p for i, p in zip(ids[ci == 0], uv[ci == 0])}
        selected = [(reference[i], p) for i, p in zip(ids[ci == view], uv[ci == view])]
        if selected:
            u, v = np.array(selected).transpose(1, 0, 2)
            pairs.append(dict(view_b=view, inliers=len(u), pixels_a=(u+.5)*images.shape[1],
                              pixels_b=(v+.5)*images.shape[1]))
    faces = np.array(payload['faces'], int)
    if initial:
        # Arbitrary coarse topology, perturbed away from ties in piecewise costs.
        coarse, faces = sphere(1)
        old = {k: np.array(v, float) for k, v in payload['initial_state'].items()}
        state = old; state['X'] = old['X'][:len(coarse)].copy()
        rng = np.random.default_rng(91)
        state['X'] += rng.normal(0, .013, state['X'].shape)
        state['C'] += rng.normal(0, .003, state['C'].shape)
        state['h'] = exp_rotation(np.array([.037, -.021, .016]))
    a = arrays(images, masks, pairs, state, faces)
    for key in ('prior_weight','smooth_weight','compactness_weight','calibration_weight','principal_sigma','log_focal_sigma'):
        if payload.get('configuration',{}).get(key) is not None:a[key]=payload['configuration'][key]
    if not initial:
        a['anchor_vertices'] = np.array(obs['anchor_vertices'], int).reshape(-1, 3)
        a['anchor_bary'] = np.array(obs['anchor_bary'], float).reshape(-1, 3)
    return state, a, payload


def perturb(state, key, index, amount):
    result = copy.deepcopy(state)
    if key in ('g', 'h'):
        d = np.zeros((len(state['g']), 3) if key == 'g' else (3,))
        d[index] = amount; result[key] = exp_rotation(d) @ state[key]
    else:
        result[key][index] += amount
    return result


def check_gradient(state, a, full):
    _, grad, stats = objective(state, a)
    worst = 0.; count = 0; eps = 1e-7
    for key, values in grad.items():
        indices = list(np.ndindex(values.shape))
        if key == 'g': indices = [i for i in indices if i[0] != 0]
        if not full and len(indices) > 15: indices = indices[::max(1, len(indices)//15)]
        for index in indices:
            plus = objective(perturb(state, key, index, eps), a)[0]
            minus = objective(perturb(state, key, index, -eps), a)[0]
            error = abs((plus-minus)/(2*eps) - values[index])
            worst = max(worst, error); count += 1
    assert worst < 2e-6, (worst, stats)
    print(f'PASS {len(state["X"])} vertices / {len(a["ci"])} observations: '
          f'{count} derivatives, maximum absolute error {worst:.3g}')
    return dict(vertices=len(state['X']), observations=len(a['ci']), derivatives=count,
                maximum_absolute_error=worst)


def main():
    # The generic path must run without importing a synthetic object fixture.
    for path in [ROOT/'code/reconstruct_images.py', *(ROOT/'code/reconstruction').glob('*.py')]:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom): names = [node.module or '']
            else: continue
            assert not any('star_' in n for n in names), (path, names)
    import star_bidirectional as legacy
    from reconstruction import losses
    assert legacy.silhouette_terms is losses.silhouette_terms
    assert legacy.visibility_term is losses.visibility_term
    print('PASS generic path has no star imports; legacy and generic kernels are identical functions')
    results = []
    for folder, initial, full in [('generic_can_refined', True, True),
                                  ('generic_can_refined', False, False),
                                  ('generic_can_depth_selected', False, False),
                                  ('generic_star_control', True, True)]:
        s, a, payload = load_case(folder, initial)
        assert valid(s, a)
        results.append(check_gradient(s, a, full))
        assert len(set(zip(a['ci'], a['pi']))) == len(a['ci'])
        assert np.allclose(s['g'] @ s['g'].transpose(0, 2, 1), np.eye(3), atol=1e-8)
        assert not payload['metric_scale_known'] and payload['geometry_error_3d'] is None
        # A joint similarity of cameras/mesh is gauge, not extra size data.
        t = copy.deepcopy(s); shift = np.array([.12, -.07, .3])
        for key in ('X', 'C'): t[key] = 1.7*(t[key] + shift)
        def projection(z):
            q = np.einsum('kij,kpj->kpi', z['g'], z['X'][None]-z['C'][:, None])
            return np.exp(z['lf'])[:, None, None]*q[..., :2]/q[..., 2, None]+z['pp'][:, None]
        normalized = step(t, {}, a)
        assert np.allclose(projection(s), projection(normalized), atol=1e-12)
        if not len(a['ci']): assert queries(s, a).shape == (0, 3)
    print('PASS unique observations, SO(3), positive depth, empty tracks, and projection-invariant gauge')
    (ROOT/'generic_checks.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
