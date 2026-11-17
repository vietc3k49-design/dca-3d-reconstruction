"""Controlled Pha 1 optimizer & robustness benchmark: Adam-Armijo vs L-BFGS-B vs Joint DCA.

Compares all three solvers from identical starting states on:
  1. Clean raw_inputs/ (both L-BFGS-B-compatible smooth objective and IPC-CCD contact objective).
  2. 25% synthetic SIFT outlier stress test (45-95 px false correspondences injected on raw_inputs/).
Outputs quantitative metrics (clean-track RMS/median px, Mask IoU, Cauchy IRLS weights,
libigl self-intersections, runtime) to --output/dca_study.json and renders visual grids.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from reconstruction.contact import ContactSystem
from reconstruction.dca import optimize_dca
from reconstruction.engine import arrays, objective, point_residuals, step, valid
from reconstruction.fields import field_coefficients
from reconstruction.initialization import initialize
from reconstruction.inputs import feature_pairs, load_images
from reconstruction.lbfgs import optimize_lbfgs
from reconstruction.mesh import subdivide
from reconstruction.optimizer import optimize
from reconstruction.point_terms import cauchy_irls_weights
from reconstruction.weak_priors import pixel_normalized_shape_priors


def build_problem(images, masks, pairs, seed, faces, use_ipc=False, cauchy_scale=None):
    width = images.shape[2]
    a = arrays(images, masks, pairs, seed, faces)
    a['prior_weight'] = 0.001
    a['compactness_weight'] = 0.0
    a['calibration_weight'] = 0.001 / (width**2)
    a.update(point_tolerance_pixels=1.0, reject_transverse_intersections=True, field_interpolation='cubic')
    a['field_coefficients'] = field_coefficients(a['fields'])
    if use_ipc:
        a['contact_model'] = ContactSystem(seed['X'], faces)
        a['contact_weight'] = 1.0 / (width**2)
    a = pixel_normalized_shape_priors(a)
    if cauchy_scale is not None:
        a['cauchy_scale_pixels'] = float(cauchy_scale)
    return a


def evaluate_metrics(state, a, images, masks, clean_track_mask=None):
    from reconstruct_images import metrics as eval_metrics
    mask_res, feat, renders = eval_metrics(state, a)
    res_norm = point_residuals(state, a)
    err_px = np.linalg.norm(res_norm, axis=1) * a['width']
    is_clean = np.ones(len(err_px), dtype=bool)
    if clean_track_mask is not None:
        is_clean = clean_track_mask[a['pi']]
    clean_err = err_px[is_clean]
    out_err = err_px[~is_clean]

    w_info = {}
    if a.get('cauchy_scale_pixels') is not None:
        w = cauchy_irls_weights(res_norm, a['width'], a['cauchy_scale_pixels'], a.get('point_tolerance_pixels', 0.))
        w_info['clean_weight_median'] = float(np.median(w[is_clean])) if np.any(is_clean) else None
        w_info['outlier_weight_median'] = float(np.median(w[~is_clean])) if np.any(~is_clean) else None
        w_info['effective_inliers'] = int(np.sum(w >= 0.5))

    from igl.predicates import find_self_intersections
    _, pairs_int, _ = find_self_intersections(np.ascontiguousarray(state['X'], float), np.ascontiguousarray(a['faces']))
    intersections = int(len(pairs_int))

    return dict(
        mask_IoU=[round(m['mask_IoU'] * 100, 2) for m in mask_res],
        mean_IoU=round(float(np.mean([m['mask_IoU'] for m in mask_res])) * 100, 2),
        all_rms_px=round(float(np.sqrt(np.mean(err_px**2))), 3) if len(err_px) else 0.,
        all_median_px=round(float(np.median(err_px)), 3) if len(err_px) else 0.,
        clean_rms_px=round(float(np.sqrt(np.mean(clean_err**2))), 3) if len(clean_err) else 0.,
        clean_median_px=round(float(np.median(clean_err)), 3) if len(clean_err) else 0.,
        outlier_rms_px=round(float(np.sqrt(np.mean(out_err**2))), 3) if len(out_err) else None,
        self_intersections=intersections,
        **w_info,
    ), renders


def inject_outliers(pairs, width, fraction=0.25, seed=42):
    """Corrupt `fraction` of SIFT tracks in view_b by 45-95 px (capped inside canvas)."""
    rng = np.random.default_rng(seed)
    corrupted = copy.deepcopy(pairs)
    total_tracks = sum(len(p['pixels_b']) for p in corrupted)
    n_bad = max(1, int(round(total_tracks * fraction)))
    bad_idx = set(rng.choice(total_tracks, size=n_bad, replace=False).tolist())
    clean_mask = np.ones(total_tracks, dtype=bool)
    offset = 0
    for p in corrupted:
        for k in range(len(p['pixels_b'])):
            gid = offset + k
            if gid in bad_idx:
                clean_mask[gid] = False
                ang = rng.uniform(0, 2 * np.pi)
                mag = rng.uniform(45.0, 95.0)
                shift = np.array([np.cos(ang), np.sin(ang)]) * mag
                p['pixels_b'][k] = np.clip(p['pixels_b'][k] + shift, 8.0, width - 8.0)
        offset += len(p['pixels_b'])
    return corrupted, clean_mask


def run_benchmark(name, images, masks, pairs, seed_fine, faces_fine, clean_track_mask, out_dir, use_ipc=False):
    results = {}
    renders_map = {}
    methods = ['adam', 'dca'] if use_ipc else ['adam', 'lbfgs', 'dca']

    for method in methods:
        s0 = copy.deepcopy(seed_fine)
        a0 = build_problem(
            images, masks, pairs, s0, faces_fine, use_ipc=use_ipc,
            cauchy_scale=3.0 if method == 'dca' else None,
        )
        t0 = time.perf_counter()
        if method == 'adam':
            res = optimize(
                s0, a0, steps=900, verbose=False, objective_fn=objective, step_fn=step, valid_fn=valid,
                learning_rates=dict(X=.003, C=.002, g=.0006, lf=.0008, pp=.0005, h=.001),
            )
            eval_a = a0
        elif method == 'lbfgs':
            res = optimize_lbfgs(
                s0, a0, steps=450, verbose=False, objective_fn=objective, step_fn=step, valid_fn=valid,
            )
            eval_a = a0
        else:
            res = optimize_dca(
                s0, a0, steps=140, verbose=False, objective_fn=objective, step_fn=step, valid_fn=valid,
                cauchy_schedule=(30., 10., 3.),
            )
            eval_a = res['a']
        elapsed = round(time.perf_counter() - t0, 2)
        metrics, renders = evaluate_metrics(res['state'], eval_a, images, masks, clean_track_mask)
        metrics.update(
            status=res['status'],
            steps=res['steps'],
            seconds=elapsed,
            visibility_violations=int(res['stats']['visibility_violations']),
            monotone=bool(res['optimization']['monotone']),
        )
        results[method] = metrics
        renders_map[method] = renders
        print(f'[{name}] {method:5s}: {metrics}', flush=True)

    # Save visual comparison strip
    cols = len(methods) + 1
    rows = len(images)
    fig, axs = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows), squeeze=False)
    for r in range(rows):
        axs[r, 0].imshow(images[r]); axs[r, 0].set_title(f'Input {r+1}'); axs[r, 0].axis('off')
        for c, m in enumerate(methods, 1):
            axs[r, c].imshow(renders_map[m][r])
            st = results[m]
            axs[r, c].set_title(f'{m.upper()} | IoU {st["mask_IoU"][r]}%\ncleanRMS {st["clean_rms_px"]}px')
            axs[r, c].axis('off')
    fig.suptitle(f'Pha 1 Study: {name}', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f'{name}_comparison.png', dpi=140)
    plt.close(fig)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path(r'C:\doan\outputs\dca_study'))
    args = p.parse_args()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(r'C:\doan\reconstruction_dca_3d\raw_inputs')
    images, masks, _ = load_images([raw_dir / 'view_01.webp', raw_dir / 'view_02.webp'], size=384)
    pairs, finfo = feature_pairs(images, masks)
    seed0, faces0, _ = initialize(images, masks, pairs)
    seed0['h'] = np.eye(3)

    # Run a clean 80-step coarse warmup so all solvers start from the exact same valid 162-vertex seed
    a_warm = build_problem(images, masks, pairs, seed0, faces0, use_ipc=True, cauchy_scale=30.)
    warm = optimize_dca(seed0, a_warm, steps=80, verbose=False, objective_fn=objective, step_fn=step, valid_fn=valid, cauchy_schedule=(30., 10.))
    seed_fine = copy.deepcopy(warm['state'])
    seed_fine['X'], faces_fine = subdivide(seed_fine['X'], faces0)

    # 1. Clean benchmark (no IPC -> compares Adam vs L-BFGS-B vs Joint DCA)
    clean_mask = np.ones(sum(len(p['pixels_a']) for p in pairs), dtype=bool)
    res_clean_noipc = run_benchmark('clean_raw_inputs_smooth', images, masks, pairs, seed_fine, faces_fine, clean_mask, out_dir, use_ipc=False)

    # 2. Clean benchmark with IPC CCD (Adam vs Joint DCA)
    res_clean_ipc = run_benchmark('clean_raw_inputs_ipc', images, masks, pairs, seed_fine, faces_fine, clean_mask, out_dir, use_ipc=True)

    # 3. 25% Gross Outlier Stress Test (14/56 tracks shifted 45-95 px)
    bad_pairs, bad_mask = inject_outliers(pairs, 384, fraction=0.25, seed=42)
    res_outliers_smooth = run_benchmark('outliers_25pct_smooth', images, masks, bad_pairs, seed_fine, faces_fine, bad_mask, out_dir, use_ipc=False)
    res_outliers_ipc = run_benchmark('outliers_25pct_ipc', images, masks, bad_pairs, seed_fine, faces_fine, bad_mask, out_dir, use_ipc=True)

    summary = dict(
        clean_smooth=res_clean_noipc,
        clean_ipc=res_clean_ipc,
        outliers_25pct_smooth=res_outliers_smooth,
        outliers_25pct_ipc=res_outliers_ipc,
        feature_info=finfo,
    )
    (out_dir / 'dca_study.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Saved DCA study to', out_dir / 'dca_study.json', flush=True)


if __name__ == '__main__':
    main()
