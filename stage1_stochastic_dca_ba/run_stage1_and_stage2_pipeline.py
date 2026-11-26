#!/usr/bin/env python3
"""
Step 2 & Step 3: Stage 1 (Paper 1: Blind COLMAP 4.2.1 CUDA + C++ Stochastic DCA-BA)
               + Stage 2 (Paper 2: Brush 3DGS 1536x1536 SH3 Surface-Rich Initialization)
on the 36-view Multi-Scale 1536x1536 RGBA dataset.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_dca_utils import (
    BRUSH_EXE,
    COLMAP_EXE,
    SOLVER_BIN,
    eval_vs_gt,
    load_sketchfab_gt_for_names,
    parse_colmap_txt,
    project,
    read_sdca_state,
    run_step,
    similarity,
    transform_sim3,
    write_colmap_sparse_txt,
    write_sdcabin1,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("OUT_DIR", str(REPO_ROOT / "workspace_runs" / "eastern_swordsman_multiscale36")))
CAP_IMG_DIR = OUT_DIR / "capture_1536" / "images"
GT_META = OUT_DIR / "gt_eval" / "cameras_meta.json"

CLEAN_SCENE = OUT_DIR / "clean_colmap_scene"
SCENE_IMG_DIR = CLEAN_SCENE / "images"
GS_OUT = OUT_DIR / "gs_out"
LOG_DIR = OUT_DIR / "logs"

TARGET_RES = 1536


def inspect_ply_header(path: Path) -> dict:
    n_verts = 0
    with path.open("rb") as f:
        while True:
            ln = f.readline().decode("ascii", errors="replace").strip()
            if ln.startswith("element vertex "):
                n_verts = int(ln.split()[2])
            elif ln == "end_header":
                break
    return {"path": str(path), "bytes": path.stat().st_size, "n_vertices": n_verts}


def run_stage1_only() -> dict:
    t0 = time.perf_counter()
    for d in (OUT_DIR, CLEAN_SCENE, GS_OUT, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    pngs = sorted(CAP_IMG_DIR.glob("*.png"))
    assert len(pngs) == 36, f"Expected 36 PNGs in {CAP_IMG_DIR}, found {len(pngs)}"

    if SCENE_IMG_DIR.exists():
        shutil.rmtree(SCENE_IMG_DIR)
    SCENE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    for p in pngs:
        shutil.copy2(p, SCENE_IMG_DIR / p.name)

    # ---------------- STAGE 1a: COLMAP 4.2.1 CUDA (Blind Multi-Scale SfM) ----------------
    db_path = OUT_DIR / "database.db"
    s1a_sparse = OUT_DIR / "stage1a_sparse"
    s1a_txt = OUT_DIR / "stage1a_txt"
    for p in (s1a_sparse, s1a_txt):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    t_ext, _ = run_step(
        "Stage 1a.1: COLMAP CUDA SIFT first_octave=0 (1536x1536, 36 multi-scale views)",
        [
            COLMAP_EXE, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(SCENE_IMG_DIR),
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.single_camera", "1",
            "--FeatureExtraction.type", "SIFT",
            "--FeatureExtraction.use_gpu", "1",
            "--FeatureExtraction.num_threads", "-1",
            "--SiftExtraction.first_octave", "0",
            "--SiftExtraction.max_num_features", "8192",
            "--SiftExtraction.peak_threshold", "0.004",
            "--default_random_seed", "42",
            "--log_target", "stderr",
        ],
        LOG_DIR / "01_feature_extractor.log",
    )
    t_mat, _ = run_step(
        "Stage 1a.2: COLMAP CUDA Guided Exhaustive Matcher (630 pairs)",
        [
            COLMAP_EXE, "exhaustive_matcher",
            "--database_path", str(db_path),
            "--FeatureMatching.guided_matching", "1",
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.num_threads", "-1",
            "--TwoViewGeometry.max_num_trials", "2000",
            "--TwoViewGeometry.random_seed", "42",
            "--default_random_seed", "42",
            "--log_target", "stderr",
        ],
        LOG_DIR / "02_exhaustive_matcher.log",
    )
    t_map, _ = run_step(
        "Stage 1a.3: COLMAP 12-Thread Mapper (36 multi-scale views)",
        [
            COLMAP_EXE, "mapper",
            "--database_path", str(db_path),
            "--image_path", str(SCENE_IMG_DIR),
            "--output_path", str(s1a_sparse),
            "--Mapper.num_threads", "-1",
            "--Mapper.multiple_models", "0",
            "--Mapper.extract_colors", "1",
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "0",
            "--Mapper.ba_local_num_images", "6",
            "--Mapper.ba_local_max_num_iterations", "12",
            "--Mapper.ba_local_max_refinements", "1",
            "--Mapper.ba_global_frames_ratio", "1.35",
            "--Mapper.ba_global_points_ratio", "1.35",
            "--Mapper.ba_global_max_num_iterations", "35",
            "--Mapper.ba_global_max_refinements", "2",
            "--Mapper.ba_global_ignore_redundant_points3D", "1",
            "--Mapper.abs_pose_max_error", "4.0",
            "--Mapper.abs_pose_min_inlier_ratio", "0.20",
            "--Mapper.random_seed", "42",
            "--default_random_seed", "42",
            "--log_target", "stderr",
        ],
        LOG_DIR / "03_mapper.log",
    )
    t_s1a = t_ext + t_mat + t_map

    model0 = s1a_sparse / "0"
    assert (model0 / "images.bin").exists(), "COLMAP mapper produced no sparse/0 model!"
    run_step(
        "COLMAP model_converter (BIN -> TXT)",
        [COLMAP_EXE, "model_converter", "--input_path", str(model0), "--output_path", str(s1a_txt), "--output_type", "TXT", "--log_target", "stderr"],
        LOG_DIR / "04_s1a_txt.log",
    )

    # ---------------- STAGE 1b: C++ STOCHASTIC DCA-BA (Paper 1) ----------------
    cameras, images, points3D = parse_colmap_txt(s1a_txt)
    n_reg = len(images)
    print(f"[Gate Check 1a] Registered {n_reg}/36 images, raw 3D points = {len(points3D)}", flush=True)
    if n_reg != 36:
        reg_names = {v["name"] for v in images.values()}
        missing = sorted({p.name for p in pngs} - reg_names)
        raise RuntimeError(f"COLMAP registered only {n_reg}/36 views! Missing: {missing}")

    kept_pids = sorted(pid for pid, pt in points3D.items() if len(pt["track"]) >= 3)
    ids = sorted(images.keys())
    idx_map = {v: i for i, v in enumerate(ids)}
    pidx_map = {v: i for i, v in enumerate(kept_pids)}

    R_list, C_list, K_list, names = [], [], [], []
    image_size = None
    for img_id in ids:
        im = images[img_id]
        cam = cameras[im["camera_id"]]
        if image_size is None:
            image_size = np.array([cam["width"], cam["height"]], dtype=np.int64)
        R_list.append(im["R"])
        C_list.append(im["C"])
        K_list.append(cam["params"])
        names.append(im["name"])

    R_raw = np.array(R_list, dtype=np.float64)
    C_raw = np.array(C_list, dtype=np.float64)
    K = np.array(K_list, dtype=np.float64)
    X_raw = np.array([points3D[pid]["xyz"] for pid in kept_pids], dtype=np.float64)
    colors = np.array([points3D[pid]["color"] for pid in kept_pids], dtype=np.uint8)

    obs = []
    for pid in kept_pids:
        for img_id, p2d_idx in points3D[pid]["track"]:
            u, v, _ = images[img_id]["points2D"][p2d_idx]
            obs.append((idx_map[img_id], pidx_map[pid], u, v))
    ob = np.array(obs, dtype=np.float64)
    ci, pi, uv = ob[:, 0].astype(np.int64), ob[:, 1].astype(np.int64), ob[:, 2:4].astype(np.float64)

    Rgt, Cgt, fx_gt = load_sketchfab_gt_for_names(GT_META, names)
    gt_eval_before = eval_vs_gt(R_raw, C_raw, Rgt, Cgt)

    s_sim, Q_sim, t_sim = similarity(C_raw, Cgt, scale=True)
    R, C, X = transform_sim3(R_raw, C_raw, X_raw, s_sim, Q_sim, t_sim)

    train = np.ones(len(ci), dtype=bool)
    for j in range(len(X)):
        js = np.flatnonzero(pi == j)
        if len(js) >= 5:
            train[js[(j * 7919 + 17) % len(js)]] = False
    cit, pit, uvt = ci[train], pi[train], uv[train]
    cie, pie, uve = ci[~train], pi[~train], uv[~train]

    prob_bin = OUT_DIR / "multiscale36_problem.bin"
    write_sdcabin1(
        dict(
            R=R, C=C, X=X, ci=cit, pi=pit, uv=uvt, K=K,
            Rgt=Rgt, Cgt=Cgt, ci_test=cie, pi_test=pie, uv_test=uve,
            anchor_idx=np.zeros((0,), dtype=np.uint32),
            anchor_xyz=np.zeros((0, 3), dtype=np.float64),
        ),
        prob_bin,
        unit=1.0,
    )

    dca_json = OUT_DIR / "dca_result.json"
    dca_state = OUT_DIR / "dca_state.bin"
    env = os.environ.copy()
    for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[k] = "1"

    t_dca_wall, _ = run_step(
        "Stage 1b: stochastic_dca_ryzen5.exe (--full-reuse --no-warmup, Paper 1)",
        [
            SOLVER_BIN,
            "--input", str(prob_bin),
            "--output", str(dca_json),
            "--state", str(dca_state),
            "--anchors", "0",
            "--seed", "11",
            "--full-reuse",
            "--no-warmup",
        ],
        LOG_DIR / "05_dca_full_reuse.log",
        env=env,
    )
    dca_meta = json.loads(dca_json.read_text(encoding="utf-8"))
    R_out, C_out, X_out = read_sdca_state(dca_state)
    s_post, Q_post, t_post = similarity(C_out, Cgt, scale=True)
    R_out_al, C_out_al, X_out_al = transform_sim3(R_out, C_out, X_out, s_post, Q_post, t_post)
    gt_eval_final = eval_vs_gt(R_out, C_out, Rgt, Cgt)

    pred_tr_in, _ = project(R, C, X, cit, pit, K)
    pred_tr_out, _ = project(R_out_al, C_out_al, X_out_al, cit, pit, K)
    err_tr_in = np.linalg.norm(pred_tr_in - uvt, axis=1)
    err_tr_out = np.linalg.norm(pred_tr_out - uvt, axis=1)

    sparse_txt0 = OUT_DIR / "sparse_txt" / "0"
    clean_sparse0 = CLEAN_SCENE / "sparse" / "0"
    for p in (sparse_txt0, clean_sparse0):
        if p.exists():
            shutil.rmtree(p)

    write_colmap_sparse_txt(sparse_txt0, R_out_al, C_out_al, X_out_al, K, image_size, names, colors, cit, pit, uvt, err_tr_out)
    clean_sparse0.mkdir(parents=True, exist_ok=True)
    run_step(
        "COLMAP model_converter (TXT -> BIN in clean_colmap_scene/sparse/0)",
        [COLMAP_EXE, "model_converter", "--input_path", str(sparse_txt0), "--output_path", str(clean_sparse0), "--output_type", "BIN", "--log_target", "stderr"],
        LOG_DIR / "06_export_clean_bin.log",
    )
    for bad in CLEAN_SCENE.glob("transforms*.json"):
        bad.unlink()

    focal_err_pct = float(100.0 * abs(K[0, 0] - fx_gt) / fx_gt)
    s1_summary = {
        "registered_images": f"{n_reg}/36",
        "clean_points3D_track_ge_3": int(len(X)),
        "fx_est_px": round(float(K[0, 0]), 4),
        "fx_gt_1536_px": round(float(fx_gt), 4),
        "focal_error_pct": round(focal_err_pct, 4),
        "stage1a_rot_mean_deg": round(float(gt_eval_before["rot_mean_deg"]), 5),
        "stage1b_rot_mean_deg": round(float(gt_eval_final["rot_mean_deg"]), 5),
        "stage1b_cam_rmse_mm": round(float(gt_eval_final.get("center_rmse", 0.0) * 1000.0), 3),
        "reproj_mean_px_before": round(float(err_tr_in.mean()), 4),
        "reproj_mean_px_after": round(float(err_tr_out.mean()), 4),
        "stage1a_colmap_s": round(t_s1a, 3),
        "stage1b_dca_internal_s": round(float(dca_meta["seconds"]), 4),
        "stage1b_dca_wall_s": round(t_dca_wall, 3),
        "dca_steps_reused": f"{dca_meta.get('steps_reusing_factorization', 0)}/{dca_meta.get('accepted', 0)}",
    }
    print("[Gate Check 1b]", json.dumps(s1_summary, indent=2), flush=True)
    return s1_summary


def run_stage2_brush(total_steps: int = 4400) -> dict:
    if GS_OUT.exists():
        shutil.rmtree(GS_OUT)
    GS_OUT.mkdir(parents=True, exist_ok=True)

    cmd_brush = [
        BRUSH_EXE,
        str(CLEAN_SCENE),
        "--total-steps", str(total_steps),
        "--growth-stop-iter", str(int(total_steps * 0.72)),
        "--refine-every", "75",
        "--growth-grad-threshold", "0.0000070",
        "--growth-select-fraction", "0.34",
        "--lr-mean", "2.6e-5",
        "--lr-mean-end", "7.0e-7",
        "--lr-scale-end", "3.0e-3",
        "--lr-coeffs-dc", "2.8e-3",
        "--sh-degree", "3",
        "--max-splats", "280000",
        "--max-resolution", str(TARGET_RES),
        "--match-alpha-weight", "0.25",
        "--export-every", str(total_steps),
        "--export-path", str(GS_OUT),
        "--export-name", "multiscale36_splats_{iter}.ply",
    ]
    print(f"[Stage 2 GPU] Running Brush 3DGS ({total_steps} steps, {TARGET_RES}x{TARGET_RES}, SH3, max 280k splats)...", flush=True)
    t_b0 = time.perf_counter()
    proc = subprocess.run(cmd_brush, capture_output=True, text=True)
    t_brush = time.perf_counter() - t_b0
    (LOG_DIR / f"07_brush_{total_steps}.log").write_text((proc.stdout + "\n" + proc.stderr).strip() + "\n", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"brush_app.exe failed (rc={proc.returncode}): {proc.stderr[-800:]}")

    ply_out = GS_OUT / f"multiscale36_splats_{total_steps}.ply"
    ply_info = inspect_ply_header(ply_out)
    return {
        "stage2_brush_s": round(t_brush, 2),
        "total_steps": total_steps,
        "ply_path": str(ply_out),
        "ply_splats": ply_info["n_vertices"],
        "ply_mb": round(ply_info["bytes"] / 1e6, 2),
    }


def main() -> int:
    t0 = time.perf_counter()
    s1 = run_stage1_only()
    s2 = run_stage2_brush(total_steps=4400)
    summary = {
        "pipeline": "Multi-Scale 36-View (1536x1536 RGBA) Stage 1 (Paper 1) + Stage 2 (Paper 2)",
        "stage1": s1,
        "stage2": s2,
        "total_stage12_s": round(time.perf_counter() - t0, 2),
    }
    (OUT_DIR / "stage12_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 72, flush=True)
    print("[STAGE 1 + STAGE 2 COMPLETE]", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
