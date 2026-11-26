#!/usr/bin/env python3
"""
Export captured Sketchfab multi-view datasets (cameras_meta.json + images/)
into ready-to-reconstruct COLMAP sparse priors (cameras.txt, images.txt, points3D.txt)
and optionally validate SIFT feature extraction via COLMAP CLI.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import os, shutil
DEFAULT_COLMAP_EXE = os.environ.get("COLMAP_EXE") or shutil.which("colmap") or "colmap"


def rotmat_to_colmap_qvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 world-to-camera rotation matrix R into COLMAP quaternion [QW, QX, QY, QZ]."""
    xyzw = Rotation.from_matrix(R).as_quat()  # scipy returns [x, y, z, w]
    q = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError(f"Invalid quaternion norm: {norm}")
    q /= norm
    if q[0] < 0.0:
        q = -q
    return q


def colmap_qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion [QW, QX, QY, QZ] back to 3x3 rotation matrix."""
    qw, qx, qy, qz = [float(v) for v in qvec]
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def resolve_colmap_image_name(file_path_str: str) -> str:
    """Normalize frame file_path (e.g. 'images/view_000.png') to path relative to <res-dir>/images."""
    norm = file_path_str.replace("\\", "/").lstrip("./")
    if norm.startswith("images/"):
        return norm[len("images/") :]
    return Path(norm).name


def parse_model_analyzer(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    patterns: dict[str, tuple[str, Any]] = {
        "rigs": (r"Rigs:\s*(\d+)", int),
        "cameras": (r"Cameras:\s*(\d+)", int),
        "frames": (r"Frames:\s*(\d+)", int),
        "registered_frames": (r"Registered frames:\s*(\d+)", int),
        "images": (r"Images:\s*(\d+)", int),
        "registered_images": (r"Registered images:\s*(\d+)", int),
        "points": (r"Points:\s*(\d+)", int),
        "observations": (r"Observations:\s*(\d+)", int),
    }
    for key, (pat, cast_fn) in patterns.items():
        m = re.search(pat, text)
        if m:
            parsed[key] = cast_fn(m.group(1))
    return parsed


def export_and_validate_priors(
    res_dir: Path,
    out_subdir: str = "sparse_gt",
    colmap_exe: Path = Path(DEFAULT_COLMAP_EXE),
    check_sift: bool = False,
    max_num_features: int = 8192,
    num_threads: int = 6,
    use_gpu: int = 0,
) -> dict[str, Any]:
    res_dir = res_dir.resolve()
    meta_path = res_dir / "cameras_meta.json"
    images_dir = res_dir / "images"

    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing cameras_meta.json at: {meta_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory at: {images_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames = meta.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in {meta_path}")

    meta_res = meta.get("resolution")
    expected_res = (int(meta_res[0]), int(meta_res[1])) if meta_res and len(meta_res) == 2 else None

    # Camera deduplication list: (width, height, fl_x, fl_y, cx, cy)
    cameras_list: list[tuple[int, int, float, float, float, float]] = []
    image_entries: list[dict[str, Any]] = []

    det_values: list[float] = []
    det_abs_errs: list[float] = []
    ortho_errs: list[float] = []
    eye_errs: list[float] = []
    qvec_roundtrip_errs: list[float] = []
    target_depths_z: list[float] = []

    seen_names: set[str] = set()

    for idx, fr in enumerate(frames):
        rel_file_path = str(fr["file_path"])
        img_path = res_dir / rel_file_path
        if not img_path.is_file():
            raise FileNotFoundError(f"Frame {idx}: image file does not exist: {img_path}")

        w = int(fr["width"])
        h = int(fr["height"])
        if expected_res is not None and (w, h) != expected_res:
            raise ValueError(
                f"Frame {idx} ({rel_file_path}): metadata width/height {(w, h)} != resolution {expected_res}"
            )

        with Image.open(img_path) as im:
            actual_w, actual_h = im.size
        if (actual_w, actual_h) != (w, h):
            raise ValueError(
                f"Frame {idx} ({rel_file_path}): actual PNG dimensions {(actual_w, actual_h)} != expected {(w, h)}"
            )

        fl_x = float(fr["fl_x"])
        fl_y = float(fr["fl_y"])
        cx = float(fr["cx"])
        cy = float(fr["cy"])
        if not (np.isfinite([fl_x, fl_y, cx, cy]).all() and fl_x > 0.0 and fl_y > 0.0):
            raise ValueError(f"Frame {idx}: invalid intrinsics fl_x={fl_x}, fl_y={fl_y}, cx={cx}, cy={cy}")

        R = np.asarray(fr["w2c_colmap_R"], dtype=np.float64)
        t = np.asarray(fr["w2c_colmap_t"], dtype=np.float64)
        eye = np.asarray(fr["eye"], dtype=np.float64)
        target = np.asarray(fr["target"], dtype=np.float64)

        if R.shape != (3, 3) or t.shape != (3,) or eye.shape != (3,) or target.shape != (3,):
            raise ValueError(
                f"Frame {idx}: unexpected pose shapes R={R.shape}, t={t.shape}, eye={eye.shape}, target={target.shape}"
            )
        if not (np.isfinite(R).all() and np.isfinite(t).all() and np.isfinite(eye).all() and np.isfinite(target).all()):
            raise ValueError(f"Frame {idx}: non-finite values in R, t, eye, or target")

        # 1. Validate orthogonality and det(R) == +1
        ortho_err = float(np.max(np.abs(R @ R.T - np.eye(3))))
        det_r = float(np.linalg.det(R))
        det_err = abs(det_r - 1.0)
        if ortho_err > 1e-7:
            raise ValueError(f"Frame {idx}: R is not orthogonal (max |R R^T - I| = {ortho_err:.3e})")
        if det_err > 1e-7:
            raise ValueError(f"Frame {idx}: det(R) != +1 (det(R) = {det_r:.12f}, |det-1| = {det_err:.3e})")

        # 2. Validate camera center consistency: -R^T @ t == eye
        recovered_eye = -R.T @ t
        eye_err = float(np.linalg.norm(recovered_eye - eye))
        if eye_err > 1e-7:
            raise ValueError(
                f"Frame {idx}: -R^T @ t != eye (L2 error = {eye_err:.3e}, got {recovered_eye}, expected {eye})"
            )

        # 3. Validate cheirality: target is in front of camera (+Z in COLMAP camera coordinates)
        target_cam = R @ target + t
        if float(target_cam[2]) <= 0.0:
            raise ValueError(f"Frame {idx}: look-at target is not in front of camera (z_cam = {target_cam[2]:.6f})")

        # 4. Convert R -> quaternion [QW, QX, QY, QZ] and verify round-trip
        qvec = rotmat_to_colmap_qvec(R)
        R_rt = colmap_qvec_to_rotmat(qvec)
        q_rt_err = float(np.max(np.abs(R_rt - R)))
        if q_rt_err > 1e-9:
            raise ValueError(f"Frame {idx}: quaternion round-trip error too large ({q_rt_err:.3e})")

        det_values.append(det_r)
        det_abs_errs.append(det_err)
        ortho_errs.append(ortho_err)
        eye_errs.append(eye_err)
        qvec_roundtrip_errs.append(q_rt_err)
        target_depths_z.append(float(target_cam[2]))

        # Assign or reuse PINHOLE camera ID
        cam_id = None
        for cid_idx, (cw, ch, cfx, cfy, ccx, ccy) in enumerate(cameras_list, start=1):
            if (
                cw == w
                and ch == h
                and abs(cfx - fl_x) <= 1e-7
                and abs(cfy - fl_y) <= 1e-7
                and abs(ccx - cx) <= 1e-7
                and abs(ccy - cy) <= 1e-7
            ):
                cam_id = cid_idx
                break
        if cam_id is None:
            cameras_list.append((w, h, fl_x, fl_y, cx, cy))
            cam_id = len(cameras_list)

        colmap_name = resolve_colmap_image_name(rel_file_path)
        if colmap_name in seen_names:
            raise ValueError(f"Duplicate image name in COLMAP export: {colmap_name}")
        seen_names.add(colmap_name)

        image_entries.append(
            {
                "image_id": idx + 1,
                "qvec": qvec,
                "tvec": t,
                "camera_id": cam_id,
                "name": colmap_name,
            }
        )

    # Write <res-dir>/<out_subdir>/{cameras.txt, images.txt, points3D.txt}
    sparse_dir = res_dir / out_subdir
    sparse_dir.mkdir(parents=True, exist_ok=True)

    cameras_txt_path = sparse_dir / "cameras.txt"
    images_txt_path = sparse_dir / "images.txt"
    points3d_txt_path = sparse_dir / "points3D.txt"

    with cameras_txt_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras_list)}\n")
        for cid_idx, (cw, ch, cfx, cfy, ccx, ccy) in enumerate(cameras_list, start=1):
            f.write(f"{cid_idx} PINHOLE {cw} {ch} {cfx:.17g} {cfy:.17g} {ccx:.17g} {ccy:.17g}\n")

    with images_txt_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_entries)}, mean observations per image: 0\n")
        for entry in image_entries:
            qv = entry["qvec"]
            tv = entry["tvec"]
            f.write(
                f"{entry['image_id']} "
                f"{qv[0]:.17g} {qv[1]:.17g} {qv[2]:.17g} {qv[3]:.17g} "
                f"{tv[0]:.17g} {tv[1]:.17g} {tv[2]:.17g} "
                f"{entry['camera_id']} {entry['name']}\n"
            )
            f.write("\n")  # Mandatory second line per image (empty POINTS2D[])

    with points3d_txt_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0, mean track length: 0\n")

    report: dict[str, Any] = {
        "res_dir": str(res_dir),
        "sparse_gt_dir": str(sparse_dir),
        "num_frames": len(image_entries),
        "num_cameras": len(cameras_list),
        "camera_model": "PINHOLE",
        "cameras": [
            {
                "camera_id": cid_idx,
                "width": cw,
                "height": ch,
                "fl_x": cfx,
                "fl_y": cfy,
                "cx": ccx,
                "cy": ccy,
            }
            for cid_idx, (cw, ch, cfx, cfy, ccx, ccy) in enumerate(cameras_list, start=1)
        ],
        "validation": {
            "all_image_dimensions_matched": True,
            "det_R_min": float(np.min(det_values)),
            "det_R_max": float(np.max(det_values)),
            "max_abs_det_R_minus_1": float(np.max(det_abs_errs)),
            "max_orthogonality_err": float(np.max(ortho_errs)),
            "max_eye_reconstruction_l2_err": float(np.max(eye_errs)),
            "max_qvec_roundtrip_err": float(np.max(qvec_roundtrip_errs)),
            "target_depth_z_min": float(np.min(target_depths_z)),
            "target_depth_z_max": float(np.max(target_depths_z)),
        },
        "files_written": {
            "cameras_txt": str(cameras_txt_path),
            "images_txt": str(images_txt_path),
            "points3D_txt": str(points3d_txt_path),
        },
    }

    # Verify exported sparse_gt with colmap model_analyzer when colmap.exe is present
    if colmap_exe.is_file():
        proc_an = subprocess.run(
            [str(colmap_exe), "model_analyzer", "--path", str(sparse_dir), "--log_target", "stderr"],
            capture_output=True,
            text=True,
        )
        an_text = (proc_an.stdout + "\n" + proc_an.stderr).strip()
        if proc_an.returncode != 0:
            raise RuntimeError(f"colmap model_analyzer failed on {sparse_dir}:\n{an_text}")
        report["colmap_model_analyzer"] = parse_model_analyzer(an_text)

    # Optional SIFT feature extraction check into a temporary SQLite database
    if check_sift:
        if not colmap_exe.is_file():
            print(f"[WARN] --check-sift requested, but COLMAP executable not found at: {colmap_exe}", file=sys.stderr)
            report["sift_check"] = {"executed": False, "reason": f"colmap.exe not found at {colmap_exe}"}
        else:
            with tempfile.TemporaryDirectory(prefix="colmap_sift_check_") as tmp_dir:
                tmp_db = Path(tmp_dir) / "sift_check.db"
                cmd = [
                    str(colmap_exe),
                    "feature_extractor",
                    "--database_path",
                    str(tmp_db),
                    "--image_path",
                    str(images_dir),
                    "--ImageReader.camera_model",
                    "PINHOLE",
                    "--FeatureExtraction.type",
                    "SIFT",
                    "--FeatureExtraction.use_gpu",
                    str(use_gpu),
                    "--FeatureExtraction.num_threads",
                    str(num_threads),
                    "--SiftExtraction.max_num_features",
                    str(max_num_features),
                    "--default_random_seed",
                    "42",
                    "--log_target",
                    "stderr",
                ]
                if len(cameras_list) == 1:
                    _, _, cfx, cfy, ccx, ccy = cameras_list[0]
                    cmd.extend(
                        [
                            "--ImageReader.single_camera",
                            "1",
                            "--ImageReader.camera_params",
                            f"{cfx:.12g},{cfy:.12g},{ccx:.12g},{ccy:.12g}",
                        ]
                    )
                else:
                    cmd.extend(["--ImageReader.single_camera", "0"])

                t0 = time.perf_counter()
                proc_sift = subprocess.run(cmd, capture_output=True, text=True)
                elapsed_s = time.perf_counter() - t0
                sift_log = (proc_sift.stdout + "\n" + proc_sift.stderr).strip()
                if proc_sift.returncode != 0:
                    raise RuntimeError(
                        f"colmap feature_extractor failed (exit={proc_sift.returncode}):\n{sift_log[-2000:]}"
                    )

                con = sqlite3.connect(str(tmp_db))
                try:
                    rows = con.execute(
                        """
                        SELECT i.name, COALESCE(k.rows, 0) AS num_kpts
                        FROM images i
                        LEFT JOIN keypoints k ON i.image_id = k.image_id
                        ORDER BY i.name ASC
                        """
                    ).fetchall()
                finally:
                    con.close()

                db_names = {r[0] for r in rows}
                expected_names = {e["name"] for e in image_entries}
                if db_names != expected_names:
                    missing = sorted(expected_names - db_names)
                    extra = sorted(db_names - expected_names)
                    raise RuntimeError(
                        f"SQLite image names mismatch vs images.txt: missing={missing}, extra={extra}"
                    )

                kpt_counts = np.array([int(r[1]) for r in rows], dtype=np.int64)
                min_idx = int(np.argmin(kpt_counts))
                max_idx = int(np.argmax(kpt_counts))

                report["sift_check"] = {
                    "executed": True,
                    "colmap_exe": str(colmap_exe),
                    "elapsed_seconds": round(elapsed_s, 4),
                    "max_num_features_limit": max_num_features,
                    "num_images_extracted": int(len(rows)),
                    "total_keypoints": int(kpt_counts.sum()),
                    "avg_keypoints_per_image": float(kpt_counts.mean()),
                    "median_keypoints_per_image": float(np.median(kpt_counts)),
                    "std_keypoints_per_image": float(kpt_counts.std()),
                    "min_keypoints": int(kpt_counts.min()),
                    "min_keypoints_image": rows[min_idx][0],
                    "max_keypoints": int(kpt_counts.max()),
                    "max_keypoints_image": rows[max_idx][0],
                    "per_image_keypoints": {r[0]: int(r[1]) for r in rows},
                }

    summary_json_path = res_dir / "colmap_priors_summary.json"
    summary_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["summary_json"] = str(summary_json_path)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert captured Sketchfab dataset (<res-dir>/cameras_meta.json) into COLMAP sparse_gt priors."
    )
    p.add_argument(
        "--res-dir",
        type=str,
        required=True,
        help="Path to resolution folder containing cameras_meta.json and images/ (e.g. .../res_1280x720).",
    )
    p.add_argument(
        "--out-subdir",
        type=str,
        default="sparse_gt",
        help="Output subdirectory inside --res-dir for COLMAP text priors (default: sparse_gt).",
    )
    p.add_argument(
        "--check-sift",
        action="store_true",
        help="Run COLMAP SIFT feature_extractor into a temp SQLite DB and report average keypoints per image.",
    )
    p.add_argument(
        "--colmap-exe",
        type=str,
        default=DEFAULT_COLMAP_EXE,
        help=f"Path to colmap executable (default: {DEFAULT_COLMAP_EXE}).",
    )
    p.add_argument(
        "--max-num-features",
        type=int,
        default=8192,
        help="SiftExtraction.max_num_features when --check-sift is enabled (default: 8192).",
    )
    p.add_argument(
        "--num-threads",
        type=int,
        default=6,
        help="FeatureExtraction.num_threads when --check-sift is enabled (default: 6).",
    )
    p.add_argument(
        "--use-gpu",
        type=int,
        choices=[0, 1],
        default=0,
        help="FeatureExtraction.use_gpu when --check-sift is enabled (default: 0 for headless stability).",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    res_dir = Path(args.res_dir)
    colmap_exe = Path(args.colmap_exe)

    report = export_and_validate_priors(
        res_dir=res_dir,
        out_subdir=args.out_subdir,
        colmap_exe=colmap_exe,
        check_sift=args.check_sift,
        max_num_features=args.max_num_features,
        num_threads=args.num_threads,
        use_gpu=args.use_gpu,
    )

    cam0 = report["cameras"][0]
    val = report["validation"]
    print("=" * 72)
    print(" COLMAP SPARSE PRIORS EXPORT & VALIDATION SUMMARY")
    print("=" * 72)
    print(f" - Resolution Dir  : {report['res_dir']}")
    print(f" - Output Priors   : {report['sparse_gt_dir']}")
    print(
        f" - Frames / Cams   : {report['num_frames']} frames, {report['num_cameras']} {report['camera_model']} camera "
        f"({cam0['width']}x{cam0['height']}, fx={cam0['fl_x']:.4f}, fy={cam0['fl_y']:.4f}, cx={cam0['cx']:.1f}, cy={cam0['cy']:.1f})"
    )
    print(
        f" - Pose Checks     : max|det(R)-1|={val['max_abs_det_R_minus_1']:.3e}, "
        f"max|RR^T-I|={val['max_orthogonality_err']:.3e}, "
        f"max||-R^T t - eye||={val['max_eye_reconstruction_l2_err']:.3e}"
    )
    print(
        f" - Target Depth +Z : [{val['target_depth_z_min']:.4f}, {val['target_depth_z_max']:.4f}]"
    )
    if "colmap_model_analyzer" in report:
        an = report["colmap_model_analyzer"]
        print(
            f" - Model Analyzer  : cameras={an.get('cameras')}, "
            f"registered_images={an.get('registered_images')}/{an.get('images')}, points={an.get('points')}"
        )
    if "sift_check" in report and report["sift_check"].get("executed"):
        sc = report["sift_check"]
        print(
            f" - SIFT Check      : avg={sc['avg_keypoints_per_image']:.2f} kpts/img "
            f"(median={sc['median_keypoints_per_image']:.1f}, min={sc['min_keypoints']} [{sc['min_keypoints_image']}], "
            f"max={sc['max_keypoints']} [{sc['max_keypoints_image']}], total={sc['total_keypoints']}, "
            f"time={sc['elapsed_seconds']:.2f}s)"
        )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
