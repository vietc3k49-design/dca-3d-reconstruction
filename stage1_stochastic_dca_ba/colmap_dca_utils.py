#!/usr/bin/env python3
"""Portable COLMAP <-> C++ Stochastic DCA-BA I/O, Sim(3) alignment, and evaluation utilities."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOLVER_BIN = str(Path(__file__).resolve().parent / "cpp_solver" / "stochastic_dca_ryzen5.exe")
COLMAP_EXE = os.environ.get("COLMAP_EXE") or shutil.which("colmap") or "colmap"
BRUSH_EXE = os.environ.get("BRUSH_EXE") or shutil.which("brush_app") or "brush_app"
SOLVER_BIN = os.environ.get("SDCA_SOLVER_BIN") or DEFAULT_SOLVER_BIN


def run_step(name: str, cmd: list[str], log_path: Path, env: dict | None = None) -> tuple[float, str]:
    print(f"\n[Pipeline] >>> Running {name} ...", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    log_path.write_text(combined + "\n", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed (exit={proc.returncode}):\n{combined[-2000:]}")
    print(f"[Pipeline] <<< Finished {name} in {elapsed:.2f}s", flush=True)
    return elapsed, combined


def colmap_qvec_to_rotmat(qvec) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def rotmat_to_colmap_qvec(R: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(R).as_quat()
    q = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def parse_colmap_txt(txt_dir: Path):
    cameras = {}
    for ln in (txt_dir / "cameras.txt").read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        cid = int(toks[0])
        cameras[cid] = {
            "model": toks[1],
            "width": int(toks[2]),
            "height": int(toks[3]),
            "params": np.array([float(x) for x in toks[4:]], dtype=np.float64),
        }

    raw_lines = [
        ln.rstrip("\r\n")
        for ln in (txt_dir / "images.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    images = {}
    for i in range(0, len(raw_lines), 2):
        toks = raw_lines[i].split()
        img_id = int(toks[0])
        qvec = [float(x) for x in toks[1:5]]
        tvec = np.array([float(x) for x in toks[5:8]], dtype=np.float64)
        Rk = colmap_qvec_to_rotmat(qvec)
        Ck = -Rk.T @ tvec
        p2_toks = raw_lines[i + 1].split()
        pts2d = [(float(p2_toks[m]), float(p2_toks[m + 1]), int(p2_toks[m + 2])) for m in range(0, len(p2_toks), 3)]
        images[img_id] = {"R": Rk, "C": Ck, "camera_id": int(toks[8]), "name": toks[9], "points2D": pts2d}

    points3D = {}
    for ln in (txt_dir / "points3D.txt").read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        pid = int(toks[0])
        xyz = np.array([float(toks[1]), float(toks[2]), float(toks[3])], dtype=np.float64)
        rgb = np.array([int(toks[4]), int(toks[5]), int(toks[6])], dtype=np.uint8)
        tr_toks = toks[8:]
        track = [(int(tr_toks[m]), int(tr_toks[m + 1])) for m in range(0, len(tr_toks), 2)]
        points3D[pid] = {"xyz": xyz, "color": rgb, "error": float(toks[7]), "track": track}

    return cameras, images, points3D


def similarity(a: np.ndarray, b: np.ndarray, scale: bool = True):
    ac = a - a.mean(0)
    bc = b - b.mean(0)
    U, S, Vt = np.linalg.svd(bc.T @ ac / len(a))
    D = np.ones(3)
    D[-1] = np.linalg.det(U @ Vt)
    Q = U @ np.diag(D) @ Vt
    s = float((S * D).sum() / np.mean(np.sum(ac * ac, axis=1))) if scale else 1.0
    t = b.mean(0) - s * Q @ a.mean(0)
    return s, Q, t


def transform_sim3(R: np.ndarray, C: np.ndarray, X: np.ndarray, s: float, Q: np.ndarray, t: np.ndarray):
    return R @ Q.T, s * (C @ Q.T) + t, s * (X @ Q.T) + t


def project(R: np.ndarray, C: np.ndarray, X: np.ndarray, ci: np.ndarray, pi: np.ndarray, K: np.ndarray):
    q = np.einsum("nij,nj->ni", R[ci], X[pi] - C[ci])
    kk = K[ci]
    uv = q[:, :2] / q[:, 2, None] * kk[:, :2] + kk[:, 2:]
    return uv, q


def eval_vs_gt(R: np.ndarray, C: np.ndarray, Rgt: np.ndarray, Cgt: np.ndarray) -> dict:
    s, Q, t = similarity(C, Cgt, scale=True)
    R_al = R @ Q.T
    C_al = s * (C @ Q.T) + t
    c_err = np.linalg.norm(C_al - Cgt, axis=1)
    rot_errs = [
        float(np.degrees(np.arccos(np.clip((float(np.trace(Rgt[k] @ R_al[k].T)) - 1.0) / 2.0, -1.0, 1.0))))
        for k in range(len(R))
    ]
    rot_arr = np.array(rot_errs)
    return {
        "sim3_scale": float(s),
        "center_rmse": float(np.sqrt(np.mean(c_err ** 2))),
        "center_max": float(c_err.max()),
        "rot_mean_deg": float(rot_arr.mean()),
        "rot_median_deg": float(np.median(rot_arr)),
        "rot_max_deg": float(rot_arr.max()),
    }


def write_sdcabin1(d: dict, path: Path, unit: float = 1.0) -> None:
    with path.open("wb") as f:
        f.write(b"SDCABIN1")
        f.write(struct.pack("<5I", len(d["R"]), len(d["X"]), len(d["ci"]), len(d["ci_test"]), len(d["anchor_idx"])))
        for k in ["R", "C", "X", "K"]:
            f.write(np.asarray(d[k], dtype="<f8").tobytes())
        for k in ["ci", "pi"]:
            f.write(np.asarray(d[k], dtype="<u4").tobytes())
        f.write(np.asarray(d["uv"], dtype="<f8").tobytes())
        for k in ["Rgt", "Cgt"]:
            f.write(np.asarray(d[k], dtype="<f8").tobytes())
        for k in ["ci_test", "pi_test"]:
            f.write(np.asarray(d[k], dtype="<u4").tobytes())
        f.write(np.asarray(d["uv_test"], dtype="<f8").tobytes())
        f.write(np.asarray(d["anchor_idx"], dtype="<u4").tobytes())
        f.write(np.asarray(d["anchor_xyz"], dtype="<f8").tobytes())
        f.write(struct.pack("<d", unit))


def read_sdca_state(path: Path):
    with path.open("rb") as f:
        magic = f.read(8)
        if magic != b"SDCAOUT1":
            raise ValueError(f"Unexpected magic header in {path}: {magic!r}")
        nc, np_pts = np.fromfile(f, dtype="<u4", count=2)
        nc, np_pts = int(nc), int(np_pts)
        data = np.fromfile(f, dtype="<f8")
    return data[: nc * 9].reshape(nc, 3, 3), data[nc * 9 : nc * 12].reshape(nc, 3), data[nc * 12 :].reshape(np_pts, 3)


def write_colmap_sparse_txt(out_dir, R, C, X, K, image_size, names, colors, ci, pi, uv, err_tr_norms_px) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    nc, npnt, nobs = len(R), len(X), len(ci)
    w, h = int(image_size[0]), int(image_size[1])
    fx, fy, cx, cy = [float(v) for v in K[0]]
    with (out_dir / "cameras.txt").open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write(f"1 PINHOLE {w} {h} {fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}\n")

    tracks = [[] for _ in range(npnt)]
    pt_err_lists = [[] for _ in range(npnt)]
    with (out_dir / "images.txt").open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Image list with two lines of data per image:\n")
        for k in range(nc):
            Rk = R[k]
            qvec = rotmat_to_colmap_qvec(Rk)
            tvec = -Rk @ C[k]
            f.write(
                f"{k + 1} {qvec[0]:.17g} {qvec[1]:.17g} {qvec[2]:.17g} {qvec[3]:.17g} "
                f"{tvec[0]:.17g} {tvec[1]:.17g} {tvec[2]:.17g} 1 {names[k]}\n"
            )
            js = np.flatnonzero(ci == k)
            pts_tokens = []
            for p2d_idx, obs_idx in enumerate(js):
                pt_idx = int(pi[obs_idx])
                pts_tokens.append(f"{float(uv[obs_idx, 0]):.17g} {float(uv[obs_idx, 1]):.17g} {pt_idx + 1}")
                tracks[pt_idx].append((k + 1, p2d_idx))
                pt_err_lists[pt_idx].append(float(err_tr_norms_px[obs_idx]))
            f.write(" ".join(pts_tokens) + "\n")

    with (out_dir / "points3D.txt").open("w", encoding="utf-8", newline="\n") as f:
        f.write("# 3D point list with one line of data per point:\n")
        for j in range(npnt):
            x, y, z = float(X[j, 0]), float(X[j, 1]), float(X[j, 2])
            r, g, b = int(colors[j, 0]), int(colors[j, 1]), int(colors[j, 2])
            pt_err = float(np.mean(pt_err_lists[j])) if pt_err_lists[j] else 0.0
            tr_str = " ".join(f"{img_id} {p2d_idx}" for img_id, p2d_idx in tracks[j])
            f.write(f"{j + 1} {x:.17g} {y:.17g} {z:.17g} {r} {g} {b} {pt_err:.10f} {tr_str}\n")
    return {"cameras": 1, "images": nc, "points3D": npnt, "observations": nobs}


def load_sketchfab_gt_for_names(meta_path: Path, names: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    by_name = {Path(fr["file_path"]).name: fr for fr in meta["frames"]}
    Rgt_list = [np.array(by_name[nm]["w2c_colmap_R"], dtype=np.float64) for nm in names]
    Cgt_list = [np.array(by_name[nm]["eye"], dtype=np.float64) for nm in names]
    return np.array(Rgt_list, dtype=np.float64), np.array(Cgt_list, dtype=np.float64), float(meta["frames"][0]["fl_x"])
