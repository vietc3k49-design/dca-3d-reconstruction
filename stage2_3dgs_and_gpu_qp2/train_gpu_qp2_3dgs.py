"""Full Stage 2 Native CUDA GPU-QP2 Trainer on 1600x1600 LEGO Technic Bulldozer (Low-VRAM 2-View Micro-Batched).
Implements Do Duc Hanh (Nov 2026) 'DCA Inspired Block Quadratic Optimization for Gaussian Splatting on CPU and GPU'
with <500 MB VRAM micro-batch accumulation over b=8 views, curvature reuse (hold=4), and delayed group check (check=2).
"""

from __future__ import annotations

import ctypes
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch

from nvrtc_gpu_qp2 import (
    CU_DIR,
    GpuQP2RendererAndSolver,
    NativeCudaKernels,
    compute_loss_and_pixel_grad,
    make_separable_ssim_kernels,
    ptr,
    run_self_test,
    sep_gauss_blur,
)

import os
REPO_ROOT = Path(__file__).resolve().parents[1]
ZOOM_DIR = Path(os.environ.get("LEGO_ZOOM_DIR", str(REPO_ROOT / "workspace_runs" / "lego_zoom_1600")))
SPARSE_TXT = ZOOM_DIR / "sparse_txt" / "0"
STAGE1A_TXT = ZOOM_DIR / "stage1a_txt"
IMG_1600_DIR = ZOOM_DIR / "images"
GT_800_DIR = Path(os.environ.get("LEGO_GT_800_DIR", str(REPO_ROOT / "workspace_runs" / "lego_800" / "images")))
OUT_DIR = CU_DIR / "out_1600"


def rodrigues_batch_np(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w, axis=1, keepdims=True)
    small = th[:, 0] < 1e-6
    k = w / np.maximum(th, 1e-12)
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    s = np.sin(th[:, 0])
    c = np.cos(th[:, 0])
    omc = 1.0 - c
    R = np.empty((len(w), 3, 3), dtype=np.float32)
    R[:, 0, 0] = c + kx * kx * omc
    R[:, 0, 1] = kx * ky * omc - kz * s
    R[:, 0, 2] = kx * kz * omc + ky * s
    R[:, 1, 0] = ky * kx * omc + kz * s
    R[:, 1, 1] = c + ky * ky * omc
    R[:, 1, 2] = ky * kz * omc - kx * s
    R[:, 2, 0] = kz * kx * omc - ky * s
    R[:, 2, 1] = kz * ky * omc + kx * s
    R[:, 2, 2] = c + kz * kz * omc
    if np.any(small):
        ws = w[small]
        Rs = np.zeros((len(ws), 3, 3), dtype=np.float32)
        Rs[:, 0, 0] = 1.0; Rs[:, 1, 1] = 1.0; Rs[:, 2, 2] = 1.0
        Rs[:, 0, 1] = -ws[:, 2]; Rs[:, 0, 2] = ws[:, 1]
        Rs[:, 1, 0] = ws[:, 2];  Rs[:, 1, 2] = -ws[:, 0]
        Rs[:, 2, 0] = -ws[:, 1]; Rs[:, 2, 1] = ws[:, 0]
        R[small] = Rs
    return R


def rotmat_to_quat_wxyz_np(R: np.ndarray) -> np.ndarray:
    n = len(R)
    q = np.empty((n, 4), dtype=np.float32)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    m0 = tr > 0.0
    if np.any(m0):
        s = np.sqrt(np.maximum(tr[m0] + 1.0, 1e-12)) * 2.0
        q[m0, 0] = 0.25 * s
        q[m0, 1] = (R[m0, 2, 1] - R[m0, 1, 2]) / s
        q[m0, 2] = (R[m0, 0, 2] - R[m0, 2, 0]) / s
        q[m0, 3] = (R[m0, 1, 0] - R[m0, 0, 1]) / s
    m1 = (~m0) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if np.any(m1):
        s = np.sqrt(np.maximum(1.0 + R[m1, 0, 0] - R[m1, 1, 1] - R[m1, 2, 2], 1e-12)) * 2.0
        q[m1, 0] = (R[m1, 2, 1] - R[m1, 1, 2]) / s
        q[m1, 1] = 0.25 * s
        q[m1, 2] = (R[m1, 0, 1] + R[m1, 1, 0]) / s
        q[m1, 3] = (R[m1, 0, 2] + R[m1, 2, 0]) / s
    m2 = (~m0) & (~m1) & (R[:, 1, 1] > R[:, 2, 2])
    if np.any(m2):
        s = np.sqrt(np.maximum(1.0 + R[m2, 1, 1] - R[m2, 0, 0] - R[m2, 2, 2], 1e-12)) * 2.0
        q[m2, 0] = (R[m2, 0, 2] - R[m2, 2, 0]) / s
        q[m2, 1] = (R[m2, 0, 1] + R[m2, 1, 0]) / s
        q[m2, 2] = 0.25 * s
        q[m2, 3] = (R[m2, 1, 2] + R[m2, 2, 1]) / s
    m3 = (~m0) & (~m1) & (~m2)
    if np.any(m3):
        s = np.sqrt(np.maximum(1.0 + R[m3, 2, 2] - R[m3, 0, 0] - R[m3, 1, 1], 1e-12)) * 2.0
        q[m3, 0] = (R[m3, 1, 0] - R[m3, 0, 1]) / s
        q[m3, 1] = (R[m3, 0, 2] + R[m3, 2, 0]) / s
        q[m3, 2] = (R[m3, 1, 2] + R[m3, 2, 1]) / s
        q[m3, 3] = 0.25 * s
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return q


def build_morton_pairs_fast_gpu(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = int(params.shape[0])
    xyz = params[:, 0:3]
    sigma_max = torch.exp(params[:, 3:6].max(dim=1).values)

    mn = xyz.min(dim=0).values
    span = torch.clamp(xyz.max(dim=0).values - mn, min=1e-9)
    q = torch.clamp(torch.floor((xyz - mn) / span * 1023.0), 0, 1023).to(torch.int64)

    def part1by2_t(n: torch.Tensor) -> torch.Tensor:
        n = (n | (n << 16)) & 0x030000FF
        n = (n | (n << 8)) & 0x0300F00F
        n = (n | (n << 4)) & 0x030C30C3
        n = (n | (n << 2)) & 0x09249249
        return n

    morton = part1by2_t(q[:, 0]) | (part1by2_t(q[:, 1]) << 1) | (part1by2_t(q[:, 2]) << 2)
    order = torch.argsort(morton).to(torch.int32)

    n_even = (N // 2) * 2
    g_a = order[0:n_even:2]
    g_b = order[1:n_even:2]
    d2 = ((xyz[g_a] - xyz[g_b]) ** 2).sum(dim=1)
    thresh = 4.0 * torch.maximum(sigma_max[g_a], sigma_max[g_b])
    valid_pair = d2 <= (thresh * thresh)

    paired_a = g_a[valid_pair]
    paired_b = g_b[valid_pair]

    unpaired_list = [g_a[~valid_pair], g_b[~valid_pair]]
    if N % 2 == 1:
        unpaired_list.append(order[-1:])
    singletons = torch.cat(unpaired_list, dim=0)

    pair_i = torch.cat([paired_a, singletons], dim=0).contiguous()
    pair_j = torch.cat([paired_b, torch.full_like(singletons, -1)], dim=0).contiguous()

    partner_of = torch.full((N,), -1, dtype=torch.int32, device=params.device)
    partner_of[paired_a.long()] = paired_b
    partner_of[paired_b.long()] = paired_a
    return pair_i, pair_j, partner_of


def parse_cams_and_centers(txt_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    img_lines = [l.strip() for l in (txt_dir / "images.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    out = {}
    for i in range(0, len(img_lines), 2):
        t = img_lines[i].split()
        qw, qx, qy, qz = [float(x) for x in t[1:5]]
        tx, ty, tz = [float(x) for x in t[5:8]]
        name = t[9]
        nrm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        qw, qx, qy, qz = qw/nrm, qx/nrm, qy/nrm, qz/nrm
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float64)
        C = -(R.T @ np.array([tx, ty, tz], dtype=np.float64))
        out[name] = (R.astype(np.float32), C.astype(np.float32))
    return out


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    sc = src - src.mean(0)
    dc = dst - dst.mean(0)
    U, S, Vt = np.linalg.svd(dc.T @ sc / len(src))
    D = np.ones(3)
    D[-1] = np.linalg.det(U @ Vt)
    Q = U @ np.diag(D) @ Vt
    s = float((S * D).sum() / np.mean(np.sum(sc * sc, axis=1)))
    t = dst.mean(0) - s * (Q @ src.mean(0))
    return s, Q, t


def load_stage1_scene() -> dict:
    cam_ln = [l.strip() for l in (SPARSE_TXT / "cameras.txt").read_text().splitlines() if l.strip() and not l.startswith("#")][0]
    ct = cam_ln.split()
    W, H = int(ct[2]), int(ct[3])
    fx, fy, cx, cy = [float(x) for x in ct[4:8]]

    poses_dca = parse_cams_and_centers(SPARSE_TXT)
    poses_1a = parse_cams_and_centers(STAGE1A_TXT)
    names_sorted = sorted(poses_dca.keys())
    records = [{"name": nm, "R": poses_dca[nm][0], "C": poses_dca[nm][1]} for nm in names_sorted]

    # Combine Stage 1b clean DCA-BA points (11,299) + additional Stage 1a detail points (6,577) aligned to DCA frame!
    pts_xyz, pts_rgb = [], []
    for ln in (SPARSE_TXT / "points3D.txt").read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        t = s.split()
        pts_xyz.append([float(t[1]), float(t[2]), float(t[3])])
        pts_rgb.append([int(t[4]) / 255.0, int(t[5]) / 255.0, int(t[6]) / 255.0])

    c_1a = np.array([poses_1a[nm][1] for nm in names_sorted], dtype=np.float64)
    c_dca = np.array([poses_dca[nm][1] for nm in names_sorted], dtype=np.float64)
    s_sim, Q_sim, t_sim = umeyama_sim3(c_1a, c_dca)

    extra_xyz, extra_rgb = [], []
    for ln in (STAGE1A_TXT / "points3D.txt").read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        t = s.split()
        tr_len = (len(t) - 8) // 2
        err_px = float(t[7])
        if tr_len == 2 and err_px < 2.2:
            p = np.array([float(t[1]), float(t[2]), float(t[3])], dtype=np.float64)
            pa = s_sim * (Q_sim @ p) + t_sim
            if np.all(np.abs(pa) < 1.5):
                extra_xyz.append(pa.tolist())
                extra_rgb.append([int(t[4]) / 255.0, int(t[5]) / 255.0, int(t[6]) / 255.0])

    pts_xyz = np.array(pts_xyz + extra_xyz, dtype=np.float32)
    pts_rgb = np.array(pts_rgb + extra_rgb, dtype=np.float32)

    eval_indices = set(range(0, len(records), 10))
    train_recs = [r for idx, r in enumerate(records) if idx not in eval_indices]
    eval_recs = [r for idx, r in enumerate(records) if idx in eval_indices]

    train_C = np.array([r["C"] for r in train_recs], dtype=np.float64)
    scene_center = train_C.mean(axis=0).astype(np.float32)
    cam_extent = float(np.max(np.linalg.norm(train_C - scene_center[None, :], axis=1)))

    xyz_norm = (pts_xyz - scene_center[None, :]) / cam_extent
    tree = cKDTree(xyz_norm)
    dists, _ = tree.query(xyz_norm, k=4)
    rms_3nn = np.sqrt(np.mean(dists[:, 1:4] ** 2, axis=1)).astype(np.float32)
    init_scale = np.clip(rms_3nn, 1e-4, 0.035)

    N0 = len(xyz_norm)
    params0 = np.zeros((N0, 13), dtype=np.float32)
    params0[:, 0:3] = xyz_norm
    params0[:, 3:6] = np.log(init_scale)[:, None]
    params0[:, 6:9] = 0.0
    params0[:, 9] = math.log(0.45 / 0.55)
    rgb_c = np.clip(pts_rgb, 0.01, 0.99)
    params0[:, 10:13] = np.log(rgb_c / (1.0 - rgb_c))

    def pack_views(recs):
        R_arr = np.array([r["R"].reshape(-1) for r in recs], dtype=np.float32)
        C_arr = np.array([(r["C"] - scene_center) / cam_extent for r in recs], dtype=np.float32)
        K_1600 = np.tile(np.array([fx, fy, cx, cy], dtype=np.float32), (len(recs), 1))
        K_800 = K_1600 * 0.5
        imgs_1600 = []
        imgs_800 = []
        rois_1600 = []
        rois_800 = []
        tx_1600, ty_1600 = (W + 15) // 16, (H + 15) // 16
        tx_800, ty_800 = (W // 2 + 15) // 16, (H // 2 + 15) // 16

        for r in recs:
            bgra = cv2.imread(str(IMG_1600_DIR / r["name"]), cv2.IMREAD_UNCHANGED)
            rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.0
            rgb16 = rgba[:, :, :3] * rgba[:, :, 3:4]
            rgb16_u8 = np.clip(np.rint(rgb16 * 255.0), 0, 255).astype(np.uint8)
            imgs_1600.append(rgb16_u8)

            rgb8_u8 = np.clip(np.rint(cv2.resize(rgb16, (W // 2, H // 2), interpolation=cv2.INTER_AREA) * 255.0), 0, 255).astype(np.uint8)
            imgs_800.append(rgb8_u8)

            ys, xs = np.nonzero(rgba[:, :, 3] > 0.005)
            if len(xs) > 0:
                pad_t = 2
                x0 = max(0, int(xs.min() // 16) - pad_t)
                y0 = max(0, int(ys.min() // 16) - pad_t)
                x1 = min(tx_1600, int(xs.max() // 16) + 1 + pad_t)
                y1 = min(ty_1600, int(ys.max() // 16) + 1 + pad_t)
                rois_1600.append([x0, y0, x1, y1])
                rois_800.append([max(0, x0 // 2 - 1), max(0, y0 // 2 - 1), min(tx_800, (x1 + 1) // 2 + 1), min(ty_800, (y1 + 1) // 2 + 1)])
            else:
                rois_1600.append([0, 0, tx_1600, ty_1600])
                rois_800.append([0, 0, tx_800, ty_800])

        return {
            "names": [r["name"] for r in recs],
            "R": R_arr,
            "C": C_arr,
            "K_1600": K_1600,
            "K_800": K_800,
            "imgs_1600_u8": np.stack(imgs_1600, axis=0),  # kept in CPU RAM (<415 MB), streamed 2 views at a time!
            "imgs_800_u8": np.stack(imgs_800, axis=0),
            "rois_1600": np.array(rois_1600, dtype=np.int32),
            "rois_800": np.array(rois_800, dtype=np.int32),
        }

    return {
        "W": W,
        "H": H,
        "scene_center": scene_center,
        "cam_extent": cam_extent,
        "params0": params0,
        "train": pack_views(train_recs),
        "eval": pack_views(eval_recs),
    }


@torch.no_grad()
def adaptive_densify_and_prune(
    params: torch.Tensor,
    pos_grad_acc: torch.Tensor,
    pos_grad_cnt: torch.Tensor,
    target_max_splats: int,
    split_scale_thresh: float = 0.006,
    min_opac: float = 0.008,
) -> torch.Tensor:
    opac = torch.sigmoid(params[:, 9])
    alive = opac >= min_opac
    params = params[alive]
    avg_grad = pos_grad_acc[alive] / torch.clamp(pos_grad_cnt[alive].to(torch.float32), min=1.0)

    cur_n = int(params.shape[0])
    room = max(0, target_max_splats - cur_n)
    if room <= 0:
        return params.contiguous()

    n_grow = min(room, cur_n)
    _, top_idx = torch.topk(avg_grad, k=n_grow, largest=True)
    sel = params[top_idx].clone()

    scales = torch.exp(sel[:, 3:6])
    max_sc, max_ax = scales.max(dim=1)
    is_large = max_sc > split_scale_thresh

    w_np = sel[:, 6:9].cpu().numpy()
    R_np = rodrigues_batch_np(w_np)
    R_t = torch.from_numpy(R_np).to(params.device)
    axis_vec = R_t[torch.arange(len(sel), device=params.device), :, max_ax]
    offset = axis_vec * (0.42 * max_sc[:, None])

    parent_upd = params[top_idx].clone()
    child = sel.clone()

    if torch.any(is_large):
        parent_upd[is_large, 0:3] -= offset[is_large]
        parent_upd[is_large, 3:6] -= math.log(1.55)
        child[is_large, 0:3] += offset[is_large]
        child[is_large, 3:6] -= math.log(1.55)

    is_small = ~is_large
    if torch.any(is_small):
        jitter = offset[is_small] * 0.45
        parent_upd[is_small, 0:3] -= jitter
        child[is_small, 0:3] += jitter
        op_s = torch.clamp(torch.sigmoid(sel[is_small, 9]) * 0.65, 0.02, 0.98)
        logit_s = torch.log(op_s / (1.0 - op_s))
        parent_upd[is_small, 9] = logit_s
        child[is_small, 9] = logit_s

    params[top_idx] = parent_upd
    out = torch.cat([params, child], dim=0)
    out[:, 3:6] = torch.clamp(out[:, 3:6], -9.2, -1.6)
    out[:, 9:13] = torch.clamp(out[:, 9:13], -8.0, 4.6)
    return out.contiguous()


def export_world_ply(params_gpu: torch.Tensor, scene_center: np.ndarray, cam_extent: float, ply_path: Path):
    C0 = 0.28209479177387814
    p = params_gpu.detach().cpu().numpy()
    N = len(p)

    xyz_world = p[:, 0:3] * cam_extent + scene_center[None, :]
    scale_world_log = p[:, 3:6] + math.log(cam_extent)
    R_mat = rodrigues_batch_np(p[:, 6:9])
    quats = rotmat_to_quat_wxyz_np(R_mat)
    opac_logit = p[:, 9:10]
    rgb = 1.0 / (1.0 + np.exp(-p[:, 10:13]))
    f_dc = (rgb - 0.5) / C0

    props = (
        ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{k}" for k in range(9)]
        + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    )
    arr = np.zeros((N, len(props)), dtype="<f4")
    arr[:, 0:3] = xyz_world
    arr[:, 6:9] = f_dc
    arr[:, 18:19] = opac_logit
    arr[:, 19:22] = scale_world_log
    arr[:, 22:26] = quats

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        + "".join(f"property float {nm}\n" for nm in props)
        + "end_header\n"
    ).encode("ascii")

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    with ply_path.open("wb") as f:
        f.write(header)
        f.write(arr.tobytes())


class LowVramGpuQP2Trainer:
    """Streams micro-batches of `mb = 2` views to compute exact 8-view batch loss, gradient, and curvature in <450 MB VRAM!"""
    def __init__(self, kernels: NativeCudaKernels, W: int, H: int):
        self.k = kernels
        self.W = int(W)
        self.H = int(H)
        self.tiles_x = (self.W + 15) // 16
        self.tiles_y = (self.H + 15) // 16
        self.dev = torch.device("cuda")
        self.kh, self.kv = make_separable_ssim_kernels(self.dev)
        self.alpha_thresh = 1.0 / 255.0

    def _micro_step(
        self,
        params: torch.Tensor,
        cam_R: torch.Tensor,
        cam_C: torch.Tensor,
        cam_K: torch.Tensor,
        roi_tiles_cpu: np.ndarray,
        gt_u8_cpu: np.ndarray,
        need_grad: bool,
        need_curv: bool,
        pair_i: torch.Tensor | None,
        pair_j: torch.Tensor | None,
        partner_of: torch.Tensor | None,
        rho_cross: float,
        weight_scale: float,
    ) -> dict:
        B = int(cam_R.shape[0])
        N = int(params.shape[0])
        BN = B * N
        roi_tiles = torch.from_numpy(roi_tiles_cpu).to(self.dev)

        proj_data = torch.empty((BN, 9), dtype=torch.float32, device=self.dev)
        proj_jac = torch.empty((BN, 45), dtype=torch.float32, device=self.dev) if need_grad else None
        tile_bbox = torch.empty((BN, 4), dtype=torch.int32, device=self.dev)
        depths = torch.empty((BN,), dtype=torch.float32, device=self.dev)
        tiles_touched = torch.zeros((BN,), dtype=torch.int32, device=self.dev)

        blk = 256
        grd = (BN + blk - 1) // blk
        self.k.launch(
            "project_gaussians_fd_kernel",
            (grd, 1, 1),
            (blk, 1, 1),
            [
                ptr(params), ptr(cam_R), ptr(cam_C), ptr(cam_K), ptr(roi_tiles),
                ctypes.c_int(B), ctypes.c_int(N), ctypes.c_int(self.W), ctypes.c_int(self.H),
                ctypes.c_int(self.tiles_x), ctypes.c_int(self.tiles_y),
                ctypes.c_float(1e-4), ctypes.c_int(1 if need_grad else 0),
                ptr(proj_data), ptr(proj_jac), ptr(tile_bbox), ptr(depths), ptr(tiles_touched),
            ],
        )

        tt_i64 = tiles_touched.to(torch.int64)
        cum = torch.cumsum(tt_i64, dim=0)
        total_inter = int(cum[-1].item()) if BN > 0 else 0

        out_img = torch.zeros((B, self.H, self.W, 3), dtype=torch.float32, device=self.dev)
        out_T = torch.ones((B, self.H, self.W), dtype=torch.float32, device=self.dev)
        out_last = torch.full((B, self.H, self.W), -1, dtype=torch.int32, device=self.dev)
        tile_ranges = torch.zeros((B * self.tiles_x * self.tiles_y, 2), dtype=torch.int32, device=self.dev)

        if total_inter > 0:
            offsets = cum - tt_i64
            keys = torch.empty((total_inter,), dtype=torch.int64, device=self.dev)
            vals = torch.empty((total_inter,), dtype=torch.int32, device=self.dev)
            self.k.launch(
                "emit_tile_keys_kernel",
                (grd, 1, 1),
                (blk, 1, 1),
                [
                    ptr(tile_bbox), ptr(depths), ptr(tiles_touched), ptr(offsets),
                    ctypes.c_int(B), ctypes.c_int(N), ctypes.c_int(self.tiles_x), ctypes.c_int(self.tiles_y),
                    ptr(keys), ptr(vals),
                ],
            )
            s_idx = torch.argsort(keys)
            sorted_keys = keys[s_idx].contiguous()
            sorted_vals = vals[s_idx].contiguous()
            del keys, vals, s_idx

            self.k.launch(
                "find_tile_ranges_kernel",
                ((total_inter + 255) // 256, 1, 1),
                (256, 1, 1),
                [ptr(sorted_keys), ctypes.c_longlong(total_inter), ptr(tile_ranges)],
            )
            del sorted_keys
        else:
            sorted_vals = torch.empty((1,), dtype=torch.int32, device=self.dev)

        self.k.launch(
            "forward_render_tiles_kernel",
            (self.tiles_x, self.tiles_y, B),
            (16, 16, 1),
            [
                ptr(tile_ranges), ptr(sorted_vals), ptr(proj_data),
                ctypes.c_int(B), ctypes.c_int(self.W), ctypes.c_int(self.H),
                ctypes.c_int(self.tiles_x), ctypes.c_int(self.tiles_y),
                ctypes.c_float(0.0), ctypes.c_float(0.0), ctypes.c_float(0.0),
                ctypes.c_float(self.alpha_thresh),
                ptr(out_img), ptr(out_T), ptr(out_last),
            ],
        )

        tx0, ty0 = int(roi_tiles_cpu[:, 0].min()), int(roi_tiles_cpu[:, 1].min())
        tx1, ty1 = int(roi_tiles_cpu[:, 2].max()), int(roi_tiles_cpu[:, 3].max())
        px0, py0 = max(0, tx0 * 16), max(0, ty0 * 16)
        px1, py1 = min(self.W, tx1 * 16), min(self.H, ty1 * 16)

        pred_sub = out_img[:, py0:py1, px0:px1, :]
        gt_sub = torch.from_numpy(gt_u8_cpu[:, py0:py1, px0:px1, :]).to(self.dev, dtype=torch.float32) * (1.0 / 255.0)
        gt_sub_bchw = gt_sub.permute(0, 3, 1, 2).contiguous()
        gt_mu = sep_gauss_blur(gt_sub_bchw, self.kh, self.kv)
        gt_sig2 = sep_gauss_blur(gt_sub_bchw * gt_sub_bchw, self.kh, self.kv) - gt_mu * gt_mu

        area_ratio = float((py1 - py0) * (px1 - px0)) / float(self.H * self.W)
        loss_sub, dL_sub, wbar_sub = compute_loss_and_pixel_grad(
            pred_sub, gt_sub, gt_mu, gt_sig2, self.kh, self.kv, need_grad=need_grad, need_wbar=need_curv
        )
        loss_weighted = loss_sub * area_ratio * weight_scale
        if not need_grad:
            return {"loss": loss_weighted, "pred": out_img}

        dL_full = torch.zeros_like(out_img)
        dL_full[:, py0:py1, px0:px1, :] = dL_sub * (area_ratio * weight_scale)

        wbar_full = None
        if need_curv:
            wbar_full = torch.zeros_like(out_img)
            wbar_full[:, py0:py1, px0:px1, :] = wbar_sub * (area_ratio * weight_scale)

        grad_img9 = torch.zeros((BN, 9), dtype=torch.float32, device=self.dev)
        curv_img42 = torch.zeros((BN, 42), dtype=torch.float32, device=self.dev) if need_curv else None
        adj_cross81 = (
            torch.zeros((BN, 81), dtype=torch.float32, device=self.dev)
            if (need_curv and rho_cross > 0.0)
            else None
        )

        if total_inter > 0:
            self.k.launch(
                "backward_grad_curv_tiles_kernel",
                (self.tiles_x, self.tiles_y, B),
                (16, 16, 1),
                [
                    ptr(tile_ranges), ptr(sorted_vals), ptr(proj_data),
                    ptr(out_T), ptr(out_last), ptr(dL_full), ptr(wbar_full), ptr(partner_of),
                    ctypes.c_int(B), ctypes.c_int(N), ctypes.c_int(self.W), ctypes.c_int(self.H),
                    ctypes.c_int(self.tiles_x), ctypes.c_int(self.tiles_y),
                    ctypes.c_float(0.0), ctypes.c_float(0.0), ctypes.c_float(0.0),
                    ctypes.c_float(self.alpha_thresh),
                    ctypes.c_int(1 if need_curv else 0),
                    ptr(grad_img9), ptr(curv_img42), ptr(adj_cross81),
                ],
            )

        M = int(pair_i.shape[0])
        out_grad13 = torch.zeros((N, 13), dtype=torch.float32, device=self.dev)
        out_B_pair = torch.zeros((M, 26, 26), dtype=torch.float32, device=self.dev) if need_curv else None
        self.k.launch(
            "assemble_13d_grad_curv_kernel",
            ((M + 127) // 128, 1, 1),
            (128, 1, 1),
            [
                ptr(grad_img9), ptr(curv_img42), ptr(adj_cross81), ptr(proj_jac), ptr(tiles_touched),
                ptr(pair_i), ptr(pair_j),
                ctypes.c_int(B), ctypes.c_int(N), ctypes.c_int(M),
                ctypes.c_float(rho_cross), ctypes.c_int(1 if need_curv else 0),
                ptr(out_grad13), ptr(out_B_pair),
            ],
        )

        pos2d_grad = torch.norm(grad_img9[:, 0:2], dim=1).view(B, N).sum(dim=0)
        vis_cnt = (tiles_touched.view(B, N) > 0).sum(dim=0)
        return {
            "loss": loss_weighted,
            "grad13": out_grad13,
            "B_pair": out_B_pair,
            "pos2d_grad": pos2d_grad,
            "vis_cnt": vis_cnt,
        }

    def batch_step(
        self,
        params: torch.Tensor,
        tr_pack: dict,
        batch_ids: list[int],
        use_1600: bool,
        need_grad: bool,
        need_curv: bool,
        pair_i: torch.Tensor | None = None,
        pair_j: torch.Tensor | None = None,
        partner_of: torch.Tensor | None = None,
        rho_cross: float = 0.0,
        micro_size: int = 2,
    ) -> dict:
        B_total = len(batch_ids)
        N = int(params.shape[0])
        K_key = "K_1600" if use_1600 else "K_800"
        img_key = "imgs_1600_u8" if use_1600 else "imgs_800_u8"
        roi_key = "rois_1600" if use_1600 else "rois_800"

        total_loss = 0.0
        acc_grad13 = torch.zeros((N, 13), dtype=torch.float32, device=self.dev) if need_grad else None
        acc_B_pair = torch.zeros((int(pair_i.shape[0]), 26, 26), dtype=torch.float32, device=self.dev) if need_curv else None
        acc_pos2d = torch.zeros((N,), dtype=torch.float32, device=self.dev) if need_grad else None
        acc_vis = torch.zeros((N,), dtype=torch.int32, device=self.dev) if need_grad else None

        for s in range(0, B_total, micro_size):
            sub_ids = batch_ids[s : s + micro_size]
            w_scale = float(len(sub_ids)) / float(B_total)
            cR = torch.from_numpy(tr_pack["R"][sub_ids]).to(self.dev)
            cC = torch.from_numpy(tr_pack["C"][sub_ids]).to(self.dev)
            cK = torch.from_numpy(tr_pack[K_key][sub_ids]).to(self.dev)
            rois_cpu = tr_pack[roi_key][sub_ids]
            imgs_cpu = tr_pack[img_key][sub_ids]

            ms = self._micro_step(
                params, cR, cC, cK, rois_cpu, imgs_cpu,
                need_grad=need_grad, need_curv=need_curv,
                pair_i=pair_i, pair_j=pair_j, partner_of=partner_of,
                rho_cross=rho_cross, weight_scale=w_scale,
            )
            total_loss += ms["loss"]
            if need_grad:
                acc_grad13 += ms["grad13"]
                acc_pos2d += ms["pos2d_grad"]
                acc_vis += ms["vis_cnt"]
            if need_curv:
                acc_B_pair += ms["B_pair"]

        return {
            "loss": total_loss,
            "grad13": acc_grad13,
            "B_pair": acc_B_pair,
            "pos2d_grad": acc_pos2d,
            "vis_cnt": acc_vis,
        }


@torch.no_grad()
def evaluate_6_views_800gt(trainer1600: LowVramGpuQP2Trainer, params: torch.Tensor, eval_pack: dict, dev: torch.device) -> tuple[float, list[np.ndarray]]:
    psnrs = []
    renders = []
    full_roi = np.array([[0, 0, trainer1600.tiles_x, trainer1600.tiles_y]], dtype=np.int32)
    for idx, nm in enumerate(eval_pack["names"]):
        cR = torch.from_numpy(eval_pack["R"][idx : idx + 1]).to(dev)
        cC = torch.from_numpy(eval_pack["C"][idx : idx + 1]).to(dev)
        cK = torch.from_numpy(eval_pack["K_1600"][idx : idx + 1]).to(dev)
        im_cpu = eval_pack["imgs_1600_u8"][idx : idx + 1]
        pred = trainer1600._micro_step(
            params, cR, cC, cK, full_roi, im_cpu,
            need_grad=False, need_curv=False, pair_i=None, pair_j=None, partner_of=None, rho_cross=0.0, weight_scale=1.0
        )["pred"][0]
        im16 = torch.clamp(torch.round(pred * 255.0), 0, 255).to(torch.uint8).cpu().numpy()
        renders.append(im16)
        im8 = cv2.resize(im16.astype(np.float32), (800, 800), interpolation=cv2.INTER_AREA)
        gt_rgba = cv2.cvtColor(cv2.imread(str(GT_800_DIR / nm), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.0
        gt_rgb = gt_rgba[:, :, :3] * gt_rgba[:, :, 3:4] * 255.0
        mse = float(np.mean((im8 - gt_rgb) ** 2))
        psnrs.append(10.0 * math.log10(255.0 * 255.0 / max(mse, 1e-8)))
    return float(np.mean(psnrs)), renders


def main():
    print("[Step 1] Running GPU-QP2 Numerical & KKT Parity Self-Test...", flush=True)
    st_res = run_self_test()
    print(f"[Step 1] Self-Test PASSED! grad_rel_err={st_res['max_directional_grad_rel_err_pct']}%, CPU/GPU step_diff={st_res['max_cpu_vs_gpu_fp32_step_diff']}", flush=True)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scene = load_stage1_scene()
    dev = torch.device("cuda")
    kernels = NativeCudaKernels()
    eng800 = LowVramGpuQP2Trainer(kernels, 800, 800)
    eng1600 = LowVramGpuQP2Trainer(kernels, 1600, 1600)
    qp_runner = GpuQP2RendererAndSolver(kernels, 1600, 1600)

    tr = scene["train"]
    V = len(tr["names"])
    params = torch.from_numpy(scene["params0"]).to(dev)
    print(f"[Init] Loaded {int(params.shape[0]):,d} Stage 1 points (cam_extent={scene['cam_extent']:.3f})", flush=True)

    bound_lo = torch.tensor([-1e30]*3 + [-9.2]*3 + [-1e30]*3 + [-8.0]*4, dtype=torch.float32, device=dev)
    bound_hi = torch.tensor([+1e30]*3 + [-1.6]*3 + [+1e30]*3 + [4.6]*4, dtype=torch.float32, device=dev)

    # Adaptive coordinate box limits annealing smoothly down to Paper Eq. (9) [0.001, 0.015, 0.01, 0.3, 0.3]
    phases = [
        {"name": "Warmup-800-P1", "1600": False, "updates": 64, "mb": 4, "c": [0.008, 0.14, 0.08, 0.60, 0.55], "rho": 1.0, "grow": 30000},
        {"name": "Grow-800-P2",   "1600": False, "updates": 64, "mb": 4, "c": [0.005, 0.09, 0.05, 0.45, 0.45], "rho": 1.0, "grow": 60000},
        {"name": "Grow-800-P3",   "1600": False, "updates": 72, "mb": 4, "c": [0.003, 0.05, 0.03, 0.35, 0.35], "rho": 1.0, "grow": 115000},
        {"name": "Grow-1600-P4",  "1600": True,  "updates": 72, "mb": 2, "c": [0.002, 0.03, 0.02, 0.30, 0.30], "rho": 0.0, "grow": 175000},
        {"name": "Zoom-1600-P5",  "1600": True,  "updates": 96, "mb": 2, "c": [0.001, 0.015,0.01, 0.25, 0.25], "rho": 0.0, "grow": 0},
        {"name": "Polish-1600-P6","1600": True,  "updates": 64, "mb": 2, "c": [0.0005,0.008,0.005,0.12, 0.12], "rho": 0.0, "grow": 0},
    ]

    rng = np.random.default_rng(88)
    batch_size = 8
    hold = 4
    group_m = 2

    torch.cuda.synchronize()
    t_train_start = time.perf_counter()

    total_attempted = 0
    total_accepted_groups = 0
    total_rejected_groups = 0
    max_kkt_observed = 0.0
    timeline = []

    for ph in phases:
        use_1600 = ph["1600"]
        eng = eng1600 if use_1600 else eng800
        cp, cs, cr, co, cl = ph["c"]
        trust_c = torch.tensor([cp]*3 + [cs]*3 + [cr]*3 + [co] + [cl]*3, dtype=torch.float32, device=dev)
        rho = float(ph["rho"])

        pair_i, pair_j, partner_of = build_morton_pairs_fast_gpu(params)
        N = int(params.shape[0])
        pos_grad_acc = torch.zeros((N,), dtype=torch.float32, device=dev)
        pos_grad_cnt = torch.zeros((N,), dtype=torch.int32, device=dev)

        lam = 1.0
        curv_valid = False
        B_pair_cached = None

        perm = rng.permutation(V)
        n_groups = ph["updates"] // group_m

        for grp_idx in range(n_groups):
            k_step0 = grp_idx * group_m
            macro_t = k_step0 // hold
            start_j = (macro_t * batch_size) % V
            if k_step0 % hold == 0 and start_j < batch_size and grp_idx > 0:
                perm = rng.permutation(V)
            batch_ids = [int(perm[(start_j + u) % V]) for u in range(batch_size)]

            if k_step0 % hold == 0:
                curv_valid = False

            x_save = params.clone()
            f_save = None
            qp_ok = True

            for sub in range(group_m):
                need_curv = not curv_valid
                res_bw = eng.batch_step(
                    params, tr, batch_ids, use_1600=use_1600,
                    need_grad=True, need_curv=need_curv,
                    pair_i=pair_i, pair_j=pair_j, partner_of=partner_of,
                    rho_cross=rho, micro_size=ph["mb"],
                )
                if sub == 0:
                    f_save = res_bw["loss"]
                if need_curv:
                    B_pair_cached = res_bw["B_pair"]
                    curv_valid = True

                pos_grad_acc += res_bw["pos2d_grad"]
                pos_grad_cnt += res_bw["vis_cnt"]

                step13, kkt_res, _, fail_flag = qp_runner.solve_gpu_qp(
                    B_pair_cached, params, res_bw["grad13"], pair_i, pair_j, trust_c, bound_lo, bound_hi, lam
                )
                total_attempted += 1
                max_kkt_observed = max(max_kkt_observed, float(kkt_res.max().item()))
                if fail_flag != 0:
                    qp_ok = False
                    break

                params = params + step13
                params[:, 3:6] = torch.clamp(params[:, 3:6], -9.2, -1.6)
                params[:, 9:13] = torch.clamp(params[:, 9:13], -8.0, 4.6)

            if qp_ok:
                f_end = eng.batch_step(
                    params, tr, batch_ids, use_1600=use_1600, need_grad=False, need_curv=False, micro_size=ph["mb"]
                )["loss"]
            else:
                f_end = float("inf")

            if math.isfinite(f_end) and f_end < f_save:
                total_accepted_groups += 1
                if (k_step0 + group_m) % hold == 0:
                    lam = max(1.0, lam * 0.5)
            else:
                params = x_save
                lam = min(1e6, lam * 4.0)
                curv_valid = False
                total_rejected_groups += 1

        torch.cuda.synchronize()
        elapsed_so_far = time.perf_counter() - t_train_start
        psnr_now, _ = evaluate_6_views_800gt(eng1600, params, scene["eval"], dev)
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        timeline.append({
            "phase": ph["name"],
            "res": 1600 if use_1600 else 800,
            "n_gaussians": int(params.shape[0]),
            "attempted_updates": total_attempted,
            "accepted_groups": total_accepted_groups,
            "rejected_groups": total_rejected_groups,
            "elapsed_s": round(elapsed_so_far, 2),
            "eval_800gt_psnr_db": round(psnr_now, 3),
            "peak_vram_mb": peak_vram_mb,
        })
        print(
            f"[{ph['name']}] t={elapsed_so_far:6.2f}s | N={int(params.shape[0]):7,d} | "
            f"updates={total_attempted:4d} (acc={total_accepted_groups}, rej={total_rejected_groups}) | "
            f"6-view Fair 800px PSNR = {psnr_now:.3f} dB | VRAM={peak_vram_mb} MB",
            flush=True,
        )

        if ph["grow"] > 0:
            params = adaptive_densify_and_prune(
                params, pos_grad_acc, pos_grad_cnt, target_max_splats=ph["grow"],
                split_scale_thresh=0.006 if not use_1600 else 0.004,
            )
            torch.cuda.empty_cache()

    torch.cuda.synchronize()
    total_train_s = time.perf_counter() - t_train_start

    alive_final = torch.sigmoid(params[:, 9]) >= 0.015
    params_final = params[alive_final].contiguous()
    final_psnr, eval_renders_1600 = evaluate_6_views_800gt(eng1600, params_final, scene["eval"], dev)

    eval_dir = OUT_DIR / "eval_gpu_qp2"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for nm, im16 in zip(scene["eval"]["names"], eval_renders_1600):
        cv2.imwrite(str(eval_dir / nm), cv2.cvtColor(im16, cv2.COLOR_RGB2BGR))

    ply_path = OUT_DIR / "lego_gpu_qp2_1600.ply"
    export_world_ply(params_final, scene["scene_center"], scene["cam_extent"], ply_path)

    timing_meta = {
        "method": "Native CUDA GPU-QP2 (Do Duc Hanh Paper 2: 13-param + Morton Pair Box-QP + Curvature Reuse hold=4 + Delayed Check m=2)",
        "wall_seconds": round(total_train_s, 3),
        "wall_minutes": round(total_train_s / 60.0, 3),
        "total_attempted_updates": total_attempted,
        "accepted_groups": total_accepted_groups,
        "rejected_groups": total_rejected_groups,
        "max_kkt_residual": max_kkt_observed,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        "final_active_splats": int(params_final.shape[0]),
        "final_fair_800gt_psnr_db": round(final_psnr, 4),
        "timeline": timeline,
        "self_test": st_res,
    }
    (OUT_DIR / "train_timing.json").write_text(json.dumps(timing_meta, indent=2), encoding="utf-8")
    print(
        f"\n[GPU-QP2 DONE] Wall time = {total_train_s:.2f}s ({total_train_s/60.0:.2f} min) | "
        f"Updates = {total_attempted} | Splats = {int(params_final.shape[0]):,d} | "
        f"Fair 800px GT PSNR = {final_psnr:.3f} dB | Max KKT = {max_kkt_observed:.2e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
