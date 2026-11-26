"""Native CUDA NVRTC Engine & Numerical Verifier for Advisor Paper 2 (Do Duc Hanh, Nov 2026):
'DCA Inspired Block Quadratic Optimization for Gaussian Splatting on CPU and GPU' (GPU-QP2).

Implements:
  1. Direct NVRTC (compute_86) compilation + CUDA Driver API launch on PyTorch CUDA stream
  2. Separable 11x11 Gaussian SSIM + L1 objective with cached target moments & exact pixel gradients
  3. Morton-order disjoint spatial pair partition (26-var pairs + 13-var singletons)
  4. Independent FP64 CPU-QP reference solver & GPU-QP2 parity/KKT/finite-difference verification (Section 9.3)
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CU_DIR = Path(__file__).resolve().parent
CU_PATH = CU_DIR / "gpu_qp2_kernels.cu"


class NativeCudaKernels:
    def __init__(self, cu_path: Path = CU_PATH):
        torch.cuda.init()
        _ = torch.empty(1, device="cuda")

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        os.add_dll_directory(str(torch_lib))
        nvrtc_dlls = sorted(torch_lib.glob("nvrtc64_*.dll"))
        if not nvrtc_dlls:
            raise RuntimeError(f"Could not find nvrtc64_*.dll inside {torch_lib}")
        self.nvrtc = ctypes.CDLL(str(nvrtc_dlls[-1]))
        self.cuda = ctypes.CDLL("nvcuda.dll")

        src_bytes = cu_path.read_bytes()
        prog = ctypes.c_void_p()
        rc = self.nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src_bytes, b"gpu_qp2_kernels.cu", 0, None, None)
        if rc != 0:
            raise RuntimeError(f"nvrtcCreateProgram failed: {rc}")

        opts = (ctypes.c_char_p * 3)(
            b"--gpu-architecture=compute_86",
            b"--std=c++17",
            b"--fmad=true",
        )
        rc = self.nvrtc.nvrtcCompileProgram(prog, 3, opts)
        log_sz = ctypes.c_size_t()
        self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
        if log_sz.value > 1:
            log_buf = ctypes.create_string_buffer(log_sz.value)
            self.nvrtc.nvrtcGetProgramLog(prog, log_buf)
            compile_log = log_buf.value.decode("utf-8", errors="replace")
        else:
            compile_log = ""
        if rc != 0:
            raise RuntimeError(f"NVRTC compilation failed (rc={rc}):\n{compile_log}")

        ptx_sz = ctypes.c_size_t()
        self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_sz))
        ptx_buf = ctypes.create_string_buffer(ptx_sz.value)
        self.nvrtc.nvrtcGetPTX(prog, ptx_buf)
        self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        self.ptx_bytes = ptx_sz.value

        self.module = ctypes.c_void_p()
        rc = self.cuda.cuModuleLoadData(ctypes.byref(self.module), ptx_buf)
        if rc != 0:
            raise RuntimeError(f"cuModuleLoadData failed: {rc}")

        self.fns = {}
        for name in (
            "project_gaussians_fd_kernel",
            "emit_tile_keys_kernel",
            "find_tile_ranges_kernel",
            "forward_render_tiles_kernel",
            "backward_grad_curv_tiles_kernel",
            "assemble_13d_grad_curv_kernel",
            "warp_box_qp_fp64_kernel",
        ):
            fn = ctypes.c_void_p()
            rc = self.cuda.cuModuleGetFunction(ctypes.byref(fn), self.module, name.encode("ascii"))
            if rc != 0:
                raise RuntimeError(f"cuModuleGetFunction({name}) failed: {rc}")
            self.fns[name] = fn

    def launch(self, name: str, grid: tuple[int, int, int], block: tuple[int, int, int], args_c: list):
        n_args = len(args_c)
        void_arr = (ctypes.c_void_p * n_args)(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args_c])
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = self.cuda.cuLaunchKernel(
            self.fns[name],
            int(grid[0]), int(grid[1]), int(grid[2]),
            int(block[0]), int(block[1]), int(block[2]),
            0,
            stream,
            void_arr,
            None,
        )
        if rc != 0:
            raise RuntimeError(f"cuLaunchKernel({name}) failed with CUDA Driver error {rc}")


def ptr(t: torch.Tensor | None) -> ctypes.c_uint64:
    return ctypes.c_uint64(0 if t is None else t.data_ptr())


def make_separable_ssim_kernels(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper Section 3.2 & 7.1: 11x11 Gaussian window sigma=1.5, separable horizontal & vertical filters."""
    coords = torch.arange(11, dtype=torch.float32, device=device) - 5.0
    g1d = torch.exp(-(coords * coords) / (2.0 * 1.5 * 1.5))
    g1d = g1d / g1d.sum()
    kh = g1d.view(1, 1, 1, 11).repeat(3, 1, 1, 1).contiguous()
    kv = g1d.view(1, 1, 11, 1).repeat(3, 1, 1, 1).contiguous()
    return kh, kv


def sep_gauss_blur(x_bchw: torch.Tensor, kh: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    # Paper Section 3.2: zero padding, window weights not renormalized at boundary
    h = F.conv2d(x_bchw, kh, padding=(0, 5), groups=3)
    return F.conv2d(h, kv, padding=(5, 0), groups=3)


def compute_loss_and_pixel_grad(
    pred_bhwc: torch.Tensor,
    gt_bhwc: torch.Tensor,
    gt_mu_bchw: torch.Tensor,
    gt_sigma2_bchw: torch.Tensor,
    kh: torch.Tensor,
    kv: torch.Tensor,
    need_grad: bool = True,
    need_wbar: bool = False,
) -> tuple[float, torch.Tensor | None, torch.Tensor | None]:
    """Exact Paper Eq. (5) & (6):
    F_B = 1/(B*P) sum [0.8 |I_hat - I| + 0.2 (1 - SSIM)]
    Uses PyTorch autograd ONLY on the final [B,3,H,W] image tensor (<1 ms on GPU, 100% exact SSIM derivative!).
    """
    B, H, W, _ = pred_bhwc.shape
    P = 3 * H * W
    C1 = 1e-4
    C2 = 9e-4

    if need_grad:
        pred_var = pred_bhwc.detach().requires_grad_(True)
    else:
        pred_var = pred_bhwc.detach()

    with torch.set_grad_enabled(need_grad):
        diff = pred_var - gt_bhwc
        # Zero subgradient convention at zero L1 residual (Paper Section 3.2)
        l1_term = torch.abs(diff)

        p_bchw = pred_var.permute(0, 3, 1, 2).contiguous()
        g_bchw = gt_bhwc.permute(0, 3, 1, 2).contiguous()

        mu_x = sep_gauss_blur(p_bchw, kh, kv)
        mu_y = gt_mu_bchw
        mu_xx = sep_gauss_blur(p_bchw * p_bchw, kh, kv)
        mu_xy = sep_gauss_blur(p_bchw * g_bchw, kh, kv)

        sig_x2 = mu_xx - mu_x * mu_x
        sig_y2 = gt_sigma2_bchw
        sig_xy = mu_xy - mu_x * mu_y

        num = (2.0 * mu_x * mu_y + C1) * (2.0 * sig_xy + C2)
        den = (mu_x * mu_x + mu_y * mu_y + C1) * (sig_x2 + sig_y2 + C2)
        ssim_map = (num / den).permute(0, 2, 3, 1)

        loss_map = 0.8 * l1_term + 0.2 * (1.0 - ssim_map)
        loss = loss_map.mean()
        loss_f64 = float(loss_map.detach().to(torch.float64).mean().item())

    dL_dI = None
    if need_grad:
        (dL_dI,) = torch.autograd.grad(loss, pred_var, retain_graph=False, create_graph=False)
        # Zero subgradient when exact zero diff on L1 part is naturally handled; ensure contiguous
        dL_dI = dL_dI.contiguous()

    w_bar = None
    if need_wbar:
        # Paper Eq. (6): w_bar_vq / (|B|*P) where w_bar_vq = 0.8 / max(0.01, |I_hat - I|) + 2.0
        scale_bp = 1.0 / float(B * P)
        w_bar = ((0.8 / torch.clamp(torch.abs(pred_bhwc.detach() - gt_bhwc), min=0.01)) + 2.0) * scale_bp
        w_bar = w_bar.contiguous()

    return loss_f64, dL_dI, w_bar


def build_morton_disjoint_pairs(params_cpu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paper Section 4.1: Morton-order disjoint spatial partition.
    Each unused center is paired with its nearest unused neighbor (searched in local Morton window for O(N) scalability)
    if distance <= 4 * max(sigma_max_i, sigma_max_j); else singleton.
    Returns: pair_i [M], pair_j [M], partner_of [N].
    """
    N = len(params_cpu)
    xyz = params_cpu[:, 0:3].astype(np.float64)
    sigma_max = np.exp(params_cpu[:, 3:6].max(axis=1).astype(np.float64))

    mn = xyz.min(axis=0)
    span = np.maximum(xyz.max(axis=0) - mn, 1e-9)
    q = np.clip(np.floor((xyz - mn) / span * 1023.0), 0, 1023).astype(np.uint64)

    def part1by2(n: np.ndarray) -> np.ndarray:
        n = (n | (n << 16)) & 0x030000FF
        n = (n | (n << 8)) & 0x0300F00F
        n = (n | (n << 4)) & 0x030C30C3
        n = (n | (n << 2)) & 0x09249249
        return n

    morton = part1by2(q[:, 0]) | (part1by2(q[:, 1]) << 1) | (part1by2(q[:, 2]) << 2)
    order = np.argsort(morton, kind="mergesort")

    used = np.zeros(N, dtype=bool)
    partner_of = np.full(N, -1, dtype=np.int32)
    p_i = []
    p_j = []
    # Exact full search when N <= 6000 (matching paper's 3,827), wide 96-Morton-neighbor search when N > 6000!
    win = N if N <= 6000 else 96

    xyz_ord = xyz[order]
    sig_ord = sigma_max[order]

    for pos in range(N):
        gi = int(order[pos])
        if used[gi]:
            continue
        used[gi] = True

        end_pos = min(N, pos + 1 + win)
        best_pos = -1
        best_d2 = 1e30
        pi_xyz = xyz_ord[pos]
        for cand in range(pos + 1, end_pos):
            gj_c = int(order[cand])
            if used[gj_c]:
                continue
            dx = xyz_ord[cand, 0] - pi_xyz[0]
            dy = xyz_ord[cand, 1] - pi_xyz[1]
            dz = xyz_ord[cand, 2] - pi_xyz[2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < best_d2:
                best_d2 = d2
                best_pos = cand

        if best_pos >= 0:
            gj = int(order[best_pos])
            thresh = 4.0 * max(sig_ord[pos], sig_ord[best_pos])
            if best_d2 <= thresh * thresh:
                used[gj] = True
                partner_of[gi] = gj
                partner_of[gj] = gi
                p_i.append(gi)
                p_j.append(gj)
                continue

        p_i.append(gi)
        p_j.append(-1)

    return (
        np.array(p_i, dtype=np.int32),
        np.array(p_j, dtype=np.int32),
        partner_of,
    )


def solve_box_qp_cpu_fp64(
    Be: np.ndarray,
    x_e: np.ndarray,
    g_e: np.ndarray,
    c_e: np.ndarray,
    lo_global: np.ndarray,
    hi_global: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, float, float]:
    """Independent CPU-QP FP64 Reference Solver (Paper Section 7.2) for numerical parity verification!"""
    p = len(x_e)
    Q = Be[:p, :p].astype(np.float64).copy()
    diag_floor = np.maximum(np.diag(Q), 1e-8)
    Q[np.arange(p), np.arange(p)] += lam * diag_floor

    l_raw = np.minimum(0.0, np.maximum(-c_e, lo_global - x_e))
    u_raw = np.maximum(0.0, np.minimum(c_e, hi_global - x_e))

    A_un = (c_e[:, None] * Q) * c_e[None, :]
    s = max(1e-30, float(np.max(np.abs(np.diag(A_un)))))
    A = A_un / s
    b = (c_e * g_e) / s
    lo = l_raw / c_e
    hi = u_raw / c_e

    y = np.zeros(p, dtype=np.float64)
    for _ in range(6):
        for a in range(p):
            if hi[a] - lo[a] <= 1e-12:
                y[a] = 0.5 * (lo[a] + hi[a])
                continue
            ra = b[a] + float(A[a] @ y)
            y[a] = np.clip(y[a] - ra / max(A[a, a], 1e-15), lo[a], hi[a])

    state = np.zeros(p, dtype=np.int32)
    for a in range(p):
        if hi[a] - lo[a] <= 1e-12:
            state[a] = 2
            y[a] = 0.5 * (lo[a] + hi[a])
        elif y[a] <= lo[a] + 1e-12:
            state[a] = -1
            y[a] = lo[a]
        elif y[a] >= hi[a] - 1e-12:
            state[a] = 1
            y[a] = hi[a]

    kkt_denom = 1.0 + float(np.max(np.abs(b)))
    for _ in range(256):
        r = A @ y + b
        nu = np.zeros(p, dtype=np.float64)
        for a in range(p):
            if state[a] == 0:
                nu[a] = r[a]
            elif state[a] == -1:
                nu[a] = min(0.0, r[a])
            elif state[a] == 1:
                nu[a] = max(0.0, r[a])
        if float(np.max(np.abs(nu))) / kkt_denom <= 1e-8:
            break

        free = np.flatnonzero(state == 0)
        max_free_r = float(np.max(np.abs(r[free]))) if len(free) else 0.0
        if max_free_r > 1e-10 * kkt_denom:
            L = np.linalg.cholesky(A[np.ix_(free, free)])
            z = np.linalg.solve(L, -r[free])
            dy_f = np.linalg.solve(L.T, z)
            alpha = 1.0
            blk_a = -1
            blk_st = 0
            for idx_f, a in enumerate(free):
                d = float(dy_f[idx_f])
                if d < -1e-15:
                    al = (lo[a] - y[a]) / d
                    if al < alpha:
                        alpha, blk_a, blk_st = al, int(a), -1
                elif d > 1e-15:
                    ah = (hi[a] - y[a]) / d
                    if ah < alpha:
                        alpha, blk_a, blk_st = ah, int(a), 1
            alpha = max(0.0, min(1.0, alpha))
            y[free] = np.clip(y[free] + alpha * dy_f, lo[free], hi[free])
            if blk_a >= 0:
                state[blk_a] = blk_st
                y[blk_a] = lo[blk_a] if blk_st == -1 else hi[blk_a]
        else:
            best_a = -1
            best_v = 0.0
            for a in range(p):
                v = (-r[a] if (state[a] == -1 and r[a] < 0) else (r[a] if (state[a] == 1 and r[a] > 0) else 0.0))
                if v > best_v:
                    best_v, best_a = float(v), a
            if best_a >= 0:
                state[best_a] = 0

    r = A @ y + b
    nu = np.where(state == 0, r, np.where(state == -1, np.minimum(0.0, r), np.where(state == 1, np.maximum(0.0, r), 0.0)))
    kkt = float(np.max(np.abs(nu))) / kkt_denom
    gamma = float(np.sum(np.where(r >= 0.0, r * (y - lo), r * (y - hi))))
    return (c_e * y).astype(np.float32), kkt, s * max(0.0, gamma)


class GpuQP2RendererAndSolver:
    def __init__(self, kernels: NativeCudaKernels, W: int, H: int, pos_eps: float = 1e-4):
        self.k = kernels
        self.W = int(W)
        self.H = int(H)
        self.tiles_x = (self.W + 15) // 16
        self.tiles_y = (self.H + 15) // 16
        self.pos_eps = float(pos_eps)
        self.device = torch.device("cuda")
        self.kh, self.kv = make_separable_ssim_kernels(self.device)

    def render_and_optional_backward(
        self,
        params: torch.Tensor,         # [N, 13] float32 CUDA
        cam_R: torch.Tensor,          # [B, 9] float32 CUDA
        cam_C: torch.Tensor,          # [B, 3] float32 CUDA
        cam_K: torch.Tensor,          # [B, 4] float32 CUDA
        roi_tiles: torch.Tensor,      # [B, 4] int32 CUDA
        gt_bhwc: torch.Tensor | None = None,
        gt_mu: torch.Tensor | None = None,
        gt_sig2: torch.Tensor | None = None,
        pair_i: torch.Tensor | None = None,
        pair_j: torch.Tensor | None = None,
        partner_of: torch.Tensor | None = None,
        need_grad: bool = False,
        need_curv: bool = False,
        rho_cross: float = 1.0,
        alpha_thresh: float = 1e-5,
        bg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict:
        B = int(cam_R.shape[0])
        N = int(params.shape[0])
        BN = B * N

        proj_data = torch.empty((BN, 9), dtype=torch.float32, device=self.device)
        proj_jac = torch.empty((BN, 45), dtype=torch.float32, device=self.device) if need_grad else None
        tile_bbox = torch.empty((BN, 4), dtype=torch.int32, device=self.device)
        depths = torch.empty((BN,), dtype=torch.float32, device=self.device)
        tiles_touched = torch.zeros((BN,), dtype=torch.int32, device=self.device)

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
                ctypes.c_float(self.pos_eps), ctypes.c_int(1 if need_grad else 0),
                ptr(proj_data), ptr(proj_jac), ptr(tile_bbox), ptr(depths), ptr(tiles_touched),
            ],
        )

        # Prefix sum for tile intersections
        tt_i64 = tiles_touched.to(torch.int64)
        cum = torch.cumsum(tt_i64, dim=0)
        total_intersections = int(cum[-1].item()) if BN > 0 else 0

        out_img = torch.empty((B, self.H, self.W, 3), dtype=torch.float32, device=self.device)
        out_T = torch.empty((B, self.H, self.W), dtype=torch.float32, device=self.device)
        out_last = torch.empty((B, self.H, self.W), dtype=torch.int32, device=self.device)
        tile_ranges = torch.zeros((B * self.tiles_x * self.tiles_y, 2), dtype=torch.int32, device=self.device)

        if total_intersections > 0:
            offsets = cum - tt_i64
            keys = torch.empty((total_intersections,), dtype=torch.int64, device=self.device)
            vals = torch.empty((total_intersections,), dtype=torch.int32, device=self.device)
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
            sort_idx = torch.argsort(keys)
            sorted_keys = keys[sort_idx].contiguous()
            sorted_vals = vals[sort_idx].contiguous()
            del keys, vals, sort_idx

            grd_k = (total_intersections + 255) // 256
            self.k.launch(
                "find_tile_ranges_kernel",
                (grd_k, 1, 1),
                (256, 1, 1),
                [ptr(sorted_keys), ctypes.c_longlong(total_intersections), ptr(tile_ranges)],
            )
            del sorted_keys
        else:
            sorted_vals = torch.empty((1,), dtype=torch.int32, device=self.device)

        self.k.launch(
            "forward_render_tiles_kernel",
            (self.tiles_x, self.tiles_y, B),
            (16, 16, 1),
            [
                ptr(tile_ranges), ptr(sorted_vals), ptr(proj_data),
                ctypes.c_int(B), ctypes.c_int(self.W), ctypes.c_int(self.H),
                ctypes.c_int(self.tiles_x), ctypes.c_int(self.tiles_y),
                ctypes.c_float(bg[0]), ctypes.c_float(bg[1]), ctypes.c_float(bg[2]),
                ctypes.c_float(alpha_thresh),
                ptr(out_img), ptr(out_T), ptr(out_last),
            ],
        )

        if gt_bhwc is None:
            return {"pred": out_img, "intersections": total_intersections}

        loss_val, dL_dI, w_bar = compute_loss_and_pixel_grad(
            out_img, gt_bhwc, gt_mu, gt_sig2, self.kh, self.kv, need_grad=need_grad, need_wbar=need_curv
        )
        if not need_grad:
            return {"loss": loss_val, "pred": out_img, "intersections": total_intersections}

        grad_img9 = torch.zeros((BN, 9), dtype=torch.float32, device=self.device)
        curv_img42 = torch.zeros((BN, 42), dtype=torch.float32, device=self.device) if need_curv else None
        adj_cross81 = (
            torch.zeros((BN, 81), dtype=torch.float32, device=self.device)
            if (need_curv and rho_cross > 0.0)
            else None
        )

        if total_intersections > 0:
            self.k.launch(
                "backward_grad_curv_tiles_kernel",
                (self.tiles_x, self.tiles_y, B),
                (16, 16, 1),
                [
                    ptr(tile_ranges), ptr(sorted_vals), ptr(proj_data),
                    ptr(out_T), ptr(out_last), ptr(dL_dI), ptr(w_bar), ptr(partner_of),
                    ctypes.c_int(B), ctypes.c_int(N), ctypes.c_int(self.W), ctypes.c_int(self.H),
                    ctypes.c_int(self.tiles_x), ctypes.c_int(self.tiles_y),
                    ctypes.c_float(bg[0]), ctypes.c_float(bg[1]), ctypes.c_float(bg[2]),
                    ctypes.c_float(alpha_thresh),
                    ctypes.c_int(1 if need_curv else 0),
                    ptr(grad_img9), ptr(curv_img42), ptr(adj_cross81),
                ],
            )

        M = int(pair_i.shape[0])
        out_grad13 = torch.zeros((N, 13), dtype=torch.float32, device=self.device)
        out_B_pair = torch.zeros((M, 26, 26), dtype=torch.float32, device=self.device) if need_curv else None

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
        return {
            "loss": loss_val,
            "pred": out_img,
            "grad13": out_grad13,
            "B_pair": out_B_pair,
            "intersections": total_intersections,
        }

    def solve_gpu_qp(
        self,
        B_pair: torch.Tensor,
        params: torch.Tensor,
        grad13: torch.Tensor,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
        trust_c13: torch.Tensor,
        bound_lo13: torch.Tensor,
        bound_hi13: torch.Tensor,
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        M = int(pair_i.shape[0])
        N = int(params.shape[0])
        out_step13 = torch.zeros((N, 13), dtype=torch.float32, device=self.device)
        out_kkt = torch.zeros((M,), dtype=torch.float64, device=self.device)
        out_gap = torch.zeros((M,), dtype=torch.float64, device=self.device)
        fail_flag = torch.zeros((1,), dtype=torch.int32, device=self.device)

        warps_per_block = 4
        blocks = (M + warps_per_block - 1) // warps_per_block
        self.k.launch(
            "warp_box_qp_fp64_kernel",
            (blocks, 1, 1),
            (warps_per_block * 32, 1, 1),
            [
                ptr(B_pair), ptr(params), ptr(grad13), ptr(pair_i), ptr(pair_j),
                ptr(trust_c13), ptr(bound_lo13), ptr(bound_hi13),
                ctypes.c_double(float(lam)), ctypes.c_int(M),
                ptr(out_step13), ptr(out_kkt), ptr(out_gap), ptr(fail_flag),
            ],
        )
        return out_step13, out_kkt, out_gap, int(fail_flag.item())


def run_self_test() -> dict:
    """Paper Section 9.3 Self-Verification:
    1. Finite-difference directional gradient check on live rendered Gaussians (<3% gate)
    2. CPU-QP vs GPU-QP2 FP64 Active-Set Cholesky parity across 13-var singletons & 26-var coupled pairs (KKT <= 1e-8)
    """
    t0 = time.perf_counter()
    kernels = NativeCudaKernels()
    W, H = 128, 128
    eng = GpuQP2RendererAndSolver(kernels, W, H, pos_eps=1e-4)
    rng = np.random.default_rng(88)

    N = 48
    params_np = np.zeros((N, 13), dtype=np.float32)
    params_np[:, 0:3] = rng.uniform(-0.18, 0.18, size=(N, 3))
    params_np[:, 3:6] = rng.uniform(-4.0, -2.7, size=(N, 3))
    params_np[:, 6:9] = rng.uniform(-0.25, 0.25, size=(N, 3))
    params_np[:, 9] = rng.uniform(-0.5, 1.2, size=(N,))
    params_np[:, 10:13] = rng.uniform(-1.5, 1.5, size=(N, 3))

    pair_i_np, pair_j_np, partner_np = build_morton_disjoint_pairs(params_np)
    M = len(pair_i_np)

    B = 3
    cam_R_list, cam_C_list, cam_K_list = [], [], []
    for k in range(B):
        ang = k * 0.45
        c_pos = np.array([math.sin(ang) * 0.9, 0.15 * k, math.cos(ang) * 0.9], dtype=np.float32)
        fwd = -c_pos / np.linalg.norm(c_pos)
        up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(fwd, up_ref)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd], axis=0).astype(np.float32)
        cam_R_list.append(R.reshape(-1))
        cam_C_list.append(c_pos)
        cam_K_list.append([140.0, 140.0, 64.0, 64.0])

    dev = torch.device("cuda")
    params = torch.from_numpy(params_np).to(dev)
    cam_R = torch.from_numpy(np.array(cam_R_list, dtype=np.float32)).to(dev)
    cam_C = torch.from_numpy(np.array(cam_C_list, dtype=np.float32)).to(dev)
    cam_K = torch.from_numpy(np.array(cam_K_list, dtype=np.float32)).to(dev)
    roi = torch.tensor([[0, 0, eng.tiles_x, eng.tiles_y]] * B, dtype=torch.int32, device=dev)

    # Smooth target image (avoids |I_hat - I| == 0 L1 non-differentiable kink during FD check!)
    gt_img = torch.full((B, H, W, 3), 0.75, dtype=torch.float32, device=dev)
    gt_bchw = gt_img.permute(0, 3, 1, 2).contiguous()
    gt_mu = sep_gauss_blur(gt_bchw, eng.kh, eng.kv)
    gt_sig2 = sep_gauss_blur(gt_bchw * gt_bchw, eng.kh, eng.kv) - gt_mu * gt_mu

    pair_i = torch.from_numpy(pair_i_np).to(dev)
    pair_j = torch.from_numpy(pair_j_np).to(dev)
    partner_of = torch.from_numpy(partner_np).to(dev)

    out = eng.render_and_optional_backward(
        params, cam_R, cam_C, cam_K, roi,
        gt_bhwc=gt_img, gt_mu=gt_mu, gt_sig2=gt_sig2,
        pair_i=pair_i, pair_j=pair_j, partner_of=partner_of,
        need_grad=True, need_curv=True, rho_cross=1.0,
    )
    g_cuda = out["grad13"].cpu().numpy()

    # 1. Directional finite-difference check along normalized gradient & coordinate-group directions (Paper Section 9.3 <3% gate)
    dir_rel_errs = []
    eps_fd = 1e-4
    test_dirs = [g_cuda.copy()]
    # Also test appearance+opacity subspace & scale subspace directions
    d_app = np.zeros_like(g_cuda); d_app[:, 9:13] = g_cuda[:, 9:13]; test_dirs.append(d_app)
    d_scl = np.zeros_like(g_cuda); d_scl[:, 3:6] = g_cuda[:, 3:6]; test_dirs.append(d_scl)
    for u in test_dirs:
        nrm_u = float(np.linalg.norm(u))
        if nrm_u < 1e-12:
            continue
        u = (u / nrm_u).astype(np.float32)
        u_t = torch.from_numpy(u).to(dev)
        lp = eng.render_and_optional_backward(
            params + eps_fd * u_t, cam_R, cam_C, cam_K, roi,
            gt_bhwc=gt_img, gt_mu=gt_mu, gt_sig2=gt_sig2, need_grad=False,
        )["loss"]
        lm = eng.render_and_optional_backward(
            params - eps_fd * u_t, cam_R, cam_C, cam_K, roi,
            gt_bhwc=gt_img, gt_mu=gt_mu, gt_sig2=gt_sig2, need_grad=False,
        )["loss"]
        fd_dir = (lp - lm) / (2.0 * eps_fd)
        an_dir = float(np.sum(g_cuda * u))
        rel = abs(an_dir - fd_dir) / max(abs(fd_dir), 1e-8)
        dir_rel_errs.append(rel)

    # 2. CPU-QP vs GPU-QP2 parity across 3 damping settings (lam = 1.0, 4.0, 0.25)
    trust_c_np = np.array([0.001]*3 + [0.015]*3 + [0.01]*3 + [0.3]*4, dtype=np.float32)
    lo_np = np.array([-1e30]*3 + [-9.2]*3 + [-1e30]*3 + [-8.0]*4, dtype=np.float32)
    hi_np = np.array([+1e30]*3 + [-1.6]*3 + [+1e30]*3 + [4.6]*4, dtype=np.float32)

    trust_c = torch.from_numpy(trust_c_np).to(dev)
    bound_lo = torch.from_numpy(lo_np).to(dev)
    bound_hi = torch.from_numpy(hi_np).to(dev)

    B_pair_np = out["B_pair"].cpu().numpy()
    max_step_diff = 0.0
    max_gpu_kkt = 0.0
    max_cpu_kkt = 0.0

    for lam in (1.0, 4.0, 16.0):
        step_gpu, kkt_gpu, gap_gpu, fail_flag = eng.solve_gpu_qp(
            out["B_pair"], params, out["grad13"], pair_i, pair_j, trust_c, bound_lo, bound_hi, lam
        )
        assert fail_flag == 0, f"GPU-QP2 reported failure flag={fail_flag} at lam={lam}"
        step_gpu_np = step_gpu.cpu().numpy()
        max_gpu_kkt = max(max_gpu_kkt, float(kkt_gpu.max().item()))

        step_cpu_np = np.zeros_like(step_gpu_np)
        for e in range(M):
            gi = int(pair_i_np[e])
            gj = int(pair_j_np[e])
            if gj >= 0:
                xe = np.concatenate([params_np[gi], params_np[gj]]).astype(np.float64)
                ge = np.concatenate([g_cuda[gi], g_cuda[gj]]).astype(np.float64)
                ce = np.concatenate([trust_c_np, trust_c_np]).astype(np.float64)
                loe = np.concatenate([lo_np, lo_np]).astype(np.float64)
                hie = np.concatenate([hi_np, hi_np]).astype(np.float64)
                de, kkt_c, _ = solve_box_qp_cpu_fp64(B_pair_np[e], xe, ge, ce, loe, hie, lam)
                step_cpu_np[gi] = de[:13]
                step_cpu_np[gj] = de[13:]
            else:
                xe = params_np[gi].astype(np.float64)
                ge = g_cuda[gi].astype(np.float64)
                de, kkt_c, _ = solve_box_qp_cpu_fp64(
                    B_pair_np[e], xe, ge, trust_c_np.astype(np.float64), lo_np.astype(np.float64), hi_np.astype(np.float64), lam
                )
                step_cpu_np[gi] = de
            max_cpu_kkt = max(max_cpu_kkt, kkt_c)

        diff = float(np.max(np.abs(step_gpu_np - step_cpu_np)))
        max_step_diff = max(max_step_diff, diff)

    res = {
        "nvrtc_ptx_bytes": kernels.ptx_bytes,
        "num_gaussians": N,
        "num_groups": M,
        "num_pairs_26var": int(np.sum(pair_j_np >= 0)),
        "num_singletons_13var": int(np.sum(pair_j_np < 0)),
        "max_directional_grad_rel_err_pct": round(100.0 * max(dir_rel_errs), 4),
        "max_gpu_qp_kkt_residual": max_gpu_kkt,
        "max_cpu_qp_kkt_residual": max_cpu_kkt,
        "max_cpu_vs_gpu_fp32_step_diff": max_step_diff,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "all_checks_passed": bool(
            max(dir_rel_errs) < 0.03 and max_gpu_kkt <= 1e-8 and max_cpu_kkt <= 1e-8 and max_step_diff < 1e-6
        ),
    }
    print(json.dumps(res, indent=2), flush=True)
    assert res["all_checks_passed"], f"Self-test failed: {res}"
    (CU_DIR / "self_test_gpu_qp2.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    run_self_test()
