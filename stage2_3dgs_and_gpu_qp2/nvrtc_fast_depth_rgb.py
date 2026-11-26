#!/usr/bin/env python3
"""Fast NVRTC CUDA 16x16 Tile Rasterizer for 3DGS Median Transmittance Depth + Alpha + RGB.

Compiles `fast_3dgs_depth.cu` in-memory via `nvrtc64_*.dll` (`compute_86`, zero `nvcc`/`cl.exe`
needed) and launches via `nvcuda.dll` on PyTorch's active CUDA stream. Automatically falls back
to a vectorized PyTorch 64x64 tile rasterizer if NVRTC compilation/loading fails for any reason.
"""

from __future__ import annotations

import ctypes
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

CU_PATH = Path(__file__).resolve().parent / "fast_3dgs_depth.cu"

_FALLBACK_CU_SOURCE = r"""
#define TILE_SIZE 16
#define TILE_PIXELS 256

extern "C" __global__ void emit_tile_keys_kernel(
    const int* __restrict__ tile_bbox,
    const int* __restrict__ tiles_touched,
    const long long* __restrict__ offsets,
    int M,
    int tiles_x,
    long long* __restrict__ out_keys
) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx >= M) return;
    if (tiles_touched[idx] <= 0) return;
    int tx0 = tile_bbox[idx * 4 + 0];
    int ty0 = tile_bbox[idx * 4 + 1];
    int tx1 = tile_bbox[idx * 4 + 2];
    int ty1 = tile_bbox[idx * 4 + 3];
    long long write_pos = offsets[idx];
    unsigned long long u_idx = (unsigned long long)(unsigned int)idx;
    for (int ty = ty0; ty < ty1; ++ty) {
        for (int tx = tx0; tx < tx1; ++tx) {
            unsigned long long tile_id = (unsigned long long)(ty * tiles_x + tx);
            out_keys[write_pos] = (long long)((tile_id << 32) | u_idx);
            ++write_pos;
        }
    }
}

extern "C" __global__ void find_tile_ranges_kernel(
    const long long* __restrict__ sorted_keys,
    long long num_intersections,
    int* __restrict__ tile_ranges
) {
    long long k = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (k >= num_intersections) return;
    int cur_tile = (int)((unsigned long long)sorted_keys[k] >> 32);
    if (k == 0) {
        tile_ranges[cur_tile * 2 + 0] = 0;
    } else {
        int prev_tile = (int)((unsigned long long)sorted_keys[k - 1] >> 32);
        if (cur_tile != prev_tile) {
            tile_ranges[prev_tile * 2 + 1] = (int)k;
            tile_ranges[cur_tile * 2 + 0] = (int)k;
        }
    }
    if (k == num_intersections - 1) {
        tile_ranges[cur_tile * 2 + 1] = (int)num_intersections;
    }
}

extern "C" __global__ void rasterize_median_depth_16x16(
    const int* __restrict__ tile_ranges,
    const long long* __restrict__ sorted_keys,
    const float* __restrict__ splat_data,
    int W,
    int H,
    int tiles_x,
    float* __restrict__ out_depth,
    float* __restrict__ out_alpha,
    float* __restrict__ out_rgb
) {
    int tx = (int)blockIdx.x;
    int ty = (int)blockIdx.y;
    int tile_id = ty * tiles_x + tx;
    int r_start = tile_ranges[tile_id * 2 + 0];
    int r_end   = tile_ranges[tile_id * 2 + 1];
    int px_i = tx * TILE_SIZE + (int)threadIdx.x;
    int py_i = ty * TILE_SIZE + (int)threadIdx.y;
    bool inside = (px_i < W && py_i < H);

    if (r_start >= r_end) {
        if (inside) {
            long long pix = (long long)py_i * (long long)W + (long long)px_i;
            out_depth[pix] = 0.0f;
            out_alpha[pix] = 0.0f;
            out_rgb[pix * 3 + 0] = 0.0f;
            out_rgb[pix * 3 + 1] = 0.0f;
            out_rgb[pix * 3 + 2] = 0.0f;
        }
        return;
    }

    float px = (float)px_i + 0.5f;
    float py = (float)py_i + 0.5f;
    float T = 1.0f;
    float cr = 0.0f, cg = 0.0f, cb = 0.0f;
    float depth_w_sum = 0.0f;
    float weight_sum = 0.0f;
    float median_depth = 0.0f;
    bool has_median = false;
    bool done = !inside;

    __shared__ float sh_u[TILE_PIXELS];
    __shared__ float sh_v[TILE_PIXELS];
    __shared__ float sh_ca[TILE_PIXELS];
    __shared__ float sh_cb[TILE_PIXELS];
    __shared__ float sh_cc[TILE_PIXELS];
    __shared__ float sh_op[TILE_PIXELS];
    __shared__ float sh_z[TILE_PIXELS];
    __shared__ float sh_r[TILE_PIXELS];
    __shared__ float sh_g[TILE_PIXELS];
    __shared__ float sh_b[TILE_PIXELS];
    int tid = (int)threadIdx.y * TILE_SIZE + (int)threadIdx.x;

    for (int base = r_start; base < r_end; base += TILE_PIXELS) {
        if (__syncthreads_and(done)) break;
        int load_k = base + tid;
        if (load_k < r_end) {
            int sv = (int)((unsigned long long)sorted_keys[load_k] & 0xFFFFFFFFULL);
            const float* sd = splat_data + (long long)sv * 10LL;
            sh_u[tid]  = sd[0];
            sh_v[tid]  = sd[1];
            sh_ca[tid] = sd[2];
            sh_cb[tid] = sd[3];
            sh_cc[tid] = sd[4];
            sh_op[tid] = sd[5];
            sh_z[tid]  = sd[6];
            sh_r[tid]  = sd[7];
            sh_g[tid]  = sd[8];
            sh_b[tid]  = sd[9];
        }
        __syncthreads();
        int count = r_end - base;
        if (count > TILE_PIXELS) count = TILE_PIXELS;
        if (!done) {
            for (int j = 0; j < count; ++j) {
                float dx = px - sh_u[j];
                float dy = py - sh_v[j];
                float pwr = 0.5f * (sh_ca[j] * dx * dx + 2.0f * sh_cb[j] * dx * dy + sh_cc[j] * dy * dy);
                if (!(pwr >= -1e-3f && pwr <= 8.0f)) continue;
                float alpha = fminf(0.99f, sh_op[j] * expf(-fmaxf(pwr, 0.0f)));
                if (alpha < (1.0f / 255.0f)) continue;
                float w = T * alpha;
                float next_T = T * (1.0f - alpha);
                float zi = sh_z[j];
                cr += w * sh_r[j];
                cg += w * sh_g[j];
                cb += w * sh_b[j];
                depth_w_sum += w * zi;
                weight_sum += w;
                if (!has_median && next_T <= 0.50f) {
                    median_depth = zi;
                    has_median = true;
                }
                T = next_T;
                if (T < 1e-4f) {
                    done = true;
                    break;
                }
            }
        }
        __syncthreads();
    }

    if (inside) {
        long long pix = (long long)py_i * (long long)W + (long long)px_i;
        float final_alpha = fminf(fmaxf(1.0f - T, 0.0f), 1.0f);
        float out_d = 0.0f;
        if (has_median) {
            out_d = median_depth;
        } else if (final_alpha >= 0.15f && weight_sum > 1e-8f) {
            out_d = depth_w_sum / weight_sum;
        }
        out_depth[pix] = out_d;
        out_alpha[pix] = final_alpha;
        out_rgb[pix * 3 + 0] = fminf(fmaxf(cr, 0.0f), 1.0f);
        out_rgb[pix * 3 + 1] = fminf(fmaxf(cg, 0.0f), 1.0f);
        out_rgb[pix * 3 + 2] = fminf(fmaxf(cb, 0.0f), 1.0f);
    }
}
"""


def _ptr(t: torch.Tensor | None) -> ctypes.c_uint64:
    return ctypes.c_uint64(0 if t is None else t.data_ptr())


class _NvrtcDepthKernels:
    """In-memory NVRTC compiler + CUDA Driver API launcher for `fast_3dgs_depth.cu`."""

    def __init__(self, device: torch.device, cu_path: Path = CU_PATH):
        torch.cuda.init()
        with torch.cuda.device(device):
            _ = torch.empty(1, device=device)

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if hasattr(os, "add_dll_directory") and torch_lib.exists():
            os.add_dll_directory(str(torch_lib))

        nvrtc_dlls = sorted(torch_lib.glob("nvrtc64_*.dll"))
        if not nvrtc_dlls:
            raise RuntimeError(f"Could not find nvrtc64_*.dll inside {torch_lib}")
        self.nvrtc = ctypes.CDLL(str(nvrtc_dlls[-1]))
        self.cuda = ctypes.CDLL("nvcuda.dll")
        self._configure_ctypes()

        src_bytes = cu_path.read_bytes() if cu_path.exists() else _FALLBACK_CU_SOURCE.encode("utf-8")

        major, minor = torch.cuda.get_device_capability(device)
        arch_candidates: list[str] = []
        for cand in (f"compute_{major}{minor}", "compute_86", "compute_75"):
            if cand not in arch_candidates:
                arch_candidates.append(cand)

        ptx_buf = None
        last_err = ""
        for arch in arch_candidates:
            prog = ctypes.c_void_p()
            rc = self.nvrtc.nvrtcCreateProgram(
                ctypes.byref(prog), src_bytes, b"fast_3dgs_depth.cu", 0, None, None
            )
            if rc != 0:
                last_err = f"nvrtcCreateProgram failed: rc={rc}"
                continue

            opts = (ctypes.c_char_p * 3)(
                f"--gpu-architecture={arch}".encode("ascii"),
                b"--std=c++17",
                b"--fmad=true",
            )
            rc = self.nvrtc.nvrtcCompileProgram(prog, 3, opts)
            log_sz = ctypes.c_size_t()
            self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
            compile_log = ""
            if log_sz.value > 1:
                log_buf = ctypes.create_string_buffer(log_sz.value)
                self.nvrtc.nvrtcGetProgramLog(prog, log_buf)
                compile_log = log_buf.value.decode("utf-8", errors="replace")

            if rc == 0:
                ptx_sz = ctypes.c_size_t()
                self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_sz))
                ptx_buf = ctypes.create_string_buffer(ptx_sz.value)
                self.nvrtc.nvrtcGetPTX(prog, ptx_buf)
                self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
                break

            self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            last_err = f"NVRTC compile failed ({arch}, rc={rc}):\n{compile_log}"

        if ptx_buf is None:
            raise RuntimeError(last_err or "NVRTC compilation failed")

        self.module = ctypes.c_void_p()
        with torch.cuda.device(device):
            rc = self.cuda.cuModuleLoadData(ctypes.byref(self.module), ptx_buf)
        if rc != 0:
            raise RuntimeError(f"cuModuleLoadData failed: rc={rc}")

        self.fns: dict[str, ctypes.c_void_p] = {}
        for name in (
            "emit_tile_keys_kernel",
            "find_tile_ranges_kernel",
            "rasterize_median_depth_16x16",
        ):
            fn = ctypes.c_void_p()
            rc = self.cuda.cuModuleGetFunction(ctypes.byref(fn), self.module, name.encode("ascii"))
            if rc != 0:
                raise RuntimeError(f"cuModuleGetFunction({name}) failed: rc={rc}")
            self.fns[name] = fn

    def _configure_ctypes(self) -> None:
        for fn_name in (
            "nvrtcCreateProgram",
            "nvrtcCompileProgram",
            "nvrtcGetProgramLogSize",
            "nvrtcGetProgramLog",
            "nvrtcGetPTXSize",
            "nvrtcGetPTX",
            "nvrtcDestroyProgram",
        ):
            getattr(self.nvrtc, fn_name).restype = ctypes.c_int

        for fn_name in ("cuModuleLoadData", "cuModuleGetFunction", "cuLaunchKernel"):
            getattr(self.cuda, fn_name).restype = ctypes.c_int

    def launch(
        self,
        name: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args_c: list[Any],
        device: torch.device,
    ) -> None:
        n_args = len(args_c)
        void_arr = (ctypes.c_void_p * n_args)(
            *[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args_c]
        )
        stream = ctypes.c_void_p(int(torch.cuda.current_stream(device).cuda_stream))
        rc = self.cuda.cuLaunchKernel(
            self.fns[name],
            ctypes.c_uint(int(grid[0])),
            ctypes.c_uint(int(grid[1])),
            ctypes.c_uint(int(grid[2])),
            ctypes.c_uint(int(block[0])),
            ctypes.c_uint(int(block[1])),
            ctypes.c_uint(int(block[2])),
            ctypes.c_uint(0),
            stream,
            void_arr,
            None,
        )
        if rc != 0:
            raise RuntimeError(f"cuLaunchKernel({name}) failed with CUDA Driver error {rc}")


_NVRTC_CACHE: dict[int, _NvrtcDepthKernels | None] = {}


def _get_nvrtc_kernels(device: torch.device) -> _NvrtcDepthKernels | None:
    dev_idx = 0 if device.index is None else int(device.index)
    if dev_idx in _NVRTC_CACHE:
        return _NVRTC_CACHE[dev_idx]
    try:
        kernels = _NvrtcDepthKernels(device=device)
        _NVRTC_CACHE[dev_idx] = kernels
        return kernels
    except Exception as exc:
        warnings.warn(
            f"[mod_nvrtc_fast_depth_rgb] NVRTC initialization failed ({exc}); "
            "falling back to vectorized PyTorch 64x64 tile rasterizer.",
            RuntimeWarning,
            stacklevel=2,
        )
        _NVRTC_CACHE[dev_idx] = None
        return None


def _eval_sh_any(
    sh_dc: torch.Tensor,
    sh_rest: torch.Tensor,
    dirs: torch.Tensor,
    sh_degree: int,
) -> torch.Tensor:
    """Evaluate spherical harmonics for degree 0, 1, 2, or 3."""
    c0 = 0.28209479177387814
    res = c0 * sh_dc
    if sh_degree >= 1 and sh_rest.shape[1] >= 3:
        c1 = 0.4886025119029199
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        res = res - c1 * y * sh_rest[:, 0] + c1 * z * sh_rest[:, 1] - c1 * x * sh_rest[:, 2]

        if sh_degree >= 2 and sh_rest.shape[1] >= 8:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            res = (
                res
                + 1.0925484305920792 * xy * sh_rest[:, 3]
                - 1.0925484305920792 * yz * sh_rest[:, 4]
                + 0.31539156525252005 * (2.0 * zz - xx - yy) * sh_rest[:, 5]
                - 1.0925484305920792 * xz * sh_rest[:, 6]
                + 0.5462742152960396 * (xx - yy) * sh_rest[:, 7]
            )
            if sh_degree >= 3 and sh_rest.shape[1] >= 15:
                res = (
                    res
                    - 0.5900435899266435 * y * (3.0 * xx - yy) * sh_rest[:, 8]
                    + 2.890611442640554 * xy * z * sh_rest[:, 9]
                    - 0.4570457994644658 * y * (4.0 * zz - xx - yy) * sh_rest[:, 10]
                    + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_rest[:, 11]
                    - 0.4570457994644658 * x * (4.0 * zz - xx - yy) * sh_rest[:, 12]
                    + 1.445305721320277 * z * (xx - yy) * sh_rest[:, 13]
                    - 0.5900435899266435 * x * (xx - 3.0 * yy) * sh_rest[:, 14]
                )
    return torch.clamp(res + 0.5, min=0.0)


def _to_cuda_f32(val: Any, device: torch.device) -> torch.Tensor:
    if isinstance(val, torch.Tensor):
        return val.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(val, dtype=np.float32), device=device, dtype=torch.float32)


def _extract_pose_params(
    pose: dict, H: int, W: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, float, float, float, float]:
    """Extract w2c (R, t) and intrinsics scaled cleanly to target (W, H)."""
    if "R" in pose and "t" in pose:
        R = _to_cuda_f32(pose["R"], device).reshape(3, 3)
        t = _to_cuda_f32(pose["t"], device).reshape(3)
    elif "w2c" in pose:
        w2c = _to_cuda_f32(pose["w2c"], device)
        R = w2c[:3, :3].contiguous()
        t = w2c[:3, 3].contiguous()
    elif "c2w" in pose:
        c2w = _to_cuda_f32(pose["c2w"], device)
        R_cw = c2w[:3, :3]
        R = R_cw.T.contiguous()
        t = -(R @ c2w[:3, 3]).contiguous()
    else:
        raise KeyError("Pose dict must contain ('R', 't'), 'w2c', or 'c2w'.")

    cam = pose.get("cam", pose)
    if "K" in cam:
        K = np.asarray(cam["K"], dtype=np.float64)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
    else:
        fx = float(cam["fx"])
        fy = float(cam.get("fy", fx))
        cx = float(cam.get("cx", 0.5 * float(cam.get("w", W))))
        cy = float(cam.get("cy", 0.5 * float(cam.get("h", H))))

    orig_w = float(cam.get("w", pose.get("w", W)))
    orig_h = float(cam.get("h", pose.get("h", H)))
    if orig_w > 0.0 and orig_h > 0.0 and (orig_w != float(W) or orig_h != float(H)):
        sx = float(W) / orig_w
        sy = float(H) / orig_h
        fx *= sx
        cx *= sx
        fy *= sy
        cy *= sy

    return R, t, fx, fy, cx, cy


def _prepare_splats(splats: dict, device: torch.device) -> dict[str, Any]:
    """Normalize splats tensors onto CUDA, prune invisible opacities, and precompute 3D covariances once."""
    xyz = _to_cuda_f32(splats["xyz"], device)
    opac = _to_cuda_f32(splats["opac"], device).reshape(-1)
    scales = _to_cuda_f32(splats["scales"], device)
    quats = _to_cuda_f32(splats["quats"], device)
    sh_dc = _to_cuda_f32(splats["sh_dc"], device)

    valid_op = opac > (1.0 / 255.0)
    if not torch.any(valid_op):
        return {"empty": True}

    xyz = xyz[valid_op].contiguous()
    opac = opac[valid_op].contiguous()
    scales = scales[valid_op].contiguous()
    quats = quats[valid_op]
    quats = quats / torch.clamp(torch.norm(quats, dim=-1, keepdim=True), min=1e-12)
    sh_dc = sh_dc[valid_op].contiguous()

    sh_channel_major = bool(splats.get("sh_channel_major", True))
    if "sh_rest" in splats and "sh_rest_raw" not in splats:
        sh_rest = _to_cuda_f32(splats["sh_rest"], device)[valid_op].contiguous()
        n_coeffs = int(sh_rest.shape[1]) if sh_rest.ndim == 3 else 0
    else:
        sh_raw = splats.get("sh_rest_raw")
        if sh_raw is None or (isinstance(sh_raw, torch.Tensor) and sh_raw.numel() == 0):
            n_coeffs = 0
            sh_rest = torch.empty((xyz.shape[0], 0, 3), dtype=torch.float32, device=device)
        else:
            sh_raw_t = _to_cuda_f32(sh_raw, device)[valid_op]
            if sh_raw_t.ndim == 3:
                n_coeffs = int(sh_raw_t.shape[1])
                sh_rest = sh_raw_t.contiguous()
            else:
                n_coeffs = int(splats.get("n_coeffs", sh_raw_t.shape[1] // 3))
                if n_coeffs > 0:
                    if sh_channel_major:
                        sh_rest = sh_raw_t[:, : 3 * n_coeffs].view(-1, 3, n_coeffs).permute(0, 2, 1).contiguous()
                    else:
                        sh_rest = sh_raw_t[:, : 3 * n_coeffs].view(-1, n_coeffs, 3).contiguous()
                else:
                    sh_rest = torch.empty((xyz.shape[0], 0, 3), dtype=torch.float32, device=device)

    if "sh_degree" in splats:
        sh_deg = int(splats["sh_degree"])
    elif n_coeffs >= 15:
        sh_deg = 3
    elif n_coeffs >= 8:
        sh_deg = 2
    elif n_coeffs >= 3:
        sh_deg = 1
    else:
        sh_deg = 0

    qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    Rq = torch.stack(
        [
            torch.stack([1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)], dim=-1),
            torch.stack([2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)], dim=-1),
            torch.stack([2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)], dim=-1),
        ],
        dim=-2,
    )
    M_mat = Rq * scales[:, None, :]
    sigma3d = (M_mat @ M_mat.transpose(1, 2)).contiguous()

    return {
        "empty": False,
        "xyz": xyz,
        "opac": opac,
        "sigma3d": sigma3d,
        "sh_dc": sh_dc,
        "sh_rest": sh_rest,
        "sh_deg": sh_deg,
    }


def _project_and_sort_view(
    prep: dict[str, Any],
    R: torch.Tensor,
    t: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    H: int,
    W: int,
    eps2d: float = 0.3,
) -> tuple[torch.Tensor, ...] | None:
    """Vectorized PyTorch CUDA projection, 2D conic computation, frustum culling, SH eval, and z-sort."""
    xyz = prep["xyz"]
    p_cam = xyz @ R.T + t[None, :]
    z = p_cam[:, 2]

    valid = z > 0.2
    if not torch.any(valid):
        return None

    xyz = xyz[valid]
    p_cam = p_cam[valid]
    z = z[valid]
    opac = prep["opac"][valid]
    sigma_v = prep["sigma3d"][valid]
    sh_dc = prep["sh_dc"][valid]
    sh_rest = prep["sh_rest"][valid]

    sigma_cam = R[None, :, :] @ sigma_v @ R.T[None, :, :]

    tan_fovx = 0.5 * W / fx
    tan_fovy = 0.5 * H / fy
    lim_x = 1.3 * tan_fovx
    lim_y = 1.3 * tan_fovy
    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    tx_z = torch.clamp(p_cam[:, 0] * inv_z, min=-lim_x, max=lim_x) * z
    ty_z = torch.clamp(p_cam[:, 1] * inv_z, min=-lim_y, max=lim_y) * z

    zero = torch.zeros_like(z)
    J = torch.stack(
        [
            torch.stack([fx * inv_z, zero, -fx * tx_z * inv_z2], dim=-1),
            torch.stack([zero, fy * inv_z, -fy * ty_z * inv_z2], dim=-1),
        ],
        dim=-2,
    )
    cov2d = J @ sigma_cam @ J.transpose(1, 2)
    cov_a = cov2d[:, 0, 0] + eps2d
    cov_b = cov2d[:, 0, 1]
    cov_c = cov2d[:, 1, 1] + eps2d
    det = cov_a * cov_c - cov_b * cov_b

    mid = 0.5 * (cov_a + cov_c)
    disc = torch.sqrt(torch.clamp(mid * mid - det, min=1e-8))
    lam_max = mid + disc
    radius = torch.ceil(3.0 * torch.sqrt(lam_max))

    u = fx * (p_cam[:, 0] * inv_z) + cx
    v = fy * (p_cam[:, 1] * inv_z) + cy

    vis = (
        (det > 1e-10)
        & (u + radius >= 0.0)
        & (u - radius < float(W))
        & (v + radius >= 0.0)
        & (v - radius < float(H))
    )
    if not torch.any(vis):
        return None

    u = u[vis]
    v = v[vis]
    radius = radius[vis]
    z_cam = z[vis]
    opac = opac[vis]
    inv_det = 1.0 / det[vis]
    conic_a = cov_c[vis] * inv_det
    conic_b = -cov_b[vis] * inv_det
    conic_c = cov_a[vis] * inv_det
    xyz = xyz[vis]
    sh_dc = sh_dc[vis]
    sh_rest = sh_rest[vis]

    cam_center = -(R.T @ t)
    dirs = xyz - cam_center[None, :]
    dirs = dirs / torch.clamp(torch.norm(dirs, dim=-1, keepdim=True), min=1e-12)
    rgbs = _eval_sh_any(sh_dc, sh_rest, dirs, prep["sh_deg"])

    order = torch.argsort(z_cam)
    return (
        u[order].contiguous(),
        v[order].contiguous(),
        radius[order].contiguous(),
        conic_a[order].contiguous(),
        conic_b[order].contiguous(),
        conic_c[order].contiguous(),
        z_cam[order].contiguous(),
        opac[order].contiguous(),
        rgbs[order].contiguous(),
    )


def _rasterize_nvrtc_16x16_view(
    kernels: _NvrtcDepthKernels,
    u: torch.Tensor,
    v: torch.Tensor,
    radius: torch.Tensor,
    conic_a: torch.Tensor,
    conic_b: torch.Tensor,
    conic_c: torch.Tensor,
    z_cam: torch.Tensor,
    opac: torch.Tensor,
    rgbs: torch.Tensor,
    H: int,
    W: int,
    out_depth: torch.Tensor,
    out_alpha: torch.Tensor,
    out_rgb: torch.Tensor,
) -> None:
    """Tile-bin sorted splats into 16x16 blocks and launch NVRTC median-depth + RGB rasterizer."""
    device = u.device
    M = int(u.shape[0])
    if M == 0:
        return

    tiles_x = (W + 15) // 16
    tiles_y = (H + 15) // 16

    tx0 = torch.clamp(torch.floor((u - radius - 0.5) / 16.0).to(torch.int32), 0, tiles_x)
    ty0 = torch.clamp(torch.floor((v - radius - 0.5) / 16.0).to(torch.int32), 0, tiles_y)
    tx1 = torch.clamp(torch.floor((u + radius + 0.5) / 16.0).to(torch.int32) + 1, 0, tiles_x)
    ty1 = torch.clamp(torch.floor((v + radius + 0.5) / 16.0).to(torch.int32) + 1, 0, tiles_y)

    tile_bbox = torch.stack([tx0, ty0, tx1, ty1], dim=-1).contiguous()
    tiles_touched = (torch.clamp(tx1 - tx0, min=0) * torch.clamp(ty1 - ty0, min=0)).to(torch.int32)

    tt_i64 = tiles_touched.to(torch.int64)
    cum = torch.cumsum(tt_i64, dim=0)
    total_intersections = int(cum[-1].item())
    if total_intersections <= 0:
        return

    offsets = (cum - tt_i64).contiguous()
    keys = torch.empty((total_intersections,), dtype=torch.int64, device=device)

    blk = 256
    grd_m = (M + blk - 1) // blk
    kernels.launch(
        "emit_tile_keys_kernel",
        (grd_m, 1, 1),
        (blk, 1, 1),
        [
            _ptr(tile_bbox),
            _ptr(tiles_touched),
            _ptr(offsets),
            ctypes.c_int(M),
            ctypes.c_int(tiles_x),
            _ptr(keys),
        ],
        device=device,
    )

    sorted_keys = torch.sort(keys).values.contiguous()
    del keys, tile_bbox, tiles_touched, offsets, tt_i64, cum

    tile_ranges = torch.zeros((tiles_x * tiles_y, 2), dtype=torch.int32, device=device)
    grd_k = (total_intersections + blk - 1) // blk
    kernels.launch(
        "find_tile_ranges_kernel",
        (grd_k, 1, 1),
        (blk, 1, 1),
        [
            _ptr(sorted_keys),
            ctypes.c_longlong(total_intersections),
            _ptr(tile_ranges),
        ],
        device=device,
    )

    splat_data = torch.stack(
        [
            u,
            v,
            conic_a,
            conic_b,
            conic_c,
            opac,
            z_cam,
            rgbs[:, 0],
            rgbs[:, 1],
            rgbs[:, 2],
        ],
        dim=-1,
    ).contiguous()

    kernels.launch(
        "rasterize_median_depth_16x16",
        (tiles_x, tiles_y, 1),
        (16, 16, 1),
        [
            _ptr(tile_ranges),
            _ptr(sorted_keys),
            _ptr(splat_data),
            ctypes.c_int(W),
            ctypes.c_int(H),
            ctypes.c_int(tiles_x),
            _ptr(out_depth),
            _ptr(out_alpha),
            _ptr(out_rgb),
        ],
        device=device,
    )


def _rasterize_pytorch_64x64_view(
    u: torch.Tensor,
    v: torch.Tensor,
    radius: torch.Tensor,
    conic_a: torch.Tensor,
    conic_b: torch.Tensor,
    conic_c: torch.Tensor,
    z_cam: torch.Tensor,
    opac: torch.Tensor,
    rgbs: torch.Tensor,
    H: int,
    W: int,
    out_depth: torch.Tensor,
    out_alpha: torch.Tensor,
    out_rgb: torch.Tensor,
    tile_size: int = 64,
    chunk: int = 512,
) -> None:
    """Bulletproof vectorized PyTorch CUDA 64x64 tile fallback with exact median transmittance depth."""
    device = u.device
    if u.numel() == 0:
        return

    u_min = u - radius
    u_max = u + radius
    v_min = v - radius
    v_max = v + radius

    for y0 in range(0, H, tile_size):
        y1 = min(y0 + tile_size, H)
        v_mask = (v_max >= y0 - 0.5) & (v_min <= y1 - 0.5)
        if not torch.any(v_mask):
            continue
        for x0 in range(0, W, tile_size):
            x1 = min(x0 + tile_size, W)
            m = v_mask & (u_max >= x0 - 0.5) & (u_min <= x1 - 0.5)
            idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            th, tw = y1 - y0, x1 - x0
            n_pix = th * tw
            py, px = torch.meshgrid(
                torch.arange(y0, y1, device=device, dtype=torch.float32) + 0.5,
                torch.arange(x0, x1, device=device, dtype=torch.float32) + 0.5,
                indexing="ij",
            )
            px_f = px.reshape(-1, 1)
            py_f = py.reshape(-1, 1)

            T = torch.ones((n_pix,), dtype=torch.float32, device=device)
            acc_rgb = torch.zeros((n_pix, 3), dtype=torch.float32, device=device)
            depth_w_sum = torch.zeros((n_pix,), dtype=torch.float32, device=device)
            weight_sum = torch.zeros((n_pix,), dtype=torch.float32, device=device)
            median_depth = torch.zeros((n_pix,), dtype=torch.float32, device=device)
            has_median = torch.zeros((n_pix,), dtype=torch.bool, device=device)

            for s in range(0, idx.numel(), chunk):
                sl = idx[s : s + chunk]
                dx = px_f - u[sl][None, :]
                dy = py_f - v[sl][None, :]
                power = -0.5 * (
                    conic_a[sl][None, :] * dx * dx
                    + 2.0 * conic_b[sl][None, :] * dx * dy
                    + conic_c[sl][None, :] * dy * dy
                )
                alpha = opac[sl][None, :] * torch.exp(torch.clamp(power, max=0.0))
                alpha = torch.where(
                    (power <= 1e-3) & (power >= -8.0) & (alpha >= (1.0 / 255.0)),
                    torch.clamp(alpha, max=0.99),
                    torch.zeros_like(alpha),
                )

                one_minus = 1.0 - alpha
                cum = torch.cumprod(one_minus, dim=1)
                T_local_before = torch.cat(
                    [torch.ones((n_pix, 1), dtype=torch.float32, device=device), cum[:, :-1]], dim=1
                )
                T_before = T[:, None] * T_local_before
                T_after = T[:, None] * cum
                weights = T_before * alpha

                z_sl = z_cam[sl]
                acc_rgb = acc_rgb + weights @ rgbs[sl]
                depth_w_sum = depth_w_sum + weights @ z_sl
                weight_sum = weight_sum + weights.sum(dim=1)

                crossed = (T_after <= 0.50) & (~has_median[:, None])
                any_crossed = torch.any(crossed, dim=1)
                if torch.any(any_crossed):
                    is_first = crossed & (torch.cumsum(crossed.to(torch.int32), dim=1) == 1)
                    z_first = is_first.to(torch.float32) @ z_sl
                    median_depth = torch.where(any_crossed, z_first, median_depth)
                    has_median = has_median | any_crossed

                T = T_after[:, -1]
                if float(T.max()) < 1e-4:
                    break

            final_alpha = torch.clamp(1.0 - T, 0.0, 1.0)
            fallback_d = torch.where(
                weight_sum > 1e-8,
                depth_w_sum / weight_sum.clamp(min=1e-8),
                torch.zeros_like(depth_w_sum),
            )
            depth_tile = torch.where(
                has_median,
                median_depth,
                torch.where(final_alpha >= 0.15, fallback_d, torch.zeros_like(depth_w_sum)),
            )

            out_depth[y0:y1, x0:x1] = depth_tile.view(th, tw)
            out_alpha[y0:y1, x0:x1] = final_alpha.view(th, tw)
            out_rgb[y0:y1, x0:x1] = torch.clamp(acc_rgb, 0.0, 1.0).view(th, tw, 3)


@torch.no_grad()
def render_36_views_median_depth_cuda(
    splats: dict,
    poses: list[dict],
    H: int = 640,
    W: int = 640,
    *,
    force_pytorch_fallback: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render multi-view median transmittance depth, opacity alpha, and RGB on CUDA.

    Args:
        splats: Dict of 3DGS tensors (`xyz`, `opac`, `scales`, `quats`, `sh_dc`, `sh_rest_raw`, ...).
        poses: List of COLMAP/camera pose dicts containing `R`, `t`, and `cam` intrinsics.
        H: Output render height.
        W: Output render width.
        force_pytorch_fallback: Force vectorized PyTorch 64x64 tile rasterization.

    Returns:
        (depths_hw, alphas_hw, rgbs_hwc) of shapes `(N_views, H, W)`, `(N_views, H, W)`,
        and `(N_views, H, W, 3)` as `torch.float32` tensors on CUDA.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("render_36_views_median_depth_cuda requires a CUDA-capable GPU.")

    xyz_raw = splats["xyz"]
    device = xyz_raw.device if isinstance(xyz_raw, torch.Tensor) and xyz_raw.device.type == "cuda" else torch.device("cuda")

    n_views = len(poses)
    depths_hw = torch.zeros((n_views, H, W), dtype=torch.float32, device=device)
    alphas_hw = torch.zeros((n_views, H, W), dtype=torch.float32, device=device)
    rgbs_hwc = torch.zeros((n_views, H, W, 3), dtype=torch.float32, device=device)
    if n_views == 0:
        return depths_hw, alphas_hw, rgbs_hwc

    prep = _prepare_splats(splats, device)
    if prep["empty"]:
        return depths_hw, alphas_hw, rgbs_hwc

    kernels = None if force_pytorch_fallback else _get_nvrtc_kernels(device)

    for v_idx, pose in enumerate(poses):
        R, t, fx, fy, cx, cy = _extract_pose_params(pose, H=H, W=W, device=device)
        proj = _project_and_sort_view(prep, R, t, fx, fy, cx, cy, H=H, W=W)
        if proj is None:
            continue
        u, v, radius, conic_a, conic_b, conic_c, z_cam, opac, rgbs = proj

        if kernels is not None:
            _rasterize_nvrtc_16x16_view(
                kernels,
                u,
                v,
                radius,
                conic_a,
                conic_b,
                conic_c,
                z_cam,
                opac,
                rgbs,
                H=H,
                W=W,
                out_depth=depths_hw[v_idx],
                out_alpha=alphas_hw[v_idx],
                out_rgb=rgbs_hwc[v_idx],
            )
        else:
            _rasterize_pytorch_64x64_view(
                u,
                v,
                radius,
                conic_a,
                conic_b,
                conic_c,
                z_cam,
                opac,
                rgbs,
                H=H,
                W=W,
                out_depth=depths_hw[v_idx],
                out_alpha=alphas_hw[v_idx],
                out_rgb=rgbs_hwc[v_idx],
            )

    return depths_hw, alphas_hw, rgbs_hwc
