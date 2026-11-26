// Native CUDA C++ Kernels (NVRTC compute_86) implementing Do Duc Hanh (Nov 2026):
// "DCA Inspired Block Quadratic Optimization for Gaussian Splatting on CPU and GPU" (GPU-QP2)
// Scalable tile-sparse EWA renderer + exact analytic/central-diff Jacobians + Warp FP64 Active-Set Cholesky Box-QP.

typedef unsigned long long size_t;

#define TILE_SIZE 16
#define TILE_PIXELS 256
#define FULL_MASK 0xffffffffu

__device__ __forceinline__ double shfl_down_f64(double val, int delta) {
    long long bits = __double_as_longlong(val);
    bits = __shfl_down_sync(FULL_MASK, bits, delta);
    return __longlong_as_double(bits);
}

__device__ __forceinline__ double shfl_f64(double val, int srcLane) {
    long long bits = __double_as_longlong(val);
    bits = __shfl_sync(FULL_MASK, bits, srcLane);
    return __longlong_as_double(bits);
}

__device__ __forceinline__ float d_sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    val += __shfl_down_sync(FULL_MASK, val, 16);
    val += __shfl_down_sync(FULL_MASK, val, 8);
    val += __shfl_down_sync(FULL_MASK, val, 4);
    val += __shfl_down_sync(FULL_MASK, val, 2);
    val += __shfl_down_sync(FULL_MASK, val, 1);
    return val;
}

__device__ __forceinline__ double warp_reduce_max_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other = shfl_down_f64(val, offset);
        if (other > val) val = other;
    }
    return shfl_f64(val, 0);
}

// Rodrigues rotation matrix R = Exp([w]_x)
__device__ __forceinline__ void rodrigues_exp(float wx, float wy, float wz, float R[9]) {
    float th2 = wx * wx + wy * wy + wz * wz;
    if (th2 < 1e-10f) {
        R[0] = 1.0f; R[1] = -wz;  R[2] = wy;
        R[3] = wz;   R[4] = 1.0f; R[5] = -wx;
        R[6] = -wy;  R[7] = wx;   R[8] = 1.0f;
    } else {
        float th = sqrtf(th2);
        float inv_th = 1.0f / th;
        float kx = wx * inv_th, ky = wy * inv_th, kz = wz * inv_th;
        float s = sinf(th), c = cosf(th), omc = 1.0f - c;
        R[0] = c + kx * kx * omc;      R[1] = kx * ky * omc - kz * s; R[2] = kx * kz * omc + ky * s;
        R[3] = ky * kx * omc + kz * s; R[4] = c + ky * ky * omc;      R[5] = ky * kz * omc - kx * s;
        R[6] = kz * kx * omc - ky * s; R[7] = kz * ky * omc + kx * s; R[8] = c + kz * kz * omc;
    }
}

// Evaluate projected mean (ux, uy) and 2D covariance (Sxx, Sxy, Syy) per Eq. (1)-(2)
__device__ __forceinline__ bool eval_proj_single(
    float mx, float my, float mz,
    float sx, float sy, float sz,
    float wx, float wy, float wz,
    const float Rv[9], const float Cv[3],
    float fx, float fy, float cx, float cy,
    int W, int H,
    float& ux, float& uy, float& Sxx, float& Sxy, float& Syy, float& z_out
) {
    float dx = mx - Cv[0], dy = my - Cv[1], dz = mz - Cv[2];
    float px = Rv[0] * dx + Rv[1] * dy + Rv[2] * dz;
    float py = Rv[3] * dx + Rv[4] * dy + Rv[5] * dz;
    float pz = Rv[6] * dx + Rv[7] * dy + Rv[8] * dz;
    z_out = pz;
    if (pz <= 0.05f) return false;

    float inv_z = 1.0f / pz;
    float inv_z2 = inv_z * inv_z;
    ux = fx * px * inv_z + cx;
    uy = fy * py * inv_z + cy;

    float Rg[9];
    rodrigues_exp(wx, wy, wz, Rg);

    // W_mat = Rv * Rg
    float M[9];
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            M[r * 3 + c] = Rv[r * 3 + 0] * Rg[0 * 3 + c]
                         + Rv[r * 3 + 1] * Rg[1 * 3 + c]
                         + Rv[r * 3 + 2] * Rg[2 * 3 + c];
        }
    }
    float e2x = expf(2.0f * fminf(fmaxf(sx, -12.0f), 2.0f));
    float e2y = expf(2.0f * fminf(fmaxf(sy, -12.0f), 2.0f));
    float e2z = expf(2.0f * fminf(fmaxf(sz, -12.0f), 2.0f));

    // Sigma_cam = M * diag(e2) * M^T (symmetric 3x3)
    float Sc00 = M[0]*M[0]*e2x + M[1]*M[1]*e2y + M[2]*M[2]*e2z;
    float Sc01 = M[0]*M[3]*e2x + M[1]*M[4]*e2y + M[2]*M[5]*e2z;
    float Sc02 = M[0]*M[6]*e2x + M[1]*M[7]*e2y + M[2]*M[8]*e2z;
    float Sc11 = M[3]*M[3]*e2x + M[4]*M[4]*e2y + M[5]*M[5]*e2z;
    float Sc12 = M[3]*M[6]*e2x + M[4]*M[7]*e2y + M[5]*M[8]*e2z;
    float Sc22 = M[6]*M[6]*e2x + M[7]*M[7]*e2y + M[8]*M[8]*e2z;

    // Perspective Jacobian D pi_v evaluated at camera-space center (Eq. 2 & Section 3.1)
    float tan_fovx = 0.5f * (float)W / fx;
    float tan_fovy = 0.5f * (float)H / fy;
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    float tx = fminf(fmaxf(px * inv_z, -lim_x), lim_x) * pz;
    float ty = fminf(fmaxf(py * inv_z, -lim_y), lim_y) * pz;

    float j00 = fx * inv_z, j02 = -fx * tx * inv_z2;
    float j11 = fy * inv_z, j12 = -fy * ty * inv_z2;

    float t00 = j00 * Sc00 + j02 * Sc02;
    float t01 = j00 * Sc01 + j02 * Sc12;
    float t02 = j00 * Sc02 + j02 * Sc22;
    float t10 = j11 * Sc01 + j12 * Sc02;
    float t11 = j11 * Sc11 + j12 * Sc12;
    float t12 = j11 * Sc12 + j12 * Sc22;

    Sxx = t00 * j00 + t02 * j02 + 0.3f;
    Sxy = t00 * 0.0f + t01 * j11 + t02 * j12;
    Syy = t11 * j11 + t12 * j12 + 0.3f;
    return true;
}

extern "C" __global__ void project_gaussians_fd_kernel(
    const float* __restrict__ params,       // [N, 13]
    const float* __restrict__ cam_R,        // [B, 9]
    const float* __restrict__ cam_C,        // [B, 3]
    const float* __restrict__ cam_K,        // [B, 4] (fx, fy, cx, cy)
    const int* __restrict__ roi_tiles,      // [B, 4] (tx_min, ty_min, tx_max, ty_max)
    int B, int N, int W, int H, int tiles_x, int tiles_y,
    float pos_eps,                          // 1e-4 * scene_scale
    int compute_jac,                        // 1 during grad/curv pass, 0 during render-only check!
    float* __restrict__ proj_data,          // [B*N, 9] (ux, uy, con_x, con_y, con_z, opac, cr, cg, cb)
    float* __restrict__ proj_jac,           // [B*N, 45] (5 x 9 central-diff Jacobian)
    int* __restrict__ tile_BBox,            // [B*N, 4] (tx0, ty0, tx1, ty1)
    float* __restrict__ depths,             // [B*N]
    int* __restrict__ tiles_touched         // [B*N]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N) return;

    int v = idx / N;
    int i = idx % N;

    const float* x = params + i * 13;
    float mx = x[0], my = x[1], mz = x[2];
    float sx = x[3], sy = x[4], sz = x[5];
    float wx = x[6], wy = x[7], wz = x[8];
    float oi = x[9];
    float lr = x[10], lg = x[11], lb = x[12];

    float ai = d_sigmoid(oi);
    if (ai < 1e-4f) {
        tiles_touched[idx] = 0;
        return;
    }

    const float* Rv = cam_R + v * 9;
    const float* Cv = cam_C + v * 3;
    float fx = cam_K[v * 4 + 0], fy = cam_K[v * 4 + 1];
    float cx = cam_K[v * 4 + 2], cy = cam_K[v * 4 + 3];

    float ux, uy, Sxx, Sxy, Syy, z_cam;
    if (!eval_proj_single(mx, my, mz, sx, sy, sz, wx, wy, wz, Rv, Cv, fx, fy, cx, cy, W, H,
                          ux, uy, Sxx, Sxy, Syy, z_cam)) {
        tiles_touched[idx] = 0;
        return;
    }

    float det = Sxx * Syy - Sxy * Sxy;
    if (det <= 1e-10f) {
        tiles_touched[idx] = 0;
        return;
    }
    float inv_det = 1.0f / det;
    float con_x = Syy * inv_det;
    float con_y = -Sxy * inv_det;
    float con_z = Sxx * inv_det;

    float mid = 0.5f * (Sxx + Syy);
    float disc = sqrtf(fmaxf(0.01f, mid * mid - det));
    float lam_max = mid + disc;
    // Paper Section 3.1: clip rectangular footprint radius (40 px at low-res, 52 px at 1600x1600)
    float max_rad = (W > 1000) ? 52.0f : 40.0f;
    float rad = fminf(ceilf(3.0f * sqrtf(lam_max)), max_rad);

    if (ux + rad < 0.0f || ux - rad >= (float)W || uy + rad < 0.0f || uy - rad >= (float)H) {
        tiles_touched[idx] = 0;
        return;
    }

    int r_tx0 = roi_tiles[v * 4 + 0], r_ty0 = roi_tiles[v * 4 + 1];
    int r_tx1 = roi_tiles[v * 4 + 2], r_ty1 = roi_tiles[v * 4 + 3];

    int tx0 = max(r_tx0, min(tiles_x, (int)floorf((ux - rad) / (float)TILE_SIZE)));
    int ty0 = max(r_ty0, min(tiles_y, (int)floorf((uy - rad) / (float)TILE_SIZE)));
    int tx1 = max(r_tx0, min(r_tx1, (int)floorf((ux + rad) / (float)TILE_SIZE) + 1));
    int ty1 = max(r_ty0, min(r_ty1, (int)floorf((uy + rad) / (float)TILE_SIZE) + 1));

    int nt = max(0, tx1 - tx0) * max(0, ty1 - ty0);
    if (nt <= 0) {
        tiles_touched[idx] = 0;
        return;
    }

    float* pd = proj_data + idx * 9;
    pd[0] = ux; pd[1] = uy;
    pd[2] = con_x; pd[3] = con_y; pd[4] = con_z;
    pd[5] = ai;
    pd[6] = d_sigmoid(lr);
    pd[7] = d_sigmoid(lg);
    pd[8] = d_sigmoid(lb);

    tile_BBox[idx * 4 + 0] = tx0;
    tile_BBox[idx * 4 + 1] = ty0;
    tile_BBox[idx * 4 + 2] = tx1;
    tile_BBox[idx * 4 + 3] = ty1;
    depths[idx] = z_cam;
    tiles_touched[idx] = nt;

    if (compute_jac) {
        // Central differences wrt 9 geometry parameters matching Paper Section 3.2!
        float g9[9] = {mx, my, mz, sx, sy, sz, wx, wy, wz};
        float* J = proj_jac + (size_t)idx * 45;
        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            float eps = (k < 3) ? pos_eps : 1e-3f;
            float orig = g9[k];
            float u_p, v_p, sxx_p, sxy_p, syy_p, zp;
            float u_m, v_m, sxx_m, sxy_m, syy_m, zm;

            g9[k] = orig + eps;
            bool ok_p = eval_proj_single(g9[0], g9[1], g9[2], g9[3], g9[4], g9[5], g9[6], g9[7], g9[8],
                                         Rv, Cv, fx, fy, cx, cy, W, H, u_p, v_p, sxx_p, sxy_p, syy_p, zp);
            g9[k] = orig - eps;
            bool ok_m = eval_proj_single(g9[0], g9[1], g9[2], g9[3], g9[4], g9[5], g9[6], g9[7], g9[8],
                                         Rv, Cv, fx, fy, cx, cy, W, H, u_m, v_m, sxx_m, sxy_m, syy_m, zm);
            g9[k] = orig;

            if (ok_p && ok_m) {
                float inv2e = 0.5f / eps;
                J[0 * 9 + k] = (u_p - u_m) * inv2e;
                J[1 * 9 + k] = (v_p - v_m) * inv2e;
                J[2 * 9 + k] = (sxx_p - sxx_m) * inv2e;
                J[3 * 9 + k] = (sxy_p - sxy_m) * inv2e;
                J[4 * 9 + k] = (syy_p - syy_m) * inv2e;
            } else {
                J[0 * 9 + k] = 0.0f; J[1 * 9 + k] = 0.0f;
                J[2 * 9 + k] = 0.0f; J[3 * 9 + k] = 0.0f; J[4 * 9 + k] = 0.0f;
            }
        }
    }
}

extern "C" __global__ void emit_tile_keys_kernel(
    const int* __restrict__ tile_BBox,
    const float* __restrict__ depths,
    const int* __restrict__ tiles_touched,
    const long long* __restrict__ offsets,
    int B, int N, int tiles_x, int tiles_y,
    long long* __restrict__ out_keys,
    int* __restrict__ out_vals
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N) return;
    int nt = tiles_touched[idx];
    if (nt <= 0) return;

    int v = idx / N;
    int tx0 = tile_BBox[idx * 4 + 0];
    int ty0 = tile_BBox[idx * 4 + 1];
    int tx1 = tile_BBox[idx * 4 + 2];
    int ty1 = tile_BBox[idx * 4 + 3];

    // IEEE-754 positive float bits preserve monotonic order!
    unsigned int depth_bits = __float_as_uint(fmaxf(depths[idx], 0.0f));
    long long write_pos = offsets[idx];
    int tiles_per_view = tiles_x * tiles_y;

    for (int ty = ty0; ty < ty1; ++ty) {
        for (int tx = tx0; tx < tx1; ++tx) {
            long long global_tile = (long long)(v * tiles_per_view + ty * tiles_x + tx);
            out_keys[write_pos] = (global_tile << 32) | (long long)depth_bits;
            out_vals[write_pos] = idx;
            ++write_pos;
        }
    }
}

extern "C" __global__ void find_tile_ranges_kernel(
    const long long* __restrict__ sorted_keys,
    long long num_intersections,
    int* __restrict__ tile_ranges // [B * tiles_x * tiles_y, 2] initialized to 0
) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
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

extern "C" __global__ void forward_render_tiles_kernel(
    const int* __restrict__ tile_ranges,
    const int* __restrict__ sorted_vals,
    const float* __restrict__ proj_data, // [B*N, 9]
    int B, int W, int H, int tiles_x, int tiles_y,
    float bg_r, float bg_g, float bg_b,
    float alpha_thresh,
    float* __restrict__ out_img,         // [B, H, W, 3]
    float* __restrict__ out_T,           // [B, H, W]
    int* __restrict__ out_last_idx       // [B, H, W]
) {
    int tx = blockIdx.x;
    int ty = blockIdx.y;
    int v = blockIdx.z;
    int global_tile = v * (tiles_x * tiles_y) + ty * tiles_x + tx;

    int r_start = tile_ranges[global_tile * 2 + 0];
    int r_end = tile_ranges[global_tile * 2 + 1];

    int px_i = tx * TILE_SIZE + threadIdx.x;
    int py_i = ty * TILE_SIZE + threadIdx.y;
    bool inside = (px_i < W && py_i < H);

    if (r_start >= r_end) {
        if (inside) {
            size_t pix = ((size_t)v * H + py_i) * W + px_i;
            out_img[pix * 3 + 0] = bg_r;
            out_img[pix * 3 + 1] = bg_g;
            out_img[pix * 3 + 2] = bg_b;
            out_T[pix] = 1.0f;
            out_last_idx[pix] = r_start - 1;
        }
        return;
    }

    float px = (float)px_i + 0.5f;
    float py = (float)py_i + 0.5f;
    float T = 1.0f;
    float cr = 0.0f, cg = 0.0f, cb = 0.0f;
    int last_pos = r_start - 1;
    bool done = !inside;

    __shared__ float sh_pd[TILE_PIXELS * 9];
    int tid = threadIdx.y * TILE_SIZE + threadIdx.x;

    for (int base = r_start; base < r_end; base += TILE_PIXELS) {
        if (__syncthreads_and(done)) break;

        int load_k = base + tid;
        if (load_k < r_end) {
            int sv = sorted_vals[load_k];
            #pragma unroll
            for (int m = 0; m < 9; ++m) sh_pd[tid * 9 + m] = proj_data[sv * 9 + m];
        }
        __syncthreads();

        int count = min(TILE_PIXELS, r_end - base);
        if (!done) {
            for (int j = 0; j < count; ++j) {
                const float* pd = sh_pd + j * 9;
                float dx = px - pd[0];
                float dy = py - pd[1];
                float pwr = 0.5f * (pd[2] * dx * dx + 2.0f * pd[3] * dx * dy + pd[4] * dy * dy);
                if (pwr > 8.0f) continue;
                float alpha = fminf(0.99f, pd[5] * expf(-pwr));
                if (alpha < alpha_thresh) continue;

                float next_T = T * (1.0f - alpha);
                if (next_T < 1e-4f) {
                    done = true;
                    break;
                }
                float w = T * alpha;
                cr += w * pd[6];
                cg += w * pd[7];
                cb += w * pd[8];
                T = next_T;
                last_pos = base + j;
            }
        }
        __syncthreads();
    }

    if (inside) {
        size_t pix = ((size_t)v * H + py_i) * W + px_i;
        out_img[pix * 3 + 0] = cr + T * bg_r;
        out_img[pix * 3 + 1] = cg + T * bg_g;
        out_img[pix * 3 + 2] = cb + T * bg_b;
        out_T[pix] = T;
        out_last_idx[pix] = last_pos;
    }
}

extern "C" __global__ void backward_grad_curv_tiles_kernel(
    const int* __restrict__ tile_ranges,
    const int* __restrict__ sorted_vals,
    const float* __restrict__ proj_data,     // [B*N, 9]
    const float* __restrict__ out_T,         // [B, H, W]
    const int* __restrict__ out_last_idx,    // [B, H, W]
    const float* __restrict__ dL_dI,         // [B, H, W, 3]
    const float* __restrict__ w_bar,         // [B, H, W, 3] (Eq. 6 weights / (B*P))
    const int* __restrict__ partner_of,      // [N] Morton partner index (-1 if singleton)
    int B, int N, int W, int H, int tiles_x, int tiles_y,
    float bg_r, float bg_g, float bg_b,
    float alpha_thresh,
    int need_curv,
    float* __restrict__ grad_img9,           // [B*N, 9]
    float* __restrict__ curv_img42,          // [B*N, 42] compact exact image-space Gram self-block
    float* __restrict__ adj_cross_img81      // [B*N, 81] exact cross Gram when partner is adjacent in tile!
) {
    int tx = blockIdx.x;
    int ty = blockIdx.y;
    int v = blockIdx.z;
    int global_tile = v * (tiles_x * tiles_y) + ty * tiles_x + tx;

    int r_start = tile_ranges[global_tile * 2 + 0];
    int r_end = tile_ranges[global_tile * 2 + 1];
    if (r_start >= r_end) return;

    int px_i = tx * TILE_SIZE + threadIdx.x;
    int py_i = ty * TILE_SIZE + threadIdx.y;
    bool inside = (px_i < W && py_i < H);

    int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
    int lane = tid & 31;

    float px = (float)px_i + 0.5f;
    float py = (float)py_i + 0.5f;

    float T_curr = 1.0f;
    int last_pos = r_start - 1;
    float dL_r = 0.0f, dL_g = 0.0f, dL_b = 0.0f;
    float wb_r = 0.0f, wb_g = 0.0f, wb_b = 0.0f;

    if (inside) {
        size_t pix = ((size_t)v * H + py_i) * W + px_i;
        T_curr = out_T[pix];
        last_pos = out_last_idx[pix];
        dL_r = dL_dI[pix * 3 + 0];
        dL_g = dL_dI[pix * 3 + 1];
        dL_b = dL_dI[pix * 3 + 2];
        if (need_curv) {
            wb_r = w_bar[pix * 3 + 0];
            wb_g = w_bar[pix * 3 + 1];
            wb_b = w_bar[pix * 3 + 2];
        }
    }

    // Back-to-front color accumulator behind current splat (including background term!)
    float S_after_r = T_curr * bg_r;
    float S_after_g = T_curr * bg_g;
    float S_after_b = T_curr * bg_b;

    // Cache previous splat's 6-phi + 3-afp + 3-wrgb in registers to capture adjacent Morton partner cross-curvature!
    int prev_gid = -2;
    float prev_phi[6] = {0, 0, 0, 0, 0, 0};
    float prev_afp[3] = {0, 0, 0};
    float prev_wrgb[3] = {0, 0, 0};
    bool prev_hit = false;

    const int CHUNK = 128;
    __shared__ float sh_pd[CHUNK * 9];
    __shared__ int sh_sv[CHUNK];

    // Traverse chunks from back to front!
    for (int chunk_end = r_end; chunk_end > r_start; chunk_end -= CHUNK) {
        int chunk_start = max(r_start, chunk_end - CHUNK);
        int count = chunk_end - chunk_start;

        if (tid < count) {
            int sv = sorted_vals[chunk_start + tid];
            sh_sv[tid] = sv;
            #pragma unroll
            for (int m = 0; m < 9; ++m) sh_pd[tid * 9 + m] = proj_data[sv * 9 + m];
        }
        __syncthreads();

        for (int j = count - 1; j >= 0; --j) {
            int global_k = chunk_start + j;
            int sv = sh_sv[j];
            int gid = sv % N;
            const float* pd = sh_pd + j * 9;

            float g9_loc[9] = {0,0,0,0,0,0,0,0,0};
            float phi[6] = {0,0,0,0,0,0};
            float afp[3] = {0,0,0};
            float wrgb[3] = {0,0,0};
            bool hit = false;

            if (inside && global_k <= last_pos) {
                float dx = px - pd[0];
                float dy = py - pd[1];
                float con_x = pd[2], con_y = pd[3], con_z = pd[4];
                float pwr = 0.5f * (con_x * dx * dx + 2.0f * con_y * dx * dy + con_z * dy * dy);
                if (pwr <= 8.0f) {
                    float ai = pd[5];
                    float alpha = fminf(0.99f, ai * expf(-pwr));
                    if (alpha >= alpha_thresh) {
                        hit = true;
                        float inv_one_minus = 1.0f / (1.0f - alpha);
                        float T_i = T_curr * inv_one_minus;
                        float cr = pd[6], cg = pd[7], cb = pd[8];

                        // Exact d(I_hat)/d(alpha) * alpha
                        afp[0] = alpha * (T_i * cr - S_after_r * inv_one_minus);
                        afp[1] = alpha * (T_i * cg - S_after_g * inv_one_minus);
                        afp[2] = alpha * (T_i * cb - S_after_b * inv_one_minus);

                        // Exact d(I_hat)/d(logit_rgb)
                        float w_blend = T_i * alpha;
                        wrgb[0] = w_blend * cr * (1.0f - cr);
                        wrgb[1] = w_blend * cg * (1.0f - cg);
                        wrgb[2] = w_blend * cb * (1.0f - cb);

                        float qx = con_x * dx + con_y * dy;
                        float qy = con_y * dx + con_z * dy;
                        phi[0] = qx;
                        phi[1] = qy;
                        phi[2] = 0.5f * qx * qx;
                        phi[3] = qx * qy;
                        phi[4] = 0.5f * qy * qy;
                        phi[5] = 1.0f - ai;

                        float dL_dfp = dL_r * afp[0] + dL_g * afp[1] + dL_b * afp[2];
                        #pragma unroll
                        for (int m = 0; m < 6; ++m) g9_loc[m] = dL_dfp * phi[m];
                        g9_loc[6] = dL_r * wrgb[0];
                        g9_loc[7] = dL_g * wrgb[1];
                        g9_loc[8] = dL_b * wrgb[2];

                        S_after_r += w_blend * cr;
                        S_after_g += w_blend * cg;
                        S_after_b += w_blend * cb;
                        T_curr = T_i;
                    }
                }
            }

            // Warp-reduce gradient (32x fewer global atomics!)
            if (__any_sync(FULL_MASK, hit)) {
                float* g_dst = grad_img9 + (size_t)sv * 9;
                #pragma unroll
                for (int m = 0; m < 9; ++m) {
                    float s_val = warp_reduce_sum_f32(g9_loc[m]);
                    if (lane == 0 && s_val != 0.0f) atomicAdd(&g_dst[m], s_val);
                }

                if (need_curv) {
                    float W_fp = hit ? (wb_r * afp[0] * afp[0] + wb_g * afp[1] * afp[1] + wb_b * afp[2] * afp[2]) : 0.0f;
                    float W_cr0 = hit ? (wb_r * afp[0] * wrgb[0]) : 0.0f;
                    float W_cr1 = hit ? (wb_g * afp[1] * wrgb[1]) : 0.0f;
                    float W_cr2 = hit ? (wb_b * afp[2] * wrgb[2]) : 0.0f;

                    float* c_dst = curv_img42 + (size_t)sv * 42;
                    int idx42 = 0;
                    // 21 upper-triangular phi*phi^T entries
                    #pragma unroll
                    for (int a = 0; a < 6; ++a) {
                        #pragma unroll
                        for (int b = a; b < 6; ++b) {
                            float v42 = warp_reduce_sum_f32(W_fp * phi[a] * phi[b]);
                            if (lane == 0 && v42 != 0.0f) atomicAdd(&c_dst[idx42], v42);
                            ++idx42;
                        }
                    }
                    // 18 cross phi * rgb entries
                    float wcr[3] = {W_cr0, W_cr1, W_cr2};
                    #pragma unroll
                    for (int ch = 0; ch < 3; ++ch) {
                        #pragma unroll
                        for (int a = 0; a < 6; ++a) {
                            float v42 = warp_reduce_sum_f32(wcr[ch] * phi[a]);
                            if (lane == 0 && v42 != 0.0f) atomicAdd(&c_dst[idx42], v42);
                            ++idx42;
                        }
                    }
                    // 3 diagonal rgb^2 entries
                    float wr0 = warp_reduce_sum_f32(hit ? (wb_r * wrgb[0] * wrgb[0]) : 0.0f);
                    float wr1 = warp_reduce_sum_f32(hit ? (wb_g * wrgb[1] * wrgb[1]) : 0.0f);
                    float wr2 = warp_reduce_sum_f32(hit ? (wb_b * wrgb[2] * wrgb[2]) : 0.0f);
                    if (lane == 0) {
                        if (wr0 != 0.0f) atomicAdd(&c_dst[39], wr0);
                        if (wr1 != 0.0f) atomicAdd(&c_dst[40], wr1);
                        if (wr2 != 0.0f) atomicAdd(&c_dst[41], wr2);
                    }

                    // Adjacent Morton-pair cross-curvature B_ij (when prev_gid == partner_of[gid])!
                    int part = partner_of[gid];
                    if (part >= 0 && prev_gid == part && adj_cross_img81 != nullptr) {
                        int first_gid = min(gid, part);
                        int first_sv = v * N + first_gid;
                        float* x_dst = adj_cross_img81 + (size_t)first_sv * 81;
                        bool both = hit && prev_hit;
                        // Store row=first_gid, col=second_gid
                        const float* pA = (gid < part) ? phi : prev_phi;
                        const float* pB = (gid < part) ? prev_phi : phi;
                        const float* aA = (gid < part) ? afp : prev_afp;
                        const float* aB = (gid < part) ? prev_afp : afp;
                        const float* rA = (gid < part) ? wrgb : prev_wrgb;
                        const float* rB = (gid < part) ? prev_wrgb : wrgb;

                        float W_ab = both ? (wb_r * aA[0]*aB[0] + wb_g * aA[1]*aB[1] + wb_b * aA[2]*aB[2]) : 0.0f;
                        for (int r = 0; r < 6; ++r) {
                            for (int c = 0; c < 6; ++c) {
                                float val = warp_reduce_sum_f32(W_ab * pA[r] * pB[c]);
                                if (lane == 0 && val != 0.0f) atomicAdd(&x_dst[r * 9 + c], val);
                            }
                            float v_r0 = warp_reduce_sum_f32(both ? (wb_r * aA[0] * rB[0] * pA[r]) : 0.0f);
                            float v_r1 = warp_reduce_sum_f32(both ? (wb_g * aA[1] * rB[1] * pA[r]) : 0.0f);
                            float v_r2 = warp_reduce_sum_f32(both ? (wb_b * aA[2] * rB[2] * pA[r]) : 0.0f);
                            if (lane == 0) {
                                if (v_r0 != 0.0f) atomicAdd(&x_dst[r * 9 + 6], v_r0);
                                if (v_r1 != 0.0f) atomicAdd(&x_dst[r * 9 + 7], v_r1);
                                if (v_r2 != 0.0f) atomicAdd(&x_dst[r * 9 + 8], v_r2);
                            }
                        }
                        for (int c = 0; c < 6; ++c) {
                            float v_c0 = warp_reduce_sum_f32(both ? (wb_r * rA[0] * aB[0] * pB[c]) : 0.0f);
                            float v_c1 = warp_reduce_sum_f32(both ? (wb_g * rA[1] * aB[1] * pB[c]) : 0.0f);
                            float v_c2 = warp_reduce_sum_f32(both ? (wb_b * rA[2] * aB[2] * pB[c]) : 0.0f);
                            if (lane == 0) {
                                if (v_c0 != 0.0f) atomicAdd(&x_dst[6 * 9 + c], v_c0);
                                if (v_c1 != 0.0f) atomicAdd(&x_dst[7 * 9 + c], v_c1);
                                if (v_c2 != 0.0f) atomicAdd(&x_dst[8 * 9 + c], v_c2);
                            }
                        }
                        float d0 = warp_reduce_sum_f32(both ? (wb_r * rA[0] * rB[0]) : 0.0f);
                        float d1 = warp_reduce_sum_f32(both ? (wb_g * rA[1] * rB[1]) : 0.0f);
                        float d2 = warp_reduce_sum_f32(both ? (wb_b * rA[2] * rB[2]) : 0.0f);
                        if (lane == 0) {
                            if (d0 != 0.0f) atomicAdd(&x_dst[6 * 9 + 6], d0);
                            if (d1 != 0.0f) atomicAdd(&x_dst[7 * 9 + 7], d1);
                            if (d2 != 0.0f) atomicAdd(&x_dst[8 * 9 + 8], d2);
                        }
                    }
                }
            }

            prev_gid = gid;
            prev_hit = hit;
            if (hit && need_curv) {
                #pragma unroll
                for (int m = 0; m < 6; ++m) prev_phi[m] = phi[m];
                #pragma unroll
                for (int m = 0; m < 3; ++m) { prev_afp[m] = afp[m]; prev_wrgb[m] = wrgb[m]; }
            }
        }
        __syncthreads();
    }
}

// Assemble 13D gradients g_i and 13x13 / 26x26 raw Gram matrices B_e across B views
extern "C" __global__ void assemble_13d_grad_curv_kernel(
    const float* __restrict__ grad_img9,        // [B*N, 9]
    const float* __restrict__ curv_img42,       // [B*N, 42]
    const float* __restrict__ adj_cross_img81,  // [B*N, 81]
    const float* __restrict__ proj_jac,         // [B*N, 45] (5 x 9)
    const int* __restrict__ tiles_touched,      // [B*N]
    const int* __restrict__ pair_i,             // [M] first Gaussian in group e
    const int* __restrict__ pair_j,             // [M] second Gaussian in group e (-1 if singleton)
    int B, int N, int M,
    float rho_cross,
    int need_curv,
    float* __restrict__ out_grad13,             // [N, 13]
    float* __restrict__ out_B_pair              // [M, 26, 26]
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= M) return;

    int gi = pair_i[e];
    int gj = pair_j[e];
    int members[2] = {gi, gj};
    int nm = (gj >= 0) ? 2 : 1;

    float B_self[2][13][13];
    if (need_curv) {
        for (int m = 0; m < nm; ++m)
            for (int r = 0; r < 13; ++r)
                for (int c = 0; c < 13; ++c)
                    B_self[m][r][c] = 0.0f;
    }

    for (int m = 0; m < nm; ++m) {
        int g = members[m];
        float g13[13] = {0,0,0,0,0,0,0,0,0,0,0,0,0};

        for (int v = 0; v < B; ++v) {
            int sv = v * N + g;
            if (tiles_touched[sv] <= 0) continue;

            const float* gi9 = grad_img9 + (size_t)sv * 9;
            const float* J = proj_jac + (size_t)sv * 45; // 5x9

            // g13[0..8] += J^T * gi9[0..4]; g13[9..12] += gi9[5..8]
            for (int k = 0; k < 9; ++k) {
                g13[k] += J[0 * 9 + k] * gi9[0] + J[1 * 9 + k] * gi9[1]
                        + J[2 * 9 + k] * gi9[2] + J[3 * 9 + k] * gi9[3] + J[4 * 9 + k] * gi9[4];
            }
            g13[9]  += gi9[5];
            g13[10] += gi9[6];
            g13[11] += gi9[7];
            g13[12] += gi9[8];

            if (need_curv) {
                const float* c42 = curv_img42 + (size_t)sv * 42;
                float G9[9][9];
                for (int r = 0; r < 9; ++r)
                    for (int c = 0; c < 9; ++c) G9[r][c] = 0.0f;

                int idx42 = 0;
                for (int a = 0; a < 6; ++a) {
                    for (int b = a; b < 6; ++b) {
                        float val = c42[idx42++];
                        G9[a][b] = val;
                        G9[b][a] = val;
                    }
                }
                for (int ch = 0; ch < 3; ++ch) {
                    for (int a = 0; a < 6; ++a) {
                        float val = c42[idx42++];
                        G9[a][6 + ch] = val;
                        G9[6 + ch][a] = val;
                    }
                }
                G9[6][6] = c42[39];
                G9[7][7] = c42[40];
                G9[8][8] = c42[41];

                // Transform 9x9 -> 13x13 via P_vi: top-left 5->9 uses J (5x9), bottom-right 4->4 is Identity!
                // First T_mid (9_img x 13): cols 0..8 = G9[:, 0..4] * J, cols 9..12 = G9[:, 5..8]
                float T_mid[9][13];
                for (int r = 0; r < 9; ++r) {
                    for (int k = 0; k < 9; ++k) {
                        T_mid[r][k] = G9[r][0] * J[0 * 9 + k] + G9[r][1] * J[1 * 9 + k]
                                    + G9[r][2] * J[2 * 9 + k] + G9[r][3] * J[3 * 9 + k] + G9[r][4] * J[4 * 9 + k];
                    }
                    T_mid[r][9]  = G9[r][5];
                    T_mid[r][10] = G9[r][6];
                    T_mid[r][11] = G9[r][7];
                    T_mid[r][12] = G9[r][8];
                }
                // Now P_vi^T * T_mid -> 13x13
                for (int r = 0; r < 9; ++r) {
                    for (int c = 0; c < 13; ++c) {
                        B_self[m][r][c] += J[0 * 9 + r] * T_mid[0][c] + J[1 * 9 + r] * T_mid[1][c]
                                         + J[2 * 9 + r] * T_mid[2][c] + J[3 * 9 + r] * T_mid[3][c]
                                         + J[4 * 9 + r] * T_mid[4][c];
                    }
                }
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 13; ++c) {
                        B_self[m][9 + r][c] += T_mid[5 + r][c];
                    }
                }
            }
        }

        float* gd = out_grad13 + g * 13;
        #pragma unroll
        for (int k = 0; k < 13; ++k) gd[k] = g13[k];
    }

    if (need_curv) {
        float* Bp = out_B_pair + (size_t)e * 676;
        for (int idx2 = 0; idx2 < 676; ++idx2) Bp[idx2] = 0.0f;

        for (int r = 0; r < 13; ++r) {
            for (int c = 0; c < 13; ++c) {
                Bp[r * 26 + c] = 0.5f * (B_self[0][r][c] + B_self[0][c][r]);
                if (nm == 2) {
                    Bp[(13 + r) * 26 + (13 + c)] = 0.5f * (B_self[1][r][c] + B_self[1][c][r]);
                }
            }
        }

        if (nm == 2 && rho_cross > 0.0f && adj_cross_img81 != nullptr) {
            int first_g = min(gi, gj);
            float Bcross[13][13];
            for (int r = 0; r < 13; ++r)
                for (int c = 0; c < 13; ++c) Bcross[r][c] = 0.0f;

            for (int v = 0; v < B; ++v) {
                int sv_i = v * N + gi;
                int sv_j = v * N + gj;
                if (tiles_touched[sv_i] <= 0 || tiles_touched[sv_j] <= 0) continue;

                int first_sv = v * N + first_g;
                const float* X9 = adj_cross_img81 + (size_t)first_sv * 81;
                const float* JA = proj_jac + (size_t)(v * N + first_g) * 45;
                const float* JB = proj_jac + (size_t)(v * N + max(gi, gj)) * 45;

                // Check if non-zero
                float any_nz = 0.0f;
                for (int t = 0; t < 81; ++t) any_nz += fabsf(X9[t]);
                if (any_nz == 0.0f) continue;

                float T_mid[9][13];
                for (int r = 0; r < 9; ++r) {
                    for (int k = 0; k < 9; ++k) {
                        T_mid[r][k] = X9[r * 9 + 0] * JB[0 * 9 + k] + X9[r * 9 + 1] * JB[1 * 9 + k]
                                    + X9[r * 9 + 2] * JB[2 * 9 + k] + X9[r * 9 + 3] * JB[3 * 9 + k]
                                    + X9[r * 9 + 4] * JB[4 * 9 + k];
                    }
                    T_mid[r][9]  = X9[r * 9 + 5];
                    T_mid[r][10] = X9[r * 9 + 6];
                    T_mid[r][11] = X9[r * 9 + 7];
                    T_mid[r][12] = X9[r * 9 + 8];
                }
                for (int r = 0; r < 9; ++r) {
                    for (int c = 0; c < 13; ++c) {
                        Bcross[r][c] += JA[0 * 9 + r] * T_mid[0][c] + JA[1 * 9 + r] * T_mid[1][c]
                                      + JA[2 * 9 + r] * T_mid[2][c] + JA[3 * 9 + r] * T_mid[3][c]
                                      + JA[4 * 9 + r] * T_mid[4][c];
                    }
                }
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 13; ++c) Bcross[9 + r][c] += T_mid[5 + r][c];
                }
            }

            for (int r = 0; r < 13; ++r) {
                for (int c = 0; c < 13; ++c) {
                    float val = rho_cross * ((gi < gj) ? Bcross[r][c] : Bcross[c][r]);
                    Bp[r * 26 + (13 + c)] = val;
                    Bp[(13 + c) * 26 + r] = val;
                }
            }
        }
    }
}

// =========================================================================================
// PAPER SECTION 7.3: GPU-QP BACKEND (gpu_box.cuh)
// One CUDA Warp (32 threads) solves one 13- or 26-variable Box-QP in FP64 shared memory!
// =========================================================================================
#define WARPS_PER_BLOCK 4

struct SharedQP26 {
    double A[26][26];
    double L[26][26];
    double b[26];
    double y[26];
    double lo[26];
    double hi[26];
    double r[26];
    double dy[26];
    int state[26];
    int free_idx[26];
};

extern "C" __global__ void warp_box_qp_fp64_kernel(
    const float* __restrict__ B_pair,        // [M, 26, 26]
    const float* __restrict__ params,        // [N, 13]
    const float* __restrict__ grad13,        // [N, 13]
    const int* __restrict__ pair_i,          // [M]
    const int* __restrict__ pair_j,          // [M] (-1 if singleton)
    const float* __restrict__ trust_c13,     // [13] Eq. (9) coordinate step limits
    const float* __restrict__ bound_lo13,    // [13] Eq. (8) global lower bounds (-1e30 if none)
    const float* __restrict__ bound_hi13,    // [13] Eq. (8) global upper bounds (+1e30 if none)
    double damping_lambda,                   // lambda >= 1
    int M,
    float* __restrict__ out_step13,          // [N, 13] FP32 step d = C * y_star
    double* __restrict__ out_kkt_res,        // [M] normalized KKT residual (Eq. 13)
    double* __restrict__ out_gap_bound,      // [M] Proposition 3 QP gap bound s * Gamma(y)
    int* __restrict__ out_fail_flag          // [1] non-zero if Cholesky/KKT fails
) {
    __shared__ SharedQP26 sh_mem[WARPS_PER_BLOCK];

    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int e = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (e >= M) return;

    SharedQP26& S = sh_mem[warp_in_block];
    int gi = pair_i[e];
    int gj = pair_j[e];
    int p = (gj >= 0) ? 26 : 13;

    const float* Be = B_pair + (size_t)e * 676;

    // 1. Form Q_e = B_e + lambda * max(diag(B_e), 1e-8) and scale by C = diag(c) (Eq. 7 & 12)
    double c_vec[26];
    if (lane < p) {
        int m_idx = (lane < 13) ? gi : gj;
        int k = (lane < 13) ? lane : (lane - 13);
        double ck = (double)trust_c13[k];
        double xk = (double)params[m_idx * 13 + k];
        double gk = (double)grad13[m_idx * 13 + k];
        double lk = fmax(-ck, (double)bound_lo13[k] - xk);
        double uk = fmin(ck, (double)bound_hi13[k] - xk);
        if (lk > 0.0) lk = 0.0;
        if (uk < 0.0) uk = 0.0;

        S.lo[lane] = lk / ck;
        S.hi[lane] = uk / ck;
        S.b[lane] = ck * gk; // unscaled by s yet
        S.y[lane] = 0.0;
        for (int col = 0; col < p; ++col) {
            int k_col = (col < 13) ? col : (col - 13);
            double c_col = (double)trust_c13[k_col];
            double q_rc = (double)Be[lane * 26 + col];
            if (lane == col) {
                double d_floor = fmax(q_rc, 1e-8);
                q_rc += damping_lambda * d_floor;
            }
            S.A[lane][col] = ck * q_rc * c_col;
        }
    }
    __syncwarp();

    double diag_abs = (lane < p) ? fabs(S.A[lane][lane]) : 0.0;
    double s_norm = fmax(1e-30, warp_reduce_max_f64(diag_abs));
    double inv_s = 1.0 / s_norm;

    if (lane < p) {
        S.b[lane] *= inv_s;
        for (int col = 0; col < p; ++col) {
            S.A[lane][col] *= inv_s;
        }
    }
    __syncwarp();

    // 2. Six feasible coordinate sweeps to identify initial working face (Section 7.2 & 7.3)
    if (lane == 0) {
        for (int sw = 0; sw < 6; ++sw) {
            for (int a = 0; a < p; ++a) {
                if (S.hi[a] - S.lo[a] <= 1e-12) {
                    S.y[a] = 0.5 * (S.lo[a] + S.hi[a]);
                    continue;
                }
                double ra = S.b[a];
                for (int col = 0; col < p; ++col) ra += S.A[a][col] * S.y[col];
                double yn = S.y[a] - ra / fmax(S.A[a][a], 1e-15);
                S.y[a] = fmin(fmax(yn, S.lo[a]), S.hi[a]);
            }
        }
        for (int a = 0; a < p; ++a) {
            if (S.hi[a] - S.lo[a] <= 1e-12) {
                S.state[a] = 2;
                S.y[a] = 0.5 * (S.lo[a] + S.hi[a]);
            } else if (S.y[a] <= S.lo[a] + 1e-12) {
                S.state[a] = -1;
                S.y[a] = S.lo[a];
            } else if (S.y[a] >= S.hi[a] - 1e-12) {
                S.state[a] = 1;
                S.y[a] = S.hi[a];
            } else {
                S.state[a] = 0;
            }
        }
    }
    __syncwarp();

    double b_inf = warp_reduce_max_f64((lane < p) ? fabs(S.b[lane]) : 0.0);
    double kkt_denom = 1.0 + b_inf;
    double final_kkt = 1.0;
    bool converged = false;

    // 3. Active-set reduced FP64 Cholesky loop (256-iteration cap, Section 7.3)
    for (int iter = 0; iter < 256; ++iter) {
        // Compute residual r = A * y + b in parallel across lanes!
        double ra = 0.0;
        double nu_a = 0.0;
        double free_ra = 0.0;
        if (lane < p) {
            ra = S.b[lane];
            for (int col = 0; col < p; ++col) ra += S.A[lane][col] * S.y[col];
            S.r[lane] = ra;
            int st = S.state[lane];
            if (st == 0) { nu_a = ra; free_ra = fabs(ra); }
            else if (st == -1) { nu_a = fmin(0.0, ra); }
            else if (st == 1)  { nu_a = fmax(0.0, ra); }
            else { nu_a = 0.0; }
        }
        __syncwarp();

        double nu_inf = warp_reduce_max_f64((lane < p) ? fabs(nu_a) : 0.0);
        final_kkt = nu_inf / kkt_denom;
        if (final_kkt <= 1e-8) {
            converged = true;
            break;
        }

        double max_free_r = warp_reduce_max_f64(free_ra);

        // If free face is NOT yet stationary, solve reduced Cholesky on free variables!
        if (max_free_r > 1e-10 * kkt_denom) {
            __shared__ int sh_nf[WARPS_PER_BLOCK];
            __shared__ int sh_chol_ok[WARPS_PER_BLOCK];
            if (lane == 0) {
                int nf = 0;
                for (int a = 0; a < p; ++a) {
                    if (S.state[a] == 0) S.free_idx[nf++] = a;
                    S.dy[a] = 0.0;
                }
                sh_nf[warp_in_block] = nf;
                sh_chol_ok[warp_in_block] = 1;

                // Reduced FP64 Cholesky L * L^T = A_FF
                for (int r = 0; r < nf; ++r) {
                    int ar = S.free_idx[r];
                    for (int c = 0; c <= r; ++c) {
                        int ac = S.free_idx[c];
                        double sum = S.A[ar][ac];
                        for (int k = 0; k < c; ++k) sum -= S.L[r][k] * S.L[c][k];
                        if (r == c) {
                            if (sum <= 1e-18) {
                                sh_chol_ok[warp_in_block] = 0;
                                break;
                            }
                            S.L[r][r] = sqrt(sum);
                        } else {
                            S.L[r][c] = sum / S.L[c][c];
                        }
                    }
                    if (!sh_chol_ok[warp_in_block]) break;
                }

                if (sh_chol_ok[warp_in_block] && nf > 0) {
                    double tmp[26];
                    for (int r = 0; r < nf; ++r) {
                        double sum = -S.r[S.free_idx[r]];
                        for (int k = 0; k < r; ++k) sum -= S.L[r][k] * tmp[k];
                        tmp[r] = sum / S.L[r][r];
                    }
                    for (int r = nf - 1; r >= 0; --r) {
                        double sum = tmp[r];
                        for (int k = r + 1; k < nf; ++k) sum -= S.L[k][r] * tmp[k];
                        tmp[r] = sum / S.L[r][r];
                        S.dy[S.free_idx[r]] = tmp[r];
                    }

                    // Line search to nearest blocking bound
                    double alpha = 1.0;
                    int block_a = -1;
                    int block_st = 0;
                    for (int r = 0; r < nf; ++r) {
                        int a = S.free_idx[r];
                        double d = S.dy[a];
                        if (d < -1e-15) {
                            double a_lo = (S.lo[a] - S.y[a]) / d;
                            if (a_lo < alpha) { alpha = a_lo; block_a = a; block_st = -1; }
                        } else if (d > 1e-15) {
                            double a_hi = (S.hi[a] - S.y[a]) / d;
                            if (a_hi < alpha) { alpha = a_hi; block_a = a; block_st = 1; }
                        }
                    }
                    alpha = fmax(0.0, fmin(1.0, alpha));
                    for (int r = 0; r < nf; ++r) {
                        int a = S.free_idx[r];
                        S.y[a] = fmin(fmax(S.y[a] + alpha * S.dy[a], S.lo[a]), S.hi[a]);
                    }
                    if (block_a >= 0) {
                        S.state[block_a] = block_st;
                        S.y[block_a] = (block_st == -1) ? S.lo[block_a] : S.hi[block_a];
                    }
                }
            }
            __syncwarp();
            if (!sh_chol_ok[warp_in_block]) break;
        } else {
            // Free face is stationary -> release active bound with largest wrong-sign multiplier!
            if (lane == 0) {
                int best_a = -1;
                double best_viol = 0.0;
                for (int a = 0; a < p; ++a) {
                    int st = S.state[a];
                    double viol = 0.0;
                    if (st == -1 && S.r[a] < 0.0) viol = -S.r[a];
                    else if (st == 1 && S.r[a] > 0.0) viol = S.r[a];
                    if (viol > best_viol) {
                        best_viol = viol;
                        best_a = a;
                    }
                }
                if (best_a >= 0) {
                    S.state[best_a] = 0;
                }
            }
            __syncwarp();
        }
    }

    // Final KKT & Proposition 3 duality gap Gamma(y) check!
    if (lane < p) {
        double ra = S.b[lane];
        for (int col = 0; col < p; ++col) ra += S.A[lane][col] * S.y[col];
        S.r[lane] = ra;
    }
    __syncwarp();

    if (lane == 0) {
        double max_nu = 0.0;
        double max_feas = 0.0;
        double gamma = 0.0;
        for (int a = 0; a < p; ++a) {
            double ra = S.r[a];
            double ya = S.y[a];
            int st = S.state[a];
            double nu = 0.0;
            if (st == 0) nu = ra;
            else if (st == -1) nu = fmin(0.0, ra);
            else if (st == 1)  nu = fmax(0.0, ra);
            if (fabs(nu) > max_nu) max_nu = fabs(nu);

            double f1 = S.lo[a] - ya;
            double f2 = ya - S.hi[a];
            if (f1 > max_feas) max_feas = f1;
            if (f2 > max_feas) max_feas = f2;

            if (ra >= 0.0) gamma += ra * (ya - S.lo[a]);
            else           gamma += ra * (ya - S.hi[a]);
        }
        final_kkt = max_nu / kkt_denom;
        out_kkt_res[e] = final_kkt;
        out_gap_bound[e] = s_norm * fmax(0.0, gamma);
        if (final_kkt > 1e-8 || max_feas > 1e-10 || !converged) {
            atomicExch(out_fail_flag, 1);
        }
    }
    __syncwarp();

    // Write raw step d = C * y_star (FP32) to disjoint Gaussian entries
    if (lane < p) {
        int m_idx = (lane < 13) ? gi : gj;
        int k = (lane < 13) ? lane : (lane - 13);
        float ck = trust_c13[k];
        out_step13[m_idx * 13 + k] = (float)((double)ck * S.y[lane]);
    }
}
