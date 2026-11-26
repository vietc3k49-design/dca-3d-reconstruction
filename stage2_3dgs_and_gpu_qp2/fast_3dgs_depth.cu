// Fast NVRTC CUDA Kernels (compute_86) for 3DGS Tile Binning & 16x16 Median Transmittance Depth + RGB Rasterization.
// Zero nvcc / cl.exe required; compiled in-memory via nvrtc64_*.dll and launched via nvcuda.dll on PyTorch CUDA stream.

#define TILE_SIZE 16
#define TILE_PIXELS 256

extern "C" __global__ void emit_tile_keys_kernel(
    const int* __restrict__ tile_bbox,      // [M, 4] (tx0, ty0, tx1, ty1)
    const int* __restrict__ tiles_touched,  // [M]
    const long long* __restrict__ offsets,  // [M]
    int M,
    int tiles_x,
    long long* __restrict__ out_keys        // [total_intersections]
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
    int* __restrict__ tile_ranges // [tiles_x * tiles_y, 2], zero-initialized
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
    const int* __restrict__ tile_ranges,       // [tiles_x * tiles_y, 2]
    const long long* __restrict__ sorted_keys, // [num_intersections] (low 32 bits = front-to-back splat index)
    const float* __restrict__ splat_data,      // [M, 10]: u, v, ca, cb, cc, opac, z_cam, r, g, b
    int W,
    int H,
    int tiles_x,
    float* __restrict__ out_depth,             // [H, W]
    float* __restrict__ out_alpha,             // [H, W]
    float* __restrict__ out_rgb                // [H, W, 3]
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
