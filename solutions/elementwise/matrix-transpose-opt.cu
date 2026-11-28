#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

__device__ __forceinline__ void store_cs_v4(float* ptr, float4 val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w)
                 : "memory");
#else
    *reinterpret_cast<float4*>(ptr) = val;
#endif
}

__device__ __forceinline__ void store_cs_f32(float* ptr, float val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    asm volatile("st.global.cs.f32 [%0], %1;" : : "l"(ptr), "f"(val) : "memory");
#else
    *ptr = val;
#endif
}

// 2D CTA tile swizzling (`kGroupY = 8`) to bound active TLB footprint and avoid
// GDDR6 partition camping on large power-of-two or rectangular row strides.
__device__ __forceinline__ void swizzle_cta_2d(int grid_x, int grid_y, int& bx, int& by) {
    constexpr int kGroupY = 8;
    const int pid = static_cast<int>(blockIdx.x) + static_cast<int>(blockIdx.y) * grid_x;
    const int group_span = kGroupY * grid_x;
    const int group_id = pid / group_span;
    const int first_by = group_id * kGroupY;
    const int group_size_y = ((grid_y - first_by) < kGroupY) ? (grid_y - first_by) : kGroupY;
    const int in_group = pid - group_id * group_span;
    bx = in_group / group_size_y;
    by = first_by + (in_group % group_size_y);
}

// ============================================================================
// Primary Fast Path (`rows % 4 == 0 && cols % 4 == 0` and 16B-aligned pointers —
// covers official `7,000 x 6,000` benchmark AND `8,192 x 8,192`!):
//   - `256` threads (`16 x 16` logical grid) process a `64 x 64` float tile (`16 KiB`
//     shared memory -> 100% occupancy `4` blocks = `1,024` threads/SM on T4).
//   - Each thread loads a `4 x 4` sub-tile via 4 coalesced `LDG.E.128` (`float4`)
//     instructions (`16` threads x `16`B = `256` contiguous bytes per half-warp),
//     transposes the `4 x 4` block in registers, and writes `4` `float4` vectors
//     to `__shared__ float4 s_v4[64][16]` at XOR-swizzled column `phys_col = ty ^ tx`,
//     eliminating 100% of shared-memory bank conflicts on both `STS.128` and `LDS.128`.
//   - Because `rows` and `cols` are multiples of 4, every `4 x 4` sub-tile on partial
//     boundary tiles (`7000 % 64 = 24`, `6000 % 64 = 48`) is either 100% in-bounds or
//     100% out-of-bounds, keeping pure `st.global.cs.v4.f32` stores everywhere!
// ============================================================================
__global__ __launch_bounds__(256) void transpose_vec4_64x64_swizzle_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols,
    int grid_x,
    int grid_y) {
    __shared__ float4 s_v4[64][16];

    int bx = 0;
    int by = 0;
    swizzle_cta_2d(grid_x, grid_y, bx, by);

    const int tile_c = bx * 64;
    const int tile_r = by * 64;

    const int tid = static_cast<int>(threadIdx.x);
    const int tx = tid & 15;   // 0..15 (`float4` column index within `64`-float tile)
    const int ty = tid >> 4;   // 0..15 (`4`-row group index within `64`-row tile)
    const int phys_col = tx ^ ty;

    const int in_r = tile_r + ty * 4;
    const int in_c = tile_c + tx * 4;
    if (in_r < rows && in_c < cols) {
        const size_t stride_in = static_cast<size_t>(cols);
        const float* in_ptr = input + static_cast<size_t>(in_r) * stride_in + in_c;

        const float4 v0 = __ldg(reinterpret_cast<const float4*>(in_ptr));
        const float4 v1 = __ldg(reinterpret_cast<const float4*>(in_ptr + stride_in));
        const float4 v2 = __ldg(reinterpret_cast<const float4*>(in_ptr + 2 * stride_in));
        const float4 v3 = __ldg(reinterpret_cast<const float4*>(in_ptr + 3 * stride_in));

        const int s_row = tx * 4;
        s_v4[s_row + 0][phys_col] = make_float4(v0.x, v1.x, v2.x, v3.x);
        s_v4[s_row + 1][phys_col] = make_float4(v0.y, v1.y, v2.y, v3.y);
        s_v4[s_row + 2][phys_col] = make_float4(v0.z, v1.z, v2.z, v3.z);
        s_v4[s_row + 3][phys_col] = make_float4(v0.w, v1.w, v2.w, v3.w);
    }
    __syncthreads();

    const int out_r = tile_c + ty * 4;
    const int out_c = tile_r + tx * 4;
    if (out_r < cols && out_c < rows) {
        const float4 o0 = s_v4[ty * 4 + 0][phys_col];
        const float4 o1 = s_v4[ty * 4 + 1][phys_col];
        const float4 o2 = s_v4[ty * 4 + 2][phys_col];
        const float4 o3 = s_v4[ty * 4 + 3][phys_col];

        const size_t stride_out = static_cast<size_t>(rows);
        float* out_ptr = output + static_cast<size_t>(out_r) * stride_out + out_c;

        store_cs_v4(out_ptr, o0);
        store_cs_v4(out_ptr + stride_out, o1);
        store_cs_v4(out_ptr + 2 * stride_out, o2);
        store_cs_v4(out_ptr + 3 * stride_out, o3);
    }
}

// ============================================================================
// General Fallback (`rows % 4 != 0 || cols % 4 != 0` or unaligned pointers):
//   Padded `32 x 33` conflict-free shared-memory tile (`32 x 8 = 256` threads)
//   with `kGroupY = 8` CTA swizzling and coalesced cache-streaming writes.
// ============================================================================
__global__ __launch_bounds__(256) void transpose_scalar_tile32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols,
    int grid_x,
    int grid_y) {
    __shared__ float s_tile[32][33];

    int bx = 0;
    int by = 0;
    swizzle_cta_2d(grid_x, grid_y, bx, by);

    const int tile_c = bx * 32;
    const int tile_r = by * 32;

    const int tx = static_cast<int>(threadIdx.x);  // 0..31
    const int ty = static_cast<int>(threadIdx.y);  // 0..7

    const int in_c = tile_c + tx;
    if (in_c < cols) {
        #pragma unroll
        for (int j = 0; j < 32; j += 8) {
            const int in_r = tile_r + ty + j;
            if (in_r < rows) {
                s_tile[ty + j][tx] =
                    __ldg(&input[static_cast<size_t>(in_r) * static_cast<size_t>(cols) + in_c]);
            }
        }
    }
    __syncthreads();

    const int out_c = tile_r + tx;
    if (out_c < rows) {
        #pragma unroll
        for (int j = 0; j < 32; j += 8) {
            const int out_r = tile_c + ty + j;
            if (out_r < cols) {
                store_cs_f32(&output[static_cast<size_t>(out_r) * static_cast<size_t>(rows) + out_c],
                             s_tile[tx][ty + j]);
            }
        }
    }
}

}  // namespace

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return;
    }

    const bool aligned_16b =
        (((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) & 15u) == 0u);

    if (aligned_16b && ((rows & 3) == 0) && ((cols & 3) == 0)) {
        const int grid_x = (cols + 63) / 64;
        const int grid_y = (rows + 63) / 64;
        const dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y));
        transpose_vec4_64x64_swizzle_kernel<<<grid, 256>>>(
            input, output, rows, cols, grid_x, grid_y);
    } else {
        const int grid_x = (cols + 31) / 32;
        const int grid_y = (rows + 31) / 32;
        const dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y));
        const dim3 block(32, 8);
        transpose_scalar_tile32_kernel<<<grid, block>>>(
            input, output, rows, cols, grid_x, grid_y);
    }
}
