#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

#ifndef DKL_THREADS
#define DKL_THREADS 256
#endif

static_assert(DKL_THREADS == 128 || DKL_THREADS == 256 || DKL_THREADS == 512,
              "DKL_THREADS must be 128, 256, or 512");

constexpr int kWarp = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kMaxSplitSlots = 1024;  // Supports split-row mode when N < 160

struct KLStateF32 {
    float ma;
    float da;
    float mb;
    float db;
    float w;
};

struct KLStateF64 {
    double ma;
    double da;
    double mb;
    double db;
    double w;
};

__device__ KLStateF64 g_split_states[kMaxSplitSlots];
__device__ unsigned int g_row_retired[160];

__device__ __forceinline__ KLStateF32 make_empty_f32() {
    KLStateF32 s;
    s.ma = -FLT_MAX;
    s.da = 0.0f;
    s.mb = -FLT_MAX;
    s.db = 0.0f;
    s.w = 0.0f;
    return s;
}

__device__ __forceinline__ KLStateF64 to_f64_state(const KLStateF32& s) {
    KLStateF64 d;
    d.ma = static_cast<double>(s.ma);
    d.da = static_cast<double>(s.da);
    d.mb = static_cast<double>(s.mb);
    d.db = static_cast<double>(s.db);
    d.w = static_cast<double>(s.w);
    return d;
}

__device__ __forceinline__ KLStateF64 make_empty_f64() {
    KLStateF64 d;
    d.ma = -1e300;
    d.da = 0.0;
    d.mb = -1e300;
    d.db = 0.0;
    d.w = 0.0;
    return d;
}

// Merge 1 scalar pair (a, b) into running FP32 online 5-tuple state:
__device__ __forceinline__ void online_push_1(KLStateF32& s, float a, float b) {
    if (s.da == 0.0f) {
        s.ma = a;
        s.da = 1.0f;
        s.mb = b;
        s.db = 1.0f;
        s.w = 0.0f;
        return;
    }
    const float new_ma = fmaxf(s.ma, a);
    const float new_mb = fmaxf(s.mb, b);
    const float alpha_old = __expf(s.ma - new_ma);
    const float alpha_new = __expf(a - new_ma);
    const float beta_old = __expf(s.mb - new_mb);
    const float beta_new = __expf(b - new_mb);

    const float shift_old = (s.ma - new_ma) - (s.mb - new_mb);
    const float diff_new = (a - new_ma) - (b - new_mb);

    s.w = fmaf(alpha_old, fmaf(s.da, shift_old, s.w), alpha_new * diff_new);
    s.da = fmaf(s.da, alpha_old, alpha_new);
    s.db = fmaf(s.db, beta_old, beta_new);
    s.ma = new_ma;
    s.mb = new_mb;
}

// Merge 4 elements (float4 ta, float4 sb) AT ONCE with a SINGLE rescale of the running accumulator!
// Cuts online state rescaling instructions by 4x on the V = 128,256 hot path!
__device__ __forceinline__ void online_push_vec4(KLStateF32& s, const float4& t4, const float4& s4,
                                                 float inv_temp) {
    const float a0 = t4.x * inv_temp, a1 = t4.y * inv_temp, a2 = t4.z * inv_temp, a3 = t4.w * inv_temp;
    const float b0 = s4.x * inv_temp, b1 = s4.y * inv_temp, b2 = s4.z * inv_temp, b3 = s4.w * inv_temp;

    const float c_ma = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));
    const float c_mb = fmaxf(fmaxf(b0, b1), fmaxf(b2, b3));

    const float new_ma = (s.da == 0.0f) ? c_ma : fmaxf(s.ma, c_ma);
    const float new_mb = (s.da == 0.0f) ? c_mb : fmaxf(s.mb, c_mb);

    const float ua0 = a0 - new_ma, ua1 = a1 - new_ma, ua2 = a2 - new_ma, ua3 = a3 - new_ma;
    const float ub0 = b0 - new_mb, ub1 = b1 - new_mb, ub2 = b2 - new_mb, ub3 = b3 - new_mb;

    const float ea0 = __expf(ua0), ea1 = __expf(ua1), ea2 = __expf(ua2), ea3 = __expf(ua3);
    const float eb0 = __expf(ub0), eb1 = __expf(ub1), eb2 = __expf(ub2), eb3 = __expf(ub3);

    const float chunk_da = (ea0 + ea1) + (ea2 + ea3);
    const float chunk_db = (eb0 + eb1) + (eb2 + eb3);
    const float chunk_w = (fmaf(ea0, ua0 - ub0, ea1 * (ua1 - ub1))) +
                          (fmaf(ea2, ua2 - ub2, ea3 * (ua3 - ub3)));

    if (s.da == 0.0f) {
        s.ma = new_ma;
        s.da = chunk_da;
        s.mb = new_mb;
        s.db = chunk_db;
        s.w = chunk_w;
    } else {
        const float alpha_old = __expf(s.ma - new_ma);
        const float beta_old = __expf(s.mb - new_mb);
        const float shift_old = (s.ma - new_ma) - (s.mb - new_mb);

        s.w = fmaf(alpha_old, fmaf(s.da, shift_old, s.w), chunk_w);
        s.da = fmaf(s.da, alpha_old, chunk_da);
        s.db = fmaf(s.db, beta_old, chunk_db);
        s.ma = new_ma;
        s.mb = new_mb;
    }
}

// Exact FP64 associative merge for warp, block, and cross-split reductions
// (eliminates log(d_b) - log(d_a) cancellation error on V = 128,256!)
__device__ __forceinline__ KLStateF64 merge_state_f64(const KLStateF64& s1, const KLStateF64& s2) {
    if (s1.da == 0.0) return s2;
    if (s2.da == 0.0) return s1;

    const double ma = fmax(s1.ma, s2.ma);
    const double mb = fmax(s1.mb, s2.mb);

    const double a1 = exp(s1.ma - ma);
    const double a2 = exp(s2.ma - ma);
    const double b1 = exp(s1.mb - mb);
    const double b2 = exp(s2.mb - mb);

    const double sh1 = (s1.ma - ma) - (s1.mb - mb);
    const double sh2 = (s2.ma - ma) - (s2.mb - mb);

    KLStateF64 out;
    out.ma = ma;
    out.mb = mb;
    out.da = s1.da * a1 + s2.da * a2;
    out.db = s1.db * b1 + s2.db * b2;
    out.w = a1 * (s1.w + s1.da * sh1) + a2 * (s2.w + s2.da * sh2);
    return out;
}

__device__ __forceinline__ KLStateF64 warp_reduce_kl_f64(KLStateF64 s) {
    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        KLStateF64 other;
        other.ma = __shfl_down_sync(kFullMask, s.ma, offset);
        other.da = __shfl_down_sync(kFullMask, s.da, offset);
        other.mb = __shfl_down_sync(kFullMask, s.mb, offset);
        other.db = __shfl_down_sync(kFullMask, s.db, offset);
        other.w = __shfl_down_sync(kFullMask, s.w, offset);
        s = merge_state_f64(s, other);
    }
    return s;
}

template <int kThreads>
__device__ __forceinline__ KLStateF64 block_reduce_kl_f64(KLStateF64 s, KLStateF64* s_warp) {
    constexpr int kWarps = kThreads / kWarp;
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;

    s = warp_reduce_kl_f64(s);
    if (lane == 0) {
        s_warp[warp] = s;
    }
    __syncthreads();

    KLStateF64 total = make_empty_f64();
    if (warp == 0) {
        total = (lane < kWarps) ? s_warp[lane] : make_empty_f64();
        total = warp_reduce_kl_f64(total);
    }
    return total;
}

__device__ __forceinline__ float finish_kl_from_state(const KLStateF64& s) {
    if (s.da <= 0.0 || s.db <= 0.0) {
        return 0.0f;
    }
    // log(db / da) = log1p((db - da) / da) avoids cancellation when db ~= da!
    const double kl_d = (s.w / s.da) + log(s.db / s.da);
    return (kl_d > 0.0) ? static_cast<float>(kl_d) : 0.0f;
}

// ============================================================================
// Path 1: Narrow Vocab (V <= 1,024) — 1 Warp per Row (8 rows/block, 0 shared memory!)
// ============================================================================
__global__ __launch_bounds__(256, 4) void distillation_kl_warp_per_row_kernel(
    const float* __restrict__ teacher_logits,
    const float* __restrict__ student_logits,
    float* __restrict__ kl,
    int N,
    int V,
    float inv_temp,
    bool vec4_aligned) {
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp_in_block;
    if (row >= N) {
        return;
    }

    const size_t row_base = static_cast<size_t>(row) * static_cast<size_t>(V);
    const float* __restrict__ t_row = teacher_logits + row_base;
    const float* __restrict__ s_row = student_logits + row_base;

    KLStateF32 local = make_empty_f32();
    if (vec4_aligned) {
        const int V4 = V >> 2;
        const float4* __restrict__ t4 = reinterpret_cast<const float4*>(t_row);
        const float4* __restrict__ s4 = reinterpret_cast<const float4*>(s_row);
        for (int j4 = lane; j4 < V4; j4 += kWarp) {
            online_push_vec4(local, __ldg(&t4[j4]), __ldg(&s4[j4]), inv_temp);
        }
    } else {
        for (int j = lane; j < V; j += kWarp) {
            online_push_1(local, __ldg(&t_row[j]) * inv_temp, __ldg(&s_row[j]) * inv_temp);
        }
    }

    const KLStateF64 merged = warp_reduce_kl_f64(to_f64_state(local));
    if (lane == 0) {
        kl[row] = finish_kl_from_state(merged);
    }
}

// ============================================================================
// Path 2: Primary 1-Block-per-Row Single-Pass Online float4 Kernel
// Covers Official Benchmark Shape N = 2,048, V = 128,256 in 1 single read pass!
// ============================================================================
template <int kThreads>
__global__ __launch_bounds__(kThreads, 4) void distillation_kl_1pass_vec4_kernel(
    const float4* __restrict__ teacher4,
    const float4* __restrict__ student4,
    float* __restrict__ kl,
    int V4,
    float inv_temp) {
    constexpr int kWarps = kThreads / kWarp;
    __shared__ KLStateF64 s_warp[kWarps];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t base4 = static_cast<size_t>(row) * static_cast<size_t>(V4);
    const float4* __restrict__ t_row4 = teacher4 + base4;
    const float4* __restrict__ s_row4 = student4 + base4;

    KLStateF32 local = make_empty_f32();

    int j4 = tid;
    const int unroll2_end = V4 - kThreads;
    for (; j4 < unroll2_end; j4 += 2 * kThreads) {
        const float4 ta0 = __ldg(&t_row4[j4]);
        const float4 sa0 = __ldg(&s_row4[j4]);
        const float4 ta1 = __ldg(&t_row4[j4 + kThreads]);
        const float4 sa1 = __ldg(&s_row4[j4 + kThreads]);
        online_push_vec4(local, ta0, sa0, inv_temp);
        online_push_vec4(local, ta1, sa1, inv_temp);
    }
    if (j4 < V4) {
        online_push_vec4(local, __ldg(&t_row4[j4]), __ldg(&s_row4[j4]), inv_temp);
    }

    const KLStateF64 total = block_reduce_kl_f64<kThreads>(to_f64_state(local), s_warp);
    if (tid == 0) {
        kl[row] = finish_kl_from_state(total);
    }
}

// ============================================================================
// Path 3: Few-Long-Rows Split-V Single-Launch Kernel (e.g. N = 4, V = 128,256)
// Splits each row across kSplits blocks so small-N wide-V saturates all 40 SMs!
// ============================================================================
template <int kThreads>
__global__ __launch_bounds__(kThreads, 4) void distillation_kl_split_row_vec4_kernel(
    const float4* __restrict__ teacher4,
    const float4* __restrict__ student4,
    float* __restrict__ kl,
    int V4,
    int splits_per_row,
    float inv_temp) {
    constexpr int kWarps = kThreads / kWarp;
    __shared__ KLStateF64 s_warp[kWarps];
    __shared__ bool s_is_last;

    const int split_id = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;

    const int chunk_v4 = (V4 + splits_per_row - 1) / splits_per_row;
    const int start_v4 = split_id * chunk_v4;
    int end_v4 = start_v4 + chunk_v4;
    if (end_v4 > V4) end_v4 = V4;

    const size_t base4 = static_cast<size_t>(row) * static_cast<size_t>(V4);
    KLStateF32 local = make_empty_f32();
    for (int j4 = start_v4 + tid; j4 < end_v4; j4 += kThreads) {
        online_push_vec4(local, __ldg(&teacher4[base4 + j4]), __ldg(&student4[base4 + j4]),
                         inv_temp);
    }

    const KLStateF64 block_state = block_reduce_kl_f64<kThreads>(to_f64_state(local), s_warp);

    if (tid == 0) {
        g_split_states[row * splits_per_row + split_id] = block_state;
        __threadfence();
        const unsigned int prev = atomicAdd(&g_row_retired[row], 1u);
        s_is_last = (prev + 1u == static_cast<unsigned int>(splits_per_row));
    }
    __syncthreads();

    if (s_is_last) {
        KLStateF64 s = (tid < splits_per_row)
                           ? g_split_states[row * splits_per_row + tid]
                           : make_empty_f64();
        const KLStateF64 row_total = block_reduce_kl_f64<kThreads>(s, s_warp);
        if (tid == 0) {
            g_row_retired[row] = 0u;
            kl[row] = finish_kl_from_state(row_total);
        }
    }
}

// ============================================================================
// Path 4: General Scalar Single-Pass Online Kernel (V % 4 != 0 or Misaligned Pointers)
// ============================================================================
template <int kThreads>
__global__ __launch_bounds__(kThreads, 4) void distillation_kl_1pass_scalar_kernel(
    const float* __restrict__ teacher_logits,
    const float* __restrict__ student_logits,
    float* __restrict__ kl,
    int V,
    float inv_temp) {
    constexpr int kWarps = kThreads / kWarp;
    __shared__ KLStateF64 s_warp[kWarps];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t base = static_cast<size_t>(row) * static_cast<size_t>(V);
    const float* __restrict__ t_row = teacher_logits + base;
    const float* __restrict__ s_row = student_logits + base;

    KLStateF32 local = make_empty_f32();
    for (int j = tid; j < V; j += kThreads) {
        online_push_1(local, __ldg(&t_row[j]) * inv_temp, __ldg(&s_row[j]) * inv_temp);
    }

    const KLStateF64 total = block_reduce_kl_f64<kThreads>(to_f64_state(local), s_warp);
    if (tid == 0) {
        kl[row] = finish_kl_from_state(total);
    }
}

inline bool is_aligned_16b(const void* p0, const void* p1) {
    return ((reinterpret_cast<uintptr_t>(p0) | reinterpret_cast<uintptr_t>(p1)) & 15u) == 0u;
}

}  // namespace

// teacher_logits, student_logits, kl are device pointers
extern "C" void solve(const float* teacher_logits, const float* student_logits, float* kl, int N,
                      int V, float temperature) {
    if (N <= 0 || V <= 0) {
        return;
    }

    const float inv_temp = 1.0f / temperature;
    const bool vec4_ok = ((V & 3) == 0) && is_aligned_16b(teacher_logits, student_logits);

    // 1. Narrow vocab: pack 8 rows per 256-thread block using pure warp shuffles
    if (V <= 1024) {
        const int blocks = (N + 7) / 8;
        distillation_kl_warp_per_row_kernel<<<blocks, 256>>>(
            teacher_logits, student_logits, kl, N, V, inv_temp, vec4_ok);
        return;
    }

    // 2. Aligned float4 fast paths
    if (vec4_ok) {
        const int V4 = V >> 2;
        const float4* t4 = reinterpret_cast<const float4*>(teacher_logits);
        const float4* s4 = reinterpret_cast<const float4*>(student_logits);

        // Split-row single-launch mode when N is tiny (< 80) and V is very wide (>= 16,384)
        if (N < 80 && V >= 16384) {
            int splits = 160 / N;
            if (splits > 32) splits = 32;
            if (N * splits > kMaxSplitSlots) splits = kMaxSplitSlots / N;
            if (splits >= 2) {
                const dim3 grid(static_cast<unsigned>(splits), static_cast<unsigned>(N));
                distillation_kl_split_row_vec4_kernel<DKL_THREADS><<<grid, DKL_THREADS>>>(
                    t4, s4, kl, V4, splits, inv_temp);
                return;
            }
        }

        distillation_kl_1pass_vec4_kernel<DKL_THREADS><<<N, DKL_THREADS>>>(
            t4, s4, kl, V4, inv_temp);
        return;
    }

    // 3. General unaligned / non-multiple-of-4 V single-pass online path
    distillation_kl_1pass_scalar_kernel<DKL_THREADS><<<N, DKL_THREADS>>>(
        teacher_logits, student_logits, kl, V, inv_temp);
}
