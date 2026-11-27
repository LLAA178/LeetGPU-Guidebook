#include <cuda_runtime.h>
#include <cmath>

namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

__global__ void max_pool2d_baseline_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int H,
                                           int W,
                                           int H_out,
                                           int W_out,
                                           int kernel_size,
                                           int stride,
                                           int padding) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc = blockIdx.z;

    if (ow >= W_out || oh >= H_out) {
        return;
    }

    const float* plane_in =
        input + static_cast<size_t>(nc) * static_cast<size_t>(H) * static_cast<size_t>(W);
    float* plane_out =
        output + static_cast<size_t>(nc) * static_cast<size_t>(H_out) * static_cast<size_t>(W_out);

    const int h_start = oh * stride - padding;
    const int w_start = ow * stride - padding;

    float max_val = -INFINITY;
    for (int kh = 0; kh < kernel_size; ++kh) {
        const int ih = h_start + kh;
        if (ih >= 0 && ih < H) {
            const float* in_row = plane_in + static_cast<size_t>(ih) * static_cast<size_t>(W);
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int iw = w_start + kw;
                if (iw >= 0 && iw < W) {
                    max_val = fmaxf(max_val, in_row[iw]);
                }
            }
        }
    }

    plane_out[static_cast<size_t>(oh) * static_cast<size_t>(W_out) + static_cast<size_t>(ow)] =
        max_val;
}

}  // namespace

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input,
                      float* output,
                      int N,
                      int C,
                      int H,
                      int W,
                      int kernel_size,
                      int stride,
                      int padding) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || kernel_size <= 0 || stride <= 0) {
        return;
    }

    const int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    if (H_out <= 0 || W_out <= 0) {
        return;
    }

    const int nc_total = N * C;
    const dim3 block(kBlockX, kBlockY, 1u);
    // Split Z if N*C > 65535
    for (int z_base = 0; z_base < nc_total; z_base += 65535) {
        const int z_chunk = (nc_total - z_base < 65535) ? (nc_total - z_base) : 65535;
        const dim3 grid(static_cast<unsigned>((W_out + kBlockX - 1) / kBlockX),
                        static_cast<unsigned>((H_out + kBlockY - 1) / kBlockY),
                        static_cast<unsigned>(z_chunk));
        const size_t in_off =
            static_cast<size_t>(z_base) * static_cast<size_t>(H) * static_cast<size_t>(W);
        const size_t out_off =
            static_cast<size_t>(z_base) * static_cast<size_t>(H_out) * static_cast<size_t>(W_out);
        max_pool2d_baseline_kernel<<<grid, block>>>(
            input + in_off, output + out_off, H, W, H_out, W_out, kernel_size, stride, padding);
    }
}
