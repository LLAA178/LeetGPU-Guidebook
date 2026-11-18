// Example (Tesla T4): nvcc -O3 -arch=sm_75 solution.cu value-clipping-benchmark.cu -o bench
#ifndef ONLINE_JUDGE
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" void solve(const float*, float*, float, float, int);

namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void sweep_cache(unsigned* data, int words) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < words) data[i] += 1;
}

constexpr int kSize = 100000;
constexpr int kGuard = 4;  // Preserve 16-byte alignment.
bool verify(const std::vector<float>& input, const std::vector<float>& result,
            int n, float lo, float hi) {
    for (int i = 0; i < static_cast<int>(result.size()); ++i) {
        int index = i - kGuard;
        if (index < 0 || index >= n) {
            if (std::isnan(result[i])) continue;
        } else if (result[i] == std::min(hi, std::max(lo, input[index]))) {
            continue;
        }
        std::fprintf(stderr, "Mismatch at output index %d for N=%d\n", index, n);
        return false;
    }
    return true;
}
}  // namespace

int main() {
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0), "device properties");
    int words = std::max(4 * 1024 * 1024, device.l2CacheSize);
    unsigned* cache;
    float *input, *storage;
    std::vector<float> host(kSize), result(kSize + 2 * kGuard);
    check(cudaMalloc(&cache, words * sizeof(unsigned)), "allocate cache sweep");
    check(cudaMemset(cache, 0, words * sizeof(unsigned)), "initialize cache sweep");
    check(cudaMalloc(&input, kSize * sizeof(float)), "allocate input");
    check(cudaMalloc(&storage, result.size() * sizeof(float)), "allocate output");
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-100.f, 100.f);
    auto sample = [&](int n, float lo, float hi) {
        std::generate_n(host.begin(), n, [&]() { return dist(rng); });
        host[0] = lo;
        if (n > 1) host[1] = hi;
        check(cudaMemcpy(input, host.data(), n * sizeof(float), cudaMemcpyHostToDevice), "upload");
        check(cudaMemset(storage, 0xff, result.size() * sizeof(float)), "poison output");
        sweep_cache<<<(words + 255) / 256, 256>>>(cache, words);
        check(cudaDeviceSynchronize(), "cache sweep");
        const auto start = std::chrono::steady_clock::now();
        solve(input, storage + kGuard, lo, hi, n);
        check(cudaDeviceSynchronize(), "solve");
        const auto stop = std::chrono::steady_clock::now();
        check(cudaMemcpy(result.data(), storage, result.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), "download");
        if (!verify(host, result, n, lo, hi)) std::exit(EXIT_FAILURE);
        return std::chrono::duration<double, std::micro>(stop - start).count();
    };
    for (int n : {1, 2, 3, 4, 5, 255, 256, 257, 1023, 1024, 1025, 99999}) {
        sample(n, -2.f, 3.f);
        sample(n, 1.f, 1.f);
    }
    std::printf("%s | 24 correctness cases passed | N=%d | cache sweep %zu MiB\n",
                device.name, kSize, words * sizeof(unsigned) >> 20);
    for (int batch = 0; batch < 3; ++batch) {
        std::vector<double> times;
        for (int i = 0; i < 17; ++i) {
            double us = sample(kSize, -51.24f, 39.51f);
            if (i >= 2) times.push_back(us);
        }
        std::sort(times.begin(), times.end());
        std::printf("batch %d | 15 samples | median %.3f us\n", batch + 1, times[7]);
    }
    check(cudaFree(cache), "free cache sweep");
    check(cudaFree(storage), "free output");
    check(cudaFree(input), "free input");
}
#endif
