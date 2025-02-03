#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <ranges>
#include <random>
#include <format>
#include <iostream>

#define C(ans) { chk_impl((ans), __FILE__, __LINE__); }

static inline void chk_impl(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        throw std::runtime_error{
            std::format("CUDA: {}: {} @ {}:{}\n",
                    cudaGetErrorName(code), cudaGetErrorString(code),
                    file, line) };
    }
}

template <unsigned S = 0> // 0,1,2,3
__device__ __forceinline__
void d_wave0(uint32_t cnt, uint32_t v[4]) {
#pragma unroll
    for (auto o = 0; o < 4; o += 2) {
        if (4 * o >= cnt) return;
        unsigned pa, pb;
        switch (S) {
            case 0: pa = 0x7430u, pb = 0x6521u; break;
            case 1: pa = 0x7520u, pb = 0x6431u; break;
            case 2:
                if (o == 2)
                    pa = 0x7531u, pb = 0x6420u;
                else
                    pa = 0x6420u, pb = 0x7531u;
                break;
            case 3: pa = 0x6420u, pb = 0x7531u; break;
            default: __builtin_unreachable();
        }
        auto cmp = __vcmpgtu4(
                __byte_perm(v[o + 0], v[o + 1], pa),
                __byte_perm(v[o + 0], v[o + 1], pb));
        v[o + 0] = __byte_perm(v[o + 0], 0, 0x3210u ^ ((cmp >>  0) & 0x1111u));
        v[o + 1] = __byte_perm(v[o + 1], 0, 0x3210u ^ ((cmp >> 16) & 0x1111u));
    }
}

template <unsigned S = 1> // 1,2,3
__device__ __forceinline__
void d_wave1(uint32_t cnt, uint32_t v[4]) {
#pragma unroll
    for (auto o = 0; o < 4; o += 2) {
        if (4 * o >= cnt) break;
        unsigned pa, pb;
        switch (S) {
            case 1: pa = 0x7610u, pb = 0x5432u; break;
            case 2:
                if (o == 2)
                    pa = 0x7632u, pb = 0x5410u;
                else
                    pa = 0x5410u, pb = 0x7632u;
                break;
            case 3: pa = 0x5410u, pb = 0x7632u; break;
            default: __builtin_unreachable();
        }
        auto cmp = __vcmpgtu4(
                __byte_perm(v[o + 0], v[o + 1], pa),
                __byte_perm(v[o + 0], v[o + 1], pb));
        v[o + 0] = __byte_perm(__byte_perm(v[o + 0], 0,
                    0x3120u ^ ((cmp >>  0) & 0x2222u)), 0, 0x3120u);
        v[o + 1] = __byte_perm(__byte_perm(v[o + 1], 0,
                    0x3120u ^ ((cmp >> 16) & 0x2222u)), 0, 0x3120u);
    }
    d_wave0<S>(cnt, v);
}

template <unsigned S = 2> // 2,3
__device__ __forceinline__
void d_wave2(uint32_t cnt, uint32_t v[4]) {
    uint32_t cmp, p, q;
    cmp = __vcmpgtu4(v[0], v[1]);
    p = __byte_perm(v[0], v[1], 0x5140u ^ ((cmp >>  0) & 0x4444u));
    q = __byte_perm(v[0], v[1], 0x7362u ^ ((cmp >> 16) & 0x4444u));
    v[0] = __byte_perm(p, q, 0x6420u), v[1] = __byte_perm(p, q, 0x7531u);
    cmp = S == 2 ? __vcmpgtu4(v[3], v[2]) : __vcmpgtu4(v[2], v[3]);
    p = __byte_perm(v[2], v[3], 0x5140u ^ ((cmp >>  0) & 0x4444u));
    q = __byte_perm(v[2], v[3], 0x7362u ^ ((cmp >> 16) & 0x4444u));
    v[2] = __byte_perm(p, q, 0x6420u), v[3] = __byte_perm(p, q, 0x7531u);
    d_wave1<S>(cnt, v);
}

__device__ __forceinline__
void d_wave3(uint32_t cnt, uint32_t v[4]) {
    uint32_t cmp, p, q;
    cmp = __vcmpgtu4(v[0], v[2]);
    p = __byte_perm(v[0], v[2], 0x5140u ^ ((cmp >>  0) & 0x4444u));
    q = __byte_perm(v[0], v[2], 0x7362u ^ ((cmp >> 16) & 0x4444u));
    v[0] = __byte_perm(p, q, 0x6420u), v[2] = __byte_perm(p, q, 0x7531u);
    cmp = __vcmpgtu4(v[1], v[3]);
    p = __byte_perm(v[1], v[3], 0x5140u ^ ((cmp >>  0) & 0x4444u));
    q = __byte_perm(v[1], v[3], 0x7362u ^ ((cmp >> 16) & 0x4444u));
    v[1] = __byte_perm(p, q, 0x6420u), v[3] = __byte_perm(p, q, 0x7531u);
    d_wave2<3>(cnt, v);
}

__device__ __forceinline__
void d_sn(uint32_t cnt, uint32_t v[4]) {
    if (cnt <= 0u) return;
    d_wave0(cnt, v);
    d_wave1(cnt, v);
    if (cnt <= 4u) return;
    d_wave2(cnt, v);
    if (cnt <= 8u) return;
    d_wave3(cnt, v);
}

struct blk_t {
    union {
        uint32_t ex[4];
        uint8_t nms[16];
    };
    uint32_t cnt;
};

__global__
void kernel(blk_t *arr, uint32_t max) {
    auto idx = threadIdx.x + (unsigned long long)blockIdx.x * blockDim.x;
    if (idx > max)
        return;
    auto blk = arr[idx];
    d_sn(blk.cnt, blk.ex);
    arr[idx] = blk;
}

int main(int argc, char *argv[]) {
    auto sz = std::stoll(argv[1]);
    auto max_n = std::stoi(argv[2]);
    auto h_arr = new blk_t[sz];
    std::mt19937_64 rnd{};
    std::uniform_int_distribution dist{ 0, 255 };
    std::poisson_distribution ndist{ 9.0 };
    std::cout << std::format("generating {} questions\n", sz);
    for (auto i = 0; i < sz; i++) {
        auto n = ndist(rnd);
        n = std::min(n, max_n);
        h_arr[i].cnt = n;
        for (auto k = 0; k < 16; k++)
            h_arr[i].nms[k] = k < n ? dist(rnd) : 0xff;
        if (sz < 50) {
            std::cout << std::format("h_arr[{}] = {{", i);
            for (auto k = 0; k < 16; k++)
                std::cout << std::format(" {:3d}", h_arr[i].nms[k]);
            std::cout << std::format(" }}\n");
        }
    }
    blk_t *d_arr;
    C(cudaMalloc(&d_arr, sz * sizeof(blk_t)));
    C(cudaMemcpy(d_arr, h_arr, sz * sizeof(blk_t), cudaMemcpyHostToDevice));
    kernel<<<(sz + 512 - 1) / 512, 512>>>(d_arr, sz);
    C(cudaDeviceSynchronize());
    C(cudaMemcpy(h_arr, d_arr, sz * sizeof(blk_t), cudaMemcpyDeviceToHost));
    std::cout << std::format("validating {} questions\n", sz);
    for (auto i = 0; i < sz; i++) {
        if (!std::ranges::is_sorted(h_arr[i].nms) || sz < 50) {
            std::cout << std::format("h_arr[{}] = {{", i);
            for (auto k = 0; k < 16; k++)
                if (k < max_n)
                    std::cout << std::format(" {:3d}", h_arr[i].nms[k]);
                else
                    std::cout << std::format(" {:02x}", h_arr[i].nms[k]);
            std::cout << std::format(" }}");
        }
        if (!std::ranges::is_sorted(h_arr[i].nms))
            std::cout << std::format("  WRONG!!!\n");
        else if (sz < 50)
            std::cout << std::format("\n");
    }
}
