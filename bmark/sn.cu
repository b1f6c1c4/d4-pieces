#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <ranges>
#include <random>
#include <format>
#include <iostream>
#include <chrono>

#include "../src/sn.cuh"

#define C(ans) { chk_impl((ans), __FILE__, __LINE__); }

static inline void chk_impl(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        throw std::runtime_error{
            std::format("CUDA: {}: {} @ {}:{}\n",
                    cudaGetErrorName(code), cudaGetErrorString(code),
                    file, line) };
    }
}

__device__ __forceinline__
void d_push_slow(uint32_t &cnt, uint32_t v[4], uint32_t nv) {
    auto mask = __vcmpne4(nv, 0xffffffffu);
    auto n = __popc(mask);
    v[cnt / 4] &= (1u << cnt % 4 * 8) - 1u;
    v[cnt / 4] |= nv << cnt % 4 * 8;
    if (cnt / 4 + 1 < 4 && cnt % 4) {
        v[cnt / 4 + 1] = ~((~nv) >> (4 - cnt % 4) * 8);
    }
    cnt += n / 8;
}

struct blk_t {
    union {
        uint32_t ex[4];
        uint8_t nms[16];
    };
    union {
        uint32_t n[2];
        uint8_t ns[8];
    };
    uint32_t cnt;
};

__global__
void kernel(blk_t *arr, uint32_t max) {
    auto idx = threadIdx.x + (unsigned long long)blockIdx.x * blockDim.x;
    if (idx > max)
        return;
    auto blk = arr[idx];
    d_push(blk.cnt, blk.ex, blk.n[0]);
    d_push(blk.cnt, blk.ex, blk.n[1]);
    d_sn(blk.cnt, blk.ex);
    arr[idx] = blk;
    if (!d_uniq_chk(blk.cnt, blk.ex))
        arr[idx].cnt = 233;
}

int main(int argc, char *argv[]) {
    auto sz = std::stoll(argv[1]);
    auto max_n = std::stoi(argv[2]);
    auto h_arr = new blk_t[sz];
    std::mt19937_64 rnd{};
    std::uniform_int_distribution dist{ 0, 254 };
    std::uniform_int_distribution xdist{ 0, 8 };
    std::poisson_distribution ndist{ 9.0 };
    std::print("generating {} questions\n", sz);
    for (auto i = 0; i < sz; i++) {
        auto n = ndist(rnd);
        n = std::min(n, max_n);
        auto x = xdist(rnd);
        x = std::min(x, 16 - n);
        h_arr[i].cnt = n;
        for (auto k = 0; k < 16; k++)
            h_arr[i].nms[k] = k < n ? dist(rnd) : 0xff;
        for (auto k = 0; k < 8; k++)
            h_arr[i].ns[k] = k < x ? dist(rnd) : 0xff;
        if (sz < 50) {
            std::print("h_arr[{}] = {{", i);
            for (auto k = 0; k < 4; k++)
                std::print(" 0x{:08x}", h_arr[i].ex[k]);
            std::print(" }}");
            std::print(" + 0x{:08x}", h_arr[i].n[0]);
            std::print(" + 0x{:08x}\n", h_arr[i].n[1]);
        }
    }
    blk_t *d_arr;
    C(cudaMalloc(&d_arr, sz * sizeof(blk_t)));
    C(cudaMemcpy(d_arr, h_arr, sz * sizeof(blk_t), cudaMemcpyHostToDevice));
        auto t1 = std::chrono::steady_clock::now();
    kernel<<<(sz + 512 - 1) / 512, 512>>>(d_arr, sz);
    C(cudaDeviceSynchronize());
        auto t2 = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        if (us < 1000)
            std::print("  => cpmpleted in {}us\n", us);
        else if (us < 1000000)
            std::print("  => cpmpleted in {:.2f}ms\n", us / 1e3);
        else
            std::print("  => cpmpleted in {:.2f}s\n", us / 1e6);
    C(cudaMemcpy(h_arr, d_arr, sz * sizeof(blk_t), cudaMemcpyDeviceToHost));
    std::print("validating {} questions\n", sz);
    for (auto i = 0; i < sz; i++) {
        auto uniq = true;
        for (auto k = 1; k < 16; k++)
            if (h_arr[i].nms[k] != 0xff && h_arr[i].nms[k - 1] == h_arr[i].nms[k])
                uniq = false;
        if (uniq != (233 != h_arr[i].cnt) || !std::ranges::is_sorted(h_arr[i].nms) || sz < 50) {
            std::print("h_arr[{}] = {{", i);
            for (auto k = 0; k < 4; k++)
                std::print(" 0x{:08x}", h_arr[i].ex[k]);
            std::print(" }} {:3}", h_arr[i].cnt);
        }
        if (uniq != (233 != h_arr[i].cnt))
            std::print("  uniq: {} vs {}", uniq, (233 != h_arr[i].cnt));
        if (!std::ranges::is_sorted(h_arr[i].nms))
            std::print("  WRONG!!!\n");
        else if (sz < 50)
            std::print("\n");
    }
}
