#include "../src/util.cuh"
#include "../src/record.cu"

#include <random>
#include <format>
#include <chrono>
#include <iostream>

template <unsigned H>
__global__
void kernel(
        R *icfg, uint32_t max, uint8_t ea,
        RX *ocfg) {
    auto idx = threadIdx.x + (unsigned long long)blockIdx.x * blockDim.x;
    if (idx > max)
        return;
    auto cfg = parse_R<H>(icfg[idx], ea);
    // if ((cfg.empty_area & 0xffu) != ea)
    //     return;
    cfg.empty_area <<= 8;
    ocfg[idx] = assemble_R<H>(cfg);
    // ocfg[idx].xaL = cfg.empty_area;
    // ocfg[idx].xaH = cfg.empty_area >> 32;
    // ocfg[idx].ea = cfg.nm_cnt;
}

template <typename ... TArgs>
static void launch(unsigned b, unsigned t, unsigned height,
        TArgs && ... args) {
    if (height == 8)
        ; // kernel<8><<<b, t>>>(std::forward<TArgs>(args)...);
    else if (height == 7)
        kernel<7><<<b, t>>>(std::forward<TArgs>(args)...);
    else if (height == 6)
        kernel<6><<<b, t>>>(std::forward<TArgs>(args)...);
    else if (height == 5)
        kernel<5><<<b, t>>>(std::forward<TArgs>(args)...);
    else if (height == 4)
        kernel<4><<<b, t>>>(std::forward<TArgs>(args)...);
    else if (height == 3)
        kernel<3><<<b, t>>>(std::forward<TArgs>(args)...);
    else if (height == 2)
        kernel<2><<<b, t>>>(std::forward<TArgs>(args)...);
    else if (height == 1)
        kernel<1><<<b, t>>>(std::forward<TArgs>(args)...);
    else
        throw std::runtime_error{ std::format("height {} not supported", height) };
}

int main(int argc, char *argv[]) {
    auto sz = std::stoll(argv[1]);
    auto h = std::stoll(argv[2]);
    R *icfg;
    RX *ocfg;
    C(cudaSetDevice(1));
    C(cudaMallocManaged(&icfg, sz * sizeof(R)));
    C(cudaMallocManaged(&ocfg, sz * sizeof(RX)));
    std::mt19937_64 rnd{};
    std::print("generating {} questions\n", sz);
    for (auto i = 0; i < sz; i++) {
        icfg[i].xaL = rnd();
        icfg[i].xaH = rnd();
        icfg[i].ex0 = rnd();
        icfg[i].ex1 = rnd();
        icfg[i].ex2 = rnd();
    }
    auto t1 = std::chrono::steady_clock::now();
    launch((sz + 512 - 1) / 512, 512, h, icfg, sz, 0xa5, ocfg);
    C(cudaDeviceSynchronize());
    auto t2 = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    if (us < 1000)
        std::print("  => completed in {}us\n", us);
    else if (us < 1000000)
        std::print("  => completed in {:.2f}ms\n", us / 1e3);
    else
        std::print("  => completed in {:.2f}s\n", us / 1e6);
    std::print("validating {} questions\n", sz);
    for (auto i = 0; i < sz; i++) {
        if (ocfg[i].ea != 0xa5 || static_cast<R>(ocfg[i]) != icfg[i]) {
            std::print("mismatch @ [{}]:\n", i);
            std::print("  xaL = 0x{:08x} vs 0x{:08x}\n", icfg[i].xaL, ocfg[i].xaL);
            std::print("  xaH = 0x{:08x} vs 0x{:08x}\n", icfg[i].xaH, ocfg[i].xaH);
            std::print("  ex0 = 0x{:08x} vs 0x{:08x}\n", icfg[i].ex0, ocfg[i].ex0);
            std::print("  ex1 = 0x{:08x} vs 0x{:08x}\n", icfg[i].ex1, ocfg[i].ex1);
            std::print("  ex2 = 0x{:08x} vs 0x{:08x}\n", icfg[i].ex2, ocfg[i].ex2);
            std::print("  ea = 0x{:02x}\n", ocfg[i].ea);
        }
    }
}
